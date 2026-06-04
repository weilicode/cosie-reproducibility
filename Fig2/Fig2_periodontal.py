import os
import torch
import scanpy as sc
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData
import anndata as ad
from sklearn.preprocessing import normalize


def cluster_and_visualize_superpixel(
    final_embeddings,
    data_dict,
    n_clusters,
    mode="joint",  # 'joint' or 'independent' or "defined"
    defined_labels=None,
    vis_basis="spatial",
    random_state=0,
    colormap=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_path=None,
    dpi=300,
    remove_title = False,
    remove_legend = False,
    remove_spine = False,
    figscale = 35
):
    """
    Perform clustering on superpixel embeddings across multiple tissue sections and visualize the results.

    Supports three clustering modes:
    
    - 'joint': All sections' embeddings are clustered together.
    
    - 'independent': Each section is clustered independently.
    
    - 'defined': Uses user-specified cluster labels.


    Parameters
    ----------
    final_embeddings : dict
        Dictionary of {section_id: np.ndarray} representing cell embeddings.

    data_dict : dict
        Dictionary of {modality: list of AnnData}, where each AnnData contains spatial coordinates.

    n_clusters : int
        Number of clusters to generate.

    mode : str, default "joint"
        Clustering mode: "joint", "independent", or "defined".

    defined_labels : dict or None
        Required if mode is "defined". A dictionary of {section_id: np.ndarray of cluster labels}.

    vis_basis : str, default "spatial"
        Key in `obsm` indicating spatial coordinates.

    random_state : int, default 0
        Random seed for KMeans clustering.

    colormap : str or list or None
        Color map used to assign RGB colors to clusters.

    swap_xy : bool, default False
        Whether to swap x and y coordinates.

    invert_x : bool, default False
        Whether to flip the image horizontally.

    invert_y : bool, default False
        Whether to flip the image vertically.

    offset : bool, default False
        Whether to shift coordinates to (0, 0).

    save_path : str or None, default None
        If specified, saves the figure(s) with this filename prefix.

    dpi : int, default 300
        DPI for the saved figure.

    remove_title : bool, default False
        Whether to remove figure title.

    remove_legend : bool, default False
        Whether to remove cluster legend.

    remove_spine : bool, default False
        Whether to remove axis borders.

    figscale : int, default 35
        Controls image figure size.

    Returns
    -------
    cluster_labels : dict
        Dictionary of {section_id: np.ndarray of cluster labels}.
    """
    import numpy as np
    from sklearn.cluster import KMeans
    import os

    adata_list = []
    embeddings = []
    coords_all = []
    section_names = []

    for section, embedding in final_embeddings.items():
        idx = int(section[1:]) - 1
        for modality, adata_list_per_mod in data_dict.items():
            if idx < len(adata_list_per_mod) and adata_list_per_mod[idx] is not None:
                adata = adata_list_per_mod[idx]
                adata_list.append(adata)
                embeddings.append(embedding)
                coords = adata.obsm[vis_basis].copy()
                if swap_xy:
                    coords = coords[:, [1, 0]]
                coords = coords.astype(int)
                if offset:
                    offset_value = coords.min(axis=0)     
                    coords -= offset_value               
                coords_all.append(coords)
                section_names.append(section)
                break

    cluster_labels = {}

    if mode == "joint":
        print("Perform joint clustering...")
        combined_embedding = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        all_clusters = kmeans.fit_predict(combined_embedding)
        start = 0
        for section, emb in zip(section_names, embeddings):
            end = start + emb.shape[0]
            cluster_labels[section] = all_clusters[start:end]
            start = end
    elif mode == "independent":
        print("Perform independent clustering...")
        for section, emb in zip(section_names, embeddings):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            cluster_labels[section] = kmeans.fit_predict(emb)
    elif mode == 'defined':
        if defined_labels is None:
            raise ValueError("If mode='defined', you must provide `defined_labels`.")
        cluster_labels = defined_labels
    else:
        raise ValueError("mode must be 'joint' or 'independent'")

    for section, coords, labels in zip(section_names, coords_all, cluster_labels.values()):
        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), fill_value=-1, dtype=int)
        for (y, x), label in zip(coords, labels):
            image[y, x] = label
        if invert_x:
            image = image[:, ::-1]
        if invert_y:
            image = image[::-1, :]
        section_save_path = None
        if save_path:
            base, ext = os.path.splitext(save_path)
            section_save_path = f"{base}_section_{section}{ext or '.png'}"
        plot_histology_clusters(
            he_clusters_image=image,
            num_he_clusters=n_clusters,
            section_title=f"Section {section} ({mode})",
            colormap=colormap,
            save_path=section_save_path,
            dpi=dpi,
            figscale = figscale,
            remove_title = remove_title,
            remove_legend = remove_legend,
            remove_spine=remove_legend, 
        )

    return cluster_labels



def plot_histology_clusters(he_clusters_image,
                            num_he_clusters,
                            section_title=None,
                            colormap=None,
                            save_path=None,
                            figscale = 35,
                            remove_title = False,
                            remove_legend = False,
                            remove_spine=False, 
                            dpi=300):
    """
    Visualize cluster maps from 2D cluster masks.

    Parameters
    ----------
    he_clusters_image : np.ndarray
        2D array of shape (H, W) where each pixel holds an integer cluster ID.

    num_he_clusters : int
        Total number of clusters (used for color assignment).

    section_title : str or None, optional
        Title shown on the figure.

    colormap : str, list, or None, optional
        Colormap for cluster coloring. If None, a default color list is used.

    save_path : str or None, optional
        Path to save the resulting image. If None, no image is saved.

    figscale : int, default 35
        Controls image figure size.

    remove_title : bool, default False
        Whether to remove the title.

    remove_legend : bool, default False
        Whether to remove the cluster legend.

    remove_spine : bool, default False
        Whether to remove the axis frame/spines.

    dpi : int, default 300
        DPI of saved image.

    Returns
    -------
    None
    """

    if colormap is None:
        color_list = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                      [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                      [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                      [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                      [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                      [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                      [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]

    elif isinstance(colormap, list):
        color_list = colormap

    else:
        cmap = cm.get_cmap(colormap)
        color_list = [ [int(255 * c) for c in to_rgb(cmap(i))] for i in range(len(cmap.colors)) ]

    image_rgb = 255 * np.ones([he_clusters_image.shape[0], he_clusters_image.shape[1], 3])
    for cluster in range(num_he_clusters):
        image_rgb[he_clusters_image == cluster] = color_list[cluster]
    image_rgb = np.array(image_rgb, dtype='uint8')

    plt.figure(figsize=(he_clusters_image.shape[1] // figscale, he_clusters_image.shape[0] // figscale))
    if remove_title:
        plt.title("")
    else:
        title = section_title if section_title else "Histology Clusters"
        plt.title(title, fontsize=18)
    plt.imshow(image_rgb, interpolation='none')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not remove_legend:
        legend_elements = [patches.Patch(facecolor=np.array(color_list[i]) / 255,
                                         label=f'Cluster {i}')
                           for i in range(num_he_clusters)]
        plt.legend(handles=legend_elements,
                   bbox_to_anchor=(1.05, 1),
                   loc='upper left',
                   borderaxespad=0.,
                   fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()



def create_normalized_adata(adata):
    """
    Create a new AnnData object by min-max scaling the expression matrix of the input `adata`.

    The values in `.X` are scaled to the range [0, 1]. If `.X` is stored in sparse format,
    it will be converted to a dense NumPy array before normalization. The original `.obs`,
    `.var`, and `.obsm` fields are preserved in the new AnnData object.

    Parameters
    ----------
    
    adata : AnnData
        The input AnnData object containing expression data in `.X`.

    Returns
    -------
    
    new_adata : AnnData
        A new AnnData object with min-max normalized `.X`, while retaining the original `.obs`, `.var`, and `.obsm` attributes.
    """
    dense_X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    scaler = MinMaxScaler()
    normalized_X = scaler.fit_transform(dense_X)
    new_adata = AnnData(X=normalized_X, obs=adata.obs.copy(), var=adata.var.copy(), obsm=adata.obsm.copy())
    return new_adata


def prepare_image(adata, molecule_name, basis, swap_xy, invert_x, invert_y, offset):
    """
    Prepare a 2D image from molecule expression and spatial coordinates in an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing spatial coordinates in `obsm[basis]` and molecule expression in `X`.

    molecule_name : str
        Name of the molecule to visualize.

    basis : str
        The key in `obsm` to use for spatial coordinates (e.g., "spatial").

    swap_xy : bool
        Whether to swap x and y coordinates.

    invert_x : bool
        Whether to flip the image horizontally.

    invert_y : bool
        Whether to flip the image vertically.

    offset : bool
        Whether to shift coordinates so that the minimum becomes (0, 0).

    Returns
    -------
    image : np.ndarray
        2D array of shape (height, width) representing the molecule intensity at each spatial location.
    """
    
    coords = adata.obsm[basis].copy()
    if swap_xy:
        coords = coords[:, [1, 0]]
    coords = coords.astype(int)
    if offset:
        offset_value = coords.min(axis=0)
        coords -= offset_value 


    values = adata[:, molecule_name].X
    if hasattr(values, "toarray"):
        values = values.toarray().flatten()
    else:
        values = np.array(values).flatten()

    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), np.nan, dtype=float)
    for (y, x), val in zip(coords, values):
        image[y, x] = val

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    return image



def plot_marker_comparison_superpixel(
    molecule_name: str,
    adata1,
    adata2,
    section1_label: str = 'Section 1',
    section2_label: str = 'Section 2',
    basis: str = 'spatial',
    colormap: str = "turbo",
    plot_style: str = "original",
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
    offset: bool = False,
    figscale: int = 35,
    dpi: int = 300,
    remove_title: bool = False,     
    remove_spine: bool = False,    
    remove_legend: bool = False,      
    save_path: str = None
):
    """
    Plot side-by-side spatial expression comparison of a target molecule at the superpixel level.

    Parameters
    ----------
    molecule_name : str
        Name of the molecule to visualize.

    adata1 : AnnData
        First AnnData object with molecule expression and spatial coordinates.

    adata2 : AnnData
        Second AnnData object with molecule expression and spatial coordinates.

    section1_label : str, default 'Section 1'
        Title label for the first section.

    section2_label : str, default 'Section 2'
        Title label for the second section.

    basis : str, default 'spatial'
        The key in `obsm` specifying spatial coordinates.

    colormap : str, default "turbo"
        Name of matplotlib colormap to use for intensity.

    plot_style : str, default "original"
        If "equal", enforce equal aspect ratio for square spatial representation.

    swap_xy : bool, default False
        Whether to swap x and y axes.

    invert_x : bool, default False
        Whether to flip the image horizontally.

    invert_y : bool, default False
        Whether to flip the image vertically.

    offset : bool, default False
        Whether to shift coordinates to align to (0, 0) origin.

    figscale : int, default 35
        Scaling factor for figure size.

    dpi : int, default 300
        Dots-per-inch for saved figure resolution.

    remove_title : bool, default False
        Whether to remove plot titles.

    remove_spine : bool, default False
        Whether to remove axes spines.

    remove_legend : bool, default False
        Whether to remove colorbar.

    save_path : str or None, default None
        If provided, save the figure to this path.

    Returns
    -------
    None
    """

    img1 = prepare_image(adata1, molecule_name, basis, swap_xy, invert_x, invert_y, offset)
    img2 = prepare_image(adata2, molecule_name, basis, swap_xy, invert_x, invert_y, offset)


    figsize1 = (img1.shape[1] / figscale, img1.shape[0] / figscale)
    figsize2 = (img2.shape[1] / figscale, img2.shape[0] / figscale)
    figsize = (figsize1[0] + figsize2[0], max(figsize1[1], figsize2[1]))

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, img, title in zip(axes, [img1, img2], [section1_label, section2_label]):
        im = ax.imshow(img, cmap=colormap, interpolation='none')
        if not remove_title:
            ax.set_title(f"{title} - {molecule_name}", fontsize=16)
        else:
            ax.set_title("")
        ax.set_xticks([])
        ax.set_yticks([])
        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)
        if plot_style == "equal":
            ax.set_aspect("equal")

        if not remove_legend:
            cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)  

    if save_path:
        base, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".png"
        save_path = base + ext
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving marker comparison to: {save_path}")
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()


def truncate_gene_expression_smartclip(adata, gene, lower=0, upper=99):
    """
    Truncate expression data based on non-zero values to reduce the impact of extreme outliers in visualization.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    
    gene : str
        Gene name (must be in adata.var_names).
    
    lower : float, optional
        Lower percentile to truncate. Default is 0.
    
    upper : float, optional
        Upper percentile to truncate. Default is 99.

    Returns
    -------
    AnnData
        New AnnData object with truncated expression values in .X.
    """

    values = adata[:, gene].X
    if hasattr(values, "toarray"):
        values = values.toarray().flatten()
    else:
        values = np.array(values).flatten()

    nonzero_values = values[values > 0]
    num_nonzero = len(nonzero_values)
    num_total = len(values)

    if num_nonzero == 0:
        values_clipped = values
    else:
        vmin = np.percentile(nonzero_values, lower) if lower > 0 else 0
        vmax = np.percentile(nonzero_values, upper)
        values_clipped = np.clip(values, vmin, vmax)

    new_adata = ad.AnnData(
        X=values_clipped[:, np.newaxis],
        obs=adata.obs.copy(),
        var=adata[:, gene].var.copy(),
        obsm=adata.obsm.copy()
    )

    return new_adata




if __name__ == "__main__":
    
        
    adata1_rna = sc.read_h5ad('./Fig2_data/Periodontal/adata_rna_s1.h5ad')
    adata2_rna = sc.read_h5ad('./Fig2_data/Periodontal/adata_rna_s2.h5ad')
    adata2_adt = sc.read_h5ad('./Fig2_data/Periodontal/adata_protein_s2.h5ad')
    
    adata1_rna.var_names_make_unique()
    adata2_rna.var_names_make_unique()
    
    adata1_he = sc.AnnData(X=adata1_rna.obsm['UNI_feature'])
    adata1_he.obsm['spatial'] = adata1_rna.obsm['spatial'].copy()
    
    
    data_dict = {
            'RNA': [adata1_rna, adata2_rna],
            'HE': [adata1_he, None],
            'Protein': [None, adata2_adt],
        }
    
    
    
    s1_emb = np.load('./Fig2_data/Periodontal/COSIE_result/s1_embedding.npy')
    s2_emb = np.load('./Fig2_data/Periodontal/COSIE_result/s2_embedding.npy')
    final_embeddings = {'s1':s1_emb, 's2':s2_emb}
    
    color_map =  [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                  [140,86,75],[227,119,194],[127,127,127],[174,199,232],
                 [107,174,214] ,[188,189,34],[255,187,120],[152,223,138],
                  [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                  [199,199,199],[219,219,141],[158,218,229], [137,69,133],
                  [31, 119, 180],[135, 206, 250]]
    
    cluster_label = cluster_and_visualize_superpixel(final_embeddings, 
                                                     data_dict,
                                                     n_clusters=12,
                                                     mode="joint", 
                                                     vis_basis="spatial",  
                                                     colormap = color_map, 
                                                     save_path = 'COSIE_clustering_periodontal.jpg',
                                                     dpi = 500,
                                                     figscale = 120)
    
    
    
    df_emb = pd.read_csv('./Fig2_data/Periodontal/StabMap_result/periodontal_harmony_embedding.csv')
    df_emb.rename(columns={df_emb.columns[0]: "cell_id"}, inplace=True)
    df_s1 = df_emb[df_emb["cell_id"].str.startswith("s1_")].reset_index(drop=True)
    df_s2 = df_emb[df_emb["cell_id"].str.startswith("s2_")].reset_index(drop=True)
    
    stabmap_embedding_s1 = df_s1.iloc[:, 1:].values  # shape: (num_s1_cells, num_PCs)
    stabmap_embedding_s2 = df_s2.iloc[:, 1:].values  # shape: (num_s2_cells, num_PCs)

    final_embeddings_stabmap = {
        's1':stabmap_embedding_s1,
        's2':stabmap_embedding_s2,
    }

    color_map_stabmap = [
        [107,174,214],  # 8
        [174,199,232],   # 9
        [140,86,75],   # 4
        [148,103,189],  # 3
        [214,39,40],    # 2
        [44,160,44],    # 1
        [227,119,194],  # 5
        [188,189,34],   # 7
        [127,127,127],   # 6
        [255,127,14],   # 0
        [255,187,120],   # 10
        [152,223,138],   # 11       
    ]
    
    cluster_label_stabmap = cluster_and_visualize_superpixel(final_embeddings_stabmap, 
                                                     data_dict,
                                                     n_clusters=12,
                                                     mode="joint", 
                                                     vis_basis="spatial",  
                                                     colormap = color_map_stabmap, 
                                                     save_path = 'StabMap_clustering_periodontal.jpg',
                                                     dpi = 500,
                                                     remove_title = True,
                                                     remove_legend = True,
                                                     remove_spine = True,
                                                     figscale = 120)
    
    
    ### Visualization of protein prediction
    adata1_protein_predicted = sc.read_h5ad('./Fig2_data/Periodontal/COSIE_result/adata1_protein_imputed.h5ad')
    adata1_protein_predicted_norm = create_normalized_adata(adata1_protein_predicted)
    
    for chosed_protein in ['KRT14', 'COLA1', 'CD31', 'CD68']:
        adata_truncate = truncate_gene_expression_smartclip(adata2_adt, chosed_protein, lower=0, upper=99)
        adata_truncate_norm = create_normalized_adata(adata_truncate)
        plot_marker_comparison_superpixel(
            chosed_protein,
            adata1_protein_predicted_norm,
            adata_truncate_norm,
            section1_label = 'Section1 predicted',
            section2_label = 'Section2 observed',
            basis = 'spatial',
            colormap = "turbo",
            figscale = 120,
            save_path = './COSIE_prediction/{}.jpg'.format(chosed_protein),
            dpi = 500)
    
    







##