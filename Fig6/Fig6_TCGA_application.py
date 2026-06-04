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
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from scipy.stats import wilcoxon
import pickle
import matplotlib as mpl


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


import numpy as np

def merge_clusters_to_new_ids(cluster_label_dict, merge_groups):
    """
    Merge specified cluster IDs into new classes with IDs starting from current max + 1.

    Parameters
    ----------
    cluster_label_dict : dict
        Dictionary of {section: np.ndarray of cluster labels}.
    merge_groups : list of list
        List of groups to be merged, e.g., [[1, 13, 22], [10, 15]].

    Returns
    -------
    new_label_dict : dict
        Updated cluster label dictionary with merged labels.
    """
    # Find global max label to start assigning new IDs
    all_labels = np.concatenate(list(cluster_label_dict.values()))
    current_max = int(all_labels.max())
    next_id = current_max + 1

    # Build merge map: old ID → new merged ID
    merge_map = {}
    for group in merge_groups:
        for cid in group:
            merge_map[cid] = next_id
        next_id += 1

    # Apply mapping to each section
    new_label_dict = {}
    for section, labels in cluster_label_dict.items():
        labels = np.array(labels)
        mapped_labels = np.array([merge_map.get(lbl, lbl) for lbl in labels])
        new_label_dict[section] = mapped_labels

    return new_label_dict

def relabel_clusters_sequentially(cluster_label_dict):
    """
    Relabel all cluster IDs across sections to contiguous integers starting from 0,
    preserving structure.

    Parameters
    ----------
    cluster_label_dict : dict
        Dictionary of {section: np.ndarray of cluster labels}.

    Returns
    -------
    relabeled_dict : dict
        Dictionary with same keys, but cluster labels relabeled to 0, 1, 2, ...
    """
    # 收集所有唯一的 cluster ID
    all_labels = np.concatenate(list(cluster_label_dict.values()))
    unique_labels = sorted(np.unique(all_labels))
    
    # 创建 old → new 映射字典
    relabel_map = {old: new for new, old in enumerate(unique_labels)}

    # 应用映射
    relabeled_dict = {}
    for section, labels in cluster_label_dict.items():
        relabeled_labels = np.vectorize(relabel_map.get)(labels)
        relabeled_dict[section] = relabeled_labels

    return relabeled_dict




def highlight_joint_clusters_all_sections(
    cluster_labels,
    data_dict,
    n_clusters,
    highlight_labels,
    vis_basis="spatial",
    colormap=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_dir=None,
    figscale=35,
    dpi=300,
    remove_title=True,
    remove_legend=True,
    remove_spine=True,
    bg_color = [200, 200, 200]
):
    """
    Visualize and highlight specified clusters across all tissue sections. For each section, this function renders a spatial plot of cell clusters, highlighting the clusters specified in `highlight_labels` using distinct colors, while rendering all other clusters in a uniform background color. 

    Parameters
    ----------
    cluster_labels : dict
        Dictionary of {section_id: np.ndarray of cluster labels} for each section.

    data_dict : dict
        Dictionary of input data.

    n_clusters : int
        Total number of clusters.

    highlight_labels : list of int
        List of cluster labels to highlight. Other clusters are rendered with background color.

    vis_basis : str, default "spatial"
        Key in `obsm` specifying the coordinate basis to use.

    colormap : str or list or None, default None
        Name of matplotlib colormap to use, or a list of RGB values. If None, uses a default palette.

    swap_xy : bool, default False
        Whether to swap x and y axes in the coordinate system.

    invert_x : bool, default False
        Whether to flip the image horizontally.

    invert_y : bool, default False
        Whether to flip the image vertically.

    offset : bool, default False
        Whether to shift coordinates to (0, 0) minimum before rendering.

    save_dir : str or None, default None
        If provided, saves each figure as a JPEG to the specified directory.

    figscale : float, default 35
        Controls the scaling of the figure size.

    dpi : int, default 300
        Resolution of the saved figure.

    remove_title : bool, default True
        Whether to remove the figure title.

    remove_legend : bool, default True
        Whether to remove the cluster legend from the plot.

    remove_spine : bool, default True
        Whether to remove the axis spines (borders around the plot).

    bg_color : list of int, default [200, 200, 200]
        RGB color used for non-highlighted clusters.

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    from matplotlib.colors import to_rgb
    from matplotlib import cm

    if colormap is None:
        base_colors = [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                       [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                       [23,190,207],[174,199,232],[255,187,120],[152,223,138],
                       [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                       [199,199,199],[219,219,141],[158,218,229],[16,60,90],
                       [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
                       [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]]
    elif isinstance(colormap, list):
        base_colors = colormap
    else:
        cmap = cm.get_cmap(colormap)
        base_colors = [[int(255 * c) for c in to_rgb(cmap(i))] for i in range(len(cmap.colors))]

    

    for section, labels in cluster_labels.items():
        idx = int(section[1:]) - 1
        coords = None
        for modality, adata_list in data_dict.items():
            if idx < len(adata_list) and adata_list[idx] is not None:
                coords = adata_list[idx].obsm[vis_basis].copy()
                if swap_xy:
                    coords = coords[:, [1, 0]]
                coords = coords.astype(int)
                if offset:
                    coords -= coords.min(axis=0)
                break
        if coords is None:
            print(f"Warning: Coordinates not found for section {section}.")
            continue

        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), fill_value=-1, dtype=int)
        for (y, x), label in zip(coords, labels):
            image[y, x] = label
        if invert_x:
            image = image[:, ::-1]
        if invert_y:
            image = image[::-1, :]

        color_list = []
        for i in range(n_clusters):
            if i in highlight_labels:
                color_list.append(base_colors[i % len(base_colors)])
            else:
                color_list.append(bg_color)

        image_rgb = 255 * np.ones((image.shape[0], image.shape[1], 3))
        for cluster in range(n_clusters):
            image_rgb[image == cluster] = color_list[cluster]
        image_rgb = image_rgb.astype("uint8")

        fig, ax = plt.subplots(figsize=(image.shape[1] // figscale, image.shape[0] // figscale))

        if not remove_title:
            ax.set_title(f"Section {section} - Highlighted Clusters", fontsize=18)

        ax.imshow(image_rgb, interpolation='none')
        ax.set_xticks([]); ax.set_yticks([])

        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)

        if not remove_legend:
            legend_elements = [
                patches.Patch(facecolor=np.array(color_list[i]) / 255, label=f'Cluster {i}')
                for i in highlight_labels
            ]
            ax.legend(handles=legend_elements,
                      bbox_to_anchor=(1.05, 1),
                      loc='upper left',
                      borderaxespad=0.,
                      fontsize=12)

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"highlighted_{section}.jpg")
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
        plt.close()





if __name__ == "__main__":
    file_path = './Fig6_data/TCGA_data'
    # adata1_rna = sc.read_h5ad('./COSIE_result_4sections/adata4_gene_imputed.h5ad')
    # adata1_rna.var_names_make_unique()
    adata1_rna_ori = sc.read_h5ad('./Fig6_data/adata_P11_LUAD_Visium_adt_istar.h5ad')
    adata1_he = sc.AnnData(X=adata1_rna_ori.obsm['UNI_feature'])
    adata1_he.obsm['spatial'] = adata1_rna_ori.obsm['spatial'].copy()
    adata2_he = sc.read_h5ad(os.path.join(file_path, 'adata_he_sample2_high_signature.h5ad'))
    adata3_he = sc.read_h5ad(os.path.join(file_path, 'adata_he_sample2_low_signature.h5ad'))

    data_dict = {
        'HE': [adata1_he, adata2_he, adata3_he],
        'RNA': [adata1_rna_ori, None, None],
    }

    s1_embedding = np.load(os.path.join(file_path,'COSIE_result_TCGA_2sections/s1_embedding.npy'))
    s2_embedding = np.load(os.path.join(file_path,'COSIE_result_TCGA_2sections/s2_embedding.npy'))
    s3_embedding = np.load(os.path.join(file_path,'COSIE_result_TCGA_2sections/s3_embedding.npy'))
    
    final_embeddings = {
        's1':s1_embedding, 
        's2':s2_embedding,
        's3':s3_embedding, 
    }

    cluster_label = cluster_and_visualize_superpixel(final_embeddings, 
                                                     data_dict,
                                                     n_clusters=25,
                                                     mode="joint", 
                                                     vis_basis="spatial",  
                                                     save_path = 'COSIE_TCGA_2sections.jpg',
                                                     figscale = 120)

    ## Virtual prediction
    adata2_gene_imputed = sc.read_h5ad(os.path.join(file_path, 'COSIE_result_TCGA_2sections/adata_high2_gene_imputed.h5ad'))
    adata3_gene_imputed = sc.read_h5ad(os.path.join(file_path, 'COSIE_result_TCGA_2sections/adata_low2_gene_imputed.h5ad')) 
    adata2_gene_imputed_norm = create_normalized_adata(adata2_gene_imputed)
    adata3_gene_imputed_norm = create_normalized_adata(adata3_gene_imputed)

    plot_marker_comparison_superpixel('TM4SF1',
                                       adata2_gene_imputed_norm, 
                                       adata3_gene_imputed_norm, 
                                       'Section s2 predicted', 
                                       'Section s3 predicted',
                                        save_path = './COSIE_TCGA_prediction/TM4SF1.jpg',
                                        dpi = 500,
                                        colormap = "turbo",
                                        figscale = 100,)
































### 