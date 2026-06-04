import os
import torch
import scanpy as sc
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from anndata import AnnData
from sklearn.cluster import KMeans
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler




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



def plot_marker_comparison(
    molecule_name: str,
    adata1,
    adata2,
    section1_label: str = 'Section 1',
    section2_label: str = 'Section 2',
    basis: str = 'spatial',
    s: int = 50,
    alpha: float = 0.9,
    colormap: str = "turbo",
    plot_style: str = "original",  # 'equal' or 'original'
    swap_xy: bool = False,
    invert_x: bool = False,
    invert_y: bool = False,
    save_path: str = None,
    dpi: int = 500,
    remove_legend = False,
    remove_spine = False,
    remove_title = False
):
    """
    Compare the spatial expression pattern of a specified molecule (e.g., gene, protein..)
    across two tissue sections, each represented by an AnnData object.

    Allows flexible visualization options, including axis inversion, coordinate swapping,
    colormap selection, and figure saving.

    Parameters
    ----------
    molecule_name : str
        The gene or protein name to visualize. Must be present in `.var` of both AnnData objects.
    
    adata1 : AnnData
        The first AnnData object, e.g., for imputed data.
    
    adata2 : AnnData
        The second AnnData object, e.g., for observed data.
    
    section1_label : str, optional
        Plot title label for the first section. Default is `'Section 1'`.
    
    section2_label : str, optional
        Plot title label for the second section. Default is `'Section 2'`.
    
    basis : str, optional
        Key in `.obsm` specifying the spatial coordinate basis (e.g., `'spatial'`). Default is `'spatial'`.
    
    s : int, optional
        Dot size in the scatter plot. Default is 50.
    
    alpha : float, optional
        Transparency level of plotted points (between 0 and 1). Default is 0.9.
    
    colormap : str, optional
        Name of the matplotlib colormap used to represent expression intensity. Default is `'viridis'`.
    
    plot_style : str, optional
        Must be one of {'equal', 'original'}.
        
        - `'equal'`: Enforces equal aspect ratio on axes.  
        - `'original'`: Keeps raw coordinate scale.  
        Default is `'original'`.
    
    swap_xy : bool, optional
        If True, swaps x and y coordinates in both sections. Default is False.
    
    invert_x : bool, optional
        If True, inverts the x-axis direction. Default is False.
    
    invert_y : bool, optional
        If True, inverts the y-axis direction. Default is False.
    
    save_path : str or None, optional
        If provided, saves the resulting figure to the specified path. The file format
        is inferred from the extension (e.g., `.pdf`, `.png`). Default is None.
    
    dpi : int, optional
        Resolution of the saved figure in dots per inch. Default is 300.

    Returns
    -------
    None
        This function does not return any value. It displays a side-by-side comparison plot of molecule expression across the two sections. If `save_path` is specified, the figure is also saved to disk.
    """
    
    # Swap XY if requested
    for adata in [adata1, adata2]:
        if swap_xy:
            coords = adata.obsm[basis][:, [1, 0]].copy()
            adata.obsm["__temp_basis__"] = coords
        else:
            adata.obsm["__temp_basis__"] = adata.obsm[basis].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for i, (adata, label, ax) in enumerate(zip([adata1, adata2],
                                               [section1_label, section2_label],
                                               axes)):
        sc.pl.embedding(
            adata,
            basis="__temp_basis__",
            color=molecule_name,
            title=None if remove_title else f'{label} - {molecule_name}',
            s=s,
            alpha=alpha,
            ax=ax,
            show=False,
            colorbar_loc=None,
            cmap=colormap
        )
    
        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()
        if plot_style == 'equal':
            ax.set_aspect('equal')
    
        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)
    
        if remove_title:
            ax.set_title("")
    
        # Remove legend and colorbar manually if possible
        if remove_legend:
            legend = ax.get_legend()
            if legend:
                legend.remove()
    
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        file_root, file_ext = os.path.splitext(save_path)
        if file_ext == "":
            file_ext = ".pdf"
        save_path_final = f"{file_root}{file_ext}"
        print(f"Saving marker comparison to: {save_path_final}")
        plt.savefig(save_path_final, dpi=dpi, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    for adata in [adata1, adata2]:
        if "__temp_basis__" in adata.obsm:
            del adata.obsm["__temp_basis__"]




def cluster_and_visualize(
    final_embeddings,
    data_dict,
    n_clusters,
    mode="joint",  # 'joint' or 'independent'
    vis_basis="spatial",
    cluster_key="cluster_labels",
    random_state=0,
    s=50,
    alpha=0.9,
    colormap="tab20",
    plot_style="original",  # 'equal' or 'original'
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    save_path=None,
    dpi=300,
    remove_legend = False,
    remove_spine = False,
    remove_title = False
):
    """
    Cluster cell embeddings and visualize the results for each tissue section.
    Supports both joint and per-section clustering modes, and offers flexible
    visualization controls (axis flip, equal scaling, saving, etc.).

    Parameters
    ----------
    final_embeddings : dict
        A dictionary mapping section names (e.g., `'s1'`, `'s2'`, ...) to 2D NumPy arrays of shape (n_cells, latent_dim), representing cell embeddings for each section.
    
    data_dict : dict
        A dictionary where each key is a modality (e.g., `'RNA'`, `'Protein'`) and each value is a list of AnnData objects, one per tissue section. If a modality is missing in a section, use `None` as a placeholder.
    
    n_clusters : int
        Number of clusters to assign using k-means.
    
    mode : str, optional
        Clustering mode. Must be one of {'joint', 'independent'}.
        
        - 'joint': Cluster all sections together.
        - 'independent': Cluster each section separately. Default is 'joint'.
    
    vis_basis : str, optional
        The key in `.obsm` to use for visualization (e.g., `'spatial'`). Default is `'spatial'`.
    
    cluster_key : str, optional
        Column name in `.obs` to store cluster assignments. Default is `'cluster_labels'`.
    
    random_state : int, optional
        Random seed for k-means reproducibility. Default is 0.
    
    s : int, optional
        Dot size in scatter plots. Default is 50.
    
    alpha : float, optional
        Point transparency (0 to 1). Default is 0.9.
    
    colormap : str, optional
        Matplotlib colormap name for cluster coloring. Default is `'tab20'`.
    
    plot_style : str, optional
        Must be one of {'equal', 'original'}.
        
        - 'equal': Enforce equal axis aspect ratio.
        - 'original': Retain raw coordinate scale. Default is `'original'`.
    
    swap_xy : bool, optional
        If True, swap x and y axes in the scatter plot. Default is False.
    
    invert_x : bool, optional
        If True, invert the x-axis. Default is False.
    
    invert_y : bool, optional
        If True, invert the y-axis. Default is False.
    
    save_path : str or None, optional
        If provided, save the figures using this prefix. Individual files will be saved for each section. Default is None (no saving).
    
    dpi : int, optional
        Resolution of saved figures in DPI (dots per inch). Default is 300.

    Returns
    -------
    cluster_labels : dict
        A dictionary mapping section IDs to arrays of assigned cluster labels.
    """
    adata_list = []
    embeddings = []
    section_names = []

    for section, embedding in final_embeddings.items():
        idx = int(section[1:]) - 1
        for modality, adata_list_per_mod in data_dict.items():
            if idx < len(adata_list_per_mod) and adata_list_per_mod[idx] is not None:
                adata_list.append(adata_list_per_mod[idx])
                embeddings.append(embedding)
                section_names.append(section)
                break

    if mode == "joint":
        print("Perform joint clustering...")
        combined_embedding = np.vstack(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        all_clusters = kmeans.fit_predict(combined_embedding).astype(str)

        # fixed order
        cluster_order = [str(i) for i in range(n_clusters)]
        color_list = [to_hex(cm.get_cmap(colormap)(i)) for i in range(n_clusters)]
        color_mapping = {str(i): color_list[i] for i in range(n_clusters)}
        # print("Color Mapping (joint):", color_mapping)

        start = 0
        for adata, emb in zip(adata_list, embeddings):
            end = start + emb.shape[0]
            adata.obs[cluster_key] = all_clusters[start:end]
            adata.obs[cluster_key] = adata.obs[cluster_key].astype(
                CategoricalDtype(categories=cluster_order, ordered=True)
            )
            adata.uns[f"{cluster_key}_colors"] = [color_mapping[cat] for cat in cluster_order]
            start = end

    elif mode == "independent":
        print("Perform independent clustering...")
        for adata, emb in zip(adata_list, embeddings):
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            clusters = kmeans.fit_predict(emb).astype(str)
            cluster_order = [str(i) for i in range(n_clusters)]
            adata.obs[cluster_key] = clusters
            adata.obs[cluster_key] = adata.obs[cluster_key].astype(
                CategoricalDtype(categories=cluster_order, ordered=True)
            )
            adata.uns[f"{cluster_key}_colors"] = [
                to_hex(cm.get_cmap(colormap)(i)) for i in range(n_clusters)
            ]
    else:
        raise ValueError("mode must be 'joint' or 'independent'")

    # Visualization
    cluster_labels = {}
    for adata, section in zip(adata_list, section_names):
        cluster_labels[section] = adata.obs[cluster_key]
        vis_coords = adata.obsm[vis_basis].copy()
        if swap_xy:
            vis_coords = vis_coords[:, [1, 0]]
            adata.obsm["__temp_basis__"] = vis_coords
            basis_to_plot = "__temp_basis__"
        else:
            basis_to_plot = vis_basis
        
        title = f"Section {section} ({mode})"

      
        fig = sc.pl.embedding(
            adata,
            basis=basis_to_plot,
            color=cluster_key,
            title=title,
            s=s,
            alpha=alpha,
            show=False,
            return_fig=True
        )
        
        ax = fig.axes[0]
        if remove_legend:
            ax.get_legend().remove()
        
        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)

        if remove_title:
            ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        
        if invert_x:
            ax.invert_xaxis()
        if invert_y:
            ax.invert_yaxis()
        if plot_style == "equal":
            ax.set_aspect("equal")
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir != "":
                os.makedirs(save_dir, exist_ok=True)
            file_root, file_ext = os.path.splitext(save_path)
            if file_ext == "":
                file_ext = ".pdf"
            section_save_path = f"{file_root}_section_{section}{file_ext}"
            print(f"Saving figure to: {section_save_path}")
            fig.savefig(section_save_path, dpi=dpi, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        

        if "__temp_basis__" in adata.obsm:
            del adata.obsm["__temp_basis__"]


    return cluster_labels




if __name__ == "__main__":
        
    file_path = './Fig4_data/'
    adata1_27me3 = sc.read_h5ad(os.path.join(file_path, 'E13_50_1_H3K27me3.h5ad'))
    adata1_27ac  = sc.read_h5ad(os.path.join(file_path, 'E13_50_1_H3K27ac.h5ad'))
    
    adata2_27me3 = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_H3K27me3.h5ad'))
    adata2_atac  = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_ATAC.h5ad'))
    adata2_rna   = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_RNA.h5ad'))
    adata2_4me3  = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_H3K4me3.h5ad'))
    
    adata1_27me3.obsm['spatial'] = adata1_27me3.obsm['spatial_local'].copy()
    adata1_27ac.obsm['spatial'] = adata1_27ac.obsm['spatial_local'].copy()
    adata2_27me3.obsm['spatial'] = adata2_27me3.obsm['spatial_local'].copy()
    adata2_atac.obsm['spatial'] = adata2_atac.obsm['spatial_local'].copy()
    adata2_rna.obsm['spatial'] = adata2_rna.obsm['spatial_local'].copy()
    adata2_4me3.obsm['spatial'] = adata2_4me3.obsm['spatial_local'].copy()
    adata2_rna.var_names_make_unique()
    
    data_dict = {
            'H3K27me3': [adata1_27me3, adata2_27me3],
            'H3K27ac': [adata1_27ac, None],
            'ATAC': [None, adata2_atac],
            'RNA': [None, adata2_rna],
            'H3K4me3': [None, adata2_4me3]
        }
    
    file_path = os.path.join('./Fig4_data/', 'COSIE_result')
    s1_embedding = np.load(os.path.join(file_path, 's1_embedding.npy'))
    s2_embedding = np.load(os.path.join(file_path, 's2_embedding.npy'))
    final_embeddings = {
        's1':s1_embedding,
        's2':s2_embedding,
    }

    cluster_labels = cluster_and_visualize(final_embeddings, 
                                       data_dict,
                                       n_clusters=11,
                                       mode="joint", 
                                       vis_basis="spatial", 
                                       s=100, 
                                       plot_style='equal',
                                       dpi = 500,
                                       save_path='COSIE_embryo.jpg')

    ## scMoMaT
    with open('./Fig4_data/scMoMaT_result/mouse_embryo.pkl', 'rb') as f:
        zs = pickle.load(f)

    final_embeddings_scmomat = {
        's1':zs[0],
        's2':zs[1],
    }


    cluster_labels_scmomat = cluster_and_visualize(final_embeddings_scmomat, 
                                       data_dict,
                                       n_clusters=11,
                                       mode="joint", 
                                       vis_basis="spatial", 
                                       s=100, 
                                       plot_style='equal',
                                       dpi = 500,
                                       save_path='scMoMaT_embryo.jpg')

    ## StabMap
    df_emb = pd.read_csv('./Fig4_data/StabMap_result/stabmap_harmony_embedding.csv')
    df_emb.rename(columns={df_emb.columns[0]: "cell_id"}, inplace=True)
    df_s1 = df_emb[df_emb["cell_id"].str.startswith("s1_")].reset_index(drop=True)
    df_s2 = df_emb[df_emb["cell_id"].str.startswith("s2_")].reset_index(drop=True)
    
    stabmap_embedding_s1 = df_s1.iloc[:, 1:].values  
    stabmap_embedding_s2 = df_s2.iloc[:, 1:].values  
    final_embeddings_stabmap = {
        's1':stabmap_embedding_s1,
        's2':stabmap_embedding_s2,
    }
        

    cluster_labels_stabmap = cluster_and_visualize(final_embeddings_stabmap, 
                                       data_dict,
                                       n_clusters=11,
                                       mode="joint", 
                                       vis_basis="spatial", 
                                       s=100, 
                                       plot_style='equal',
                                       dpi = 500,
                                       save_path='StabMap_embryo.jpg')


    ## SpaMosaic
    adata_spamosaic = sc.read_h5ad('./Fig4_data/SpaMosaic_result/ad_epi_s1_2.h5ad')
    spamosaic_embedding_s1 = adata_spamosaic.obsm['merged_emb'][:adata1_27me3.shape[0], :]
    spamosaic_embedding_s2 = adata_spamosaic.obsm['merged_emb'][adata1_27me3.shape[0]:, :]
    final_embeddings_spamosaic = {
        's1':spamosaic_embedding_s1,
        's2':spamosaic_embedding_s2,
    }
    cluster_labels_spamosaic = cluster_and_visualize(final_embeddings_spamosaic, 
                                       data_dict,
                                       n_clusters=11,
                                       mode="joint", 
                                       vis_basis="spatial", 
                                       s=100, 
                                       plot_style='equal',
                                       dpi = 500,
                                       save_path='SpaMosaic_embryo.jpg')


    os.makedirs("./ATAC_section1_prediction", exist_ok=True)
    os.makedirs("./H3K4me3_section1_prediction", exist_ok=True)
    os.makedirs("./RNA_section1_prediction", exist_ok=True)
    os.makedirs("./H3K27ac_section2_prediction", exist_ok=True)
    file_path = os.path.join('./Fig4_data/', 'COSIE_result')
    ## Virtual prediction
    adata1_ATAC_imputed = sc.read_h5ad(os.path.join(file_path, 'adata1_ATAC_imputed.h5ad'))
    adata1_ATAC_imputed_norm = create_normalized_adata(adata1_ATAC_imputed)
    adata2_ATAC_norm = create_normalized_adata(adata2_atac)

    gene_list = ['Sox2', 'Alb', 'Col11a1', 'Tbx20']
    
    for gene in gene_list:
        plot_marker_comparison(gene, 
                               adata1_ATAC_imputed_norm, 
                               adata2_ATAC_norm, 
                               section1_label = 'Section s1 Imputed ATAC', 
                               section2_label = 'Section s2 Observed ATAC', 
                               save_path = './ATAC_section1_prediction/{}.jpg'.format(gene),
                               dpi = 500,
                               s=270)

    adata1_H3K4me3_imputed = sc.read_h5ad(os.path.join(file_path,  'adata1_H3K4me3_imputed.h5ad'))
    adata1_H3K4me3_imputed_norm = create_normalized_adata(adata1_H3K4me3_imputed)
    adata2_H3K4me3_norm = create_normalized_adata(adata2_4me3)
    for gene in gene_list:
        plot_marker_comparison(gene, 
                               adata1_H3K4me3_imputed_norm, 
                               adata2_H3K4me3_norm, 
                               section1_label = 'Section s1 Imputed H3K4me3', 
                               section2_label = 'Section s2 Observed H3K4me3', 
                               save_path = './H3K4me3_section1_prediction/{}.jpg'.format(gene),
                               dpi = 500,
                               s=270,)


    adata1_RNA_imputed = sc.read_h5ad(os.path.join(file_path, 'adata1_RNA_imputed.h5ad'))
    adata1_RNA_imputed_norm = create_normalized_adata(adata1_RNA_imputed)
    adata2_rna_norm = create_normalized_adata(adata2_rna)
    for gene in gene_list:
        plot_marker_comparison(gene, 
                               adata1_RNA_imputed_norm, 
                               adata2_rna_norm, 
                               section1_label = 'Section s1 Imputed RNA', 
                               section2_label = 'Section s2 Observed RNA', 
                               save_path = './RNA_section1_prediction/{}.jpg'.format(gene),
                               dpi = 500,
                               s=270,)


    adata2_H3K27ac_imputed = sc.read_h5ad(os.path.join(file_path,  'adata2_H3K27ac_imputed.h5ad'))
    adata2_H3K27ac_imputed_norm = create_normalized_adata(adata2_H3K27ac_imputed)
    adata1_27ac_norm = create_normalized_adata(adata1_27ac)

    for gene in gene_list:
        plot_marker_comparison(gene, 
                               adata1_27ac_norm, 
                               adata2_H3K27ac_imputed_norm, 
                               section1_label = 'Section s1 Observed H3K27ac', 
                               section2_label = 'Section s2 Imputed H3K27ac', 
                               save_path = './H3K27ac_section2_prediction/{}.jpg'.format(gene),
                               dpi = 500,
                               s=270,)












### 

