import os
import torch
import scanpy as sc
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from anndata import AnnData

from sklearn.cluster import KMeans
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
from pandas.api.types import CategoricalDtype

from matplotlib import patches
from matplotlib.colors import to_rgb
import matplotlib as mpl


def cluster_and_visualize_single_embedding(
    embedding,
    spatial_coords,
    n_clusters,
    title="",
    save_path=None,
    colormap=None,
    dpi=300,
    figscale=35,
    invert_x=False,
    invert_y=False,
    swap_xy=False,
    offset=False,
    remove_title=False,
    remove_legend=False,
    remove_spine=False
):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from matplotlib import patches
    from matplotlib.colors import to_rgb
    from matplotlib import cm

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(embedding)

    coords = spatial_coords.copy()
    if swap_xy:
        coords = coords[:, [1, 0]]
    coords = coords.astype(int)
    if offset:
        coords -= coords.min(axis=0)

    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), fill_value=-1, dtype=int)
    for (y, x), label in zip(coords, labels):
        image[y, x] = label

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    # Coloring
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
        
    image_rgb = 255 * np.ones([image.shape[0], image.shape[1], 3])
    for cluster in range(n_clusters):
        image_rgb[image == cluster] = color_list[cluster % len(color_list)]
    image_rgb = np.array(image_rgb, dtype='uint8')

    # Plot
    plt.figure(figsize=(image.shape[1] // figscale, image.shape[0] // figscale))
    if not remove_title:
        plt.title(title, fontsize=18)
    plt.imshow(image_rgb, interpolation='none')
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not remove_legend:
        legend_elements = [patches.Patch(facecolor=np.array(color_list[i % len(color_list)]) / 255,
                                         label=f'Cluster {i}')
                           for i in range(n_clusters)]
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




if __name__ == "__main__":

    adata1_rna_enhanced = sc.read_h5ad('./Fig3_data/Mouse_hippocampus_enhancement/adata1_rna_enhanced.h5ad')
    sc.pp.normalize_total(adata1_rna_enhanced)
    sc.pp.log1p(adata1_rna_enhanced)
    sc.pp.scale(adata1_rna_enhanced)
    sc.tl.pca(adata1_rna_enhanced, n_comps=50)
    
    color_map =  [[255,127,14],[44,160,44],[214,39,40],[148,103,189],
                  [140,86,75],[227,119,194],[127,127,127],[188,189,34],
                 [107,174,214] ,[174,199,232],[255,187,120],[152,223,138],
                  [255,152,150],[197,176,213],[196,156,148],[247,182,210],
                  [199,199,199],[219,219,141],[158,218,229], [137,69,133],
                  [31, 119, 180],[135, 206, 250]]
    
    num_clusters = 21
    cluster_and_visualize_single_embedding(
        embedding = adata1_rna_enhanced.obsm['X_pca'], 
        spatial_coords = adata1_rna_enhanced.obsm['spatial'], 
        n_clusters=num_clusters,
        offset= True,
        swap_xy = True,
        invert_x = True,
        remove_title=True,
        remove_legend=True,
        remove_spine=True,
        colormap=color_map,
        save_path='COSIE_Enhanced_rna_clustering.jpg',
        figscale = 50)

    ## Virtual prediction
    adata2_metabolite_imputed=sc.read_h5ad('./Fig3_data/Mouse_hippocampus_enhancement/adata2_metabolite_imputed.h5ad')
    adata1_meta = sc.read_h5ad('./Fig3_data/Mouse_hippocampus/section3/adata_section3_meta.h5ad')
    adata2_metabolite_imputed_norm = create_normalized_adata(adata2_metabolite_imputed)
    adata1_meta_norm = create_normalized_adata(adata1_meta)
    for metabolite in ['V1476', 'V1198', 'V1025', 'V10']:
        plot_marker_comparison_superpixel(
            metabolite,
            adata2_metabolite_imputed_norm,
            adata1_meta_norm,
            section1_label = 'Section2 Predicted',
            section2_label = 'Section1 Observed',
            basis = 'spatial',
            colormap = "turbo",
            offset = True,
            swap_xy=True, 
            figscale = 50,
            invert_x = True,
            dpi = 500,  
            save_path = './COSIE_hippocampus_enhancement/{}.jpg'.format(metabolite))
    




    





##