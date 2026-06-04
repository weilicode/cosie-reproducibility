import pandas as pd
import anndata as ad
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm
import gc
import joblib
import pickle
import numpy as np
import seaborn as sns
import scanpy as sc
import symphonypy as sp
import matplotlib.pyplot as plt
import os
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from matplotlib.cm import get_cmap
from matplotlib import patches
from pandas.api.types import CategoricalDtype
from matplotlib.colors import to_rgb


def plot_histology_clusters(
    he_clusters_image,
    num_he_clusters,
    section_title=None,
    colormap=None,
    cluster_names=None,
    save_path=None,
    figscale=35,
    remove_title=False,
    remove_legend=False,
    remove_spine=False,
    dpi=300,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib import cm
    from matplotlib.colors import to_rgb

    # ---- colors ----
    if colormap is None:
        color_list = [
            [255,127,14],[44,160,44],[214,39,40],[148,103,189],
            [140,86,75],[227,119,194],[127,127,127],[188,189,34],
            [23,190,207],[174,199,232],[255,187,120],[152,223,138],
            [255,152,150],[197,176,213],[196,156,148],[247,182,210],
            [199,199,199],[219,219,141],[158,218,229],[16,60,90],
            [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
            [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0],
        ]
    elif isinstance(colormap, list):
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap)
        color_list = [
            [int(255 * c) for c in to_rgb(cmap(i))]
            for i in range(num_he_clusters)
        ]

    if len(color_list) < num_he_clusters:
        raise ValueError(
            f"Color list has {len(color_list)} colors but "
            f"{num_he_clusters} clusters are present."
        )

    # ---- RGB image ----
    h, w = he_clusters_image.shape
    image_rgb = np.ones((h, w, 3), dtype=np.uint8) * 255

    for c in range(num_he_clusters):
        image_rgb[he_clusters_image == c] = color_list[c]

    # ---- plot ----
    plt.figure(figsize=(w // figscale, h // figscale))
    if not remove_title:
        plt.title(section_title or "Histology Clusters", fontsize=18)


    plt.imshow(image_rgb, interpolation="none")
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ---- legend ----
    if not remove_legend:
        legend_elements = []
        for i in range(num_he_clusters):
            label = (
                cluster_names[i]
                if cluster_names is not None and i < len(cluster_names)
                else f"Cluster {i}"
            )
            legend_elements.append(
                patches.Patch(
                    facecolor=np.array(color_list[i]) / 255,
                    label=str(label),
                )
            )

        plt.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=12,
        )

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close()


def visualize_superpixel_from_adata(
    adata,
    obs_key,
    vis_basis="spatial",
    colormap=None,
    legend_labels=None,
    swap_xy=False,
    invert_x=False,
    invert_y=False,
    offset=False,
    save_path=None,
    dpi=300,
    remove_title=False,
    remove_legend=False,
    remove_spine=False,
    figscale=35,
    title=None,
):
    import numpy as np

    # ---- coords ----
    coords = adata.obsm[vis_basis].copy()
    if swap_xy:
        coords = coords[:, [1, 0]]
    coords = coords.astype(int)

    if offset:
        coords -= coords.min(axis=0)

    # ---- labels ----
    labels = adata.obs[obs_key]
    cluster_names = None

    if hasattr(labels.dtype, "categories"):
        # categorical labels
        label_codes = labels.cat.codes.values

        if legend_labels is not None:
            cluster_names = legend_labels
            num_clusters = len(legend_labels)
        else:
            cluster_names = list(labels.cat.categories)
            num_clusters = len(cluster_names)

    else:
        # numeric labels
        label_codes = labels.astype(int).values
        num_clusters = int(label_codes.max()) + 1
        cluster_names = (
            legend_labels
            if legend_labels is not None
            else [f"Cluster {i}" for i in range(num_clusters)]
        )

    # ---- build image ----
    max_y, max_x = coords.max(axis=0) + 1
    image = np.full((max_y, max_x), fill_value=-1, dtype=int)

    for (y, x), lab in zip(coords, label_codes):
        if lab >= 0:
            image[y, x] = lab

    if invert_x:
        image = image[:, ::-1]
    if invert_y:
        image = image[::-1, :]

    # ---- plot ----
    plot_histology_clusters(
        he_clusters_image=image,
        num_he_clusters=num_clusters,
        section_title=title if title is not None else obs_key,
        colormap=colormap,
        cluster_names=cluster_names,
        save_path=save_path,
        dpi=dpi,
        figscale=figscale,
        remove_title=remove_title,
        remove_legend=remove_legend,
        remove_spine=remove_spine,
    )




def assign_group_from_clusters(adata, cluster_key, group_dict, new_key="group_label"):
    """Map cluster ids -> group names, save to adata.obs[new_key]."""
    cluster_ids = adata.obs[cluster_key].astype(int).values
    labels = np.array([None] * adata.n_obs, dtype=object)

    for group_name, clusters in group_dict.items():
        mask = np.isin(cluster_ids, clusters)
        labels[mask] = group_name

    if np.any(labels == None):
        missing_clusters = np.unique(cluster_ids[labels == None])
        raise ValueError(f"Some clusters are not assigned to any group: {missing_clusters}")

    categories = list(group_dict.keys())
    adata.obs[new_key] = labels
    adata.obs[new_key] = adata.obs[new_key].astype(CategoricalDtype(categories=categories))
    
    return adata.obs[new_key]



if __name__ == "__main__":
    
    adata_adt = sc.read_h5ad('./Fig7_data/adata_P24_LUAD_Visium_adt_istar.h5ad')
    adata_pre = sc.read_h5ad('./Fig7_data/P24_virtual_prediction.h5ad')
    assigned_labels = np.load('./Fig7_data/adata_P24_LUAD_HE_mapping_assigned_labels.npy')
    adata_pre.obsm['spatial'] = adata_adt.obsm['spatial'].copy()
    adata_pre.obs['labels'] = assigned_labels.copy()
    
    
    group_dict = {
        "Macrophages": [11],
        "Bronchus": [3],
        "Vessels": [8,10],
        "Normal lung": [1,4,9,13,15,18,21,23],
        "Pneumocytes": [17],
        "Tumor": [2,5,6,7,14,16,19,20],
        "Fibrous tissue": [0,12,22],
        "Lymphoid aggregates": [24],
    }
    
    colormap = [
        [255, 127, 14],   # Macrophages
        [188, 189, 34],  # Bronchus
        [220, 20, 60], # Vessels
        [173, 216, 230], # Normal lung
        [77, 175, 74],   # Pneumocyte
        [148, 103, 189],   # Tumor
        [247, 182, 210], # Fibrous tissue
        [139, 69, 19],   # Lymphoid aggregates
        [0, 191, 255],   # Fibrous+tumor
    ]
    
    legend_labels = [
        f"{name} (clusters {','.join(map(str, clusters))})"
        for name, clusters in group_dict.items()
    ]
    
    cluster_key = "labels"        # original cluster ID
    group_key   = "transferred_labels"        # After mapping
    
    
    group_labels = assign_group_from_clusters(
            adata_pre,
            cluster_key=cluster_key,
            group_dict=group_dict,
            new_key=group_key,
        )
    
    visualize_superpixel_from_adata(
            adata_pre,
            obs_key=group_key,
            colormap=colormap,
            legend_labels=legend_labels,
            swap_xy=False,
            figscale=100,
            save_path='P24_HE_label_transfer.jpg',
        )
    
    
    
    
    
    
    



















### 