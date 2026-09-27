import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from matplotlib import patches, cm
from matplotlib.colors import to_rgb
from sklearn.cluster import KMeans


def plot_histology_clusters(cluster_image, n_clusters, title=None, colormap=None, save_path=None,
                            figscale=35, remove_title=False, remove_legend=False,
                            remove_spine=False, dpi=300):

    if colormap is None:
        color_list = [
            [255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],
            [227,119,194],[127,127,127],[188,189,34],[23,190,207],[174,199,232],
            [255,187,120],[152,223,138],[255,152,150],[197,176,213],[196,156,148],
            [247,182,210],[199,199,199],[219,219,141],[158,218,229],[16,60,90],
            [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
            [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]
        ]
    elif isinstance(colormap, list):
        color_list = colormap
    else:
        cmap = cm.get_cmap(colormap)
        color_list = [[int(255*c) for c in to_rgb(cmap(i))] for i in range(cmap.N)]

    image_rgb = np.full((*cluster_image.shape, 3), 255, dtype=np.uint8)
    for cluster in range(n_clusters):
        image_rgb[cluster_image == cluster] = color_list[cluster % len(color_list)]

    fig, ax = plt.subplots(figsize=(cluster_image.shape[1]/figscale, cluster_image.shape[0]/figscale))

    if not remove_title:
        ax.set_title(title if title else "Clusters", fontsize=18)

    ax.imshow(image_rgb, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])

    if remove_spine:
        for spine in ax.spines.values():
            spine.set_visible(False)

    if not remove_legend:
        handles = [patches.Patch(facecolor=np.array(color_list[i % len(color_list)])/255,
                                 label=f"Cluster {i}") for i in range(n_clusters)]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left",
                  borderaxespad=0, fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)


def cluster_and_visualize_superpixel(final_embeddings, data_dict, n_clusters, mode="joint",
                                     defined_labels=None, vis_basis="spatial", random_state=0,
                                     colormap=None, swap_xy=False, invert_x=False, invert_y=False,
                                     offset=False, save_path=None, dpi=300, remove_title=False,
                                     remove_legend=False, remove_spine=False, figscale=35):

    embeddings, coords_dict, section_names = [], {}, []

    for section, embedding in final_embeddings.items():
        idx = int(section[1:]) - 1

        for _, adata_list in data_dict.items():
            if idx < len(adata_list) and adata_list[idx] is not None:
                adata = adata_list[idx]
                coords = adata.obsm[vis_basis].copy()

                if swap_xy:
                    coords = coords[:, [1, 0]]

                coords = coords.astype(int)

                if offset:
                    coords -= coords.min(axis=0)

                if embedding.shape[0] != coords.shape[0]:
                    raise ValueError(f"{section}: embedding/coordinate mismatch "
                                     f"({embedding.shape[0]} vs {coords.shape[0]}).")

                embeddings.append(embedding)
                coords_dict[section] = coords
                section_names.append(section)
                break

    if mode == "joint":
        print("Perform joint clustering...")
        combined_embedding = np.vstack(embeddings)
        labels_all = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(combined_embedding)

        cluster_labels = {}
        start = 0
        for section, embedding in zip(section_names, embeddings):
            end = start + embedding.shape[0]
            cluster_labels[section] = labels_all[start:end]
            start = end

    elif mode == "independent":
        print("Perform independent clustering...")
        cluster_labels = {
            section: KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(embedding)
            for section, embedding in zip(section_names, embeddings)
        }

    elif mode == "defined":
        if defined_labels is None:
            raise ValueError("If mode='defined', `defined_labels` must be provided.")
        cluster_labels = defined_labels

    else:
        raise ValueError("mode must be 'joint', 'independent', or 'defined'.")

    for section in section_names:
        coords = coords_dict[section]
        labels = np.asarray(cluster_labels[section])

        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), -1, dtype=int)

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
            image, n_clusters,
            title=f"Section {section} ({mode})",
            colormap=colormap,
            save_path=section_save_path,
            figscale=figscale,
            remove_title=remove_title,
            remove_legend=remove_legend,
            remove_spine=remove_spine,
            dpi=dpi
        )

    return cluster_labels


def merge_clusters_to_new_ids(cluster_labels, merge_groups):
    all_labels = np.concatenate(list(cluster_labels.values()))
    next_id = int(all_labels.max()) + 1
    merge_map = {}

    for group in merge_groups:
        for cluster_id in group:
            merge_map[cluster_id] = next_id
        next_id += 1

    return {
        section: np.array([merge_map.get(label, label) for label in labels])
        for section, labels in cluster_labels.items()
    }


def relabel_clusters_sequentially(cluster_labels):
    all_labels = np.concatenate(list(cluster_labels.values()))
    unique_labels = sorted(np.unique(all_labels))
    mapping = {old: new for new, old in enumerate(unique_labels)}

    return {
        section: np.array([mapping[label] for label in labels])
        for section, labels in cluster_labels.items()
    }


def highlight_joint_clusters_all_sections(cluster_labels, data_dict, n_clusters, highlight_labels,
                                           vis_basis="spatial", colormap=None, swap_xy=False,
                                           invert_x=False, invert_y=False, offset=False,
                                           save_dir=None, figscale=35, dpi=300,
                                           remove_title=True, remove_legend=True,
                                           remove_spine=True, bg_color=(200,200,200)):

    if colormap is None:
        base_colors = [
            [255,127,14],[44,160,44],[214,39,40],[148,103,189],[140,86,75],
            [227,119,194],[127,127,127],[188,189,34],[23,190,207],[174,199,232],
            [255,187,120],[152,223,138],[255,152,150],[197,176,213],[196,156,148],
            [247,182,210],[199,199,199],[219,219,141],[158,218,229],[16,60,90],
            [128,64,7],[22,80,22],[107,20,20],[74,52,94],[70,43,38],
            [114,60,97],[64,64,64],[94,94,17],[12,95,104],[0,0,0]
        ]
    elif isinstance(colormap, list):
        base_colors = colormap
    else:
        cmap = cm.get_cmap(colormap)
        base_colors = [[int(255*c) for c in to_rgb(cmap(i))] for i in range(cmap.N)]

    for section, labels in cluster_labels.items():
        idx = int(section[1:]) - 1
        coords = None

        for _, adata_list in data_dict.items():
            if idx < len(adata_list) and adata_list[idx] is not None:
                coords = adata_list[idx].obsm[vis_basis].copy()

                if swap_xy:
                    coords = coords[:, [1, 0]]

                coords = coords.astype(int)

                if offset:
                    coords -= coords.min(axis=0)

                break

        if coords is None:
            raise ValueError(f"No spatial coordinates found for {section}.")

        if len(labels) != coords.shape[0]:
            raise ValueError(f"{section}: labels/coordinates length mismatch.")

        max_y, max_x = coords.max(axis=0) + 1
        image = np.full((max_y, max_x), -1, dtype=int)

        for (y, x), label in zip(coords, labels):
            image[y, x] = label

        if invert_x:
            image = image[:, ::-1]
        if invert_y:
            image = image[::-1, :]

        color_list = [
            base_colors[i % len(base_colors)] if i in highlight_labels else list(bg_color)
            for i in range(n_clusters)
        ]

        image_rgb = np.full((*image.shape, 3), 255, dtype=np.uint8)
        for cluster in range(n_clusters):
            image_rgb[image == cluster] = color_list[cluster]

        fig, ax = plt.subplots(figsize=(image.shape[1]/figscale, image.shape[0]/figscale))

        if not remove_title:
            ax.set_title(f"Section {section} - Highlighted clusters", fontsize=18)

        ax.imshow(image_rgb, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])

        if remove_spine:
            for spine in ax.spines.values():
                spine.set_visible(False)

        if not remove_legend:
            handles = [
                patches.Patch(facecolor=np.array(color_list[i])/255, label=f"Cluster {i}")
                for i in highlight_labels
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.05,1), loc="upper left", frameon=False)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"highlighted_{section}.jpg")
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()
        plt.close(fig)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def format_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", labelsize=11)


def save_fig(fig, save_path):
    fig.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def plot_bar(df, x, y, ylabel, save_path, ylim, color="#F28C8C"):
    fig, ax = plt.subplots(figsize=(2.7, 3.2))
    ax.bar(df[x], df[y], color=color, edgecolor="none", width=0.55)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    format_axis(ax)
    plt.tight_layout()
    save_fig(fig, save_path)


def plot_metric_boxplot(datasets, metric8, metric112, ylabel, save_path, ylim,
                        color="#F28C8C", color_light="#FAD4D4"):

    rows = []

    for sample, modality, res8, res112 in datasets:
        rows.append(pd.DataFrame({
            "Sample": sample, "Modality": modality,
            "Resolution": "8 μm", "Value": res8[metric8]
        }))
        rows.append(pd.DataFrame({
            "Sample": sample, "Modality": modality,
            "Resolution": "112 μm", "Value": res112[metric112]
        }))

    plot_df = pd.concat(rows, ignore_index=True).dropna(subset=["Value"])
    plot_df["Group"] = plot_df["Sample"] + "-" + plot_df["Modality"]

    group_order = ["S1-Protein", "S2-Protein", "S3-RNA", "S4-Protein", "S4-RNA"]

    fig, ax = plt.subplots(figsize=(7.8, 5))

    sns.boxplot(
        data=plot_df,
        x="Group",
        y="Value",
        hue="Resolution",
        order=group_order,
        palette={"8 μm": color, "112 μm": color_light},
        width=0.55,
        showcaps=False,
        showfliers=False,
        boxprops={"linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        medianprops={"color": "black", "linewidth": 1.8},
        ax=ax
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    format_axis(ax)
    ax.tick_params(axis="x", labelrotation=25)
    ax.legend(frameon=False, title="", fontsize=11)

    plt.tight_layout()
    save_fig(fig, save_path)


if __name__ == "__main__":

    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42

    data_root = "./Fig6_data"
    result_root = os.path.join(data_root, "COSIE_result")
    quantitative_root = "./Fig6_lung_all_quantitative"
    os.makedirs(quantitative_root, exist_ok=True)

    adata1_rna = sc.read_h5ad(os.path.join(data_root, "adata_P15_LUAD_Visium_rna_istar.h5ad"))
    adata2_rna = sc.read_h5ad(os.path.join(data_root, "adata_P12_LUAD_Visium_rna_istar.h5ad"))
    adata3_adt = sc.read_h5ad(os.path.join(data_root, "adata_P24_LUAD_Visium_adt_istar.h5ad"))
    adata4_adt = sc.read_h5ad(os.path.join(data_root, "adata_P11_LUAD_Visium_adt_istar.h5ad"))

    adata1_he = sc.AnnData(X=adata1_rna.obsm["UNI_feature"])
    adata2_he = sc.AnnData(X=adata2_rna.obsm["UNI_feature"])
    adata3_he = sc.AnnData(X=adata3_adt.obsm["UNI_feature"])
    adata4_he = sc.AnnData(X=adata4_adt.obsm["UNI_feature"])

    adata1_he.obsm["spatial"] = adata1_rna.obsm["spatial"].copy()
    adata2_he.obsm["spatial"] = adata2_rna.obsm["spatial"].copy()
    adata3_he.obsm["spatial"] = adata3_adt.obsm["spatial"].copy()
    adata4_he.obsm["spatial"] = adata4_adt.obsm["spatial"].copy()

    data_dict = {
        "HE": [adata1_he, adata2_he, adata3_he, adata4_he],
        "RNA": [adata1_rna, adata2_rna, None, None],
        "Protein": [None, None, adata3_adt, None]
    }

    final_embeddings = {
        f"s{i}": np.load(os.path.join(result_root, f"s{i}_embedding.npy"))
        for i in range(1, 5)
    }

    for i, adata_he in enumerate([adata1_he, adata2_he, adata3_he, adata4_he], start=1):
        assert final_embeddings[f"s{i}"].shape[0] == adata_he.n_obs

    cluster_labels = cluster_and_visualize_superpixel(
        final_embeddings, data_dict, n_clusters=25, mode="joint",
        vis_basis="spatial", dpi=500, figscale=120
    )

    cluster_labels = merge_clusters_to_new_ids(cluster_labels, [[1,13,22], [10,15]])
    cluster_labels = relabel_clusters_sequentially(cluster_labels)

    assert len(np.unique(np.concatenate(list(cluster_labels.values())))) == 22

    color_map = [
        [174,199,232],[76,144,133],[64,64,64],[255,187,120],
        [137,69,133],[227,119,194],[127,127,127],[188,189,34],
        [196,156,148],[44,160,44],[214,39,40],[152,223,138],
        [255,152,150],[197,176,213],[22,80,22],[23,190,207],
        [199,199,199],[219,219,141],[128,64,7],[255,127,14],
        [247,182,210],[148,103,189]
    ]

    cluster_and_visualize_superpixel(
        final_embeddings, data_dict, n_clusters=22, mode="defined",
        defined_labels=cluster_labels, vis_basis="spatial", colormap=color_map,
        save_path="COSIE_lung.jpg", dpi=500, figscale=120
    )

    for cluster_id in [10,18,19,20,21]:
        highlight_joint_clusters_all_sections(
            cluster_labels=cluster_labels,
            data_dict=data_dict,
            n_clusters=22,
            highlight_labels=[cluster_id],
            vis_basis="spatial",
            colormap=color_map,
            figscale=220,
            dpi=500,
            save_dir=os.path.join("highlight", f"cluster_{cluster_id}"),
            bg_color=(200,200,200)
        )

    s1_pro_8 = load_pkl(os.path.join(result_root, "correlation_metrics_P15_s1_protein.pkl"))
    s1_pro_112 = load_pkl(os.path.join(result_root, "correlation_metrics_P15_s1_protein_112.pkl"))
    s2_pro_8 = load_pkl(os.path.join(result_root, "correlation_metrics_P12_s2_protein.pkl"))
    s2_pro_112 = load_pkl(os.path.join(result_root, "correlation_metrics_P12_s2_protein_112.pkl"))
    s3_rna_8 = load_pkl(os.path.join(result_root, "correlation_metrics_P24_RNA_1000hvg.pkl"))
    s3_rna_112 = load_pkl(os.path.join(result_root, "correlation_metrics_P24_RNA_1000hvg_112.pkl"))
    s4_pro_8 = load_pkl(os.path.join(result_root, "correlation_metrics_P11_protein.pkl"))
    s4_pro_112 = load_pkl(os.path.join(result_root, "correlation_metrics_P11_protein_112.pkl"))
    s4_rna_8 = load_pkl(os.path.join(result_root, "correlation_metrics_P11_RNA_1000hvg.pkl"))
    s4_rna_112 = load_pkl(os.path.join(result_root, "correlation_metrics_P11_RNA_1000hvg_112.pkl"))

    metric_datasets = [
        ("S1", "Protein", s1_pro_8, s1_pro_112),
        ("S2", "Protein", s2_pro_8, s2_pro_112),
        ("S3", "RNA", s3_rna_8, s3_rna_112),
        ("S4", "Protein", s4_pro_8, s4_pro_112),
        ("S4", "RNA", s4_rna_8, s4_rna_112)
    ]

    lung_res = pd.read_csv(os.path.join(result_root, "Clustering_metrics.csv"))
    clustering_df = lung_res[lung_res["annotation_key"] == "annotations"].copy().sort_values("sid")
    clustering_df = clustering_df[["sid", "ari", "nmi"]]

    plot_bar(
        clustering_df,
        x="sid",
        y="ari",
        ylabel="ARI",
        save_path=os.path.join(quantitative_root, "ari_barplot.png"),
        ylim=(0, 0.85)
    )

    plot_bar(
        clustering_df,
        x="sid",
        y="nmi",
        ylabel="NMI",
        save_path=os.path.join(quantitative_root, "nmi_barplot.png"),
        ylim=(0, 0.85)
    )

    plot_metric_boxplot(
        metric_datasets,
        metric8="gene_pcc",
        metric112="gene_pcc_112",
        ylabel="Gene-wise PCC",
        save_path=os.path.join(quantitative_root, "pcc_boxplot.png"),
        ylim=(-0.1, 1)
    )

    plot_metric_boxplot(
        metric_datasets,
        metric8="gene_spearman",
        metric112="gene_spearman_112",
        ylabel="Gene-wise Spearman",
        save_path=os.path.join(quantitative_root, "spearman_boxplot.png"),
        ylim=(-0.15, 1)
    )