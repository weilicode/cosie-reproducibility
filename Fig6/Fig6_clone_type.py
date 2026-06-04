import os
import re

import scanpy as sc
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# -----------------------------
# paths
# -----------------------------
data_dir = "./Fig6_data/"
save_dir = "./"
os.makedirs(save_dir, exist_ok=True)

adata_path = os.path.join(data_dir, "adata_P11_LUAD_Visium_rna.h5ad")
clone_path = os.path.join(data_dir, "spot_clone_type.csv")


# -----------------------------
# load data
# -----------------------------
adata = sc.read_h5ad(adata_path)
clone_df = pd.read_csv(clone_path)

clone_df = clone_df[
    clone_df["Sample_ID"].str.upper() == "P11_ADC"
].copy()


# -----------------------------
# add clone annotation
# -----------------------------
clone_info = clone_df.set_index("Barcode")[["Clone"]]
adata.obs = adata.obs.join(clone_info, how="left")

adata.obs["Clone"] = (
    adata.obs["Clone"]
    .astype("category")
    .cat.add_categories("none_consider")
    .fillna("none_consider")
)


# -----------------------------
# spatial coordinate
# -----------------------------
adata.obsm["spatial_plot"] = adata.obsm["spatial"].copy()
adata.obsm["spatial_plot"] = adata.obsm["spatial_plot"][:, [1, 0]]


# -----------------------------
# colors
# -----------------------------
label_color_dict = {
    "Ref": "#D3D3D3",
    "Clone A1": "#B2E2B2",
    "Clone C1": "#FFD700",
    "Clone C2": "#FEC44F",
    "Clone B1": "#6495ED",
    "Clone B2": "#1E5AAC",
    "Clone D1": "#FB6A4A",
    "Clone D2": "#EF3B2C",
    "Clone D3": "#CB181D",
    "none_consider": "#D3D3D3",
}

adata.obs["Clone"] = adata.obs["Clone"].astype("category")
adata.uns["Clone_colors"] = [
    label_color_dict[x]
    for x in adata.obs["Clone"].cat.categories
]


# -----------------------------
# plot
# -----------------------------
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

fig = sc.pl.embedding(
    adata,
    basis="spatial_plot",
    color="Clone",
    s=30,
    title="Annotations-Clone",
    frameon=False,
    show=False,
    return_fig=True
)

fig.savefig(
    os.path.join(save_dir, "clone_annotation.png"),
    dpi=500,
    bbox_inches="tight"
)

plt.show()
plt.close()