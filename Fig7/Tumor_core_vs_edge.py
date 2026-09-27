import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

if __name__ == "__main__":
    tcga_file = "./Fig7_data/DEG_TCGA.csv"
    cptac_file = "./Fig7_data/DEG_CPTAC.csv"
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    df_tcga = pd.read_csv(tcga_file)
    df_cptac = pd.read_csv(cptac_file)

    req_cols = ["gene", "logFC_inner_vs_core"]
    for name, df in [("TCGA", df_tcga), ("CPTAC", df_cptac)]:
        miss = [c for c in req_cols if c not in df.columns]
        if len(miss) > 0:
            raise ValueError(f"{name} missing columns: {miss}")

    overlap_genes = sorted(set(df_tcga["gene"].dropna()) & set(df_cptac["gene"].dropna()))
    if len(overlap_genes) == 0:
        raise ValueError("No overlapping genes found between TCGA and CPTAC.")

    df_tcga_plot = df_tcga[df_tcga["gene"].isin(overlap_genes)].copy()
    df_tcga_plot["dataset"] = "TCGA"
    df_cptac_plot = df_cptac[df_cptac["gene"].isin(overlap_genes)].copy()
    df_cptac_plot["dataset"] = "CPTAC"
    plot_df = pd.concat([df_tcga_plot, df_cptac_plot], axis=0, ignore_index=True)

    plot_df["logFC_inner_vs_core"] = plot_df["logFC_inner_vs_core"].clip(lower=-0.95, upper=0.95)

    tcga_medians = plot_df[plot_df["dataset"] == "TCGA"].groupby("gene")["logFC_inner_vs_core"].median().sort_values(ascending=True)
    sorted_genes = tcga_medians.index.tolist()

    direction_colors = {"core": "#CBB6E9", "inner": "#E8B5C8"}
    medians = plot_df.groupby(["gene", "dataset"])["logFC_inner_vs_core"].median()

    fig, ax = plt.subplots(figsize=(7, max(6, 0.33 * len(sorted_genes))))

    y_base = np.arange(len(sorted_genes))
    offset = 0.17
    box_width = 0.25

    for i, gene in enumerate(sorted_genes):
        for dataset in ["TCGA", "CPTAC"]:
            if dataset == "TCGA":
                dy = +offset
                edge_color = "#3A3A3A"
                line_style = "-"
                line_width = 1.4
            else:
                dy = -offset
                edge_color = "#8A8A8A"
                line_style = "--"
                line_width = 1.2

            vals = plot_df.loc[(plot_df["gene"] == gene) & (plot_df["dataset"] == dataset), "logFC_inner_vs_core"].dropna().values
            if len(vals) == 0:
                continue

            median_val = medians.loc[(gene, dataset)]
            direction = "core" if median_val > 0 else "inner"
            color = direction_colors[direction]

            ax.boxplot(
                vals,
                positions=[i + dy],
                vert=False,
                widths=box_width,
                patch_artist=True,
                showfliers=False,
                showcaps=False,
                boxprops=dict(facecolor=color, edgecolor=edge_color, linewidth=line_width, linestyle=line_style),
                medianprops=dict(color=edge_color, linewidth=1.6),
                whiskerprops=dict(color=edge_color, linewidth=1.0, linestyle=line_style),
                capprops=dict(linewidth=0),
            )

    ax.axvline(0, linestyle="--", color="gray", linewidth=1)

    ax.set_yticks(y_base)
    ax.set_yticklabels(sorted_genes)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Patient-level logFC (core vs inner band)")
    ax.set_ylabel("")
    ax.set_title("TCGA and CPTAC")

    for i in range(len(sorted_genes) - 1):
        ax.axhline(i + 0.5, color="lightgray", lw=0.5, alpha=0.45)

    direction_legend = [
        Patch(facecolor=direction_colors["core"], edgecolor="#3A3A3A", label="Core enriched"),
        Patch(facecolor=direction_colors["inner"], edgecolor="#3A3A3A", label="Inner-band enriched"),
    ]
    dataset_legend = [
        Patch(facecolor="white", edgecolor="#3A3A3A", linewidth=1.4, linestyle="-", label="TCGA"),
        Patch(facecolor="white", edgecolor="#8A8A8A", linewidth=1.2, linestyle="--", label="CPTAC"),
    ]

    leg1 = ax.legend(handles=direction_legend, frameon=False, loc="upper right", bbox_to_anchor=(1.36, 1.00))
    ax.add_artist(leg1)
    ax.legend(handles=dataset_legend, frameon=False, loc="upper right", bbox_to_anchor=(1.25, 0.88))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    jpg_path = os.path.join(out_dir, "Core_inner_DEG.jpg")

    plt.savefig(jpg_path, dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()

