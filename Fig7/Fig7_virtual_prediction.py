import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


if __name__ == "__main__":

    data_dir = "./Fig7_data"
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    color_8 = "#F28C8C"
    color_112 = "#FAD4D4"
    color_giga_8 = "#6B8EC1"
    color_giga_112 = "#C7D4EA"


    def format_axis(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(axis="both", labelsize=11)


    def plot_resolution_boxplot(file8, file112, title_prefix, output_prefix):
        df8 = pd.read_csv(os.path.join(data_dir, file8))
        df112 = pd.read_csv(os.path.join(data_dir, file112))

        for metric in ["PCC", "Spearman", "SSIM"]:

            plot_df = pd.concat([
                pd.DataFrame({"Resolution": "8 μm", "Value": df8[metric]}),
                pd.DataFrame({"Resolution": "112 μm", "Value": df112[metric]})
            ], ignore_index=True).dropna()

            fig, ax = plt.subplots(figsize=(4, 5))

            sns.boxplot(
                data=plot_df,
                x="Resolution",
                y="Value",
                hue="Resolution",
                palette={
                    "8 μm": color_8,
                    "112 μm": color_112
                },
                legend=False,
                width=0.5,
                showcaps=False,
                showfliers=False,
                boxprops={"linewidth": 1.5},
                whiskerprops={"linewidth": 1.5},
                medianprops={"color": "black", "linewidth": 1.8},
                ax=ax
            )

            ax.set_ylabel(metric)
            ax.set_xlabel("")
            ax.set_ylim((0, 1) if metric == "SSIM" else (-0.1, 1))
            ax.set_title(f"{title_prefix}: {metric}")

            format_axis(ax)
            plt.tight_layout()

            plt.savefig(
                os.path.join(out_dir, f"{output_prefix}_{metric.lower()}.jpg"),
                dpi=500,
                bbox_inches="tight"
            )

            plt.show()
            plt.close()


    def plot_protein_cosie_gigatime(file8, file112):

        df8 = pd.read_csv(os.path.join(data_dir, file8))
        df112 = pd.read_csv(os.path.join(data_dir, file112))

        # Display names
        gene_to_protein = {
            "SDC1": "CD138",
            "CD274": "PD-L1",
            "ITGAX": "CD11c",
            "PDCD1": "PD-1",
            "CD3E": "CD3",
            "CD8A": "CD8",
            "FCGR3A": "CD16",
            "EPCAM": "EpCAM",
            "PTPRC": "CD45",
            "HLA-DRA": "HLA-DR",
            "PECAM1": "CD31",
            "VIM": "Vimentin"
        }

        df8["Protein"] = df8["gt_protein"].map(lambda x: gene_to_protein.get(x, x))
        df112["Protein"] = df112["gt_protein"].map(lambda x: gene_to_protein.get(x, x))

        protein_order = df8["Protein"].tolist()

        assert df8["Protein"].is_unique
        assert df112["Protein"].is_unique
        assert set(df8["Protein"]) == set(df112["Protein"])

        df8 = df8.set_index("Protein").reindex(protein_order).reset_index()
        df112 = df112.set_index("Protein").reindex(protein_order).reset_index()

        metric_cols = {
            "PCC": ("cosie_pcc", "gigatime_pcc"),
            "Spearman": ("cosie_spearman", "gigatime_spearman"),
            "SSIM": ("cosie_ssim", "gigatime_ssim")
        }

        groups = [
            "COSIE-8 μm",
            "COSIE-112 μm",
            "GigaTIME-8 μm",
            "GigaTIME-112 μm"
        ]

        colors = {
            "COSIE-8 μm": color_8,
            "COSIE-112 μm": color_112,
            "GigaTIME-8 μm": color_giga_8,
            "GigaTIME-112 μm": color_giga_112
        }

        for metric, (cosie_col, gigatime_col) in metric_cols.items():

            plot_df = []

            for _, row in df8.iterrows():
                plot_df.append({
                    "Protein": row["Protein"],
                    "Group": "COSIE-8 μm",
                    "Value": row[cosie_col]
                })
                plot_df.append({
                    "Protein": row["Protein"],
                    "Group": "GigaTIME-8 μm",
                    "Value": row[gigatime_col]
                })

            for _, row in df112.iterrows():
                plot_df.append({
                    "Protein": row["Protein"],
                    "Group": "COSIE-112 μm",
                    "Value": row[cosie_col]
                })
                plot_df.append({
                    "Protein": row["Protein"],
                    "Group": "GigaTIME-112 μm",
                    "Value": row[gigatime_col]
                })

            plot_df = pd.DataFrame(plot_df)

            x = np.arange(len(protein_order))
            bar_width = 0.18
            offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

            fig, ax = plt.subplots(figsize=(18, 5))

            for k, group in enumerate(groups):

                vals = (
                    plot_df[plot_df["Group"] == group]
                    .set_index("Protein")
                    .reindex(protein_order)["Value"]
                    .values
                )

                ax.bar(
                    x + offsets[k],
                    vals,
                    width=bar_width,
                    color=colors[group],
                    edgecolor="none",
                    linewidth=0.6,
                    label=group
                )

            ax.set_ylabel(metric)
            ax.set_xlabel("")
            ax.set_title(f"P24_LUAD Protein-level {metric}")

            ax.set_xticks(x)
            ax.set_xticklabels(protein_order, rotation=30, ha="right")

            if metric == "SSIM":
                ax.set_ylim(0, 1)

            ax.legend(
                frameon=False,
                ncol=4,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18)
            )

            ax.axhline(0, color="black", linewidth=1.2)

            format_axis(ax)

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    out_dir,
                    f"HE_input_protein_COSIE_GigaTIME_{metric.lower()}.jpg"
                ),
                dpi=500,
                bbox_inches="tight"
            )

            plt.show()
            plt.close()


    plot_resolution_boxplot(
        "P24_HE_input_RNA_metric_8um.csv",
        "P24_HE_input_RNA_metric_112um.csv",
        "HE input → RNA",
        "HE_input_RNA"
    )


    plot_protein_cosie_gigatime(
        "P24_HE_input_protein_COSIE_GigaTIME_8um.csv",
        "P24_HE_input_protein_COSIE_GigaTIME_112um.csv"
    )


    plot_resolution_boxplot(
        "P24_RNA_input_protein_metric_8um.csv",
        "P24_RNA_input_protein_metric_112um.csv",
        "RNA input → Protein",
        "RNA_input_protein"
    )


    plot_resolution_boxplot(
        "P24_protein_input_RNA_metric_8um.csv",
        "P24_protein_input_RNA_metric_112um.csv",
        "Protein input → RNA",
        "protein_input_RNA"
    )