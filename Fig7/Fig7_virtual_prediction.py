import os
import pickle
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams["pdf.fonttype"] = 42

if __name__ == "__main__":
    
    data_dir = "./Fig7_data"
    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    
    
    def plot_boxplot(
        plot_df,
        x,
        y,
        palette,
        ylabel,
        save_name,
        figsize=(4, 5),
        ylim=None,
        width=0.5,
    ):
        plot_df = plot_df.dropna(subset=[y])
    
        fig, ax = plt.subplots(figsize=figsize)
    
        sns.boxplot(
            data=plot_df,
            x=x,
            y=y,
            palette=palette,
            width=width,
            showcaps=False,
            boxprops={"linewidth": 1.5},
            whiskerprops={"linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 1.8},
            showfliers=False,
            ax=ax,
        )
    
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
    
        if ylim is not None:
            ax.set_ylim(*ylim)
    
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=11)
    
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
    
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, save_name), bbox_inches="tight")
        plt.show()
        plt.close()
    
    
    # ============================================================
    # 1. Gene-wise PCC: 8 μm vs 112 μm
    # ============================================================
    with open(os.path.join(data_dir, "correlation_metrics_P24_HE_only_RNA.pkl"), "rb") as f:
        results_8 = pickle.load(f)
    
    with open(os.path.join(data_dir, "correlation_metrics_P24_HE_only_RNA_112um.pkl"), "rb") as f:
        results_112 = pickle.load(f)
    
    plot_df_pcc = pd.concat(
        [
            pd.DataFrame({
                "Resolution": "8 μm",
                "Value": results_8["gene_pcc"],
            }),
            pd.DataFrame({
                "Resolution": "112 μm",
                "Value": results_112["gene_pcc_112"],
            }),
        ],
        ignore_index=True,
    )
    
    plot_boxplot(
        plot_df=plot_df_pcc,
        x="Resolution",
        y="Value",
        palette={
            "8 μm": "#F28C8C",
            "112 μm": "#FAD4D4",
        },
        ylabel="Gene-wise PCC",
        save_name="P24_gene_pcc.jpg",
        figsize=(4, 5),
        ylim=(0, 1),
        width=0.5,
    )
    
    
    # ============================================================
    # 2. Gene false positive: COSIE
    # ============================================================
    fp_gene = pd.read_csv(
        os.path.join(data_dir, "P24_LUAD_HE_fp_per_gene.csv"),
        index_col=0,
    ).squeeze()
    
    plot_df_gene_fp = pd.DataFrame({
        "Method": "COSIE",
        "Value": fp_gene,
    }).dropna()
    
    plot_boxplot(
        plot_df=plot_df_gene_fp,
        x="Method",
        y="Value",
        palette={"COSIE": "#F28C8C"},
        ylabel="False positives",
        save_name="P24_gene_false_positive_COSIE.jpg",
        figsize=(2.2, 5),
        ylim=(0, 0.21),
        width=0.45,
    )
    
    
    # ============================================================
    # 3. Protein false positive: COSIE vs GigaTIME
    # ============================================================
    fp_pro_cosie = pd.read_csv(
        os.path.join(data_dir, "fp_cosie_protein.csv")
    )
    
    fp_pro_gigatime = pd.read_csv(
        os.path.join(data_dir, "fp_gigatime_protein.csv")
    )
    
    plot_df_protein_fp = pd.concat(
        [
            pd.DataFrame({
                "Method": "COSIE",
                "Value": fp_pro_cosie["FP_mean_over_thresholds"],
            }),
            pd.DataFrame({
                "Method": "GigaTIME",
                "Value": fp_pro_gigatime["FP_mean_over_thresholds"],
            }),
        ],
        ignore_index=True,
    ).dropna()
    
    plot_boxplot(
        plot_df=plot_df_protein_fp,
        x="Method",
        y="Value",
        palette={
            "COSIE": "#F28C8C",
            "GigaTIME": "#6B8EC1",
        },
        ylabel="False Positive (FP)",
        save_name="P24_protein_false_positive.jpg",
        figsize=(4, 5),
        ylim=(0, 0.32),
        width=0.5,
    )

   
    df_112 = pd.read_csv('./Fig7_data/P24_LUAD_comparison_with_gigatime_112um.csv')
    df_8 = pd.read_csv('./Fig7_data/P24_LUAD_comparison_with_gigatime_8um.csv')
    protein_order = df_8["gt_protein"].tolist()
    gene_to_protein = {
        "SDC1": "CD138",  #
        "CD274": "PD-L1", #
        "ITGAX": "CD11c",  #
        "PDCD1": "PD-1",  # 
        "FCGR3A": "CD16",   #
        "CD3E": "CD3",  #
        "CD8A": "CD8", #
        "EPCAM": "EpCAM",
        "PTPRC": "CD45",  #
        "HLA-DRA": "HLA-DR", #
        "PECAM1": "CD31",  #
        "VIM": "Vimentin",  #
    }
    
    
    
    df_112["gt_protein"] = df_112["gt_protein"].map(lambda x: gene_to_protein.get(x, x))
    df_8["gt_protein"] = df_8["gt_protein"].map(lambda x: gene_to_protein.get(x, x))
    protein_order = df_8["gt_protein"].tolist()
    
    plot_df = []
    
    for _, row in df_8.iterrows():
        plot_df.append({
            "Protein": row["gt_protein"],
            "Group": "COSIE-8 μm",
            "Value": row["cosie_pcc"]
        })
        plot_df.append({
            "Protein": row["gt_protein"],
            "Group": "GigaTIME-8 μm",
            "Value": row["gigatime_pcc"]
        })
    
    for _, row in df_112.iterrows():
        plot_df.append({
            "Protein": row["gt_protein"],
            "Group": "COSIE-112 μm",
            "Value": row["cosie_pcc"]
        })
        plot_df.append({
            "Protein": row["gt_protein"],
            "Group": "GigaTIME-112 μm",
            "Value": row["gigatime_pcc"]
        })
    
    plot_df = pd.DataFrame(plot_df)
    
    # -----------------------------
    # Plot
    # -----------------------------
    groups = ["COSIE-8 μm", "COSIE-112 μm", "GigaTIME-8 μm", "GigaTIME-112 μm"]
    
    colors = {
        "COSIE-8 μm": "#F28C8C",
        "COSIE-112 μm": "#FAD4D4",
        "GigaTIME-8 μm": "#6B8EC1",
        "GigaTIME-112 μm": "#C7D4EA",
    }
    
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
            edgecolor="none",  #"black",
            linewidth=0.6,
            label=group
        )
    
    ax.set_ylabel("PCC")
    ax.set_xlabel("")
    ax.set_title("P24_LUAD Protein-level PCC")
    ax.set_xticks(x)
    ax.set_xticklabels(protein_order, rotation=30, ha="right")
    # ax.set_ylim(0, 1)
    
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.axhline(0, color="black", linewidth=1.2)
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    
    plt.tight_layout()
    
    plt.savefig(
        "P24_protein_pcc_barplot_8um_112um_combined.jpg",
        bbox_inches="tight"
    )
    
    plt.show()
    plt.close()
        
        
    










