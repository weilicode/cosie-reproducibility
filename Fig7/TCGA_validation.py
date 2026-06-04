import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import spearmanr, kruskal, mannwhitneyu

mpl.rcParams["pdf.fonttype"] = 42


def format_p(p):
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def trim_group(group, value_col="Lymphoid aggregates", lower_q=0.05, upper_q=0.95):
    if len(group) < 10:
        return group.copy()
    low = group[value_col].quantile(lower_q)
    high = group[value_col].quantile(upper_q)
    return group[(group[value_col] >= low) & (group[value_col] <= high)].copy()


if __name__ == "__main__":

    df = pd.read_csv("./Fig7_data/TCGA_LUAD_celltype_overall_proportion_with_metadata.csv")

    # =====================================================
    # 1. TIL vs Lymphoid aggregates correlation
    # =====================================================
    x_col = "Lymphoid aggregates"
    y_col = "TIL.Regional.Fraction"

    if x_col in df.columns and y_col in df.columns:

        tmp = df[[x_col, y_col]].dropna().copy()

        if tmp.shape[0] >= 10:

            tmp[y_col] = tmp[y_col] / 100.0

            r, p = spearmanr(tmp[x_col], tmp[y_col])

            fig, ax = plt.subplots(figsize=(4.5, 4.5))

            sns.regplot(
                data=tmp,
                x=x_col,
                y=y_col,
                ax=ax,
                ci=95,
                scatter_kws={
                    "s": 20,
                    "alpha": 0.75,
                    "color": "#9CB65A",
                    "edgecolor": "#8DA44C",
                },
                line_kws={
                    "color": "#8AA54A",
                    "linewidth": 2.2,
                },
            )

            for collection in ax.collections:
                if collection.__class__.__name__ == "PolyCollection":
                    collection.set_facecolor("#DCE6C2")
                    collection.set_alpha(0.45)

            sns.despine(ax=ax)

            ax.set_title(f"{y_col}\nSpearman r={r:.3f}, p={p:.2e}", fontsize=10)
            ax.set_xlabel("Lymphoid aggregates proportion")
            ax.set_ylabel(y_col)
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")
            ax.set_box_aspect(1)

            plt.tight_layout()
            plt.savefig(
                "./TIL.Regional.Fraction_correlation.jpg",
                bbox_inches="tight",
            )
            plt.show()
            plt.close()

    # =====================================================
    # 2. Immune subtype vs Lymphoid aggregates
    # =====================================================
    cluster_col = "Immune.Subtype"
    value_col = "Lymphoid aggregates"

    df2 = df.copy()
    df2[cluster_col] = df2[cluster_col].astype(str).str.strip()
    df2.loc[df2[cluster_col].isin(["nan", "None", ""]), cluster_col] = np.nan
    df2 = df2.dropna(subset=[cluster_col, value_col]).copy()

    df_trim = (
        df2.groupby(cluster_col, group_keys=False)
        .apply(lambda g: trim_group(g, value_col=value_col))
        .reset_index(drop=True)
    )

    cluster_order = ["C4", "C1", "C2", "C3", "C6"]
    df_plot = df_trim[df_trim[cluster_col].isin(cluster_order)].copy()

    # overall Kruskal-Wallis
    groups = [
        df_plot.loc[df_plot[cluster_col] == st, value_col].values
        for st in cluster_order
        if np.sum(df_plot[cluster_col] == st) > 0
    ]

    stat_kw, p_kw = kruskal(*groups)

    # pairwise C4 vs others
    pairwise_p = {}
    c4 = df_plot.loc[df_plot[cluster_col] == "C4", value_col].dropna()

    for st in cluster_order:
        if st == "C4":
            continue

        other = df_plot.loc[df_plot[cluster_col] == st, value_col].dropna()

        if len(c4) == 0 or len(other) == 0:
            continue

        _, p = mannwhitneyu(c4, other, alternative="two-sided")
        pairwise_p[st] = p

    colors = {
        "C4": "#4C78A8",
        "C1": "#E45756",
        "C2": "#E45756",
        "C3": "#E45756",
        "C6": "#E45756",
    }

    sns.set_style("white", {
        "xtick.bottom": True,
        "ytick.left": True,
    })

    fig, ax = plt.subplots(figsize=(4.8, 3.8))

    sns.boxplot(
        data=df_plot,
        x=cluster_col,
        y=value_col,
        order=cluster_order,
        palette=colors,
        width=0.45,
        showfliers=False,
        ax=ax,
        boxprops={
            "edgecolor": "#666666",
            "linewidth": 1.2,
            "alpha": 0.9,
        },
        medianprops={
            "color": "#333333",
            "linewidth": 1.6,
        },
        whiskerprops={
            "color": "#666666",
            "linewidth": 1.0,
        },
        capprops={
            "linewidth": 0,
        },
    )

    # 星号：C4 vs others
    y_max = df_plot[value_col].max()
    y_min = df_plot[value_col].min()
    y_range = y_max - y_min if y_max > y_min else 1.0

    x_positions = {st: i for i, st in enumerate(cluster_order)}

    for st in cluster_order:
        if st == "C4" or st not in pairwise_p:
            continue

        group_vals = df_plot.loc[df_plot[cluster_col] == st, value_col]
        y = group_vals.max() + 0.04 * y_range

        ax.text(
            x_positions[st],
            y,
            format_p(pairwise_p[st]),
            ha="center",
            va="bottom",
            fontsize=13,
            color="#333333",
        )

    ax.axvspan(-0.45, 0.45, color="#7FA9CF", alpha=0.30, zorder=0)

    ax.set_xlabel("")
    ax.set_ylabel("Lymphoid aggregates proportion", fontsize=11)
    ax.set_title(f"Kruskal-Wallis p = {p_kw:.2e}", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(-0.55, len(cluster_order) - 0.45)
    ax.tick_params(axis="both", labelsize=10)

    sns.despine(ax=ax)

    plt.tight_layout()
    plt.savefig(
        "./Immune_subtype_box_clean.jpg",
        bbox_inches="tight",
    )
    plt.show()
    plt.close()