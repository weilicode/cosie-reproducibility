import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42


data_dir = Path("./Fig7_data")

group_col = "STK11"
group_order = ["WT", "Mut"]
features = ["CD45", "CD3", "CD8"]

colors = {
    "WT": "#BDBDBD",
    "Mut": "#2E86C1",
}


def plot_stk11_protein(cohort):
    plot_df = pd.read_csv(data_dir / f"{cohort}_STK11_protein_density_plot.csv")
    stats_df = pd.read_csv(data_dir / f"{cohort}_STK11_protein_density_stats.csv")

    out_dir = Path(f"./{cohort}_STK11_protein")
    out_dir.mkdir(parents=True, exist_ok=True)

    for feat in features:

        row = stats_df.loc[stats_df["feature"] == feat]

        if len(row) > 0:
            pval = row["pval"].values[0]
            fdr = row["pval_adj"].values[0]
            p_text = f"P = {pval:.2e}\nFDR = {fdr:.2e}"
        else:
            p_text = "P = NA\nFDR = NA"

        df_tmp = plot_df[[group_col, feat]].dropna().copy()
        df_tmp[feat] = df_tmp[feat].astype(float)

        tmp_list = []

        for group, g in df_tmp.groupby(group_col):
            if len(g) <= 10:
                tmp_list.append(g.copy())
            else:
                q05 = g[feat].quantile(0.05)
                q95 = g[feat].quantile(0.95)
                tmp_list.append(
                    g[(g[feat] >= q05) & (g[feat] <= q95)].copy()
                )

        df_tmp = pd.concat(tmp_list, axis=0, ignore_index=True)

        x = np.arange(len(group_order))
        fig, ax = plt.subplots(figsize=(2.5, 3.85))

        medians = []

        for i, group in enumerate(group_order):

            vals = df_tmp.loc[df_tmp[group_col] == group, feat].values

            if len(vals) == 0:
                medians.append(np.nan)
                continue

            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            med = np.median(vals)
            medians.append(med)

            ax.plot(
                [i, i],
                [q1, q3],
                lw=5,
                color=colors[group],
                alpha=0.7,
                solid_capstyle="round",
                zorder=2,
            )

            ax.scatter(
                i,
                med,
                s=130,
                color=colors[group],
                edgecolor="white",
                linewidth=1.2,
                zorder=3,
            )

            jitter = np.random.uniform(-0.06, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=10,
                color=colors[group],
                alpha=0.18,
                edgecolors="none",
                linewidths=0,
                zorder=1,
            )

        ax.plot(
            x,
            medians,
            color="gray",
            lw=1.8,
            alpha=0.6,
            zorder=1,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(group_order)
        ax.set_xlim(-0.5, len(group_order) - 0.5)
        ax.set_ylabel("Protein density")
        ax.set_title(feat, fontsize=13)

        ax.text(
            0.5,
            1.05,
            p_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        save_path = out_dir / f"{feat}_STK11_Mut_vs_WT_{cohort}.jpg"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        print("Saved:", save_path)


if __name__ == "__main__":
    plot_stk11_protein("TCGA")
    plot_stk11_protein("CPTAC")