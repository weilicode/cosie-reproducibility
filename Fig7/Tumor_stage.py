import os
import re
from pathlib import Path
from typing import List, Optional, Dict
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu
from scipy.stats import kruskal

if __name__ == "__main__":

    density_cols = ["protein_PD-L1__density", "protein_CD3__density", "protein_CD8__density", "protein_CD45__density"]
    TCGA_protein_with_stage = pd.read_csv('./Fig7_data/TCGA_protein_with_stage.csv')    
    df_stage = pd.read_csv('./Fig7_data/TCGA_protein_density_with_stage.csv')
    out_dir = "./TCGA_protein_with_stage"
    os.makedirs(out_dir, exist_ok=True)
    
    stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    
    colors = {
        "Stage I": "#D8BFD8",
        "Stage II": "#A569BD",
        "Stage III": "#7D3C98",
        "Stage IV": "#4A235A",
    }
    
    for feat in density_cols:
    
        row = TCGA_protein_with_stage.loc[TCGA_protein_with_stage["feature"] == feat]
    
        if len(row) > 0:
            rho = row["spearman_rho"].values[0]
            fdr = row["trend_FDR"].values[0]
            fdr_text = f"ρ = {rho:.2f}, Trend FDR = {fdr:.2e}"
        else:
            fdr_text = "Trend FDR = NA"
    
        df_plot = df_stage[["stage", feat]].dropna().copy()
        df_plot[feat] = df_plot[feat].astype(float)
    
        df_plot = (
            df_plot
            .groupby("stage", group_keys=False)
            .apply(
                lambda g: g.copy() if len(g) <= 10 else g[
                    (g[feat] >= g[feat].quantile(0.05)) &
                    (g[feat] <= g[feat].quantile(0.95))
                ].copy()
            )
        )
    
        protein_name = feat.replace("protein_", "").replace("__density", "")
    
        x = np.arange(len(stage_order))
    
        fig, ax = plt.subplots(figsize=(5, 5))
    
        medians = []
    
        for i, stage in enumerate(stage_order):
    
            vals = df_plot.loc[
                df_plot["stage"] == stage,
                feat
            ].values
    
            if len(vals) == 0:
                medians.append(np.nan)
                continue
    
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            med = np.median(vals)
            medians.append(med)
    
            # IQR line
            ax.plot(
                [i, i],
                [q1, q3],
                lw=5,
                color=colors[stage],
                alpha=0.7,
                solid_capstyle="round",
                zorder=2
            )
    
            # median point
            ax.scatter(
                i,
                med,
                s=130,
                color=colors[stage],
                edgecolor="white",
                linewidth=1.2,
                zorder=3
            )
    
            # raw points
            jitter = np.random.uniform(-0.06, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=10,
                color=colors[stage],
                alpha=0.18,
                # zorder=1
                edgecolors='none',   
                linewidths=0         
            )
    
        # connect medians
        ax.plot(
            x,
            medians,
            color="gray",
            lw=1.8,
            alpha=0.6
        )
    
        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_xlim(-0.5, len(stage_order) - 1 + 0.5)
        ax.set_ylabel("Protein density")
        ax.set_title(protein_name, fontsize=13)
    
        ax.text(
            0.5,
            1.05,
            fdr_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10
        )
    
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
        plt.tight_layout()
    
        save_path = os.path.join(out_dir, f"{protein_name}_TCGA.jpg")
        plt.savefig(save_path,dpi=500, bbox_inches="tight")
        plt.show()
        plt.close()
    
        print("Saved:", save_path)


    CPTAC_protein_with_stage = pd.read_csv('./Fig7_data/CPTAC_protein_with_stage.csv')
    df_stage = pd.read_csv('./Fig7_data/CPTAC_protein_density_with_stage.csv')
    out_dir = "./CPTAC_protein_with_stage"
    os.makedirs(out_dir, exist_ok=True)
    
    stage_order = ["Stage I", "Stage II", "Stage III", "Stage IV"]
    
    colors = {
        "Stage I": "#D8BFD8",
        "Stage II": "#A569BD",
        "Stage III": "#7D3C98",
        "Stage IV": "#4A235A",
    }
    
    for feat in density_cols:
    
        row = CPTAC_protein_with_stage.loc[CPTAC_protein_with_stage["feature"] == feat]
    
        if len(row) > 0:
            rho = row["spearman_rho"].values[0]
            fdr = row["trend_FDR"].values[0]
            fdr_text = f"ρ = {rho:.2f}, Trend FDR = {fdr:.2e}"
        else:
            fdr_text = "Trend FDR = NA"
    
        df_plot = df_stage[["stage", feat]].dropna().copy()
        df_plot[feat] = df_plot[feat].astype(float)
    
        df_plot = (
            df_plot
            .groupby("stage", group_keys=False)
            .apply(
                lambda g: g.copy() if len(g) <= 10 else g[
                    (g[feat] >= g[feat].quantile(0.05)) &
                    (g[feat] <= g[feat].quantile(0.95))
                ].copy()
            )
        )
    
        protein_name = feat.replace("protein_", "").replace("__density", "")
    
        x = np.arange(len(stage_order))
    
        fig, ax = plt.subplots(figsize=(5, 5))
    
        medians = []
    
        for i, stage in enumerate(stage_order):
    
            vals = df_plot.loc[
                df_plot["stage"] == stage,
                feat
            ].values
    
            if len(vals) == 0:
                medians.append(np.nan)
                continue
    
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            med = np.median(vals)
            medians.append(med)
    
            # IQR line
            ax.plot(
                [i, i],
                [q1, q3],
                lw=5,
                color=colors[stage],
                alpha=0.7,
                solid_capstyle="round",
                zorder=2
            )
    
            # median point
            ax.scatter(
                i,
                med,
                s=130,
                color=colors[stage],
                edgecolor="white",
                linewidth=1.2,
                zorder=3
            )
    
            # raw points
            jitter = np.random.uniform(-0.06, 0.06, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=10,
                color=colors[stage],
                alpha=0.18,
                # zorder=1
                edgecolors='none',   
                linewidths=0         
            )
    
        # connect medians
        ax.plot(
            x,
            medians,
            color="gray",
            lw=1.8,
            alpha=0.6
        )
    
        ax.set_xticks(x)
        ax.set_xticklabels(stage_order)
        ax.set_xlim(-0.5, len(stage_order) - 1 + 0.5)
        ax.set_ylabel("Protein density")
        ax.set_title(protein_name, fontsize=13)
    
        ax.text(
            0.5,
            1.05,
            fdr_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=10
        )
    
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
        plt.tight_layout()
    
        save_path = os.path.join(out_dir, f"{protein_name}_CPTAC.jpg")
        plt.savefig(save_path, dpi=500, bbox_inches="tight")
        plt.show()
        plt.close()
    
        print("Saved:", save_path)
        