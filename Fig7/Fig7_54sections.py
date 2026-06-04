import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __name__ == "__main__":

    clustering_result = pd.read_csv('./Fig7_data/COSIE_Foundation_54sections_clustering.csv')
    
    plot_df = pd.concat([
        pd.DataFrame({
            "Metric": "ARI",
            "Value": clustering_result["ari_merged"].values
        }),
        pd.DataFrame({
            "Metric": "NMI",
            "Value": clustering_result["nmi_merged"].values
        })
    ], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=(5,5))
    
    palette = {
        "ARI": "#F28C8C",
        "NMI": "#F28C8C"
    }
    
    sns.violinplot(
        data=plot_df,
        x="Metric",
        y="Value",
        inner="quartile",
        cut=0,
        linewidth=1.5,
        palette=palette,
        ax=ax
    )
    
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0,1)
    
    ax.tick_params(axis="x", labelrotation=35, labelsize=12)
    ax.tick_params(axis="y", labelsize=11)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    
    plt.tight_layout()
    
    plt.savefig(
        "ARI_NMI_violin.jpg",
        bbox_inches="tight"
    )
    
    plt.show()
    plt.close()



    ## Tumor fraction
    df_plot = pd.read_csv('./Fig7_data/54sections_ct_proportions.csv')
    df_plot["stage_merge"] = df_plot["tumor_stage"].replace({
        "Normal": "Normal/AAH",
        "AAH": "Normal/AAH"
    })
    stage_order = ["Normal/AAH", "AIS", "MIA", "LUAD"]
    
    df_plot = df_plot[
        df_plot["stage_merge"].isin(stage_order)
    ].copy()
    value_col = "label_6_proportion"
    
    df2 = df_plot[df_plot["stage_merge"].isin(stage_order)].copy()
    
    # -----------------------------
    # aesthetics
    # -----------------------------
    inner_radius = 0.28
    
    ring_step = 0.078      
    point_jitter = 0.0035
    
    lw_bg = 26
    lw_fg = 26
    
    bg_alpha = 0.13
    fg_alpha = 0.52
    point_alpha = 0.60
    
    vmin = 0
    vmax = 0.70
    
    arc_gap_deg = 20
    arc_span = 2 * np.pi - np.deg2rad(arc_gap_deg)
    
    colors = {
        "Normal/AAH": "#4F93D2",
        "AIS": "#5FAE7D",
        "MIA": "#F07C13",
        "LUAD": "#9C2FB0",
    }
    
    radii = inner_radius + np.arange(len(stage_order)) * ring_step
    
    fig = plt.figure(figsize=(8, 7))
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    
    theta_bg = np.linspace(0, arc_span, 1500)
    
    for i, stage in enumerate(stage_order):
        vals = df2.loc[df2["stage_merge"] == stage, value_col].dropna().values
        vals = np.clip(vals, vmin, vmax)
    
        r = radii[i]
        c = colors[stage]
    
        # light background arc
        ax.plot(
            theta_bg,
            np.full_like(theta_bg, r),
            color=c,
            alpha=bg_alpha,
            linewidth=lw_bg,
            solid_capstyle= 'butt',  #"round",
            zorder=1,
        )
    
        # median progress arc
        med = np.median(vals)
        med_theta = (med - vmin) / (vmax - vmin) * arc_span
    
        theta_med = np.linspace(
            0,
            med_theta,
            max(5, int(1500 * med_theta / arc_span))
        )
    
        ax.plot(
            theta_med,
            np.full_like(theta_med, r),
            color=c,
            alpha=fg_alpha,
            linewidth=lw_fg,
            solid_capstyle= 'butt',  # "round",
            zorder=4,
        )
    
        # individual section dots
        theta_vals = (vals - vmin) / (vmax - vmin) * arc_span
    
        rng = np.random.default_rng(100 + i)
        rr = r + rng.uniform(-point_jitter, point_jitter, len(theta_vals))
    
        ax.scatter(
            theta_vals,
            rr,
            s=110,
            color=c,
            edgecolor="white",
            linewidth=0.3,
            alpha=point_alpha,
            zorder=5,
        )
    
    # center white circle
    center = plt.Circle(
        (0, 0),
        inner_radius - 0.045,
        transform=ax.transData._b,
        color="white",
        zorder=20,
    )
    ax.add_artist(center)
    
    ax.text(
        0,
        0,
        "Label 6\nproportion",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        transform=ax.transData._b,
        zorder=21,
    )
    
    # ticks
    tick_vals = np.linspace(vmin, vmax, 5)
    tick_theta = (tick_vals - vmin) / (vmax - vmin) * arc_span
    
    ax.set_xticks(tick_theta)
    ax.set_xticklabels([f"{x:.2f}" for x in tick_vals], fontsize=10)
    
    ax.set_ylim(0, radii[-1] + ring_step * 1.4)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    
    # legend
    legend_elements = [
        Line2D(
            [0], [0],
            color=colors[s],
            lw=8,
            alpha=0.75,
            label=s,
        )
        for s in stage_order
    ]
    
    ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.05, 0.5),
        frameon=False,
        fontsize=12,
    )
    
    plt.tight_layout()
    plt.savefig(
        "Tumor_proportion_trend.jpg",
        bbox_inches="tight"
    )
    plt.show()


























###