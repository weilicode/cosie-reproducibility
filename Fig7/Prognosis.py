import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
if __name__ == "__main__":
    
    boot_cidx_all = np.load('./Fig7_data/C_index_results_all/bootstrap_1000_cidx_median.npy')
    boot_cidx = np.load('./Fig7_data/C_index_results_all/bootstrap_1000_cidx_stageI_only_median.npy')
    
    all_stage = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_stage_median.npy')
    all_age = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_age_median.npy')
    all_gender = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_gender_median.npy')
    all_race = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_race_median.npy')
    all_egfr = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_EGFR_mut_median.npy')
    all_stk11 = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_STK11_mut_median.npy')
    all_kras = np.load('./Fig7_data/C_index_results_all/CPTAC_all_stages_bootstrap_KRAS_mut_median.npy')

    stage1_age = np.load('./Fig7_data/C_index_results_all/CPTAC_stageI_bootstrap_age_median.npy')
    stage1_gender = np.load('./Fig7_data/C_index_results_all/CPTAC_stageI_bootstrap_gender_median.npy')
    stage1_race = np.load('./Fig7_data/C_index_results_all/CPTAC_stageI_bootstrap_race_median.npy')
    stage1_egfr = np.load('./Fig7_data/C_index_results_all/CPTAC_stageI_bootstrap_EGFR_mut_median.npy')
    stage1_stk11 = np.load('./Fig7_data/C_index_results_all/CPTAC_stageI_bootstrap_STK11_mut_median.npy')
    stage1_kras = np.load('./Fig7_data/C_index_results_all/CPTAC_stageI_bootstrap_KRAS_mut_median.npy')

    methods = {
        "COSIE": {
            "all": boot_cidx_all,
            "stage1": boot_cidx,
        },
        "Stage": {
            "all": all_stage,
            "stage1": None,   
        },
        "KRAS": {
            "all": all_kras,
            "stage1": stage1_kras,
        },
        "STK11": {
            "all": all_stk11,
            "stage1": stage1_stk11,
        },
        "EGFR": {
            "all": all_egfr,
            "stage1": stage1_egfr,
        },
        "Age": {
            "all": all_age,
            "stage1": stage1_age,
        },
        "Race": {
            "all": all_race,
            "stage1": stage1_race,
        },
        "Gender": {
            "all": all_gender,
            "stage1": stage1_gender,
        },
        
    }
    
    group_info = [
        ("All stages", "all"),
        ("Stage I only", "stage1"),
    ]
    
    colors = {
        "COSIE":   "#F28C8C",  
        "Stage":   "#A9D6E5",  # soft cyan
        "Age":     "#B7B5E4",  # lavender
        "Gender":  "#BFD8B8",  
        "Race":    "#F3D08B",  # pastel gold
        "EGFR":    "#8FB9D8",  # muted blue
        "STK11":   "#F2B6A0",  # soft salmon
        "KRAS":    "#D8CFC4",  # warm gray
    }
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    group_centers = np.arange(len(group_info))
    group_total_width = 0.82
    
    for gi, (group_label, group_key) in enumerate(group_info):
        valid_methods = [
            m for m in methods
            if methods[m][group_key] is not None
        ]
    
        n = len(valid_methods)
        bar_width = group_total_width / n
        offsets = (np.arange(n) - (n - 1) / 2) * bar_width
    
        for j, method in enumerate(valid_methods):
            arr = np.asarray(methods[method][group_key], dtype=float)
    
            mean_val = np.nanmean(arr)
            std_val = np.nanstd(arr)
    
            ax.bar(
                group_centers[gi] + offsets[j],
                mean_val,
                yerr=std_val,
                width=bar_width * 0.9,
                capsize=4,
                color=colors[method],
                edgecolor="black",
                linewidth=0.8,
                label=method if gi == 0 else None,
            )
    
    ax.set_ylabel("C-index", fontsize=12)
    ax.set_xticks(group_centers)
    ax.set_xticklabels([g[0] for g in group_info], fontsize=11)
    ax.set_title("CPTAC external validation", fontsize=13)
    ax.set_ylim(0.3, 0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    handles, labels = ax.get_legend_handles_labels()
    existing = set(labels)
    
    for method in methods:
        if method not in existing:
            ax.bar(
                np.nan, np.nan,
                color=colors[method],
                edgecolor="black",
                linewidth=0.8,
                label=method
            )
    
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    plt.tight_layout()
    plt.savefig("C_index.jpg", bbox_inches="tight")
    
    plt.show()


    from scipy.stats import wilcoxon
    from statsmodels.stats.multitest import multipletests
    
    comparisons_all = {
        "Stage": all_stage,
        "Age": all_age,
        "Gender": all_gender,
        "Race": all_race,
        "EGFR": all_egfr,
        "STK11": all_stk11,
        "KRAS": all_kras
    }
    
    pvals=[]
    
    for name, arr in comparisons_all.items():
    
        _, p = wilcoxon(
            boot_cidx_all,
            arr,
            alternative="greater"
        )
    
        print(name, p)
        pvals.append(p)
    
    
    adj_p = multipletests(
        pvals,
        method="fdr_bh"
    )[1]
    
    print("\nAll stages FDR-adjusted:")
    for n,p in zip(comparisons_all.keys(), adj_p):
        print(n,p)


    comparisons_stage1 = {
        "Age": stage1_age,
        "Gender": stage1_gender,
        "Race": stage1_race,
        # "Mutation": stage1_mut,
        "EGFR": stage1_egfr,
        "STK11": stage1_stk11,
        "KRAS": stage1_kras
    }
    
    pvals=[]
    
    for name, arr in comparisons_stage1.items():
    
        _, p = wilcoxon(
            boot_cidx,
            arr,
            alternative="greater"
        )
    
        print(name, p)
        pvals.append(p)
    
    
    adj_p = multipletests(
        pvals,
        method="fdr_bh"
    )[1]
    
    print("\nStage I FDR-adjusted:")
    for n,p in zip(comparisons_stage1.keys(), adj_p):
        print(n,p)
    
