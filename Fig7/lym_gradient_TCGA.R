library(survival)
library(glue)
library(dplyr)
library(tidyr)
library(glmnet)
setwd("~/Desktop/RA/COSIE/COSIE-Survival_MIL/200um_version/TCGA")

plot_surv_if_sig <- function(dat,
                             dir,
                             file_name,
                             title,
                             conf.int = TRUE) {
  plot_dat <- dat[, c("OS.Time", "OS", "component"), drop = FALSE]
  plot_dat <- plot_dat[complete.cases(plot_dat), , drop = FALSE]
  colnames(plot_dat) <- c("time", "status", "group")
  plot_dat$group <- factor(plot_dat$group)
  
  fit_risk <- survival::survfit(
    survival::Surv(time, status) ~ group,
    data = plot_dat
  )
  
  p <- survminer::ggsurvplot(
    fit_risk,
    data = plot_dat,
    conf.int = conf.int,
    pval = TRUE,
    risk.table = FALSE,
    title = title
  )
  
  ggplot2::ggsave(
    filename = file.path(dir, file_name),
    plot = p$plot,
    width = 6,
    height = 5,
    dpi = 300
  )
  
  return(p)
}



cox_component_group <- function(df, 
                                dir,
                                adj_var,
                                col_prefix_lst, 
                                startwith = TRUE,
                                df_cols = NULL,
                                thre_lst = list(c(0.3, 0.7)),
                                sig_thre = 0.05,
                                min_n = 10) {
  if (is.null(df_cols)) {
    df_cols <- colnames(df)
  }
  
  for (col_prefix in col_prefix_lst) {
    if (startwith) {
      auto_cols <- df_cols[startsWith(df_cols, col_prefix)]
    } else {
      auto_cols <- df_cols[endsWith(df_cols, col_prefix)]
    }
    
    if (length(auto_cols) == 0) {
      message("No columns matched prefix: ", col_prefix)
      next
    }
    
    prefix_clean <- gsub("[^A-Za-z0-9_]", "_", col_prefix)
    
    for (thre in thre_lst) {
      q_low  <- thre[1]
      q_high <- thre[2]
      thre_label <- paste0("q_low", q_low, "_high", q_high)
      
      # create subfolder: dir_root / prefix_clean / thre_label
      sub_dir <- file.path(dir, prefix_clean, thre_label)
      if (!dir.exists(sub_dir)) {
        dir.create(sub_dir, recursive = TRUE, showWarnings = FALSE)
      }
      
      for (col_name in auto_cols) {
        # derive short name by stripping prefix
        if (startwith) {
          short_name <- sub(paste0("^", gsub("([.|()\\^{}+$*?])", "\\\\\\1", col_prefix)), "", col_name)
        } else {
          short_name <- sub(paste0(gsub("([.|()\\^{}+$*?])", "\\\\\\1", col_prefix), "$"), "", col_name)
        }
        short_name <- gsub("^_+|_+$", "", short_name)
        if (nchar(short_name) == 0) short_name <- col_name
        
        ct_cols_now <- col_name
        selected_cols <- c("OS.Time", "OS", ct_cols_now, adj_var)
        miss_cols <- setdiff(selected_cols, colnames(df))
        if (length(miss_cols) > 0) {
          message("Skipping ", col_name, ": missing columns: ", paste(miss_cols, collapse = ", "))
          next
        }
        
        dat_sub <- df[, selected_cols, drop = FALSE]
        dat_sub <- dat_sub[complete.cases(dat_sub), , drop = FALSE]
        
        if (nrow(dat_sub) < min_n) {
          message("Skipping ", col_name, ": too few complete cases.")
          next
        }
        
        high_cut <- quantile(dat_sub[[ct_cols_now]], q_high, na.rm = TRUE)
        low_cut  <- quantile(dat_sub[[ct_cols_now]], q_low,  na.rm = TRUE)
        
        high_idx <- dat_sub[[ct_cols_now]] >= high_cut
        low_idx  <- dat_sub[[ct_cols_now]] <= low_cut
        
        dat_extreme <- dat_sub[high_idx | low_idx, , drop = FALSE]
        print(paste(col_prefix, thre_label, col_name, nrow(dat_extreme)))
        
        if (nrow(dat_extreme) < min_n) {
          message("Skipping ", col_name, ": too few patients after filtering.")
          next
        }
        if (sum(high_idx) == 0 || sum(low_idx) == 0) {
          message("Skipping ", col_name, ": no valid high/low groups.")
          next
        }
        
        dat_extreme$component <- ifelse(high_idx[high_idx | low_idx], "high", "low")
        dat_extreme$component <- factor(dat_extreme$component, levels = c("low", "high"))
        
        form <- as.formula(
          paste("Surv(OS.Time, OS) ~", paste(c(ct_cols_now, adj_var), collapse = " + "))
        )
        
        cox_fit <- try(coxph(form, data = dat_extreme), silent = TRUE)
        if (inherits(cox_fit, "try-error")) {
          message("Skipping ", col_name, ": coxph failed.")
          next
        }
        
        cox_sum <- summary(cox_fit)
        coef_mat <- cox_sum$coefficients
        coef_mat_ct <- coef_mat[rownames(coef_mat) %in% ct_cols_now, , drop = FALSE]
        ct_pvals <- coef_mat_ct[, "Pr(>|z|)"]
        plot_flag <- any(!is.na(ct_pvals) & ct_pvals < sig_thre)
        
        if (plot_flag) {
          plot_surv_if_sig(
            dat = dat_extreme,
            dir = sub_dir,
            title = paste0(col_name, " (", thre_label, ")"),
            file_name = paste0(short_name, "_sig_surv.pdf")
          )
        }
        
        out <- c(
          paste0("Prefix: ", col_prefix),
          paste0("Column: ", col_name),
          paste0("High cutoff quantile: ", q_high),
          paste0("Low cutoff quantile: ", q_low),
          paste0("N high: ", sum(dat_extreme$component == "high")),
          paste0("N low: ", sum(dat_extreme$component == "low")),
          "",
          capture.output(summary(cox_fit))
        )
        
        writeLines(
          out,
          file.path(sub_dir, paste0(short_name, "_component_surv_summary.txt"))
        )
      }
    }
  }
}



df <- read.csv("TCGA_LUAD_boundary_celltype_gradients_OLS_updated_with_metadata.csv", row.names = 1)
df$demographic.race[df$demographic.race %in% c("not reported", "american indian or alaska native", "Unknown", "Asia", "")] <- "Others"
df$stage[df$stage == ""] <- "Unknown"
df$EGFR_Mut[df$EGFR_Mut == ""] <- "WT"
df$STK11_Mut[df$STK11_Mut == ""] <- "WT"
df$KRAS_Mut[df$KRAS_Mut == ""] <- "WT"



adj <- c( "demographic.age_at_index", "demographic.gender", "demographic.race",
          "EGFR_Mut", "STK11_Mut", "KRAS_Mut", "stage")

dir_root <- "Spatial_feature"


#################################### CT gradient ####################################
col_prefix_lst <- c("inner_r1_5")

thre_lst <- list( c(0.2, 0.8))

cox_component_group(
  df = df, col_prefix_lst = col_prefix_lst, 
  dir = dir_root,
  adj_var = adj, 
  thre_lst = thre_lst,
  sig_thre = 0.1
)

