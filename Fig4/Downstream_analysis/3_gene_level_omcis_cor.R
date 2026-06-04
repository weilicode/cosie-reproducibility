setwd("/mnt/HDD1B/Users/liran/10_integration_wei/0_downstream/Result_all")


seurat_s1 <- readRDS("seurat_slide1.rds")
seurat_s2 <- readRDS( "seurat_slide2.rds")


DefaultAssay(seurat_s1) <- 'rna_imputed.csv'
# seurat_s1 <- NormalizeData(seurat_s1)

counts_matrix <- GetAssayData(seurat_s1, slot = "counts")
seurat_s1[["rna_imputed.csv"]] <- CreateAssayObject(data = log1p(counts_matrix))


DefaultAssay(seurat_s2) <- 'RNA.csv'
# seurat_s2 <- NormalizeData(seurat_s2)

counts_matrix <- GetAssayData(seurat_s2, slot = "counts")
seurat_s2[["rna_imputed.csv"]] <- CreateAssayObject(data = log1p(counts_matrix))

# Load required libraries
library(Seurat)
library(ggplot2)
library(patchwork)  # For combining plots
library(dplyr)



########################
# correlation relationship
########################
# Define output directory
output_folder <- "0_gene_correlation_comparison_log_atac_var_gene"
dir.create(output_folder, showWarnings = FALSE)

# Function to generate all possible 2-way comparisons (choose 2)
generate_omic_pairs <- function(assays) {
  combn(assays, 2, simplify = FALSE)  # Get all combinations of 2
}

# Define available assays for both slides
assays_s1 <- c("atac_imputed.csv", "H3K27ac.csv", "H3K27me3.csv", "H3K4me3_imputed.csv", "rna_imputed.csv")
assays_s2 <- c("ATAC.csv", "H3K27ac_imputed.csv", "H3K27me3.csv", "H3K4me3.csv", "RNA.csv")

# Generate all valid 2-way comparisons
omic_pairs_s1 <- generate_omic_pairs(assays_s1)
omic_pairs_s2 <- generate_omic_pairs(assays_s2)



# Function to find the top 2000 most variable genes
get_top_variable_genes <- function(seurat_obj, omic1, omic2, n = 2000) {
  # Extract expression matrices
  expr_omic1 <- GetAssayData(seurat_obj, assay = omic1, slot = "data")
  expr_omic2 <- GetAssayData(seurat_obj, assay = omic2, slot = "data")
  
  # Identify common genes
  common_genes <- intersect(rownames(expr_omic1), rownames(expr_omic2))
  
  if (length(common_genes) < n) {
    warning(paste("Fewer than", n, "common genes found for", omic1, "vs", omic2))
  }
  
  # Find top variable genes in seurat_s1
  seurat_obj_subset <- subset(seurat_obj, features = common_genes)
  variable_genes <- FindVariableFeatures(seurat_obj_subset, selection.method = "vst", nfeatures = n)
  
  # Return top 2000 variable genes from common genes
  return(intersect(common_genes, VariableFeatures(variable_genes)))
}

# Function to compute correlation per gene for a given omic pair using predefined gene list
compute_gene_correlations <- function(seurat_obj, omic1, omic2, gene_list) {
  Idents(seurat_obj) <- seurat_obj$all_clusters  # Set cluster identities
  
  # Extract expression matrices
  expr_omic1 <- GetAssayData(seurat_obj, assay = omic1, slot = "data")[gene_list, ]
  expr_omic2 <- GetAssayData(seurat_obj, assay = omic2, slot = "data")[gene_list, ]
  
  # Compute Pearson correlation for each gene
  gene_correlations <- sapply(gene_list, function(gene) {
    return(cor(expr_omic1[gene,], expr_omic2[gene,], method = "pearson"))
  })
  
  # Convert to data frame
  gene_correlations_df <- data.frame(
    gene = gene_list,
    correlation = gene_correlations
  )
  
  return(na.omit(gene_correlations_df))  # Remove NA values
}


# Function to compare correlations across slides and generate density scatter plots
compare_gene_correlations <- function(seurat_s1, seurat_s2, omic_pairs_s1, omic_pairs_s2) {
  for (i in seq_along(omic_pairs_s1)) {
    omic_s1_1 <- omic_pairs_s1[[i]][1]
    omic_s1_2 <- omic_pairs_s1[[i]][2]
    omic_s2_1 <- omic_pairs_s2[[i]][1]
    omic_s2_2 <- omic_pairs_s2[[i]][2]
    
    print(paste0("Processing: ", omic_s1_1, " vs ", omic_s1_2, " (Slide 1) | ", omic_s2_1, " vs ", omic_s2_2, " (Slide 2)"))
    
    # Get gene list from Slide 1
    top_variable_genes <- get_top_variable_genes(seurat_s1, omic_s1_1, omic_s1_2, 2000)
    
    # Compute correlations
    cor_s1 <- compute_gene_correlations(seurat_s1, omic_s1_1, omic_s1_2, top_variable_genes)
    cor_s2 <- compute_gene_correlations(seurat_s2, omic_s2_1, omic_s2_2, top_variable_genes)
    
    # Merge correlations by gene
    df_correlation_plot <- merge(cor_s1, cor_s2, by = "gene", suffixes = c("_s1", "_s2"))
    
    # Compute Pearson correlation between slides
    pearson_r <- cor(df_correlation_plot$correlation_s1, df_correlation_plot$correlation_s2, method = "pearson")
    
    # Generate density scatter plot
    p <- ggplot(df_correlation_plot, aes(x = correlation_s1, y = correlation_s2)) +
      geom_bin2d(bins = 100) + 
      scale_fill_gradient(low = "blue", high = "yellow") +
      annotate("text", x = min(df_correlation_plot$correlation_s1), y = max(df_correlation_plot$correlation_s2), 
               label = paste0("r = ", round(pearson_r, 2)), hjust = 0, size = 6) +
      theme_minimal() +
      labs(
        x = paste0(omic_s1_1, " - ", omic_s1_2, " (Slide 1)"),
        y = paste0(omic_s2_1, " - ", omic_s2_2, " (Slide 2)"),
        title = paste0(omic_s1_1, " vs ", omic_s1_2, " | ", omic_s2_1, " vs ", omic_s2_2)
      )
    
    # Save plots
    pdf(paste0(output_folder, "/correlation_", omic_s1_1, "_", omic_s1_2, "_vs_", omic_s2_1, "_", omic_s2_2, ".pdf"), width = 5, height = 5)
    print(p)
    dev.off()
    
    png(paste0(output_folder, "/correlation_", omic_s1_1, "_", omic_s1_2, "_vs_", omic_s2_1, "_", omic_s2_2, ".png"), width = 500, height = 500)
    print(p)
    dev.off()
  }
}

compare_gene_correlations <- function(seurat_s1, seurat_s2, omic_pairs_s1, omic_pairs_s2) {
  for (i in seq_along(omic_pairs_s1)) {
    omic_s1_1 <- omic_pairs_s1[[i]][1]
    omic_s1_2 <- omic_pairs_s1[[i]][2]
    omic_s2_1 <- omic_pairs_s2[[i]][1]
    omic_s2_2 <- omic_pairs_s2[[i]][2]
    
    print(paste0("Processing: ", omic_s1_1, " vs ", omic_s1_2, " (Slide 1) | ", omic_s2_1, " vs ", omic_s2_2, " (Slide 2)"))
    
    # Get gene list from Slide 1
    top_variable_genes <- get_top_variable_genes(seurat_s1, omic_s1_1, omic_s1_2, 2000)
    
    # Compute correlations
    cor_s1 <- compute_gene_correlations(seurat_s1, omic_s1_1, omic_s1_2, top_variable_genes)
    cor_s2 <- compute_gene_correlations(seurat_s2, omic_s2_1, omic_s2_2, top_variable_genes)
    
    # Merge correlations by gene
    df_correlation_plot <- merge(cor_s1, cor_s2, by = "gene", suffixes = c("_s1", "_s2"))
    
    # ✅ Save merged correlation data to CSV
    write.csv(df_correlation_plot,
              file = paste0(output_folder, "/correlation_", omic_s1_1, "_", omic_s1_2, "_vs_", omic_s2_1, "_", omic_s2_2, ".csv"),
              row.names = FALSE)
    
    # Compute Pearson correlation between slides
    pearson_r <- cor(df_correlation_plot$correlation_s1, df_correlation_plot$correlation_s2, method = "pearson")
    
    # Generate density scatter plot
    p <- ggplot(df_correlation_plot, aes(x = correlation_s1, y = correlation_s2)) +
      geom_bin2d(bins = 100) + 
      scale_fill_gradient(low = "blue", high = "yellow") +
      annotate("text", x = min(df_correlation_plot$correlation_s1), y = max(df_correlation_plot$correlation_s2), 
               label = paste0("r = ", round(pearson_r, 2)), hjust = 0, size = 6) +
      theme_minimal() +
      labs(
        x = paste0(omic_s1_1, " - ", omic_s1_2, " (Slide 1)"),
        y = paste0(omic_s2_1, " - ", omic_s2_2, " (Slide 2)"),
        title = paste0(omic_s1_1, " vs ", omic_s1_2, " | ", omic_s2_1, " vs ", omic_s2_2)
      )
    
    # Save plots
    pdf(paste0(output_folder, "/correlation_", omic_s1_1, "_", omic_s1_2, "_vs_", omic_s2_1, "_", omic_s2_2, ".pdf"), width = 5, height = 5)
    print(p)
    dev.off()
    
    png(paste0(output_folder, "/correlation_", omic_s1_1, "_", omic_s1_2, "_vs_", omic_s2_1, "_", omic_s2_2, ".png"), width = 500, height = 500)
    print(p)
    dev.off()
  }
}


# Run analysis for Seurat S1 and Seurat S2
DefaultAssay(seurat_s1) <- 'atac_imputed.csv'
DefaultAssay(seurat_s2) <- 'ATAC.csv'
compare_gene_correlations(seurat_s1, seurat_s2, omic_pairs_s1, omic_pairs_s2)
