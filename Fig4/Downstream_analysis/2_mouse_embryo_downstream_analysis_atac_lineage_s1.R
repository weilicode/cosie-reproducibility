setwd('/mnt/HDD1/Users/liran/10_integration_wei/0_downstream/Result_all')

library(Seurat)
library(Matrix)
library(tidyverse)
library(slingshot)
library(pheatmap)


seurat_s1 <- readRDS(file = "seurat_slide1.rds")

#####################
# lineage
#####################
max(seurat_s1_sub@images$slice1@coordinates$row)
min(seurat_s1_sub@images$slice1@coordinates$row)


seurat_s1_sub <- subset(seurat_s1, cells = rownames(seurat_s1@meta.data[seurat_s1$all_clusters == 1 | seurat_s1$all_clusters == 8,]))

seurat_s1_sub2 <- subset(seurat_s1_sub, cells = rownames(seurat_s1_sub@images$slice1@coordinates[seurat_s1_sub@images$slice1@coordinates$row > 25,]))

seurat_s1_sub <- seurat_s1_sub2
DefaultAssay(seurat_s1_sub) <- 'atac_imputed.csv'
seurat_s1_sub <-  FindVariableFeatures(seurat_s1_sub)
seurat_s1_sub <- ScaleData(seurat_s1_sub)
seurat_s1_sub <- RunPCA(seurat_s1_sub)
seurat_s1_sub <- FindNeighbors(seurat_s1_sub, dims = 1:30)
seurat_s1_sub <- FindClusters(seurat_s1_sub, resolution = 0.5)
seurat_s1_sub <- RunUMAP(seurat_s1_sub, dims = 1:30)

DimPlot(seurat_s1_sub, group.by = 'all_clusters')

group_by_name = 'all_clusters'
reduction_fun =  'umap'
outputname = 's1_embbedding_atac'
output_dir = 'lineage_atac_s1/'
dir.create(output_dir, showWarnings = FALSE)

p3 <- DimPlot(seurat_s1_sub,label=TRUE,pt.size=0.5,reduction = reduction_fun)
print(p3)

pdf(paste0(output_dir, outputname, "_clusters_umap_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 4)
print(p3)
dev.off()

p4 <- SpatialPlot(seurat_s1_sub, label = FALSE, label.size = 3, crop = FALSE, pt.size.factor = 4.5, image.alpha = 0.5, stroke = 0)
p4$layers[[1]]$aes_params <- c(p4$layers[[1]]$aes_params, shape=22)
print(p4)


pdf(paste0(output_dir, outputname, "_clusters_spatial_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 5)
print(p4)
dev.off()

sshot <- slingshot(data = Embeddings(seurat_s1_sub,reduction = reduction_fun), clusterLabels = seurat_s1_sub$all_clusters)

pt <- slingPseudotime(sshot)

seurat_s1_sub@meta.data <- cbind(seurat_s1_sub@meta.data, pt)

p1 <- FeaturePlot(seurat_s1_sub,features = 'Lineage1', reduction = reduction_fun) + scale_color_viridis_c()
print(p1)

pdf(paste0(output_dir, outputname,"_umap_slingshot_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 4)
print(p1)
dev.off()
# 
seurat_s1_sub$Lineage1[is.na(seurat_s1_sub$Lineage1)] <- 0
p3 <- SpatialPlot(seurat_s1_sub, label = FALSE, label.size = 3, features = 'Lineage1',crop = FALSE, pt.size.factor = 4.5,  image.alpha = 0.5, stroke = 0, min.cutoff = 8, max.cutoff = 100)
p3$layers[[1]]$aes_params <- c(p3$layers[[1]]$aes_params, shape=22)
p3

pdf(paste0(output_dir, outputname,"_clusters_spatial_slingshot_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 5)
print(p3)
dev.off()

################# 
# subset
################# 
source('/mnt/HDD1/Users/liran/05_FFPE/FFPEA08/scripts/lm_filter_cells_by_mask.R')

result <- filter_cells_by_mask(seurat_s1_sub, "./spatial_s1/tissue_lowres_image_mask.png")

seurat_s1_sub2 <- result$filtered_seurat

outputname = 's1_embbedding_atac_sub'
output_dir = '0_lineage_atac_s1_sub/'
dir.create(output_dir, showWarnings = FALSE)


p4 <- SpatialPlot(seurat_s1_sub2, label = FALSE, label.size = 3, crop = FALSE, pt.size.factor = 4.5, image.alpha = 0.5, stroke = 0, group.by = group_by_name)
#p4$layers[[1]]$aes_params <- c(p4$layers[[1]]$aes_params, shape=22)
print(p4)


pdf(paste0(output_dir, outputname, "_clusters_spatial_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 5)
print(p4)
dev.off()

seurat_s1_sub <- seurat_s1_sub2

sshot <- slingshot(data = Embeddings(seurat_s1_sub,reduction = reduction_fun), clusterLabels = seurat_s1_sub$all_clusters)

pt <- slingPseudotime(sshot)

seurat_s1_sub@meta.data <- cbind(seurat_s1_sub@meta.data, pt)

p1 <- FeaturePlot(seurat_s1_sub,features = 'Lineage1', reduction = reduction_fun) + scale_color_viridis_c()
print(p1)

pdf(paste0(output_dir, outputname,"_umap_slingshot_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 4)
print(p1)
dev.off()
# 
seurat_s1_sub$Lineage1[is.na(seurat_s1_sub$Lineage1)] <- 0
p3 <- SpatialPlot(seurat_s1_sub, label = FALSE, label.size = 3, features = 'Lineage1',crop = FALSE, pt.size.factor = 4.5,  image.alpha = 0.5, stroke = 0, , min.cutoff = 'q35')
#p3$layers[[1]]$aes_params <- c(p3$layers[[1]]$aes_params, shape=22)
p3

pdf(paste0(output_dir, outputname,"_clusters_spatial_slingshot_", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 5)
print(p3)
dev.off()



# df for this
min(seurat_s1_sub$Lineage1)
max(seurat_s1_sub$Lineage1)
seurat_s1_sub$pt <- -seurat_s1_sub$Lineage1+39
p3 <- SpatialPlot(seurat_s1_sub, label = FALSE, label.size = 3, features = 'pt',crop = FALSE, pt.size.factor = 4.5,  image.alpha = 0.5, stroke = 0, max.cutoff = 32)
#p3$layers[[1]]$aes_params <- c(p3$layers[[1]]$aes_params, shape=22)
print(p3)
pdf(paste0(output_dir, outputname,"_clusters_spatial_slingshot_revers_pt", group_by_name, '.',reduction_fun, ".pdf"), width = 5, height = 5)
print(p3)
dev.off()
# 
# DefaultAssay(seurat_s1_sub) <- 'RNA.csv'
# 
# heamap.df.rna <- data.frame('pt' = seurat_s1_sub$pt)
# heamap.df.rna <- cbind(heamap.df.rna, data.frame(t(seurat_s1_sub@assays$RNA.csv@data)))
# dim(heamap.df.rna)
# 
# heamap.df.rna2 = heamap.df.rna[order(heamap.df.rna[,1]),]
# 
# heamap.df.rna2[1:5, 1:5]
# 
# heamap.df.rna.no.pt = heamap.df.rna2[,-1]
# heamap.df.rna.no.pt = t(heamap.df.rna.no.pt)
# 
# ########
# # select genes for plot
# gene_cor_df <- data.frame('gene' = rownames(heamap.df.rna.no.pt), 'cor'=0)
# rownames(gene_cor_df) <- rownames(heamap.df.rna.no.pt)
# 
# 
# for (i in 1:dim(gene_cor_df)[1]){
#   gene = rownames(gene_cor_df)[i]
#   gene_cor_df[i,2] = cor(heamap.df.rna2$pt, heamap.df.rna2[,gene])
# }
# 
# gene_cor_df_no_na <- na.omit(gene_cor_df)
# dim(gene_cor_df_no_na)
# gene_cor_df_no_na = gene_cor_df_no_na[order(gene_cor_df_no_na[,2], decreasing = TRUE),]
# 
# gene_cor_df_select <- gene_cor_df_no_na[gene_cor_df_no_na$cor > 0.1,]
# dim(gene_cor_df_select)
# dim(gene_cor_df_no_na)
# 
# final_list1 <- gene_cor_df_select$gene
# 
# 
# heatmap <- pheatmap(heamap.df.rna.no.pt[final_list1[1:50],], cluster_cols = FALSE, cluster_rows = FALSE, show_colnames = FALSE)
# 
# pdf(paste0("./lineage_s1/", outputname,"_clusters_spatial_slingshot_", group_by_name, '.',reduction_fun, "RNA_pos_cor_0.1_heatmap_1_50_order_by_cor.pdf"), width = 6.5, height = 9)
# print(heatmap)
# # print(rna_out)
# dev.off()
# 
# 
# heatmap <- pheatmap(heamap.df.rna.no.pt[final_list1[1:50],], cluster_cols = FALSE, show_colnames = FALSE)
# 
# pdf(paste0("./lineage_s1/", outputname,"_clusters_spatial_slingshot_", group_by_name, '.',reduction_fun, "RNA_pos_cor_0.1_heatmap_1_50_cluster_rows.pdf"), width = 6.5, height = 9)
# print(heatmap)
# # print(rna_out)
# dev.off()

####################
# dotplot all
################### 
# plot_gene_lineage <- function(genelist, file_name){
#   final_list <- genelist
#   final_list <- intersect(rownames(seurat_s1_sub@assays$RNA.csv@data), final_list)
#   final_list <- intersect(rownames(seurat_s1_sub@assays$ATAC.csv@data), final_list)
#   final_list <- intersect(rownames(seurat_s1_sub@assays$H3K27ac_imputed.csv@data), final_list)
#   final_list <- intersect(rownames(seurat_s1_sub@assays$H3K4me3.csv@data), final_list)  
#   final_list <- intersect(rownames(seurat_s1_sub@assays$H3K27me3_imputed.csv@data), final_list) 
#   
#   
#   df.plot <- data.frame('cell' = rownames(seurat_s1_sub@meta.data),
#                         'pt' = seurat_s1_sub$pt,
#                         'H3K27me3_imputed' = colSums(seurat_s1_sub@assays$H3K27me3_imputed.csv@data[final_list,])/sum(seurat_s1_sub@assays$H3K27me3_imputed.csv@data[final_list,]),
#                         'H3K4me3' = colSums(seurat_s1_sub@assays$H3K4me3.csv@data[final_list,])/sum(seurat_s1_sub@assays$H3K4me3.csv@data[final_list,]),
#                         'H3K27ac_imputed' = colSums(seurat_s1_sub@assays$H3K27ac_imputed.csv@data[final_list,])/sum(seurat_s1_sub@assays$H3K27ac_imputed.csv@data[final_list,]),
#                         'ATAC' = colSums(seurat_s1_sub@assays$ATAC.csv@data[final_list,])/sum(seurat_s1_sub@assays$ATAC.csv@data[final_list,]))
#                     
#   
# 
#   
# 
#   df.plot <- df.plot[df.plot$pt>7,]
#   df_long <- pivot_longer(
#     data = df.plot[-1],
#     cols = -pt, # Select all columns except pseudotime to pivot
#     names_to = "condition", # This will be the new column for the condition names
#     values_to = "score" # This will be the new column for the scores
#   )
#   
#   df_long_2 <- df_long[df_long$pt!=0,]
#   
#   p <- ggplot(data = df_long_2, aes(x = pt, y = score, color = condition)) +
#     geom_point(alpha = 0.5) + # Plots the points with some transparency
#     geom_smooth(method = "loess", se = TRUE, aes(fill = condition), alpha = 0.1) + # Adds a smooth line with a confidence interval
#     # scale_color_manual(values = c("rna" = "red", "h3k27me3" = "green", "h3k27mac" = "blue")) +
#     labs(x = "Pseudo-time", y = "Score", title = "Chromatin opening") +
#     theme_minimal() + # Sets a minimal theme
#     theme(legend.title = element_blank()) # Removes the legend title
#   
#   print(p)
#   
#   pdf(paste0("./lineage_s1/", outputname,"_clusters_spatial_slingshot_", group_by_name, '.',reduction_fun, "RNA_", file_name, "_dot.pdf"), width = 6.5, height = 7)
#   print(p)
#   dev.off()
# }
# 
# length(final_list1)
# plot_gene_lineage(final_list1, 'pos_cor_0.1')



################## 
#dot plot per gene
################## 
plot_one_gene_lineage_no_RNA_nor <- function(gene) {
  final_list <- gene
  
  if (length(final_list) != 0) {
    df.plot <- data.frame(
      'cell' = rownames(seurat_s1_sub@meta.data),
      'pt' = seurat_s1_sub$pt,
      'H3K27me3' = seurat_s1_sub@assays$H3K27me3.csv@data[final_list, ],
      'H3K4me3_imputed' = seurat_s1_sub@assays$H3K4me3_imputed.csv@data[final_list, ],
      'H3K27ac' = seurat_s1_sub@assays$H3K27ac.csv@data[final_list, ],
      'ATAC_imputed' = seurat_s1_sub@assays$atac_imputed.csv@data[final_list, ]
    )
    
    # Normalize each chromatin modality separately
    df.plot$H3K27me3_scale <- (df.plot$H3K27me3 - min(df.plot$H3K27me3)) / (max(df.plot$H3K27me3) - min(df.plot$H3K27me3))
    df.plot$H3K4me3_imputed_scale <- (df.plot$H3K4me3_imputed - min(df.plot$H3K4me3_imputed)) / (max(df.plot$H3K4me3_imputed) - min(df.plot$H3K4me3_imputed))
    df.plot$H3K27ac_scale <- (df.plot$H3K27ac - min(df.plot$H3K27ac)) / (max(df.plot$H3K27ac) - min(df.plot$H3K27ac))
    df.plot$ATAC_imputed_scale <- (df.plot$ATAC_imputed - min(df.plot$ATAC_imputed)) / (max(df.plot$ATAC_imputed) - min(df.plot$ATAC_imputed))
    
    # Keep only the scaled columns
    df.plot <- df.plot[, c("cell", "pt", "H3K27me3_scale", "H3K4me3_imputed_scale", "H3K27ac_scale", "ATAC_imputed_scale")]    
    # Filter pseudotime threshold
    # df.plot <- df.plot[df.plot$pt > 7, ]
    
    # Convert data to long format
    df_long <- pivot_longer(
      data = df.plot[-1],  # Exclude 'cell' column
      cols = -pt,  # Select all columns except pseudotime to pivot
      names_to = "condition",  # New column for condition names
      values_to = "score"  # New column for scores
    )
    
    df_long_2 <- df_long[df_long$pt != 0, ]
    
    # Plot
    p <- ggplot(data = df_long_2, aes(x = pt, y = score, color = condition)) +
      geom_smooth(method = "loess", se = TRUE, aes(fill = condition), alpha = 0.1) + # Smooth line with confidence interval
      labs(x = "Pseudo-time", y = "Score", title = "Chromatin opening") +
      theme_minimal() +
      theme(legend.title = element_blank()) +
      labs(title = gene)
    
    print(p)
    
    # Save as PDF
    pdf(paste0(output_dir, outputname, "_lineage_noRNA_normalize_", gene, ".pdf"), width = 6.5, height = 6)
    print(p)
    dev.off()
  }
}

plot_one_gene_lineage_no_RNA_nor('Sox2')

gene_to_look <- c('Pou4f1','Lhx5', 'Neurod2', 'Sox2', 'E2f2', 'Car10', 'Ank3', 'Gria2')


for (gene_name in gene_to_look){
  plot_one_gene_lineage_no_RNA_nor(c(gene_name))
}





