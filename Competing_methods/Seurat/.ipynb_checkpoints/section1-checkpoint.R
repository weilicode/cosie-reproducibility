setwd("~/Desktop/RA/COSIE/Spatial_mux_seq")
library(reticulate)
library(anndata)
library(Seurat)
library(ggplot2)
library(Matrix)

me3_sce <- read_h5ad("section1/E13_50_1_H3K27me3.h5ad")
ac_sce  <- read_h5ad("section1/E13_50_1_H3K27ac.h5ad")

me3_mat <- t(me3_sce$X)
ac_mat  <- t(ac_sce$X)
me3_mat <- as(me3_mat, "dgCMatrix")
ac_mat  <- as(ac_mat, "dgCMatrix")

# 2. Add feature/cell names
rownames(me3_mat) <- me3_sce$var_names
colnames(me3_mat) <- me3_sce$obs_names

rownames(ac_mat) <- ac_sce$var_names
colnames(ac_mat) <- ac_sce$obs_names


# 4. Create Seurat object with two assays
obj <- CreateSeuratObject(counts = me3_mat, assay = "H3K27me3")
obj[["H3K27ac"]] <- CreateAssayObject(counts = ac_mat)

# 5. Process H3K27me3
DefaultAssay(obj) <- "H3K27me3"
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, reduction.name = "pca_me3")

# 6. Process H3K27ac
DefaultAssay(obj) <- "H3K27ac"
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, reduction.name = "pca_ac")

# 7. WNN integration
obj <- FindMultiModalNeighbors(
  obj,
  reduction.list = list("pca_me3", "pca_ac"),
  dims.list = list(1:50, 1:50)
)

obj <- RunUMAP(
  obj,
  nn.name = "weighted.nn",
  reduction.name = "wnn.umap",
  reduction.key = "wnnUMAP_"
)

spatial_local <- me3_sce$obsm["spatial_local"]$spatial_local
spatial_local <- as.matrix(spatial_local)
rownames(spatial_local) <- me3_sce$obs_names
spatial_local <- spatial_local[colnames(obj), ]

obj$spatial_x <- spatial_local[, 1]
obj$spatial_y <- spatial_local[, 2]


emb_me3 <- Embeddings(obj, "pca_me3")[, 1:50]
emb_ac  <- Embeddings(obj, "pca_ac")[, 1:50]

emb_combined <- cbind(emb_me3, emb_ac)

set.seed(1234)
km <- kmeans(emb_combined, centers = 11, nstart = 50)

obj$kmeans_11 <- factor(km$cluster)

df_plot <- obj@meta.data

p <- ggplot(df_plot, aes(x = spatial_x, y = spatial_y, color = kmeans_11)) +
  geom_point(size = 2, alpha = 0.8) +
  coord_fixed() +
  theme_classic() +
  labs(color = "K-means cluster")
ggsave("section1/seurat_kmeans_plot.png", plot = p, width = 6, height = 5, dpi = 300)

write.csv(df_plot, "section1/seurat_kmeans_df.csv", row.names = TRUE)
saveRDS(emb_combined, "section1/seurat_emb.rds")






