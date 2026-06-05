setwd("~/Desktop/RA/COSIE/Spatial_mux_seq")
library(reticulate)
library(anndata)
library(Seurat)
library(ggplot2)
library(Matrix)

get_mat <- function(sce) {
  mat <- t(sce$X)
  mat <- as(mat, "dgCMatrix")
  
  rownames(mat) <- as.character(sce$var_names)
  colnames(mat) <- as.character(sce$obs_names)
  
  return(mat)
}


me3_sce <- read_h5ad("section2/E13_50_3_H3K27me3.h5ad")
ac_sce  <- read_h5ad("section2/E13_50_3_H3K4me3.h5ad")
rna_sce <- read_h5ad("section2/E13_50_3_RNA.h5ad")
atac_sce <- read_h5ad("section2/E13_50_3_ATAC.h5ad")


me3_mat  <- get_mat(me3_sce)
ac_mat   <- get_mat(ac_sce)
rna_mat  <- get_mat(rna_sce)
atac_mat <- get_mat(atac_sce)


obj <- CreateSeuratObject(counts = rna_mat, assay = "RNA")

obj[["H3K27me3"]] <- CreateAssayObject(counts = me3_mat)
obj[["H3K27ac"]]  <- CreateAssayObject(counts = ac_mat)
obj[["ATAC"]]     <- CreateAssayObject(counts = atac_mat)

# -----------------------------
# RNA preprocessing
# -----------------------------
DefaultAssay(obj) <- "RNA"
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, reduction.name = "pca_rna")

# -----------------------------
# H3K27me3 preprocessing
# -----------------------------
DefaultAssay(obj) <- "H3K27me3"
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, reduction.name = "pca_me3")

# -----------------------------
# H3K27ac preprocessing
# -----------------------------
DefaultAssay(obj) <- "H3K27ac"
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, reduction.name = "pca_ac")

# -----------------------------
# H3K27ac preprocessing
# -----------------------------
DefaultAssay(obj) <- "ATAC"
obj <- NormalizeData(obj)
obj <- FindVariableFeatures(obj, nfeatures = 3000)
obj <- ScaleData(obj)
obj <- RunPCA(obj, reduction.name = "pca_atac")


# -----------------------------
# 4-modal WNN integration
# -----------------------------
obj <- FindMultiModalNeighbors(
  obj,
  reduction.list = list(
    "pca_rna",
    "pca_me3",
    "pca_ac",
    "pca_atac"
  ),
  dims.list = list(
    1:50,
    1:50,
    1:50,
    1:50
  )
)

obj <- RunUMAP(
  obj,
  nn.name = "weighted.nn",
  reduction.name = "wnn.umap",
  reduction.key = "wnnUMAP_"
)


# -----------------------------
# Optional: k-means to fixed 10 clusters
# -----------------------------
set.seed(1234)

emb_rna  <- Embeddings(obj, "pca_rna")[, 1:50]
emb_me3  <- Embeddings(obj, "pca_me3")[, 1:50]
emb_ac   <- Embeddings(obj, "pca_ac")[, 1:50]
emb_atac <- Embeddings(obj, atac_reduction)[, 1:50]

emb_combined <- cbind(emb_rna, emb_me3, emb_ac, emb_atac)

km <- kmeans(emb_combined, centers = 11, nstart = 50)

obj$kmeans_11 <- factor(km$cluster)

# -----------------------------
# Add spatial coordinates
# -----------------------------
spatial_local <- as.matrix(me3_sce$obsm["spatial_local"]$spatial_local)

rownames(spatial_local) <- as.character(me3_sce$obs_names)

spatial_local <- spatial_local[colnames(obj), ]

obj$spatial_x <- spatial_local[, 1]
obj$spatial_y <- spatial_local[, 2]

df_plot <- obj@meta.data


# Spatial plot: k-means cluster
p2 <- ggplot(df_plot, aes(x = spatial_x, y = spatial_y, color = kmeans_11)) +
  geom_point(size = 2, alpha = 0.8) +
  coord_fixed() +
  theme_classic() +
  labs(color = "K-means cluster")

ggsave("section2/seurat_kmeans_plot.png", plot = p2, width = 6, height = 5, dpi = 300)

# -----------------------------
# Save outputs
# -----------------------------
write.csv(df_plot, "section2/seurat_kmeans_df.csv", row.names = TRUE)
saveRDS(emb_combined, "section2/seurat_emb.rds")



