setwd("/mnt/HDD1/Users/liran/10_integration_wei/0_benckmark_new/")


library(StabMap)
library(Seurat)
library(Matrix)
library(SingleCellExperiment)
library(data.table)
library(ggplot2)
library(harmony)

# File paths
base_path <- "./data/spatial_mux_seq/"

# Initialize runtime tracking
runtime_log <- data.frame(
  Step = character(),
  Start_Time = character(),
  End_Time = character(),
  Duration_Seconds = numeric(),
  stringsAsFactors = FALSE
)

# Function to log runtime
log_runtime <- function(step_name, start_time, end_time) {
  duration <- as.numeric(difftime(end_time, start_time, units = "secs"))
  runtime_log <<- rbind(runtime_log, data.frame(
    Step = step_name,
    Start_Time = format(start_time, "%Y-%m-%d %H:%M:%S"),
    End_Time = format(end_time, "%Y-%m-%d %H:%M:%S"),
    Duration_Seconds = round(duration, 3),
    stringsAsFactors = FALSE
  ))
  cat(sprintf("[%s] %s completed in %.3f seconds\n",
              format(end_time, "%H:%M:%S"), step_name, duration))
}

# Set working directory
cat("Starting StabMap integration with runtime tracking...\n")
overall_start <- Sys.time()

# Step 1: Data Loading
step_start <- Sys.time()
# Load each CSV as matrix
# Section 1 data
s1_H3K27me3 <- fread(paste0(base_path, "s1_H3K27me3.csv"))
s1_H3K27ac <- fread(paste0(base_path, "s1_H3K27ac.csv"))

# Section 2 data
s2_H3K27me3 <- fread(paste0(base_path, "s2_H3K27me3.csv"))
s2_ATAC <- fread(paste0(base_path, "s2_ATAC.csv"))
s2_RNA <- fread(paste0(base_path, "s2_RNA.csv"))
s2_H3K4me3 <- fread(paste0(base_path, "s2_H3K4me3.csv"))

step_end <- Sys.time()
log_runtime("Data Loading", step_start, step_end)


# Step 2: Data preparation
step_start <- Sys.time()
# Convert to matrix and set rownames
df_to_matrix <- function(df) {
  mat <- as.matrix(df[,-1])  # remove first column (cell names)
  rownames(mat) <- df[[1]]   # set rownames from first column
  return(t(mat))             # transpose to features × cells
}

# Process Section 1 matrices
mat_s1_H3K27me3 <- df_to_matrix(s1_H3K27me3)
mat_s1_H3K27ac <- df_to_matrix(s1_H3K27ac)

# Process Section 2 matrices
mat_s2_H3K27me3 <- df_to_matrix(s2_H3K27me3)
mat_s2_ATAC <- df_to_matrix(s2_ATAC)
mat_s2_RNA <- df_to_matrix(s2_RNA)
mat_s2_H3K4me3 <- df_to_matrix(s2_H3K4me3)

# ---- Slide 1 ----
colnames(mat_s1_H3K27ac) <- colnames(mat_s1_H3K27me3)
common_cells_s1 <- Reduce(intersect, list(colnames(mat_s1_H3K27me3), colnames(mat_s1_H3K27ac)))

mat_s1 <- rbind(
  {
    mat <- mat_s1_H3K27me3[, common_cells_s1, drop = FALSE]
    rownames(mat) <- paste0(rownames(mat), "__H3K27me3")
    mat
  },
  {
    mat <- mat_s1_H3K27ac[, common_cells_s1, drop = FALSE]
    rownames(mat) <- paste0(rownames(mat), "__H3K27ac")
    mat
  }
)

sce_s1 <- SingleCellExperiment(
  assays = list(logcounts = mat_s1)
)

# ---- Slide 2 ----
colnames(mat_s2_ATAC) <- colnames(mat_s2_H3K27me3)
colnames(mat_s2_RNA) <- colnames(mat_s2_H3K27me3)
colnames(mat_s2_H3K4me3) <- colnames(mat_s2_H3K27me3)


common_cells_s2 <- Reduce(intersect, list(
  colnames(mat_s2_H3K27me3),
  colnames(mat_s2_ATAC),
  colnames(mat_s2_RNA),
  colnames(mat_s2_H3K4me3)
))

mat_s2 <- rbind(
  {
    mat <- mat_s2_H3K27me3[, common_cells_s2, drop = FALSE]
    rownames(mat) <- paste0(rownames(mat), "__H3K27me3")
    mat
  },
  {
    mat <- mat_s2_ATAC[, common_cells_s2, drop = FALSE]
    rownames(mat) <- paste0(rownames(mat), "__ATAC")
    mat
  },
  {
    mat <- mat_s2_RNA[, common_cells_s2, drop = FALSE]
    rownames(mat) <- paste0(rownames(mat), "__RNA")
    mat
  },
  {
    mat <- mat_s2_H3K4me3[, common_cells_s2, drop = FALSE]
    rownames(mat) <- paste0(rownames(mat), "__H3K4me3")
    mat
  }
)

sce_s2 <- SingleCellExperiment(
  assays = list(logcounts = mat_s2)
)

# Rename cells to avoid overlapping names
colnames(sce_s1) <- paste0("s1_", colnames(sce_s1))
colnames(sce_s2) <- paste0("s2_", colnames(sce_s2))

# Build list for StabMap
expr_list <- list(
  Section1 = logcounts(sce_s1),
  Section2 = logcounts(sce_s2)
)

# Calculate total features
feature_names <- lapply(expr_list, rownames)
all_features <- unlist(feature_names)
unique_features <- length(unique(all_features))
print(paste("Total unique features in expr_list:", unique_features))

# Check modality overlaps with UpSet plot
mosaicDataUpSet(expr_list, plot = TRUE)
ggsave("spatial_mux_seq_upset_plot.pdf", width = 8, height = 10)

step_end <- Sys.time()
log_runtime("Data preparation", step_start, step_end)



# Step 3: Run stabmap
step_start <- Sys.time()
# Run StabMap
stab_result <- stabMap(expr_list, projectAll = TRUE, maxFeatures = length(all_features))

# Save StabMap embedding
write.csv(as.data.frame(stab_result), file = "spatial_mux_seq_stabmap_embedding.csv")

step_end <- Sys.time()
log_runtime("StabMap Integration", step_start, step_end)

# Step 4: Run harmony
step_start <- Sys.time()
# Prepare Harmony input from stab_result matrix
colnames(stab_result) <- paste0("PC_", seq_len(ncol(stab_result)))
seu_stab <- CreateSeuratObject(counts = t(stab_result))
seu_stab[["pca"]] <- CreateDimReducObject(
  embeddings = stab_result,
  key = "PC_",
  assay = DefaultAssay(seu_stab)
)

# Add batch info
sample_labels <- c(
  setNames(rep("Section1", ncol(sce_s1)), colnames(sce_s1)),
  setNames(rep("Section2", ncol(sce_s2)), colnames(sce_s2))
)
seu_stab$batch <- sample_labels[Cells(seu_stab)]

# Run Harmony with all dimensions
seu_stab <- RunHarmony(
  object = seu_stab,
  group.by.vars = "batch",
  # No dims.use parameter = use all dims
  plot_convergence = TRUE,
  nclust = 50,
  max_iter = 15,
  early_stop = TRUE,
  project.dim = FALSE
)

# Save Harmony embedding
harmony_matrix <- Embeddings(seu_stab, "harmony")
write.csv(harmony_matrix, file = "spatial_mux_seq_harmony_embedding.csv")

step_end <- Sys.time()
log_runtime("Run Harmony", step_start, step_end)


# UMAP before Harmony - use all dimensions
all_dims <- 1:ncol(Embeddings(seu_stab, "pca"))
seu_stab <- RunUMAP(seu_stab, reduction = "pca", dims = all_dims)
p_before <- DimPlot(seu_stab, group.by = "batch", reduction = "umap")
ggsave("spatial_mux_seq_umap_before_harmony.pdf", plot = p_before, width = 5, height = 4)

# UMAP after Harmony - use all dimensions
all_dims <- 1:ncol(Embeddings(seu_stab, "harmony"))
seu_stab <- RunUMAP(seu_stab, reduction = "harmony", dims = all_dims)
p_after <- DimPlot(seu_stab, group.by = "batch", reduction = "umap")
ggsave("spatial_mux_seq_umap_after_harmony.pdf", plot = p_after, width = 5, height = 4)

# Save the Seurat object for further analysis
saveRDS(seu_stab, file = "spatial_mux_seq_integrated.rds")



# Calculate overall runtime
overall_end <- Sys.time()
overall_duration <- as.numeric(difftime(overall_end, overall_start, units = "secs"))

# Add overall runtime to log
runtime_log <- rbind(runtime_log, data.frame(
  Step = "TOTAL_RUNTIME",
  Start_Time = format(overall_start, "%Y-%m-%d %H:%M:%S"),
  End_Time = format(overall_end, "%Y-%m-%d %H:%M:%S"),
  Duration_Seconds = round(overall_duration, 3),
  stringsAsFactors = FALSE
))

# Add additional summary columns
runtime_log$Duration_Minutes <- round(runtime_log$Duration_Seconds / 60, 2)
runtime_log$Duration_Hours <- round(runtime_log$Duration_Seconds / 3600, 4)
runtime_log$Percentage_of_Total <- round((runtime_log$Duration_Seconds / overall_duration) * 100, 2)

# Save runtime log to CSV
write.csv(runtime_log, file = paste0("mouse_embryo_runtime_log.csv"), row.names = FALSE)