# Load necessary libraries
library(Seurat)
library(Matrix)
library(tidyverse)

# Define directories
data_dir <- "./exported_data"

# List all exported data
files <- list.files(data_dir, pattern = "\\.csv$", full.names = TRUE)

# Separate files by type
expression_files <- files[!grepl("_metadata|_final", files)]
metadata_files <- files[grepl("_metadata", files)]
embedding_files <- files[grepl("_final", files)]

# Function to read an expression matrix and **transpose it**
read_expression_matrix <- function(file) {
  df <- read.csv(file, row.names = 1)
  return(as.matrix(t(df)))  # Transpose: rows = genes, cols = cells
}

# Function to read embedding data **and assign correct row names**
read_embedding <- function(file, cell_barcodes) {
  df <- read.csv(file, row.names = NULL)
  embedding_matrix <- as.matrix(t(df))
  colnames(embedding_matrix) <- cell_barcodes  # Assign correct cell barcodes
  return(embedding_matrix)
}

# Separate Slide 1 and Slide 2 data
slide1_files <- expression_files[grepl("^.*/s1_", expression_files)]
slide2_files <- expression_files[grepl("^.*/s2_", expression_files)]

slide1_metadata_file <- metadata_files[grepl("^.*/s1_", metadata_files)]
slide2_metadata_file <- metadata_files[grepl("^.*/s2_", metadata_files)]

slide1_embedding_files <- embedding_files[grepl("^.*/s1_", embedding_files)]
slide2_embedding_files <- embedding_files[grepl("^.*/s2_", embedding_files)]

# Function to create a multi-omics Seurat object
create_seurat <- function(expression_list, metadata_file, embedding_list, slide_name) {
  first_modality <- expression_list[1]
  seurat_obj <- CreateSeuratObject(counts = read_expression_matrix(first_modality), 
                                   assay = gsub(".*/s[12]_", "", first_modality))
  
  for (modality in expression_list[-1]) {
    # modality <- slide1_files[5]
    assay_name <- gsub(".*/s[12]_", "", modality)  # Extract modality name
    matrix <- read_expression_matrix(modality)
    print(all(colnames(matrix) == colnames(seurat_obj)))
    seurat_obj[[assay_name]] <- CreateAssayObject(counts = matrix)
  }
  
  # Add metadata if available
  if (length(metadata_file) > 0) {
    metadata <- read.csv(metadata_file[1], row.names = 1)
    seurat_obj <- AddMetaData(seurat_obj, metadata)
  }
  
  # Get cell barcodes from first modality
  cell_barcodes <- colnames(seurat_obj)
  
  # Add embeddings as a new `DimReduc` object with correct row names
  for (embedding_file in embedding_list) {
    
    embedding_name <- gsub(".*/s[12]_", "", embedding_file)
    embedding_matrix <- read_embedding(embedding_file, cell_barcodes)
    seurat_obj[[embedding_name]] <- CreateAssayObject(counts = embedding_matrix)
  }
  # Store slide information
  seurat_obj$slide <- slide_name
  
  return(seurat_obj)
}

# Create Seurat objects for Slide 1 and Slide 2
seurat_s1 <- create_seurat(slide1_files, slide1_metadata_file, slide1_embedding_files, "Slide1")
seurat_s2 <- create_seurat(slide2_files, slide2_metadata_file, slide2_embedding_files, "Slide2")


######### add spatial information
add_image <- function(object, file.path, filter.matrix = TRUE, slice = "slice1", assay = 'final.csv'){
  image <- Read10X_Image(image.dir = file.path, filter.matrix = filter.matrix)
  image <- image[Cells(x = object)]
  DefaultAssay(object = image) <- assay
  object[[slice]] <- image
  return(object)
}

seurat_s1 <- add_image(seurat_s1, './spatial_s1/')
seurat_s2 <- add_image(seurat_s2, './spatial_s2/')



# Save Seurat objects
saveRDS(seurat_s1, file = "seurat_slide1.rds")
saveRDS(seurat_s2, file = "seurat_slide2.rds")
