import anndata as ad
import pandas as pd
import numpy as np
import os

# Define paths
raw_data_dir = "./raw_data"
imputed_data_dir = "./imputed_data"
embedding_dir = "./embedding"
output_dir = "./exported_data"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to save AnnData matrix to CSV
def save_matrix_to_csv(adata, filename):
    """ Saves expression matrix from AnnData as a CSV file """
    df = pd.DataFrame(adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X, 
                      index=adata.obs.index, columns=adata.var.index)
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"))
    print(f"Saved: {filename}.csv")

# Function to save embeddings to CSV
def save_embedding_to_csv(embedding_file, filename):
    """ Saves numpy embedding matrix as a CSV file """
    embedding = np.load(embedding_file)
    df = pd.DataFrame(embedding)
    df.to_csv(os.path.join(output_dir, f"{filename}.csv"), index=False)
    print(f"Saved: {filename}.csv")

# Function to save cell annotations (all_clusters)
def save_metadata_to_csv(adata, filename):
    """ Saves cell metadata from AnnData as a CSV file """
    if "all_clusters" in adata.obs:
        df = adata.obs[["all_clusters"]]
        df.to_csv(os.path.join(output_dir, filename + "_metadata.csv"))
        print(f"Saved: {filename}_metadata.csv")

# Process raw data
for file in os.listdir(raw_data_dir):
    if file.endswith(".h5ad"):
        adata = ad.read_h5ad(os.path.join(raw_data_dir, file))
        save_matrix_to_csv(adata, file.replace(".h5ad", ""))

# Process imputed data (expression matrices + metadata)
for file in os.listdir(imputed_data_dir):
    if file.endswith(".h5ad"):
        adata = ad.read_h5ad(os.path.join(imputed_data_dir, file))
        save_matrix_to_csv(adata, file.replace(".h5ad", ""))
        save_metadata_to_csv(adata, file.replace(".h5ad", ""))

# Process embeddings
for file in os.listdir(embedding_dir):
    if file.endswith(".npy"):
        save_embedding_to_csv(os.path.join(embedding_dir, file), file.replace(".npy", ""))
