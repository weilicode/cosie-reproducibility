## Please refer to https://github.com/JinmiaoChenLab/SpatialGlue for installation

import os
import torch
import pandas as pd
import scanpy as sc
from SpatialGlue.preprocess import clr_normalize_each_cell, pca, lsi
import SpatialGlue
import numpy as np

adata1_27me3 = sc.read_h5ad('E13_50_1_H3K27me3.h5ad')
adata1_27ac  = sc.read_h5ad('E13_50_1_H3K27ac.h5ad')

adata1_27me3.obsm['spatial'] = adata1_27me3.obsm['spatial_local'].copy()
adata1_27ac.obsm['spatial'] = adata1_27ac.obsm['spatial_local'].copy()

data_type = 'embryo'

# Fix random seed
from SpatialGlue.preprocess import fix_seed
random_seed = 2022
fix_seed(random_seed)


sc.pp.highly_variable_genes(adata1_27me3, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata1_27me3, target_sum=1e4)
sc.pp.log1p(adata1_27me3)
sc.pp.scale(adata1_27me3)
adata1_27me3_high =  adata1_27me3[:, adata1_27me3.var['highly_variable']]
adata1_27me3.obsm['feat'] = pca(adata1_27me3_high, n_comps=50)


sc.pp.highly_variable_genes(adata1_27ac, flavor="seurat_v3", n_top_genes=3000)
sc.pp.normalize_total(adata1_27ac, target_sum=1e4)
sc.pp.log1p(adata1_27ac)
sc.pp.scale(adata1_27ac)

adata1_27ac_high =  adata1_27ac[:, adata1_27ac.var['highly_variable']]
adata1_27ac.obsm['feat'] = pca(adata1_27ac_high, n_comps=50)

import numpy as np
import scipy.sparse as sp

# ATAC
adata1_27ac = adata1_27ac[adata1_27ac.obs_names].copy()

if "X_lsi" not in adata1_27ac.obsm_keys():

    X = adata1_27ac.X
    col_sum = np.array(X.sum(axis=0)).ravel() if sp.issparse(X) else X.sum(axis=0)
    keep_var = col_sum > 0
    adata1_27ac = adata1_27ac[:, keep_var].copy()
    X = adata1_27ac.X
    row_sum = np.array(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
    keep_obs = row_sum > 0
    adata1_27ac = adata1_27ac[keep_obs, :].copy()
    lsi(adata1_27ac, use_highly_variable=False, n_components=51)

adata1_27ac.obsm["feat"] = adata1_27ac.obsm["X_lsi"].copy()


adata1_27me3 = adata1_27me3[adata1_27me3.obs_names].copy()

if "X_lsi" not in adata1_27me3.obsm_keys():

    X = adata1_27me3.X
    col_sum = np.array(X.sum(axis=0)).ravel() if sp.issparse(X) else X.sum(axis=0)
    keep_var = col_sum > 0
    adata1_27me3 = adata1_27me3[:, keep_var].copy()
    X = adata1_27me3.X
    row_sum = np.array(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
    keep_obs = row_sum > 0
    adata1_27me3 = adata1_27me3[keep_obs, :].copy()
    lsi(adata1_27me3, use_highly_variable=False, n_components=51)

adata1_27me3.obsm["feat"] = adata1_27me3.obsm["X_lsi"].copy()
from SpatialGlue.preprocess import construct_neighbor_graph
data = construct_neighbor_graph(adata1_27ac, adata1_27me3, datatype=data_type)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define model
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue
model = Train_SpatialGlue(data, datatype=data_type, device=device)

# train model
output = model.train()

np.save('embryo_s1_spatialglue.npy', output['SpatialGlue'])



