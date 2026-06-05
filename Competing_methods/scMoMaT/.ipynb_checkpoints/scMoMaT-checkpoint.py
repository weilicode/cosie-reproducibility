# Please refer to https://github.com/PeterZZQ/scMoMaT for installation


import sys, os

import numpy as np
# from umap import UMAP
import time
import torch
import matplotlib.pyplot as plt
import pandas as pd  
import scanpy as sc
import scipy.sparse as sps
import scipy.io as sio
from os.path import join
import h5py
import warnings


import scmomat 
from scmomat import preprocess
from sklearn.cluster import KMeans
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.cm import get_cmap
from pandas.api.types import CategoricalDtype

adata1_27me3 = sc.read_h5ad('E13_50_1_H3K27me3.h5ad')
adata1_27ac  = sc.read_h5ad('E13_50_1_H3K27ac.h5ad')

adata2_27me3 = sc.read_h5ad('E13_50_3_H3K27me3.h5ad')
adata2_atac  = sc.read_h5ad('E13_50_3_ATAC.h5ad')
adata2_rna   = sc.read_h5ad('E13_50_3_RNA.h5ad')
adata2_4me3  = sc.read_h5ad('E13_50_3_H3K4me3.h5ad')

adata1_27me3.obsm['spatial'] = adata1_27me3.obsm['spatial_local'].copy()
adata1_27ac.obsm['spatial'] = adata1_27ac.obsm['spatial_local'].copy()
adata2_27me3.obsm['spatial'] = adata2_27me3.obsm['spatial_local'].copy()
adata2_atac.obsm['spatial'] = adata2_atac.obsm['spatial_local'].copy()
adata2_rna.obsm['spatial'] = adata2_rna.obsm['spatial_local'].copy()
adata2_4me3.obsm['spatial'] = adata2_4me3.obsm['spatial_local'].copy()

adata1_27me3.var_names_make_unique() 
adata1_27ac.var_names_make_unique() 
adata2_27me3.var_names_make_unique() 
adata2_atac.var_names_make_unique() 
adata2_rna.var_names_make_unique() 
adata2_4me3.var_names_make_unique() 


adata1_27me3 = mark_meta(add_prefix(adata1_27me3, "batch1-H3K27me3"), "H3K27me3", "batch1")
adata1_27ac  = mark_meta(add_prefix(adata1_27ac,  "batch1-H3K27ac"),  "H3K27ac",  "batch1")

adata2_rna   = mark_meta(add_prefix(adata2_rna,   "batch2-RNA"),      "RNA",      "batch2")
adata2_atac  = mark_meta(add_prefix(adata2_atac,  "batch2-ATAC"),     "ATAC",     "batch2")
adata2_27me3 = mark_meta(add_prefix(adata2_27me3, "batch2-H3K27me3"), "H3K27me3", "batch2")
adata2_4me3  = mark_meta(add_prefix(adata2_4me3,  "batch2-H3K4me3"),  "H3K4me3",  "batch2")


hvg_rna = get_hvg(adata2_rna, topk=3000)
rna_list = [None, scmo_preprocess(adata2_rna, hvg_rna, modality="RNA", log=True)]

# ATAC 
hvp_atac = get_hvg(adata2_atac, topk=3000)
atac_list = [None, scmo_preprocess(adata2_atac, hvp_atac, modality="ATAC_gene_score", log=True)]

#  H3K27ac（batch1）
hvp_27ac = get_hvg(adata1_27ac, topk=3000)
h3k27ac_list = [scmo_preprocess(adata1_27ac, hvp_27ac, modality="H3K27ac", log=True), None]

# H3K4me3（batch2）
hvp_4me3 = get_hvg(adata2_4me3, topk=3000)
h3k4me3_list = [None, scmo_preprocess(adata2_4me3, hvp_4me3, modality="H3K4me3", log=True)]


# H3K27me3（ batch1  batch2 ）
adata1_27me3, adata2_27me3 = preprocess_shared_hvp(adata1_27me3, adata2_27me3, topk=3000)
h3k27me3_list = [
    scmo_preprocess(adata1_27me3, adata1_27me3.var_names, modality="H3K27me3", log=True),
    scmo_preprocess(adata2_27me3, adata2_27me3.var_names, modality="H3K27me3", log=True)
]

feats_name = {
    "rna": hvg_rna,  #  batch2
    "h3k27me3": adata1_27me3.var_names,  # unify  feature space
    "h3k27ac": hvp_27ac,
    "h3k4me3": hvp_4me3,
    "atac":hvp_atac,
}

counts = {
    "feats_name": feats_name,
    "nbatches": 2,
    "rna": rna_list,               # [None, RNA matrix from batch2]
    "h3k27me3": h3k27me3_list,     # [batch1, batch2]
    "h3k27ac": h3k27ac_list,       # [batch1, None]
    "h3k4me3": h3k4me3_list,       # [None, batch2]
    "atac": atac_list              # [None, batch2]
}



#------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Number of latent dimensions, key hyper-parameter, 20~30 works for most of the cases.
K = 30
#------------------------------------------------------------------------------------------------------------------------------------
# NOTE: Here we list other parameters in the function for illustration purpose, most of these parameters are set as default value.
# weight on regularization term, default value
lamb = 0.001 
# number of total iterations, default value
T = 4000
# print the result after each ``interval'' iterations, default value
interval = 1000
# batch size for each iteraction, default value
batch_size = 0.1
# learning rate, default value
lr = 1e-2
# random seed, default value
seed = 0
# running device, can be CPU or GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#------------------------------------------------------------------------------------------------------------------------------------

start_time = time.time()
model = scmomat.scmomat_model(counts = counts, K = K, batch_size = batch_size, interval = interval, lr = lr, lamb = lamb, seed = seed, device = device)
losses = model.train_func(T = T)
end_time = time.time()
print("running time: " + str(end_time - start_time))

# Plot loss function
x = np.linspace(0, T, int(T/interval)+1)
plt.plot(x, losses)

zs = model.extract_cell_factors()


import pickle

with open("scMoMaT_mouse_embryo.pkl", "wb") as f:
    pickle.dump(zs, f)




