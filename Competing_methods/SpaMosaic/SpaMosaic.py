# Please refer to https://github.com/JinmiaoChenLab/SpaMosaic for installation


import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import scanpy as sc
from os.path import join

import pickle
import numpy as np
import time
from spamosaic.framework import SpaMosaic
import spamosaic.utils as utls
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
import os, random, sys
import scipy.io as sio
from scipy import sparse
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.decomposition import PCA
from typing import List, Optional

import torch
from torch import Tensor
from sklearn.utils.extmath import randomized_svd
import scanpy.external as sce
from sklearn.decomposition import PCA
torch.use_deterministic_algorithms(False)


adata1_27me3 = sc.read_h5ad('E13_50_1_H3K27me3.h5ad')
adata1_27ac = sc.read_h5ad('E13_50_1_H3K27ac.h5ad')
adata2_27me3 = sc.read_h5ad('E13_50_3_H3K27me3.h5ad')
adata2_atac = sc.read_h5ad('E13_50_3_ATAC.h5ad')
adata2_rna = sc.read_h5ad('E13_50_3_RNA.h5ad')
adata2_4me3 = sc.read_h5ad('E13_50_3_H3K4me3.h5ad')


adata1_27me3.obs['src'] = 's1'
adata1_27ac.obs['src'] = 's1'
adata2_27me3.obs['src'] = 's2'
adata2_atac.obs['src'] = 's2'
adata2_rna.obs['src'] = 's2'
adata2_4me3.obs['src'] = 's2'
adata1_27me3.obs_names = [name + '_s1' for name in adata1_27me3.obs_names]
adata1_27ac.obs_names = [name + '_s1' for name in adata1_27ac.obs_names]
adata2_27me3.obs_names = [name + '_s2' for name in adata2_27me3.obs_names]
adata2_atac.obs_names = [name + '_s2' for name in adata2_atac.obs_names]
adata2_rna.obs_names = [name + '_s2' for name in adata2_rna.obs_names]
adata2_4me3.obs_names = [name + '_s2' for name in adata2_4me3.obs_names]

adata1_27me3.obsm['spatial'] = adata1_27me3.obsm['spatial_local']
adata1_27ac.obsm['spatial'] = adata1_27ac.obsm['spatial_local']
adata2_27me3.obsm['spatial'] = adata2_27me3.obsm['spatial_local']
adata2_atac.obsm['spatial'] = adata2_atac.obsm['spatial_local']
adata2_rna.obsm['spatial'] = adata2_rna.obsm['spatial_local']
adata2_4me3.obsm['spatial'] = adata2_4me3.obsm['spatial_local']


adata1_27me3.var_names_make_unique()
adata1_27ac.var_names_make_unique()
adata2_27me3.var_names_make_unique()
adata2_atac.var_names_make_unique()
adata2_rna.var_names_make_unique()
adata2_4me3.var_names_make_unique()



input_key = 'dimred_bc'

input_dict = {
    'H3K27ac':[adata1_27ac, None],
    'H3K27me3':[adata1_27me3, adata2_27me3],
    'ATAC': [None, adata2_atac],
    'rna': [None, adata2_rna],
    'adata2_4me3': [None, adata2_4me3]
}


RNA_preprocess(input_dict['rna'], batch_corr=False, favor='scanpy', n_hvg=3000, batch_key='src', key=input_key)  
Epigenome_preprocess(input_dict['H3K27ac'], batch_corr=False,  n_peak=3000, batch_key='src', key=input_key)  
Epigenome_preprocess(input_dict['H3K27me3'], batch_corr=True,  n_peak=3000, batch_key='src', key=input_key) 
Epigenome_preprocess(input_dict['ATAC'], batch_corr=False,  n_peak=3000, batch_key='src', key=input_key)  
Epigenome_preprocess(input_dict['adata2_4me3'], batch_corr=False,  n_peak=3000, batch_key='src', key=input_key)  

start_time = time.time()

model = SpaMosaic(
    modBatch_dict=input_dict, input_key=input_key,
    batch_key='src', intra_knns=10, inter_knn_base=10, w_g=0.8,
    seed=1234,
    device='cuda:0'
)

model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)


ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
ad_mosaic = sc.concat(ad_embs)
end_time = time.time()
print("running time: " + str(end_time - start_time))

ad_mosaic.write('ad_epi_s1_2_new.h5ad')


