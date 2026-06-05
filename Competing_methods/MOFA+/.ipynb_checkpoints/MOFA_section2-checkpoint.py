## Please refer to https://biofam.github.io/MOFA2/ for installation

import numpy as np
import pandas as pd
import scanpy as sc

from matplotlib import pyplot as plt
import seaborn as sns
import muon as mu
from pathlib import Path
import os


adata2_27me3 = sc.read_h5ad('E13_50_3_H3K27me3.h5ad')
adata2_atac  = sc.read_h5ad('E13_50_3_ATAC.h5ad')
adata2_rna   = sc.read_h5ad('E13_50_3_RNA.h5ad')
adata2_4me3  = sc.read_h5ad('E13_50_3_H3K4me3.h5ad')
adata2_27me3.obsm['spatial'] = adata2_27me3.obsm['spatial_local'].copy()
adata2_atac.obsm['spatial'] = adata2_atac.obsm['spatial_local'].copy()
adata2_rna.obsm['spatial'] = adata2_rna.obsm['spatial_local'].copy()
adata2_4me3.obsm['spatial'] = adata2_4me3.obsm['spatial_local'].copy()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sc.pp.normalize_total(adata2_27me3)
sc.pp.log1p(adata2_27me3)
sc.pp.scale(adata2_27me3)
sc.tl.pca(adata2_27me3, n_comps=50)

adata2_27me3_pca = sc.AnnData(X=adata2_27me3.obsm['X_pca'])
adata2_27me3_pca.obsm["spatial"] = adata2_27me3.obsm["spatial"].copy()

sc.pp.normalize_total(adata2_atac)
sc.pp.log1p(adata2_atac)
sc.pp.scale(adata2_atac)
sc.tl.pca(adata2_atac, n_comps=50)
adata2_atac_pca = sc.AnnData(X=adata2_atac.obsm['X_pca'])
adata2_atac_pca.obsm["spatial"] = adata2_atac.obsm["spatial"].copy()

sc.pp.normalize_total(adata2_rna)
sc.pp.log1p(adata2_rna)
sc.pp.scale(adata2_rna)
sc.tl.pca(adata2_rna, n_comps=50)
adata2_rna_pca = sc.AnnData(X=adata2_rna.obsm['X_pca'])
adata2_rna_pca.obsm["spatial"] = adata2_rna.obsm["spatial"].copy()

sc.pp.normalize_total(adata2_4me3)
sc.pp.log1p(adata2_4me3)
sc.pp.scale(adata2_4me3)
sc.tl.pca(adata2_4me3, n_comps=50)
adata2_4me3_pca = sc.AnnData(X=adata2_4me3.obsm['X_pca'])
adata2_4me3_pca.obsm["spatial"] = adata2_4me3.obsm["spatial"].copy()

mdata_pca = mu.MuData({'adata2_27me3_pca': adata2_27me3_pca,  'adata2_atac_pca': adata2_atac_pca, 'adata2_rna_pca':adata2_rna_pca, 'adata2_4me3_pca':adata2_4me3_pca})

mu.tl.mofa(mdata_pca, outfile="MOFA_mouse_embryo_s2_pca.hdf5", n_factors=10)

np.save('mofa_emb_mouse_embryo_s2.npy',mdata_pca.obsm['X_mofa'])








