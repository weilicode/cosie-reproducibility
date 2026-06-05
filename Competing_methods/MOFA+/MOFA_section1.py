## Please refer to https://biofam.github.io/MOFA2/ for installation

import numpy as np
import pandas as pd
import scanpy as sc

from matplotlib import pyplot as plt
import seaborn as sns
import muon as mu
from pathlib import Path
import os

adata1_27me3 = sc.read_h5ad('E13_50_1_H3K27me3.h5ad')
adata1_27ac  = sc.read_h5ad('E13_50_1_H3K27ac.h5ad')
adata1_27me3.obsm['spatial'] = adata1_27me3.obsm['spatial_local'].copy()
adata1_27ac.obsm['spatial'] = adata1_27ac.obsm['spatial_local'].copy()

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sc.pp.normalize_total(adata1_27me3)
sc.pp.log1p(adata1_27me3)
sc.pp.scale(adata1_27me3)
sc.tl.pca(adata1_27me3, n_comps=50)

adata1_27me3_pca = sc.AnnData(X=adata1_27me3.obsm['X_pca'])
adata1_27me3_pca.obsm["spatial"] = adata1_27me3.obsm["spatial"].copy()

sc.pp.normalize_total(adata1_27ac)
sc.pp.log1p(adata1_27ac)
sc.pp.scale(adata1_27ac)
sc.tl.pca(adata1_27ac, n_comps=50)

adata1_27ac_pca = sc.AnnData(X=adata1_27ac.obsm['X_pca'])
adata1_27ac_pca.obsm["spatial"] = adata1_27ac.obsm["spatial"].copy()
mdata_pca = mu.MuData({'HK27me3': adata1_27me3_pca,  'HK27ac': adata1_27ac_pca})

mu.tl.mofa(mdata_pca, outfile="MOFA_mouse_embryo_s1_pca.hdf5", n_factors=10)
np.save('mofa_emb_mouse_embryo_s1.npy',mdata_pca.obsm['X_mofa'])






