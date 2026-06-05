## Please refer to https://github.com/kpcoleman/miso for installation

from miso.hist_features import get_features
from miso.utils import *
from miso import Miso
from PIL import Image
import pandas as pd
import numpy as np
import scanpy as sc
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
Image.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None
import torch
import random

seed=100
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("CUDA is available. GPU:", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    print("CUDA is not available. Using CPU.")


file_path = '../../../../project/SpatialMultimodal/all_data/Spatial_mux_seq'


adata2_27me3 = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_H3K27me3.h5ad'))
adata2_atac  = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_ATAC.h5ad'))
adata2_rna   = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_RNA.h5ad'))
adata2_4me3  = sc.read_h5ad(os.path.join(file_path, 'E13_50_3_H3K4me3.h5ad'))
adata2_27me3.obsm['spatial'] = adata2_27me3.obsm['spatial_local'].copy()
adata2_atac.obsm['spatial'] = adata2_atac.obsm['spatial_local'].copy()
adata2_rna.obsm['spatial'] = adata2_rna.obsm['spatial_local'].copy()
adata2_4me3.obsm['spatial'] = adata2_4me3.obsm['spatial_local'].copy()
adata2_27me3_data = preprocess(adata2_27me3,modality='atac')
adata2_atac_data = preprocess(adata2_atac,modality='atac')
adata2_rna_data = preprocess(adata2_rna,modality='rna')
adata2_4me3_data = preprocess(adata2_4me3,modality='atac')

model = Miso([adata2_27me3_data, adata2_atac_data, adata2_rna_data, adata2_4me3_data],ind_views='all',combs='all',sparse=True,device=device)
model.train()

np.save('emb_embryo_s2.npy', model.emb)











## 