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



adata1_27me3 = sc.read_h5ad('E13_50_1_H3K27me3.h5ad')
adata1_27ac  = sc.read_h5ad('E13_50_1_H3K27ac.h5ad')
adata1_27me3.obsm['spatial'] = adata1_27me3.obsm['spatial_local'].copy()
adata1_27ac.obsm['spatial'] = adata1_27ac.obsm['spatial_local'].copy()
adata1_27me3_data = preprocess(adata1_27me3,modality='atac')
adata1_27ac_data = preprocess(adata1_27ac,modality='atac')



model = Miso([adata1_27me3_data, adata1_27ac_data],ind_views='all',combs='all',sparse=True,device=device)
model.train()

np.save('emb_embryo_s1.npy', model.emb)











## 