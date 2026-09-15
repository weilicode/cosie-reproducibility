## Please refer to https://github.com/prov-gigatime/GigaTIME/tree/main for pretrained checkpoint
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import glob
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import torchvision
from torchvision.utils import save_image
from collections import OrderedDict
from datetime import datetime
import numpy as np
import torch
from scipy import stats
import random
import archs
import losses
from metrics import iou_score
from utils import AverageMeter, str2bool
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch
import torch.distributed as dist
import os
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, DistributedSampler
from scipy.stats import pearsonr, spearmanr
from prov_data import *
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
import warnings, os, sys
from easydict import EasyDict as edict
import albumentations as geometric
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

SEED = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

print("Seeded. Torch:", torch.__version__, "CUDA:", torch.version.cuda)

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


channel_names = [
    'DAPI',
    'TRITC',   # background
    'Cy5',     # background
    'PD-1',
    'CD14',
    'CD4',
    'T-bet',
    'CD34',
    'CD68',
    'CD16',
    'CD11c',
    'CD138',
    'CD20',
    'CD3',
    'CD8',
    'PD-L1',
    'CK',
    'Ki67',
    'Tryptase',
    'Actin-D',
    'Caspase3-D',
    'PHH3-B',
    'Transgelin'
]

common_channel_list = channel_names
def parse_args():
    config = edict()

    config.name = "gigatime_inference"
    config.output_dir = "/gigatime_wsi_patches_P24_LUAD_Visium/gigatime_output"
    config.patch_dir = "/gigatime_wsi_patches_P24_LUAD_Visium"   
    config.batch_size = 12                        
    config.arch = "gigatime"
    config.input_channels = 3
    config.num_classes = 23
    config.input_w = 256 
    config.input_h = 256 
    config.patch_suffix = "_he.png"

    return config

config = parse_args()

val_transform = Compose([
    geometric.Resize(config.input_h, config.input_w),
    transforms.Normalize(),
])



class HEPatchInferenceDataset(Dataset):
    def __init__(self, patch_dir, transform, patch_suffix="_he.png"):
        self.patch_dir = patch_dir
        self.transform = transform
        self.patch_paths = sorted(
            glob.glob(os.path.join(patch_dir, f"*{patch_suffix}"))
        )

        if len(self.patch_paths) == 0:
            raise ValueError(f"No patch files found in {patch_dir} with suffix {patch_suffix}")

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, idx):
        patch_path = self.patch_paths[idx]
        img = np.array(Image.open(patch_path).convert("RGB"))

        augmented = self.transform(image=img, mask=img)
        img = augmented["image"].astype("float32")
        img = img.transpose(2, 0, 1)  # HWC -> CHW

        patch_name = os.path.basename(patch_path)
        return img, patch_name



inference_dataset = HEPatchInferenceDataset(
    patch_dir=config.patch_dir,
    transform=val_transform,
    patch_suffix=config.patch_suffix
)

inference_loader = DataLoader(
    inference_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=0,
    drop_last=False
)

print("Number of inference patches:", len(inference_dataset))

def plot_inference_patch(idx=0):
    img, patch_name = inference_dataset[idx]

    he_img = img[:3].copy()
    he_img = he_img * std[:, None, None] + mean[:, None, None]
    he_img = np.clip(he_img, 0, 1)
    he_img = np.transpose(he_img, (1, 2, 0))

    plt.figure(figsize=(5, 5))
    plt.imshow(he_img)
    plt.title(patch_name)
    plt.axis("off")
    plt.show()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = archs.__dict__[config.arch](config.num_classes, config.input_channels)

repo_id = "prov-gigatime/GigaTIME"
local_dir = snapshot_download(repo_id=repo_id)

weights_path = os.path.join(local_dir, "model.pth")
state_dict = torch.load(weights_path, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("=> loaded model", config.arch)
print("Model device:", next(model.parameters()).device)

os.makedirs(os.path.join(config.output_dir, "models", config.name), exist_ok=True)

with open(os.path.join(config.output_dir, "models", config.name, "config.yml"), "w") as f:
    yaml.dump(dict(config), f)

def do_inference(input_image, model, window_size=256):
    b, c, h, w = input_image.shape
    output_logits = torch.zeros(
        b, config.num_classes, h, w,
        device=input_image.device
    )

    with torch.no_grad():
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = input_image[:, :, i:i + window_size, j:j + window_size]
                logits = model(window)
                output_logits[:, :, i:i + window_size, j:j + window_size] = logits

    return output_logits


def plot_patch_predictions(idx=0, channels_to_show=None, show_binary=False):
    if channels_to_show is None:
        channels_to_show = ["CD3", "CD8", "CD20", "CK", "PD-L1", "Ki67"]

    imgs, patch_names = next(iter(inference_loader))
    imgs = imgs.to(device)

    with torch.no_grad():
        logits = do_inference(imgs, model, window_size=256)
        probs = torch.sigmoid(logits)
        binary = (probs > 0.5).float()

    probs_np = probs.cpu().numpy()
    binary_np = binary.cpu().numpy()

    he_img = imgs[idx, :3].cpu().numpy()
    he_img = he_img * std[:, None, None] + mean[:, None, None]
    he_img = np.clip(he_img, 0, 1)
    he_img = np.transpose(he_img, (1, 2, 0))

    n = len(channels_to_show)
    n_cols = 1 + n
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(he_img)
    axes[0].set_title(f"H&E\n{patch_names[idx]}")
    axes[0].axis("off")

    for i, ch_name in enumerate(channels_to_show, start=1):
        ch_idx = common_channel_list.index(ch_name)
        if show_binary:
            axes[i].imshow(binary_np[idx, ch_idx], cmap="gray")
            axes[i].set_title(f"{ch_name}\nbinary")
        else:
            axes[i].imshow(probs_np[idx, ch_idx], cmap="viridis")
            axes[i].set_title(f"{ch_name}\nprob")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def parse_patch_name(patch_name):
    """
    Expected patch name format:
        {y}_{x}_{patch_h}_{patch_w}_he.png
    """
    stem = patch_name.replace("_he.png", "")
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected patch name format: {patch_name}")

    y = int(parts[0])
    x = int(parts[1])
    patch_h = int(parts[2])
    patch_w = int(parts[3])

    return y, x, patch_h, patch_w


def run_gigatime_inference(
    loader,
    model,
    output_dir,
    save_probs=True,
    save_binary=False,
    threshold=0.5,
    window_size=256
):
    os.makedirs(output_dir, exist_ok=True)
    pred_records = []

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(loader), desc="Running GigaTIME inference")

        for imgs, patch_names in loader:
            imgs = imgs.to(device)

            logits = do_inference(imgs, model, window_size=window_size)
            probs = torch.sigmoid(logits)

            if save_binary:
                binary = (probs > threshold).float()

            probs_np = probs.cpu().numpy()
            if save_binary:
                binary_np = binary.cpu().numpy()

            batch_size = probs_np.shape[0]

            for i in range(batch_size):
                patch_name = patch_names[i]
                y, x, patch_h, patch_w = parse_patch_name(patch_name)

                stem = patch_name.replace("_he.png", "")
                record = {
                    "patch_name": patch_name,
                    "x": x,
                    "y": y,
                    "patch_h": patch_h,
                    "patch_w": patch_w,
                }

                if save_probs:
                    prob_path = os.path.join(output_dir, f"{stem}_gigatime_prob.npy")
                    np.save(prob_path, probs_np[i])
                    record["prob_path"] = prob_path

                if save_binary:
                    binary_path = os.path.join(output_dir, f"{stem}_gigatime_binary.npy")
                    np.save(binary_path, binary_np[i])
                    record["binary_path"] = binary_path

                pred_records.append(record)

            pbar.update(1)

        pbar.close()

    pred_df = pd.DataFrame(pred_records)
    pred_df.to_csv(os.path.join(output_dir, "prediction_index.csv"), index=False)

    return pred_df


prediction_output_dir = os.path.join(config.output_dir, "models", config.name, "patch_predictions")

pred_df = run_gigatime_inference(
    loader=inference_loader,
    model=model,
    output_dir=prediction_output_dir,
    save_probs=True,
    save_binary=True,   
    threshold=0.5,
    window_size=256
)




