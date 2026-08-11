import os
import json
import numpy as np
import h5py
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

def load_coco_data(base_dir, 
                   pca_features=True):
    base_dir = Path(base_dir)
    data = {}

    with h5py.File(base_dir / "coco2014_captions.h5", "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    for split in ("train", "val"):
        feat_suffix = "_pca" if pca_features else ""
        feat_file = base_dir / f"{split}2014_vgg16_fc7{feat_suffix}.h5"
        with h5py.File(feat_file, "r") as f:
            data[f"{split}_features"] = np.asarray(f["features"])

    with open(base_dir / "coco2014_vocab.json", "r") as f:
        dict_data = json.load(f)
        data.update(dict_data)   

    for split in ("train", "val"):
        url_file = base_dir / f"{split}2014_urls.txt"
        with open(url_file, "r") as f:
            data[f"{split}_urls"] = np.asarray([line.strip() for line in f])

    return data


class CocoDataset(Dataset):
    def __init__(self, 
                 data, 
                 split='train',
                 captions_per_image=5):
        self.data = data
        self.split = split

        self.captions = data[f'{split}_captions']
        self.features = data[f'{split}_features']

        self.num_samples = self.captions.shape[0]
        self.num_images = self.features.shape[0]
        self.captions_per_image = captions_per_image

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        cap = self.captions[idx]
        img_idx = idx // self.captions_per_image

        if img_idx >= self.num_images:
            img_idx = idx % self.num_images
        feat = self.features[img_idx]

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(cap, dtype=torch.long)

def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded