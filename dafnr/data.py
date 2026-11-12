import os
import random
from glob import glob

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find(directory: str, stem: str) -> str:
    for ext in EXTS:
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            return p
    hits = glob(os.path.join(directory, stem + ".*"))
    if not hits:
        raise FileNotFoundError(f"no file for '{stem}' in {directory}")
    return hits[0]


class SegDataset(Dataset):
    def __init__(self, root: str, split: str, size: int = 224, train: bool = False, mask_suffix: str = "",
                 aug_prob: float = 0.5, rotation: float = 30.0, scale=(0.75, 1.25)):
        self.root, self.size, self.train = root, size, train
        self.mask_suffix, self.p, self.rotation, self.scale = mask_suffix, aug_prob, rotation, scale
        with open(os.path.join(root, "splits", f"{split}.txt")) as f:
            self.ids = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.ids)

    def _load(self, idx):
        sid = self.ids[idx]
        img = Image.open(_find(os.path.join(self.root, "images"), sid)).convert("RGB")
        mask = Image.open(_find(os.path.join(self.root, "masks"), sid + self.mask_suffix)).convert("L")
        img = TF.resize(img, [self.size, self.size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        img = TF.to_tensor(img)
        mask = torch.from_numpy((np.array(mask) > 127).astype(np.float32))[None]
        return sid, img, mask

    def _augment(self, img, mask):
        if random.random() < self.p:
            img, mask = TF.hflip(img), TF.hflip(mask)
        if random.random() < self.p:
            img, mask = TF.vflip(img), TF.vflip(mask)
        angle = random.uniform(-self.rotation, self.rotation) if random.random() < self.p else 0.0
        scale = random.uniform(*self.scale) if random.random() < self.p else 1.0
        if angle != 0.0 or scale != 1.0:
            img = TF.affine(img, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                            interpolation=InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                             interpolation=InterpolationMode.NEAREST)
        return img, mask

    def __getitem__(self, idx):
        sid, img, mask = self._load(idx)
        if self.train:
            img, mask = self._augment(img, mask)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
        return {"id": sid, "image": img, "mask": mask}
