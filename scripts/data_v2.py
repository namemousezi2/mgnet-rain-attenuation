import hashlib
import numpy as np, pandas as pd, cv2, torch
from torch.utils.data import Dataset
from common import imread01


def is_val(path):
    """Deterministic 10 % of TRAIN rows used for model selection (never the test split)."""
    return int(hashlib.md5(str(path).encode()).hexdigest(), 16) % 10 == 0


class TripletDataset(Dataset):
    """Returns frames (3,3,H,W) BGR in [0,1], clean (3,H,W), label, index.
    train=True -> random 256x256 crop + horizontal flip (no resizing, keeps geometry)."""
    def __init__(self, csv, subset="all", crop=256, train=False):
        df = pd.read_csv(csv)
        if subset == "train": df = df[~df.curr.map(is_val)]
        elif subset == "val": df = df[df.curr.map(is_val)]
        self.df = df.reset_index(drop=True); self.crop = crop; self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        imgs = [imread01(r.prev), imread01(r.curr), imread01(r.next), imread01(r.clean)]
        h = min(x.shape[0] for x in imgs); w = min(x.shape[1] for x in imgs)
        imgs = [x[:h, :w] for x in imgs]
        if self.train or self.crop:
            c = self.crop
            if h < c or w < c:
                imgs = [cv2.resize(x, (max(w, c), max(h, c))) for x in imgs]; h, w = imgs[0].shape[:2]
            y0 = np.random.randint(0, h - c + 1) if self.train else (h - c) // 2
            x0 = np.random.randint(0, w - c + 1) if self.train else (w - c) // 2
            imgs = [x[y0:y0 + c, x0:x0 + c] for x in imgs]
            if self.train and np.random.rand() < 0.5: imgs = [x[:, ::-1] for x in imgs]
        t = [torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))) for x in imgs]
        return torch.stack(t[:3]), t[3], int(r.label), i


class ClsDataset(Dataset):
    """Classifier input: RGB 224x224 in [0,1] (same preprocessing as the original training)."""
    def __init__(self, csv, subset="all", train=False):
        df = pd.read_csv(csv)
        if subset == "train": df = df[~df.curr.map(is_val)]
        elif subset == "val": df = df[df.curr.map(is_val)]
        self.df = df.reset_index(drop=True); self.train = train

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(str(r.curr))
        img = cv2.cvtColor(cv2.resize(img, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.train and np.random.rand() < 0.5: img = img[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))), int(r.label), i


def cls_input_from_bgr01(frame_bgr01):
    """BGR [0,1] (H,W,3) numpy -> classifier tensor (1,3,224,224)."""
    img = cv2.cvtColor(cv2.resize(frame_bgr01, (224, 224)), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[None]
