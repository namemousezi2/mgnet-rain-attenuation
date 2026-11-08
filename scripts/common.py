"""Shared helpers for the v2 (reproducible) experiments.

All scripts are run from the PROJECT ROOT, e.g.
    python experiments_v2/make_split.py
"""
import os, sys, json, random
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments_v2" / "results"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"
SYN_DIR = DATA / "training_synthetic"
VID_DIR = DATA / "video_dataset_synthetic"
REAL_DIR = DATA / "real_video_triplets"          # created by build_real_triplets.py
SPA_TEST = DATA / "spa_data" / "real_test_1000"   # official SPA-Data real test set (1000 pairs)
CLASSES = ["light", "moderate", "heavy"]

# Synthetic backgrounds held out for testing (never seen in training)
TEST_BACKGROUNDS = ["sample_street.png", "sample_park.png"]
# Real videos: frames in the last TEST_FRACTION of each video (by time) are test frames
TEST_FRACTION = 0.25


def seed_all(seed=0):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def p(rel):
    """Normalise a relative path stored with Windows separators."""
    return str(rel).replace("\\", "/")


def imread01(path):
    """BGR float32 in [0,1] (the derainer was trained on BGR, see training/dataset.py)."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def to_tensor(img):
    import torch
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b):
    """SSIM of Wang et al. (2004): 11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03, data range 1,
    computed per channel and averaged (no scikit-image dependency)."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)
    vals = []
    for c in range(a.shape[2]):
        x, y = a[..., c], b[..., c]
        mx, my = g(x), g(y)
        sxx = g(x * x) - mx * mx; syy = g(y * y) - my * my; sxy = g(x * y) - mx * my
        m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
        vals.append(m[5:-5, 5:-5].mean())
    return float(np.mean(vals))


def psnr_y(a, b):
    ya = cv2.cvtColor((a * 255).round().astype(np.uint8), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    yb = cv2.cvtColor((b * 255).round().astype(np.uint8), cv2.COLOR_BGR2YCrCb)[..., 0].astype(np.float64)
    mse = np.mean((ya - yb) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def save_json(obj, name):
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("  saved", OUT / name)
