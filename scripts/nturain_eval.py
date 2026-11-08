import json, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
def shp(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
def psnr(a, g, mk): return float(10 * np.log10(1 / (np.mean(((np.clip(a, 0, 1) - g) ** 2)[sl][mk[sl]]) + 1e-12)))
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8); mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mv); big = np.isin(lab, [k for k in range(1, n) if stats[k, 4] > 120]).astype(np.uint8); return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
def repl(x, m, th, thin=True):
    d = (x - m).mean(2); mk = (d > th / 255).astype(np.uint8)
    if thin: mk = mk & (1 - cv2.morphologyEx(mk, cv2.MORPH_OPEN, K5))
    mk = cv2.dilate(mk, K3).astype(bool); y = x.copy(); y[mk] = m[mk]; return y
def clamp(x, m, dl): return np.minimum(x, m + dl)
def variants(x, m9, m15):
    V = {"A10": repl(x, m9, 10), "A6": repl(x, m9, 6), "A4": repl(x, m9, 4), "A3": repl(x, m9, 3), "A6_nothin": repl(x, m9, 6, False), "A4_m15": repl(x, m15, 4), "clamp.04": clamp(x, m9, 0.04), "clamp.02": clamp(x, m9, 0.02), "clamp.01": clamp(x, m9, 0.01), "full_m9": m9.copy()}
    return V

import torch, torch.nn as nn, sys, os, time, torchvision; torchvision.__version__ = "0.99.0"
import baselines_v2 as B
dev = "cuda"
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
from mg_net import MG
ck = ROOT / "experiments_v2" / "checkpoints"
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
mf6 = MF().to(dev); mf6.load_state_dict(torch.load(ck / "student_mf6.pth", map_location=dev)["model_state_dict"]); mf6.eval()
mg = MG().to(dev); mg.load_state_dict(torch.load(ck / "student_mg.pth", map_location=dev)["model_state_dict"]); mg.eval()
ND = ROOT / "data" / "compare model" / "SPAC-SupplementaryMaterials-master" / "Dataset_Testing_Synthetic"
def frames(d):
    fs = sorted([p for p in Path(d).rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp")], key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)])
    return [cv2.imread(str(p)).astype(np.float32) / 255 for p in fs]
import re
def psnr_full(a, g): return float(10 * np.log10(1 / (np.mean((np.clip(a, 0, 1) - g) ** 2) + 1e-12)))
def ssim(a, g):
    a = cv2.cvtColor(np.clip(a, 0, 1), cv2.COLOR_BGR2GRAY).astype(np.float64); g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = 0.01 ** 2, 0.03 ** 2; f = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu1, mu2 = f(a), f(g); s11 = f(a * a) - mu1 ** 2; s22 = f(g * g) - mu2 ** 2; s12 = f(a * g) - mu1 * mu2
    return float((((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))).mean())
R = {}
for tag in ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"]:
    try:
        rain = frames(ND / (tag + "_Rain")); gt = frames(ND / (tag + "_GT")); sp = frames(ND / (tag + "_spacCNN"))
    except Exception as e: print(tag, "load error", e); continue
    n = min(len(rain), len(gt)); print(tag, "frames", len(rain), len(gt), len(sp), "size", rain[0].shape, flush=True)
    g0 = cv2.cvtColor(gt[10], cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(gt[26 if n > 26 else n - 1], cv2.COLOR_BGR2GRAY)
    (dx, dy), _ = cv2.phaseCorrelate(g0.astype(np.float32), g1.astype(np.float32)); shift16 = float(np.hypot(dx, dy))
    acc = {}
    centers = [int(v) for v in np.linspace(22, n - 23, 8)] if n > 46 else []
    for i in centers:
        fr = [rain[i + j] for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0)
        m15 = np.median(np.stack([rain[min(max(i + j, 0), n - 1)] for j in range(-21, 22, 6)]), 0); P = motion_mask(x, m15)
        win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
            MGo = mg(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); S7 = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        V = {"in": x, "MG": MGo, "mf7": S7, "A4": repl(x, m9, 4), "median9": m9}
        o = {k: (v if k == "in" else np.where(P[..., None], x, v)) for k, v in V.items()}
        o["median9_noprot"] = m9; o["SPAC_published"] = sp[i] if i < len(sp) else None
        for k, v in o.items():
            if v is None: continue
            acc.setdefault(k, []).append((psnr_full(v, gt[i]), ssim(v, gt[i])))
    R[tag] = {"shift_px_over_16_frames": shift16, **{k: [float(np.mean([a[0] for a in v])), float(np.mean([a[1] for a in v]))] for k, v in acc.items()}}
    print(tag, "shift16=%.1f" % shift16, {k: [round(x, 2) for x in v] for k, v in R[tag].items() if k != "shift_px_over_16_frames"}, flush=True)
json.dump(R, open(OUT / "nturain_eval.json", "w"), indent=1)
print("\nPSNR/SSIM per clip (camera shift px/16 frames)")
for t, r in R.items(): print(t, "shift %.1f" % r["shift_px_over_16_frames"], "  ".join(f"{k}:{v[0]:.2f}/{v[1]:.3f}" for k, v in r.items() if k != "shift_px_over_16_frames"))
print("DONE")
