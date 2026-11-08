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
M = {}
for n in ["prenet_ft", "restormer"]:
    m, mult = B.LOADERS[n](); M[n] = (m.to(dev).eval(), mult); print("loaded", n, flush=True)
def base(n, x):
    m, mult = M[n]
    try:
        with torch.no_grad(): return np.clip(B.infer(m, mult, x), 0, 1)
    except Exception:
        torch.cuda.empty_cache()
        with torch.no_grad(): return np.clip(B.infer(m, mult, x, tile=256), 0, 1)
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
for _k in [k for k in sys.modules if k == "networks" or k.startswith("networks.")]: del sys.modules[_k]
from networks.derain_net import DerainNet
sv = DerainNet(n_features=32, n_resblocks=8).to(dev); sv.load_state_dict(torch.load(str(S2 / "model_states" / "derainer_rho05.pt"), map_location=dev)); sv.eval()
SRC = Path(__file__).parent / "asf_src"; sys.path.insert(0, str(SRC)); cwd = os.getcwd(); os.chdir(SRC)
from Network import TOFlow
asf = TOFlow().to(dev); sd = torch.load(str(SRC / "toflow_models_mine" / "denoising_best.pkl"), map_location=dev); asf.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False); asf.eval(); os.chdir(cwd)
@torch.no_grad()
def run_asf(fr):
    t = torch.from_numpy(np.stack([f[:, :, ::-1] for f in fr]).copy()).permute(0, 3, 1, 2)[None].to(dev)
    Hh = t.shape[-2]; ph = (32 - Hh % 32) % 32; t = torch.nn.functional.pad(t.flatten(0, 1), (0, 0, 0, ph), mode="reflect").view(1, 3, 3, Hh + ph, -1)
    o = asf(t)[0, :, :Hh].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); return np.ascontiguousarray(o[:, :, ::-1])
def outs_at(c, i):
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0); m15 = np.median(np.stack([getf(c, i + j) for j in range(-21, 22, 6)]), 0); P = motion_mask(x, m15)
    win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
        S7 = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); S6 = mf6(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); MGo = mg(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        t7 = torch.from_numpy(np.stack([getf(c, i + j) for j in range(-3, 4)])).permute(3, 0, 1, 2)[None].to(dev)
        S2o = sv(t7)[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()[3]
    V = {"in": x, "MG": MGo, "mf7": S7, "mf6": S6, "A4": repl(x, m9, 4), "full_m9": m9.copy()}
    o = {k: (v if k == "in" else np.where(P[..., None], x, v)) for k, v in V.items()}
    o["S2VD"] = S2o; o["PReNet"] = base("prenet_ft", x); o["Restormer"] = base("restormer", x); o["ASF"] = run_asf([getf(c, i + j) for j in (-1, 0, 1)])
    return o
def scene(tag, c, st, idxs):
    s = st[sl]; acc = {}
    for i in idxs:
        o1, o2 = outs_at(c, i), outs_at(c, i + 1)
        for k in o1:
            acc.setdefault("sh_" + k, []).append(shp(o1[k], st) / (shp(o1["in"], st) + 1e-9))
            acc.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s].mean()))
    f_in = float(np.mean(acc["in"])); r = {k: (float(np.mean(v)) if k.startswith("sh_") else float(np.mean(v)) / f_in) for k, v in acc.items()}; r["flicker_in"] = f_in
    print(tag, {k: round(v, 4) for k, v in r.items()}, flush=True); return r
c = cv2.VideoCapture(str(D / "iphon_raining.MOV")); fps = c.get(5); st = smask(c, 5 * fps, 10 * fps)
R = {}
R["mid"] = scene("iphon_mid", c, st, [int(t * fps) for t in range(55, 121, 8)])
R["heavy"] = scene("iphon_heavy", c, st, [int(t * fps) for t in range(125, 241, 12)])
R["dry_old"] = scene("DRY_old_8.3to9.7", c, smask(c, 5 * fps, 7.5 * fps), [int(t * fps) for t in np.linspace(8.3, 9.7, 8)])
R["dry_new"] = scene("DRY_new_1.3to1.8", c, smask(c, 1.0 * fps, 2.3 * fps), [int(t * fps) for t in np.linspace(1.3, 1.8, 8)])
json.dump(R, open(OUT / "new_iph_all.json", "w"), indent=1)
meths = [k for k in R["mid"] if not k.startswith("sh_") and k not in ("in", "flicker_in")]
print("\nREMOVED % (mean of mid, heavy) | ref=old, ref=new | sharp")
for k in meths:
    row = []
    for ref in ("dry_old", "dry_new"):
        v = []
        for sc in ("mid", "heavy"):
            fr_, fd_ = R[sc]["flicker_in"], R[ref]["flicker_in"]; S = R[sc][k]; rho = R[ref][k]
            v.append(1 - (S - rho * fd_ / fr_) / (1 - fd_ / fr_))
        row.append(100 * np.mean(v))
    print(f"{k:10s} {row[0]:6.1f} {row[1]:6.1f}  sharp {np.mean([R['mid']['sh_'+k], R['heavy']['sh_'+k]]):.3f}")
print("DONE")
