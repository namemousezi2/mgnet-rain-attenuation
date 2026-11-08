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
import torch, torch.nn as nn
dev = "cuda"
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
mf = MF().to(dev); mf.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
from mg_net import MG
mf6 = MG().to(dev); mf6.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); mf6.eval()
R = {}
def outs_at(c, i):
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0); m15 = np.median(np.stack([getf(c, i + j) for j in range(-21, 22, 6)]), 0); P = motion_mask(x, m15)
    win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad(): X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]; S = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); S6 = mf6(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    V = {"in": x, "A10": repl(x, m9, 10), "A4": repl(x, m9, 4), "S7": S, "MG": S6, "full_m9": m9.copy()}
    return {k: (v if k == "in" else np.where(P[..., None], x, v)) for k, v in V.items()}
def scene(tag, c, N, st, idxs):
    s = st[sl]; acc = {}
    for i in idxs:
        o1, o2 = outs_at(c, i), outs_at(c, i + 1)
        for k in o1:
            f = float(np.abs(o1[k] - o2[k]).mean(2)[sl][s].mean()); acc.setdefault(k, []).append(f)
    base = np.mean(acc["in"]); R[tag] = {k: float(np.mean(v) / base) for k, v in acc.items()}; R[tag]["flicker_in"] = float(base)
    print(tag, {k: round(v, 4) for k, v in R[tag].items()}, flush=True)
c = cv2.VideoCapture(str(D / "iphon_raining.MOV")); fps = c.get(5); N = int(c.get(7)); st = smask(c, 5 * fps, 10 * fps)
for bn, rg in {"iphon_mid": range(55, 121, 8), "iphon_heavy": range(125, 241, 12)}.items(): scene(bn, c, N, st, [int(t * fps) for t in rg])
scene("DRY_iphon_8to10s", c, N, smask(c, 5 * fps, 7.5 * fps), [int(t * fps) for t in np.linspace(8.3, 9.7, 8)])
for k in HELD:
    dn = CCTV[k]["best_dry"]; rm, cr, N = med(D / k); gtm, cd, Nd = med(D / dn); GT, w = align(rm, gtm); st = smask(cd, 0, Nd - 2, w); scene(k[:22], cr, N, st, list(np.linspace(30, N - 31, 8).astype(int)))
rn, an = "iphone_light rain2", "iphone_after_rain2"; pr = cv2.VideoCapture(str(D / (rn + ".MOV"))); N = int(pr.get(7)); rm = med(D / (rn + ".MOV"))[0]; gtm, pa, Na = med(D / (an + ".MOV")); GT, w = align(rm, gtm); scene(rn, pr, N, smask(pa, 0, Na - 2, w), list(np.linspace(30, N - 31, 8).astype(int)))
rm, ca, Na = med(D / "cctv_shop_after+rain.mp4"); gtm, cb, Nb = med(D / "cctv_shop_before+rain.mp4"); GT, w = align(rm, gtm); sd = smask(cb, 0, Nb - 2, w); scene("DRY_shop_after", ca, Na, sd, list(np.linspace(30, Na - 31, 8).astype(int)))
rm, c2, N2 = med(D / "cctv_shop2.mp4"); scene("shop2_rain", c2, N2, sd, list(np.linspace(30, N2 - 31, 8).astype(int)))
json.dump(R, open(OUT / "flicker_mg.json", "w"), indent=1); print("DONE")
