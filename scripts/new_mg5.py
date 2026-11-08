"""Distil the sharpness-preserving masked-temporal-median (A) into a single-frame student: target T = x except thin bright streak pixels -> 9-frame median.
Train: CCTV clips (minus 3 held-out) + iPhone light rain1 (+ dry frames).  Test (never trained): 3 CCTV clips, iphon_raining mid/heavy, iPhone light rain2.
Metrics (independent 15-frame/stride-3 streak detector, same median for in/out): streak density ratio out/in, bg_change (|out-x| outside streak mask), static sharpness ratio, static PSNR gain."""
import cv2, numpy as np, json, torch, time, copy, glob, os
import torch.nn.functional as F
from common import ROOT, OUT
from order_synth import infer as sinfer
from student_net import load_student
import baselines_nv as BN
dev = BN.dev; W, H = 640, 360; D = ROOT / "data" / "new_clips"; BUDGET = 0.4 * 3600; P = 160; sl = (slice(16, H - 16), slice(24, W - 24))
CCTV = json.load(open(OUT / "cctv_pairs_check.json"))["pairs"]
def getf(c, i):
    c.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = c.read(); return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 if ok else None
def med(p, n=60):
    c = cv2.VideoCapture(str(p)); N = int(c.get(7)); fr = [getf(c, i) for i in np.linspace(0, N - 2, n).astype(int)]; return np.median(np.stack([f for f in fr if f is not None]), 0), c, N
def gb(a, b):
    g = []
    for k in range(3):
        A = np.stack([a[..., k].ravel(), np.ones(a[..., k].size)], 1); g.append(np.linalg.lstsq(A, b[..., k].ravel(), rcond=None)[0])
    return g
def app(x, g): return np.stack([x[..., k] * g[k][0] + g[k][1] for k in range(3)], -1)
def align(rm, gtm):
    w = np.eye(2, 3, dtype=np.float32); ga, ra = cv2.cvtColor(gtm, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rm, cv2.COLOR_BGR2GRAY)
    cc, w = cv2.findTransformECC(ra, ga, w, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    G = cv2.warpAffine(gtm, w, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT); return app(G, gb(rm[sl], G[sl])), w
def smask(c, a, b, wmat=None):
    fr = [getf(c, i) for i in np.linspace(a, b, 24).astype(int)]
    if wmat is not None: fr = [cv2.warpAffine(f, wmat, (W, H), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT) for f in fr]
    return cv2.erode((np.stack(fr).std(0).mean(2) < 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); K3 = np.ones((3, 3), np.uint8)
def streak_mask(x, m, th): d = (x - m).mean(2); mk = (d > th).astype(np.uint8); thin = mk & (1 - cv2.morphologyEx(mk, cv2.MORPH_OPEN, K5)); return cv2.dilate(thin, K3).astype(bool)
def nbr(c, i, N, step, half): return [getf(c, min(max(i + k * step, 0), N - 1)) for k in range(-half, half + 1)]
def streak_mask5(x, m, th): d = (x - m).mean(2); mk = (d > th).astype(np.uint8); thin = mk & (1 - cv2.morphologyEx(mk, cv2.MORPH_OPEN, K5)); return cv2.dilate(thin, np.ones((5, 5), np.uint8)).astype(bool)
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8); mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mv); big = np.isin(lab, [k for k in range(1, n) if stats[k, 4] > 120]).astype(np.uint8); return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
def target(c, i, N):
    fr = nbr(c, i, N, 2, 4); x = fr[4]; m9 = np.median(np.stack(fr), 0); P = motion_mask(x, np.median(np.stack(nbr(c, i, N, 6, 3)), 0)); mk = streak_mask5(x, m9, 4 / 255) & ~P; T = x.copy(); T[mk] = m9[mk]
    return np.concatenate(fr[2:7] + [x - m9], 2), T, mk
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
trainX, trainT, trainM = [], [], []
def add(c, N, n):
    for i in np.linspace(8, N - 9, n).astype(int):
        x, T, mk = target(c, int(i), N); trainX.append(x.astype(np.float16)); trainT.append(T.astype(np.float16)); trainM.append(mk.astype(np.uint8))
for k in [k for k in CCTV if k.startswith("cctv_") and "black" not in k and k not in HELD]:
    c = cv2.VideoCapture(str(D / k)); add(c, int(c.get(7)), 35)
for dn in ["cctv_before_rain_1-11_1.mp4", "cctv_before_rain_1-11_2.mp4", "cctv_after_rain_1-11_1.mp4", "cctv_after_rain_1-11_3.mp4"]:
    c = cv2.VideoCapture(str(D / dn)); add(c, int(c.get(7)), 20)
c = cv2.VideoCapture(str(D / "iphone_light rain1.MOV")); add(c, int(c.get(7)), 50)
print("train samples", len(trainX), "streak px frac", float(np.mean([m.mean() for m in trainM])), flush=True)
def gk(sig, k):
    x = torch.arange(k, dtype=torch.float32) - k // 2; g = torch.exp(-x ** 2 / (2 * sig ** 2)); g = g / g.sum(); return (g[:, None] * g[None, :]).to(dev)[None, None].repeat(3, 1, 1, 1)
import torch.nn as nn
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
from mg_net import MG
CK = ROOT / "experiments_v2" / "checkpoints"
ITERS = 21738; SEEDS = [0, 1]
def to5(X):
    # X: (B,18,H,W) with 5 frames in channels 0:15 -> replace residual-to-9-frame-median by residual-to-5-frame-median
    B = X.shape[0]; fr = X[:, :15].reshape(B, 5, 3, X.shape[2], X.shape[3]); m5 = fr.median(1).values
    return torch.cat([X[:, :15], X[:, 6:9] - m5], 1)
def train5(seed):
    torch.manual_seed(seed); m = MG().to(dev); m.train()
    opt = torch.optim.Adam(m.parameters(), 1e-3); rng = np.random.default_rng(seed + 200); it = 0
    while it < ITERS:
        xs, ts, ms = [], [], []
        for _ in range(12):
            j = rng.integers(len(trainX)); yy = rng.integers(0, H - P); xx = rng.integers(0, W - P); fl = rng.random() < 0.5
            cr = lambda a: np.ascontiguousarray((a[yy:yy + P, xx:xx + P][:, ::-1]) if fl else a[yy:yy + P, xx:xx + P])
            xs.append(cr(trainX[j])); ts.append(cr(trainT[j])); ms.append(cr(trainM[j]))
        X = torch.from_numpy(np.stack(xs).astype(np.float32)).permute(0, 3, 1, 2).to(dev); T = torch.from_numpy(np.stack(ts).astype(np.float32)).permute(0, 3, 1, 2).to(dev); Mk = torch.from_numpy(np.stack(ms).astype(np.float32))[:, None].to(dev)
        out = m(to5(X)); loss = ((out - T).abs().mean(1, keepdim=True) * (1 + 20 * Mk)).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); it += 1
        if it % 1000 == 0: print("train5", seed, it, round(float(loss), 4), flush=True)
        if it > 0.7 * ITERS:
            for g in opt.param_groups: g["lr"] = 3e-4
    m.eval(); torch.save({"model_state_dict": m.state_dict()}, CK / f"mg5_s{seed}.pth"); print("trained5", seed, flush=True); return m
def load(p):
    mm = MG().to(dev); mm.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return mm.eval()
MODELS = {"MG9": (load(CK / "student_mg.pth"), False)}
for sd in SEEDS: MODELS[f"MG5_s{sd}"] = (train5(sd), True)
def hf(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
def outs(c, i, prot):
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0); win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
        o = {"in": x}; X5 = to5(X)
        for kk, (mm, five) in MODELS.items(): o[kk] = mm(X5 if five else X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    o["full_m9"] = m9.copy(); o["full_m5"] = np.median(np.stack(fr[2:7]), 0)
    if prot:
        P = motion_mask(x, np.median(np.stack([getf(c, i + j) for j in range(-21, 22, 6)]), 0)); o = {k: (v if k == "in" else np.where(P[..., None], x, v)) for k, v in o.items()}
    nr = cv2.dilate((np.abs(x - m9).mean(2) > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    return o, nr
def scene(c, st, idxs, prot, shmask):
    s = st[sl]; fl = {}; sh = {}
    for i in idxs:
        o1, nr = outs(c, i, prot); o2, _ = outs(c, i + 1, prot)
        for k in o1:
            fl.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s].mean())); sh.setdefault(k, []).append(hf(o1[k], st if shmask == "static" else nr))
    return {k: float(np.mean(v)) for k, v in fl.items()}, {k: float(np.mean(v)) for k, v in sh.items()}
def removed(fin, fdry, f, fd):
    out = {}
    for k in f:
        if k == "in": continue
        S = f[k] / fin; rho = fd[k] / fd["in"]; out[k] = float(1 - (S * fin - rho * fd["in"]) / (fin - fd["in"]))
    return out
R = {}
# 1) natural iPhone (same protocol as Table 3: protection on, static-pixel sharpness, dry reference = 1.3-1.8 s)
c = cv2.VideoCapture(str(D / "iphon_raining.MOV")); fps = c.get(5); st = smask(c, 5 * fps, 10 * fps)
fd, shd = scene(c, smask(c, 1.0 * fps, 2.3 * fps), [int(t * fps) for t in np.linspace(1.3, 1.8, 8)], True, "static")
for bn, rg in {"iphon_mid": range(55, 121, 8), "iphon_heavy": range(125, 241, 12)}.items():
    f, sh = scene(c, st, [int(t * fps) for t in rg], True, "static"); rm = removed(f["in"], fd["in"], f, fd)
    R[bn] = {"removed": rm, "sharp": {k: sh[k] / sh["in"] for k in sh if k != "in"}}; print(bn, R[bn], flush=True)
# 2) artificial rain H01 and J01 (same protocol as Tables 4-5: no protection, non-rain-pixel sharpness)
ND = D / "New rain GT and light medium and heavy"; JD = D / "backyard jungle"
for tag, folder, dry, rains in (("H01", ND, "H01_dry.MOV", ["H01_hose_light.MOV", "H01_hose_medium.MOV", "H01_hose_medium_heavy.MOV", "H01_hose_heavy.MOV"]), ("J01", JD, "GT Jungle.MOV", ["Jungle Light.MOV", "Jungle Medium.MOV", "Jungle Heavy.MOV"])):
    dm, cd, Nd = med(folder / dry); sd0 = smask(cd, 0, Nd - 2); ix = lambda N: list(np.linspace(40, N - 41, 8).astype(int))
    fdry, shdry = scene(cd, sd0, ix(Nd), False, "nonrain")
    for rn in rains:
        rm_, cr, N = med(folder / rn); GT, w = align(rm_, dm); stt = smask(cd, 0, Nd - 2, w)
        f, sh = scene(cr, stt, ix(N), False, "nonrain"); R[rn] = {"removed": removed(f["in"], fdry["in"], f, fdry), "sharp": {k: sh[k] / sh["in"] for k in sh if k != "in"}, "flicker_in": f["in"], "flicker_dry": fdry["in"]}; print(rn, R[rn], flush=True)
json.dump(R, open(OUT / "new_mg5.json", "w"), indent=1); print("DONE")

# 4) detector kept-rate on CCTV shop scenes (as Table 6): MG9, MG5, plain medians, with / without protection
from ultralytics import YOLO
yl = YOLO("yolov8n.pt")
def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.25)[0]; return [(int(b.cls), float(b.conf), b.xyxy[0].cpu().numpy()) for b in r.boxes if int(b.cls) in (0, 1, 2, 3, 5, 7)]
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1); return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
Y = {}
for name in ["cctv_shop1.mp4", "cctv_shop2.mp4", "cctv_shop3.mp4", "cctv_shop_before+rain.mp4", "cctv_shop_after+rain.mp4"]:
    c = cv2.VideoCapture(str(D / name)); N = int(c.get(7)); rows = []
    def gf(i):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(min(max(i, 0), N - 1))); ok, f = c.read(); return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    for i in np.linspace(24, N - 25, 50).astype(int):
        o, _ = outs(c, int(i), False); x = o["in"]; P_ = motion_mask(x, np.median(np.stack([gf(i + j) for j in range(-21, 22, 6)]), 0)); dx = det(x); r = {}
        for k, y in o.items():
            if k == "in": continue
            for tag, yy in ((k, y), (k + "_prot", np.where(P_[..., None], x, y))):
                dy = det(yy); r[tag] = float(np.mean([any(d[0] == e[0] and iou(d[2], e[2]) > 0.5 for e in dy) for d in dx])) if dx else np.nan
        rows.append(r)
    Y[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}; print("YOLO", name, {k: round(v, 3) for k, v in Y[name].items()}, flush=True)
json.dump(Y, open(OUT / "new_mg5_yolo.json", "w"), indent=1); print("DONE2")
