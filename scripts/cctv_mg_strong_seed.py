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
    fr = nbr(c, i, N, 2, 4); x = fr[4]; m9 = np.median(np.stack(fr), 0); P = motion_mask(x, np.median(np.stack(nbr(c, i, N, 6, 3)), 0)); mk = streak_mask5(x, m9, 2 / 255) & ~P; T = x.copy(); T[mk] = m9[mk]
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
import sys; SEED = int(sys.argv[1])
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
from mg_net import MG
base, _ = load_student("student_xb"); base = base.to(dev).eval()
teacher = MF().to(dev); teacher.load_state_dict(torch.load(ROOT / 'experiments_v2' / 'checkpoints' / 'student_mf7.pth', map_location=dev)['model_state_dict']); teacher.eval()
import sys; SEED = int(sys.argv[1]); torch.manual_seed(SEED); m = MG().to(dev); m.train(); print("MG params K", sum(p.numel() for p in m.parameters()) / 1e3, flush=True)
opt = torch.optim.Adam(m.parameters(), 1e-3); rng = np.random.default_rng(SEED); t0 = time.time(); it = 0
while time.time() - t0 < BUDGET:
    xs, ts, ms = [], [], []
    for _ in range(12):
        j = rng.integers(len(trainX)); yy = rng.integers(0, H - P); xx = rng.integers(0, W - P); fl = rng.random() < 0.5
        cr = lambda a: np.ascontiguousarray((a[yy:yy + P, xx:xx + P][:, ::-1]) if fl else a[yy:yy + P, xx:xx + P])
        xs.append(cr(trainX[j])); ts.append(cr(trainT[j])); ms.append(cr(trainM[j]))
    X = torch.from_numpy(np.stack(xs).astype(np.float32)).permute(0, 3, 1, 2).to(dev); T = torch.from_numpy(np.stack(ts).astype(np.float32)).permute(0, 3, 1, 2).to(dev); Mk = torch.from_numpy(np.stack(ms).astype(np.float32))[:, None].to(dev)
    with torch.no_grad(): Tt = teacher(X)
    out = m(X); loss = ((out - T).abs().mean(1, keepdim=True) * (1 + 20 * Mk)).mean() + 0.0 * (out - Tt).abs().mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); it += 1
    if it == int(0.7 * 1e9): pass
    if it % 300 == 0: print("train", it, round(float(loss), 4), flush=True)
    if time.time() - t0 > 0.7 * BUDGET: 
        for g in opt.param_groups: g["lr"] = 3e-4
m.eval(); torch.save({"model_state_dict": m.state_dict()}, ROOT / "experiments_v2" / "checkpoints" / f"student_mg_strong_s{SEED}.pth"); print("trained", it, flush=True)
