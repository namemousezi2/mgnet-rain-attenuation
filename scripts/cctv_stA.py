"""Distil the sharpness-preserving masked-temporal-median (A) into a single-frame student: target T = x except thin bright streak pixels -> 9-frame median.
Train: CCTV clips (minus 3 held-out) + iPhone light rain1 (+ dry frames).  Test (never trained): 3 CCTV clips, iphon_raining mid/heavy, iPhone light rain2.
Metrics (independent 15-frame/stride-3 streak detector, same median for in/out): streak density ratio out/in, bg_change (|out-x| outside streak mask), static sharpness ratio, static PSNR gain."""
import cv2, numpy as np, json, torch, time, copy, glob, os
import torch.nn.functional as F
from common import ROOT, OUT
from order_synth import infer as sinfer
from student_net import load_student
import baselines_nv as BN
dev = BN.dev; W, H = 640, 360; D = ROOT / "data" / "new_clips"; BUDGET = 0.25 * 3600; P = 160; sl = (slice(16, H - 16), slice(24, W - 24))
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
def target(c, i, N):
    fr = nbr(c, i, N, 2, 4); x = fr[4]; m9 = np.median(np.stack(fr), 0); mk = streak_mask(x, m9, 10 / 255); T = x.copy(); T[mk] = m9[mk]; return x, T, mk
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
trainX, trainT, trainM = [], [], []
def add(c, N, n):
    for i in np.linspace(8, N - 9, n).astype(int):
        x, T, mk = target(c, int(i), N); trainX.append(x); trainT.append(T); trainM.append(mk.astype(np.float32))
for k in [k for k in CCTV if k.startswith("cctv_") and "black" not in k and k not in HELD]:
    c = cv2.VideoCapture(str(D / k)); add(c, int(c.get(7)), 60)
for dn in ["cctv_before_rain_1-11_1.mp4", "cctv_before_rain_1-11_2.mp4", "cctv_after_rain_1-11_1.mp4", "cctv_after_rain_1-11_3.mp4"]:
    c = cv2.VideoCapture(str(D / dn)); add(c, int(c.get(7)), 20)
c = cv2.VideoCapture(str(D / "iphone_light rain1.MOV")); add(c, int(c.get(7)), 80)
print("train samples", len(trainX), "streak px frac", float(np.mean([m.mean() for m in trainM])), flush=True)
def gk(sig, k):
    x = torch.arange(k, dtype=torch.float32) - k // 2; g = torch.exp(-x ** 2 / (2 * sig ** 2)); g = g / g.sum(); return (g[:, None] * g[None, :]).to(dev)[None, None].repeat(3, 1, 1, 1)
base, _ = load_student("student_xb"); base = base.to(dev).eval(); m = copy.deepcopy(base).train(); opt = torch.optim.Adam(m.parameters(), 1e-4); rng = np.random.default_rng(1); t0 = time.time(); it = 0
while time.time() - t0 < BUDGET:
    xs, ts, ms = [], [], []
    for _ in range(8):
        j = rng.integers(len(trainX)); yy = rng.integers(0, H - P); xx = rng.integers(0, W - P); fl = rng.random() < 0.5
        cr = lambda a: np.ascontiguousarray((a[yy:yy + P, xx:xx + P][:, ::-1]) if fl else a[yy:yy + P, xx:xx + P])
        xs.append(cr(trainX[j])); ts.append(cr(trainT[j])); ms.append(cr(trainM[j]))
    X = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).to(dev); T = torch.from_numpy(np.stack(ts)).permute(0, 3, 1, 2).to(dev); Mk = torch.from_numpy(np.stack(ms))[:, None].to(dev)
    out = m(X); loss = ((out - T).abs().mean(1, keepdim=True) * (1 + 6 * Mk)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); it += 1
    if it % 300 == 0: print("train", it, round(float(loss), 4), flush=True)
m.eval(); torch.save({"cfg": (64, 8), "model_state_dict": m.state_dict()}, ROOT / "experiments_v2" / "checkpoints" / "student_stA.pth"); print("trained", it, flush=True)
def shp(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
def psnr(a, g, mk): return float(10 * np.log10(1 / (np.mean(((np.clip(a, 0, 1) - g) ** 2)[sl][mk[sl]]) + 1e-12)))
R = {}
def run(tag, c, idxs, N, g, st):
    rows = []
    for i in idxs:
        fr = nbr(c, i, N, 3, 7); x = fr[7]; m15 = np.median(np.stack(fr), 0); sm = streak_mask(x, m15, 12 / 255)
        with torch.no_grad(): yb = np.clip(sinfer(base, 2, x), 0, 1); ys = np.clip(sinfer(m, 2, x), 0, 1)
        T = x.copy(); fr9 = nbr(c, i, N, 2, 4); m9 = np.median(np.stack(fr9), 0); mk9 = streak_mask(x, m9, 10 / 255); T[mk9] = m9[mk9]
        dens = lambda y: float(streak_mask(y, m15, 12 / 255)[sl].mean()); d0 = dens(x) + 1e-9; nm = ~cv2.dilate(sm.astype(np.uint8), K5).astype(bool)
        row = {}
        for k, y in {"xb": yb, "stA": ys, "A_target": T}.items():
            row["dens_" + k] = dens(y) / d0; row["bg_" + k] = float(np.abs(y - x).mean(2)[sl][nm[sl]].mean()); row["sh_" + k] = shp(y, st) / shp(x, st); row["g_" + k] = psnr(y, g, st) - psnr(x, g, st)
        rows.append(row)
    R[tag] = dict(static_frac=float(st[sl].mean()), **{k: float(np.mean([r[k] for r in rows])) for k in rows[0]}); print(tag, {k: round(v, 3) for k, v in R[tag].items()}, flush=True)
c = cv2.VideoCapture(str(D / "iphon_raining.MOV")); fps = c.get(5); N = int(c.get(7)); gt = np.median(np.stack([getf(c, i) for i in np.linspace(5 * fps, 10 * fps, 80).astype(int)]), 0); st = smask(c, 5 * fps, 10 * fps)
for bn, (a, b) in {"iphon_mid": (55, 120), "iphon_heavy": (125, 240)}.items(): run(bn, c, [int(t * fps) for t in range(a, b + 1, 10)], N, gt, st)
for k in HELD:
    dn = CCTV[k]["best_dry"]; rm, cr, N = med(D / k); gtm, cd, Nd = med(D / dn); GT, w = align(rm, gtm); st = smask(cd, 0, Nd - 2, w); run(k[:22], cr, list(np.linspace(12, N - 13, 12).astype(int)), N, GT, st)
rn, an = "iphone_light rain2", "iphone_after_rain2"; pr = cv2.VideoCapture(str(D / (rn + ".MOV"))); N = int(pr.get(7)); rm = med(D / (rn + ".MOV"))[0]; gtm, pa, Na = med(D / (an + ".MOV")); GT, w = align(rm, gtm); run(rn, pr, list(np.linspace(12, N - 13, 12).astype(int)), N, GT, smask(pa, 0, Na - 2, w))
json.dump(R, open(OUT / "cctv_stA.json", "w"), indent=1); print("DONE")
