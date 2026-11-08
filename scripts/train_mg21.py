"""MG-21: identical to the strong-target MG-Net except that the blend operand and the
training target use a 21-frame temporal median (stride 2) instead of a 9-frame one.
Nothing else changes: same thin-structure test (theta = 2/255), same motion protection of
the target, same 18-channel input layout, same architecture, same schedule, same budget.
The point of the experiment: the output is x + alpha*(m - x) + 0.05*tanh(r), so the flicker
reduction of MG-Net is structurally capped by the median it blends with. Swapping m9 for m21
raises that cap from about 66% to about 77%; the question is what it costs in detection."""
import cv2, numpy as np, json, torch, time
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch.nn as nn
from mg_net import MG
dev = "cuda"; BUDGET = 0.4 * 3600; P = 160
CCTV = json.load(open(OUT / "cctv_pairs_check.json"))["pairs"]
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
def nbr(c, i, N, step, half): return [getf(c, min(max(i + k * step, 0), N - 1)) for k in range(-half, half + 1)]
def streak_mask5(x, m, th):
    d = (x - m).mean(2); mk = (d > th).astype(np.uint8)
    thin = mk & (1 - cv2.morphologyEx(mk, cv2.MORPH_OPEN, K5))
    return cv2.dilate(thin, np.ones((5, 5), np.uint8)).astype(bool)
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8)
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st_, _ = cv2.connectedComponentsWithStats(mv)
    big = np.isin(lab, [k for k in range(1, n) if st_[k, 4] > 120]).astype(np.uint8)
    return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
HALF = 10                      # 21 frames at stride 2  ->  offsets -20 .. +20
def target(c, i, N):
    fr = nbr(c, i, N, 2, HALF); x = fr[HALF]; m = np.median(np.stack(fr), 0)
    Pm = motion_mask(x, np.median(np.stack(nbr(c, i, N, 6, 3)), 0))
    mk = streak_mask5(x, m, 2 / 255) & ~Pm
    T = x.copy(); T[mk] = m[mk]
    stack = np.concatenate(fr[HALF - 2:HALF + 3] + [x - m], 2)   # 5 frames at -4,-2,0,+2,+4 plus (x - m21)
    return stack, T, mk
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
trainX, trainT, trainM = [], [], []
def add(c, N, n):
    for i in np.linspace(2 * HALF, N - 2 * HALF - 1, n).astype(int):
        x, T, mk = target(c, int(i), N)
        trainX.append(x.astype(np.float16)); trainT.append(T.astype(np.float16)); trainM.append(mk.astype(np.uint8))
for k in [k for k in CCTV if k.startswith("cctv_") and "black" not in k and k not in HELD]:
    c = cv2.VideoCapture(str(D / k)); add(c, int(c.get(7)), 35)
for dn in ["cctv_before_rain_1-11_1.mp4", "cctv_before_rain_1-11_2.mp4", "cctv_after_rain_1-11_1.mp4", "cctv_after_rain_1-11_3.mp4"]:
    c = cv2.VideoCapture(str(D / dn)); add(c, int(c.get(7)), 20)
c = cv2.VideoCapture(str(D / "iphone_light rain1.MOV")); add(c, int(c.get(7)), 50)
print("train samples", len(trainX), "streak px frac", float(np.mean([m.mean() for m in trainM])), flush=True)
m = MG().to(dev); m.train(); print("MG params K", sum(p.numel() for p in m.parameters()) / 1e3, flush=True)
opt = torch.optim.Adam(m.parameters(), 1e-3); rng = np.random.default_rng(1); t0 = time.time(); it = 0
while time.time() - t0 < BUDGET:
    xs, ts, ms = [], [], []
    for _ in range(12):
        j = rng.integers(len(trainX)); yy = rng.integers(0, H - P); xx = rng.integers(0, W - P); fl = rng.random() < 0.5
        cr = lambda a: np.ascontiguousarray((a[yy:yy + P, xx:xx + P][:, ::-1]) if fl else a[yy:yy + P, xx:xx + P])
        xs.append(cr(trainX[j])); ts.append(cr(trainT[j])); ms.append(cr(trainM[j]))
    X = torch.from_numpy(np.stack(xs).astype(np.float32)).permute(0, 3, 1, 2).to(dev)
    T = torch.from_numpy(np.stack(ts).astype(np.float32)).permute(0, 3, 1, 2).to(dev)
    Mk = torch.from_numpy(np.stack(ms).astype(np.float32))[:, None].to(dev)
    out = m(X); loss = ((out - T).abs().mean(1, keepdim=True) * (1 + 20 * Mk)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); it += 1
    if it % 300 == 0: print("train", it, round(float(loss), 4), flush=True)
    if time.time() - t0 > 0.7 * BUDGET:
        for g in opt.param_groups: g["lr"] = 3e-4
m.eval(); torch.save({"model_state_dict": m.state_dict()}, ROOT / "experiments_v2" / "checkpoints" / "student_mg21.pth")
print("trained iterations", it, flush=True); print("DONE")
