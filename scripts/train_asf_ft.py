"""ASF-Net-ft: fine-tune the ASF-Net backbone (Xue et al., 2025) on OUR clips with OUR supervision,
so that the second recent video derainer in the paper is no longer evaluated zero-shot.

Held identical to MG-Net and to S2VD-ft: the training clips, the masked-temporal-median target
(theta = 2/255, thin-structure test, motion protection), the mask-weighted L1 loss with lambda = 20,
and the 0.4 h budget on the same GPU. Differs: the architecture (flow-aligned TOFlow, 3.09M
parameters, 3 frames in, centre frame out) and the start point (released weights, lr 1e-4).
The SpyNet flow estimator is frozen, as it is a pretrained component and not part of the comparison.

This is not the full ASF-Net training procedure; it answers only how their architecture compares
with the gated blend when both see the same data, target and budget."""
import cv2, numpy as np, json, torch, time, sys, os
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch.nn as nn
import torchvision; torchvision.__version__ = "0.99.0"   # Network.py gates the deformable-conv path on this
dev = "cuda"; BUDGET = 0.4 * 3600; P = 160; BS = 2; NCROP = 2      # P must be a multiple of 32
SRC = Path(__file__).parent / "asf_src"; sys.path.insert(0, str(SRC)); cwd = os.getcwd(); os.chdir(SRC)
from Network import TOFlow
net = TOFlow().to(dev)
sd_ = torch.load(str(SRC / "toflow_models_mine" / "denoising_best.pkl"), map_location=dev)
net.load_state_dict({k.replace("module.", "", 1): v for k, v in sd_.items()}, strict=False)
os.chdir(cwd)
nfr = 0
for n_, p_ in net.named_parameters():
    if "spynet" in n_.lower() or "flow" in n_.lower(): p_.requires_grad = False; nfr += 1
print("ASF params K", sum(p.numel() for p in net.parameters()) / 1e3,
      "| frozen tensors", nfr, "| trainable K", sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e3, flush=True)
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
rng0 = np.random.default_rng(0)
trainX, trainT, trainM = [], [], []     # 3 consecutive frames (RGB), target for the centre, streak mask
def add(c, N, n):
    for i in np.linspace(8, N - 9, n).astype(int):
        i = int(i)
        f9 = nbr(c, i, N, 2, 4); x = f9[4]; m9 = np.median(np.stack(f9), 0)
        Pm = motion_mask(x, np.median(np.stack(nbr(c, i, N, 6, 3)), 0))
        mk = streak_mask5(x, m9, 2 / 255) & ~Pm
        T = x.copy(); T[mk] = m9[mk]
        clip = np.stack([f[:, :, ::-1] for f in nbr(c, i, N, 1, 1)])      # 3 frames, BGR -> RGB
        Tr = T[:, :, ::-1]
        for _ in range(NCROP):
            yy = int(rng0.integers(0, H - P)); xx = int(rng0.integers(0, W - P))
            trainX.append((clip[:, yy:yy + P, xx:xx + P] * 255).astype(np.uint8))
            trainT.append((Tr[yy:yy + P, xx:xx + P] * 255).astype(np.uint8))
            trainM.append(mk[yy:yy + P, xx:xx + P].astype(np.uint8))
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
for k in [k for k in CCTV if k.startswith("cctv_") and "black" not in k and k not in HELD]:
    c = cv2.VideoCapture(str(D / k)); add(c, int(c.get(7)), 35)
for dn in ["cctv_before_rain_1-11_1.mp4", "cctv_before_rain_1-11_2.mp4", "cctv_after_rain_1-11_1.mp4", "cctv_after_rain_1-11_3.mp4"]:
    c = cv2.VideoCapture(str(D / dn)); add(c, int(c.get(7)), 20)
c = cv2.VideoCapture(str(D / "iphone_light rain1.MOV")); add(c, int(c.get(7)), 50)
print("train crops", len(trainX), "streak px frac", float(np.mean([m.mean() for m in trainM])), flush=True)
net.train()
for m_ in net.modules():
    if m_.__class__.__name__.lower().find("spynet") >= 0: m_.eval()
opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], 1e-4)
rng = np.random.default_rng(1); t0 = time.time(); it = 0
while time.time() - t0 < BUDGET:
    j = rng.integers(len(trainX), size=BS)
    X = torch.from_numpy(np.stack([trainX[k] for k in j]).astype(np.float32) / 255).permute(0, 1, 4, 2, 3).to(dev)  # B,T,C,H,W
    T = torch.from_numpy(np.stack([trainT[k] for k in j]).astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)
    Mk = torch.from_numpy(np.stack([trainM[k] for k in j]).astype(np.float32))[:, None].to(dev)
    try:
        out = net(X)
    except Exception as e:
        print("FORWARD FAILED:", type(e).__name__, e, flush=True); raise
    if out.dim() == 5: out = out[:, 1]
    loss = ((out - T).abs().mean(1, keepdim=True) * (1 + 20 * Mk)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0); opt.step(); it += 1
    if it == 1: print("first iteration OK, out shape", tuple(out.shape), flush=True)
    if it % 200 == 0: print("train", it, round(float(loss), 4), "elapsed_min", round((time.time() - t0) / 60, 1), flush=True)
    if time.time() - t0 > 0.7 * BUDGET:
        for g in opt.param_groups: g["lr"] = 3e-5
net.eval(); torch.save(net.state_dict(), ROOT / "experiments_v2" / "checkpoints" / "asf_ft.pth")
print("trained iterations", it, flush=True); print("DONE")
