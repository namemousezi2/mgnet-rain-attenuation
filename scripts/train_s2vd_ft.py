"""S2VD-ft: fine-tune the S2VD derain backbone (Yue et al., 2021) on OUR clips with OUR
supervision, so that the comparison in the paper is no longer zero-shot.

What is held identical to MG-Net: the training clips, the masked-temporal-median target
(theta = 2/255, thin-structure test, motion protection), the mask-weighted L1 loss with
lambda = 20, and the 0.4 h training budget on the same GPU.
What differs: the architecture (S2VD's 3D-conv DerainNet, 525K parameters, 7 consecutive
frames in and out) and the starting point (the authors' released weights, fine-tuned at
lr 1e-4 rather than trained from scratch).

This is NOT the full S2VD training procedure, which is semi-supervised and uses its own
generator and losses. It answers one question only: given the same data, the same target and
the same budget, how does their backbone compare with the gated blend of MG-Net."""
import cv2, numpy as np, json, torch, time, sys
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch.nn as nn
dev = "cuda"; BUDGET = 0.4 * 3600; P = 160; BS = 4; NCROP = 2
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
from networks.derain_net import DerainNet
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
trainX, trainT, trainM = [], [], []     # 7-frame clips, target for the centre frame, streak mask
def add(c, N, n):
    for i in np.linspace(8, N - 9, n).astype(int):
        i = int(i)
        f9 = nbr(c, i, N, 2, 4); x = f9[4]; m9 = np.median(np.stack(f9), 0)
        Pm = motion_mask(x, np.median(np.stack(nbr(c, i, N, 6, 3)), 0))
        mk = streak_mask5(x, m9, 2 / 255) & ~Pm
        T = x.copy(); T[mk] = m9[mk]
        clip = np.stack(nbr(c, i, N, 1, 3))          # 7 consecutive frames, centre = x
        for _ in range(NCROP):
            yy = int(rng0.integers(0, H - P)); xx = int(rng0.integers(0, W - P))
            trainX.append((clip[:, yy:yy + P, xx:xx + P] * 255).astype(np.uint8))
            trainT.append((T[yy:yy + P, xx:xx + P] * 255).astype(np.uint8))
            trainM.append(mk[yy:yy + P, xx:xx + P].astype(np.uint8))
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
for k in [k for k in CCTV if k.startswith("cctv_") and "black" not in k and k not in HELD]:
    c = cv2.VideoCapture(str(D / k)); add(c, int(c.get(7)), 35)
for dn in ["cctv_before_rain_1-11_1.mp4", "cctv_before_rain_1-11_2.mp4", "cctv_after_rain_1-11_1.mp4", "cctv_after_rain_1-11_3.mp4"]:
    c = cv2.VideoCapture(str(D / dn)); add(c, int(c.get(7)), 20)
c = cv2.VideoCapture(str(D / "iphone_light rain1.MOV")); add(c, int(c.get(7)), 50)
print("train crops", len(trainX), "streak px frac", float(np.mean([m.mean() for m in trainM])), flush=True)
net = DerainNet(n_features=32, n_resblocks=8).to(dev)
net.load_state_dict(torch.load(str(S2 / "model_states" / "derainer_rho05.pt"), map_location=dev))
print("S2VD params K", sum(p.numel() for p in net.parameters()) / 1e3, flush=True)
net.train()
opt = torch.optim.Adam(net.parameters(), 1e-4); rng = np.random.default_rng(1); t0 = time.time(); it = 0
while time.time() - t0 < BUDGET:
    j = rng.integers(len(trainX), size=BS)
    X = torch.from_numpy(np.stack([trainX[k] for k in j]).astype(np.float32) / 255).permute(0, 4, 1, 2, 3).to(dev)   # B,3,7,H,W
    T = torch.from_numpy(np.stack([trainT[k] for k in j]).astype(np.float32) / 255).permute(0, 3, 1, 2).to(dev)      # B,3,H,W
    Mk = torch.from_numpy(np.stack([trainM[k] for k in j]).astype(np.float32))[:, None].to(dev)
    out = net(X)[:, :, 3]                                   # centre frame of the 7
    loss = ((out - T).abs().mean(1, keepdim=True) * (1 + 20 * Mk)).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); it += 1
    if it % 200 == 0: print("train", it, round(float(loss), 4), "elapsed_min", round((time.time() - t0) / 60, 1), flush=True)
    if time.time() - t0 > 0.7 * BUDGET:
        for g in opt.param_groups: g["lr"] = 3e-5
net.eval(); torch.save(net.state_dict(), ROOT / "experiments_v2" / "checkpoints" / "s2vd_ft.pth")
print("trained iterations", it, flush=True); print("DONE")
