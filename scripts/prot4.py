import cv2, numpy as np, json, torch, torch.nn as nn, glob
from common import ROOT, OUT
from ultralytics import YOLO
dev = "cuda"; W, H = 640, 360; D = ROOT / "data" / "new_clips"; fn = D / "cctv_rain_1-11_2.mp4"
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
def L(n):
    q = MF().to(dev); q.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / n, map_location=dev)["model_state_dict"]); return q.eval()
from mg_net import MG
def LM():
    q = MG().to(dev); q.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); return q.eval()
m6 = LM(); m7 = L("student_mf7.pth"); yl = YOLO("yolov8n.pt")
K5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)); K3 = np.ones((3, 3), np.uint8)
def smask(x, mm, th): d = (x - mm).mean(2); mk = (d > th).astype(np.uint8); thin = mk & (1 - cv2.morphologyEx(mk, cv2.MORPH_OPEN, K5)); return cv2.dilate(thin, K3).astype(bool)

def prot_mask(x, m9):
    mv = (np.abs(x - m9).mean(2) > 0.08).astype(np.uint8); mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mv); big = np.isin(lab, [k for k in range(1, n) if stats[k, 4] > 300]).astype(np.uint8)
    return cv2.dilate(big, np.ones((15, 15), np.uint8)).astype(bool)
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8); mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mv); big = np.isin(lab, [k for k in range(1, n) if stats[k, 4] > 120]).astype(np.uint8); return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.25)[0]; return [(int(b.cls), float(b.conf), b.xyxy[0].cpu().numpy()) for b in r.boxes if int(b.cls) in (0, 1, 2, 3, 5, 7)]
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1); return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
RES = {}
for name in ["cctv_shop1.mp4", "cctv_shop2.mp4", "cctv_shop3.mp4", "cctv_shop_before+rain.mp4", "cctv_shop_after+rain.mp4"]:
    c = cv2.VideoCapture(str(D / name)); N = int(c.get(7))
    def gf(i):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(min(max(i, 0), N - 1))); ok, f = c.read(); return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    rows = []
    for i in np.linspace(24, N - 25, 50).astype(int):
        fr = [gf(i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0); mk = smask(x, m9, 10 / 255); A = x.copy(); A[mk] = m9[mk]
        win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad(): X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]; S6 = m6(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); S7 = m7(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        m15_ = np.median(np.stack([gf(i + j) for j in range(-21, 22, 6)]), 0); P = motion_mask(x, m15_); S6p = np.where(P[..., None], x, S6); S7p = np.where(P[..., None], x, S7)
        dx = det(x); outs = {"S6": S6, "S6p": S6p, "M9": m9.copy(), "M9p": np.where(P[..., None], x, m9)}; r = dict(n_in=len(dx))
        bm = np.zeros((H, W), bool)
        for d in dx: x1, y1, x2, y2 = [int(v) for v in d[2]]; bm[max(y1, 0):y2, max(x1, 0):x2] = True
        fr15 = [gf(i + j) for j in range(-21, 22, 6)]; m15 = np.median(np.stack(fr15), 0); d0 = smask(x, m15, 12 / 255).mean() + 1e-9
        for k, y in outs.items():
            dy = det(y); r["n_" + k] = len(dy); r["kept_" + k] = float(np.mean([any(d[0] == e[0] and iou(d[2], e[2]) > 0.5 for e in dy) for d in dx])) if dx else np.nan
            ch = np.abs(y - x).mean(2); r["chgbox_" + k] = float(ch[bm].mean()) if bm.any() else np.nan; r["chgout_" + k] = float(ch[~bm].mean()); r["dens_" + k] = float(smask(np.clip(y, 0, 1), m15, 12 / 255).mean() / d0)
        r["prot_frac"] = float(P.mean()); rows.append(r)
    RES[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}; print(name, {k: round(v, 4) for k, v in RES[name].items()}, flush=True)
json.dump(RES, open(OUT / "prot4.json", "w"), indent=1); print("DONE")
