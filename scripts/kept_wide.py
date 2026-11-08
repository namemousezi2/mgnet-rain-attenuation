"""Detector retention on a wider set of scenes: the five shop scenes plus the three
held-out street CCTV clips and the two iPhone road clips."""
import json, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"; NS = [3, 5, 9, 15, 21]
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
NET = {"MG": loadmg(ck / "student_mg.pth")}
for t, f in (("MGs_s1", "student_mg_strong.pth"), ("MGs_s2", "student_mg_strong_s2.pth"), ("MGs_s3", "student_mg_strong_s3.pth")):
    if (ck / f).exists(): NET[t] = loadmg(ck / f)
import torch.nn as nn
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
METH = ["med%d" % N for N in NS] + list(NET) + ["mf7"]
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8)
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st_, _ = cv2.connectedComponentsWithStats(mv)
    big = np.isin(lab, [k for k in range(1, n) if st_[k, 4] > 120]).astype(np.uint8)
    return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
CA = {}
def gf(c, cid, i):
    k = (cid, int(i))
    if k not in CA:
        if len(CA) > 400: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]
def outs(c, cid, i):
    fr = [gf(c, cid, i + j) for j in range(-8, 9, 2)]; x = fr[4]; o = {"in": x}
    for N in NS:
        h = (N - 1) // 2; o["med%d" % N] = np.median(np.stack([gf(c, cid, i + j) for j in range(-2 * h, 2 * h + 1, 2)]), 0)
    win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
        for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf7"] = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    o["_P"] = motion_mask(x, o["med15"]); return o
from ultralytics import YOLO
yl = YOLO("yolov8n.pt")
def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.25)[0]
    return [(int(b.cls), float(b.conf), b.xyxy[0].cpu().numpy()) for b in r.boxes if int(b.cls) in (0, 1, 2, 3, 5, 7)]
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
PL = ROOT / "data" / "new_place_video_rain"
SCENES = [("shop1", D / "cctv_shop1.mp4"), ("shop2", D / "cctv_shop2.mp4"), ("shop3", D / "cctv_shop3.mp4"),
          ("shop_before", D / "cctv_shop_before+rain.mp4"), ("shop_after", D / "cctv_shop_after+rain.mp4"),
          ("street_heavy", D / "cctv_heavy_rain_1-11_1.mp4"), ("street_medium", D / "cctv_medium_rain_1-11_4.mp4"),
          ("street_rain", D / "cctv_rain_1-11_3.mp4"),
          ("front_road", PL / "front_road_rain.MOV"), ("back_road", PL / "back_road_rain.MOV")]
Y = {}; NDET = {}
for name, p in SCENES:
    if not p.exists(): print("missing", name, flush=True); continue
    c = cv2.VideoCapture(str(p)); N = int(c.get(7)); rows = []; nd = 0
    for i in np.linspace(26, N - 27, 30).astype(int):
        o = outs(c, name, int(i)); x = o["in"]; P_ = o["_P"]; dx = det(x); nd += len(dx)
        if not dx: continue
        r = {}
        for k in METH:
            for tag, yy in ((k, o[k]), (k + "_prot", np.where(P_[..., None], x, o[k]))):
                r[tag] = float(np.mean([any(d[0] == e[0] and iou(d[2], e[2]) > 0.5 for e in det(yy)) for d in dx]))
        rows.append(r)
    CA.clear(); NDET[name] = nd
    if len(rows) < 5: print("skip", name, "input detections", nd, flush=True); continue
    Y[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("%-14s n_frames_used %2d  input_dets %4d  " % (name, len(rows), nd), {k: round(Y[name][k], 3) for k in METH}, flush=True)
json.dump({"Y": Y, "n_det": NDET}, open(OUT / "kept_wide.json", "w"), indent=1)
print("\n=== kept-rate over %d scenes (bare | with protection) ===" % len(Y))
for k in METH:
    b = np.mean([Y[n][k] for n in Y]); pr = np.mean([Y[n][k + "_prot"] for n in Y])
    print("  %-8s %.3f | %.3f" % (k, b, pr))
s = [k for k in NET if k.startswith("MGs_")]
if len(s) > 1:
    v = [np.mean([Y[n][k] for n in Y]) for k in s]; vp = [np.mean([Y[n][k + "_prot"] for n in Y]) for k in s]
    print("  MGs seeds: %.3f +- %.3f | %.3f +- %.3f" % (np.mean(v), np.std(v), np.mean(vp), np.std(vp)))
print("DONE")
