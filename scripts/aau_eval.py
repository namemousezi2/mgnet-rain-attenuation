"""AAU RainSnow (Bahnsen & Moeslund 2019): public traffic-surveillance rain/snow video
with per-instance ground truth. Does rain attenuation help or hurt a detector?
Metric: AP50 and recall of YOLOv8n against the dataset's own boxes, not our kept-rate."""
import json, cv2, numpy as np, re, sys
from pathlib import Path
from common import ROOT, OUT
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
NFRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 25
AAU = ROOT / "data" / "new_clips" / "AAU"
NS = [3, 5, 9, 15, 21]
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
NET = {"MG": loadmg(ck / "student_mg.pth"), "MGs": loadmg(ck / "student_mg_strong.pth")}
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
METH = ["med%d" % N for N in NS] + ["MG", "MGs", "mf7"]
ALL = ["in"] + METH + [m + "_prot" for m in METH]
ann = json.load(open(AAU / "aauRainSnow-rgb.json"))
ID2Y = {1: 0, 2: 1, 3: 2, 4: 3, 6: 5, 8: 7}          # COCO json id -> YOLOv8 class index
IMG = {im["id"]: im for im in ann["images"]}
GT = {}
for a in ann["annotations"]:
    if a["category_id"] in ID2Y and not a.get("iscrowd"):
        x, y, w, h = a["bbox"]; GT.setdefault(a["image_id"], []).append((ID2Y[a["category_id"]], [x, y, x + w, y + h]))
SEQ = {}
for im in ann["images"]:
    SEQ.setdefault("/".join(im["file_name"].split("/")[:2]), []).append(im)
probe = json.load(open(OUT / "aau_probe.json"))
USE = sorted(k for k, v in probe.items() if (v["drift"] is not None and v["drift"] == v["drift"]) and v["drift"] < 1.0 and v["bright"] > 0.25)
print("sequences used:", len(USE), flush=True)
from ultralytics import YOLO
yl = YOLO("yolov8n.pt")
def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.05)[0]
    return [(int(b.cls), float(b.conf), b.xyxy[0].cpu().numpy().tolist()) for b in r.boxes if int(b.cls) in ID2Y.values()]
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8)
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mv)
    big = np.isin(lab, [k for k in range(1, n) if st[k, 4] > 120]).astype(np.uint8)
    return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
DETS = {k: [] for k in ALL}; NGT = {c: 0 for c in ID2Y.values()}
for seq in USE:
    loc, name = seq.split("/")
    cap = cv2.VideoCapture(str(AAU / seq / "cam1.mkv")); N = int(cap.get(7))
    mpath = AAU / loc / (name + "-mask.png"); roi = None
    if mpath.exists():
        mm = cv2.imread(str(mpath), 0)
        if mm is not None: roi = mm > 127
    ims = sorted(SEQ[seq], key=lambda im: im["file_name"])
    pick = [ims[i] for i in np.linspace(0, len(ims) - 1, min(NFRAME, len(ims))).astype(int)]
    cache = {}
    def gf(i):
        i = int(min(max(i, 0), N - 1))
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            cache[i] = f.astype(np.float32) / 255 if ok else None
        return cache[i]
    inside = lambda b: roi is None or roi[int(min(max((b[1] + b[3]) / 2, 0), roi.shape[0] - 1)), int(min(max((b[0] + b[2]) / 2, 0), roi.shape[1] - 1))]
    used = 0
    for im in pick:
        fi = int(re.search(r"cam1-(\d+)\.png", im["file_name"]).group(1))
        g = [(c, b) for c, b in GT.get(im["id"], []) if inside(b)]
        if not g: continue
        fr = [gf(fi + j) for j in range(-8, 9, 2)]
        if any(f is None for f in fr): continue
        x = fr[4]; o = {"in": x}
        for Nw in NS:
            h = (Nw - 1) // 2; w_ = [gf(fi + j) for j in range(-2 * h, 2 * h + 1, 2)]
            if any(f is None for f in w_): w_ = fr
            o["med%d" % Nw] = np.median(np.stack(w_), 0)
        win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
        with torch.no_grad():
            X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
            for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            o["mf7"] = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        P = motion_mask(x, o["med15"])
        for k in METH: o[k + "_prot"] = np.where(P[..., None], x, o[k])
        key = (seq, im["id"])
        for c, b in g: NGT[c] += 1
        for k in ALL:
            for cls, sc, b in det(o[k]):
                if inside(b): DETS[k].append((key, cls, sc, b))
        used += 1
    cache.clear(); print("  %-28s %3d frames used" % (seq, used), flush=True)
GTBY = {}
for seq in USE:
    for im in SEQ[seq]:
        loc, name = seq.split("/")
        GTBY[(seq, im["id"])] = [(c, b) for c, b in GT.get(im["id"], [])]
def ap50(dets):
    aps = {}
    for c in sorted(NGT):
        if NGT[c] == 0: continue
        d = sorted([x for x in dets if x[1] == c], key=lambda z: -z[2])
        taken = {}; tp = []; fp = []
        for key, cls, sc, b in d:
            gts = [(i, gb) for i, (gc, gb) in enumerate(GTBY.get(key, [])) if gc == c and (key, i) not in taken]
            best, bi = 0, None
            for i, gb in gts:
                v = iou(b, gb)
                if v > best: best, bi = v, i
            if best >= 0.5: taken[(key, bi)] = 1; tp.append(1); fp.append(0)
            else: tp.append(0); fp.append(1)
        if not tp: aps[c] = 0.0; continue
        tp = np.cumsum(tp); fp = np.cumsum(fp)
        rec = tp / NGT[c]; prec = tp / np.maximum(tp + fp, 1e-9)
        mrec = np.concatenate([[0], rec, [rec[-1]]]); mpre = np.concatenate([[0], prec, [0]])
        for i in range(len(mpre) - 2, -1, -1): mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps[c] = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
    return aps
NAME = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
print("\nground-truth instances:", {NAME[c]: n for c, n in NGT.items() if n})
res = {}
for k in ALL:
    a = ap50(DETS[k]); m = float(np.mean(list(a.values()))) if a else 0.0
    rec = len({(key, 1) for key, cls, sc, b in DETS[k] if sc > 0.25})
    res[k] = {"mAP50": m, "per_class": {NAME[c]: round(v, 4) for c, v in a.items()}, "n_det": len(DETS[k])}
    print("%-12s mAP50 %.4f   %s" % (k, m, {NAME[c]: round(v, 3) for c, v in a.items()}), flush=True)
json.dump({"res": res, "n_gt": {NAME[c]: n for c, n in NGT.items()}, "sequences": USE}, open(OUT / "aau_eval.json", "w"), indent=1)
b = res["in"]["mAP50"]
print("\n=== mAP50 relative to the rainy input (%.4f) ===" % b)
for k in METH: print("  %-10s bare %+6.1f%%   with protection %+6.1f%%" % (k, 100 * (res[k]["mAP50"] / b - 1), 100 * (res[k + "_prot"]["mAP50"] / b - 1)))
print("DONE")
