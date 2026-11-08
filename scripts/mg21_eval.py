"""Place MG-21 on the flicker-detection plane of Table 9, under exactly the protocol used there.
Flicker and sharpness: the six natural-rain clips of Table 8, twenty windows per clip, the same
windows for every method, bare output, no motion protection. Kept-rate: the five CCTV shop scenes
of Table 6, reported bare and with the optional protection step. Compared methods are kept to the
four that decide the question: the 9- and 21-frame medians, MG-strong (seed 1) and MG-21."""
import json, cv2, numpy as np, sys
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RNG = np.random.RandomState(0)
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
MGS = loadmg(ck / "student_mg_strong.pth")
p21 = ck / "student_mg21.pth"
if not p21.exists(): print("MISSING", p21, "- run run_mg21.bat first"); sys.exit(1)
MG21 = loadmg(p21)
METH = ["med9", "med21", "MGs", "MG21"]
def shp(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
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
        if len(CA) > 600: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]
def outs(c, cid, i):
    f9 = [gf(c, cid, i + j) for j in range(-8, 9, 2)]
    f21 = [gf(c, cid, i + j) for j in range(-20, 21, 2)]
    x = f9[4]; o = {"in": x}
    o["med9"] = np.median(np.stack(f9), 0); o["med21"] = np.median(np.stack(f21), 0)
    m15 = np.median(np.stack([gf(c, cid, i + j) for j in range(-14, 15, 2)]), 0)
    w9 = torch.from_numpy(np.stack(f9)).permute(0, 3, 1, 2).to(dev)
    md9 = torch.from_numpy(o["med9"]).permute(2, 0, 1).to(dev)
    md21 = torch.from_numpy(o["med21"]).permute(2, 0, 1).to(dev)
    xt = w9[4]
    with torch.no_grad():
        X9 = torch.cat([w9[2], w9[3], w9[4], w9[5], w9[6], xt - md9], 0)[None]
        X21 = torch.cat([w9[2], w9[3], w9[4], w9[5], w9[6], xt - md21], 0)[None]
        o["MGs"] = MGS(X9)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["MG21"] = MG21(X21)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    o["_P"] = motion_mask(x, m15)
    return o
def drift(c, cid, i, n=16):
    a = cv2.cvtColor(gf(c, cid, i), cv2.COLOR_BGR2GRAY); b = cv2.cvtColor(gf(c, cid, i + n), cv2.COLOR_BGR2GRAY)
    (dx, dy), _ = cv2.phaseCorrelate(a, b); return float(np.hypot(dx, dy))
PL = ROOT / "data" / "new_place_video_rain"; NC = ROOT / "data" / "new_clips"
CLIPS = [("front_road", PL / "front_road_rain.MOV"), ("back_road", PL / "back_road_rain.MOV"),
         ("house", PL / "already_rain.MOV"),
         ("p4_1931", NC / "3 footage raining new place" / "IMG_1931.MOV"),
         ("p4_1933", NC / "3 footage raining new place" / "IMG_1933.MOV"),
         ("p4_1934", NC / "3 footage raining new place" / "IMG_1934.MOV")]
R = {}
for cid, p in CLIPS:
    if not p.exists(): print("missing", cid, flush=True); continue
    c = cv2.VideoCapture(str(p)); fps = c.get(5); N = int(c.get(7)); sec = N / fps
    ok = [i for i in [int(t * fps) for t in np.linspace(10, sec - 10, 4 * NWIN)] if drift(c, cid, i - 8) < 1.0][:NWIN]
    if len(ok) < 3: print("skip (camera not static)", cid, flush=True); continue
    st = smask(c, ok[0] / fps - 3, ok[0] / fps + 3); s_ = st[sl]; per = {}
    for i in ok:
        o1, o2 = outs(c, cid, i), outs(c, cid, i + 1)
        for k in [k for k in o1 if k != "_P"]:
            per.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s_].mean()))
            per.setdefault("sh_" + k, []).append(shp(o1[k], st) / (shp(o1["in"], st) + 1e-9))
    CA.clear(); R[cid] = {k: [float(x) for x in v] for k, v in per.items()}
    print("done", cid, "n windows", len(ok), flush=True)
def boot(v, f, n=2000):
    v = np.asarray(v); idx = RNG.randint(0, len(v), (n, len(v)))
    s = np.sort([f(v[j]) for j in idx]); return float(s[int(0.025 * n)]), float(s[int(0.975 * n)])
print("\n=== flicker removed %% (raw, no dry correction) over %d clips ===" % len(R))
FL = {}
for k in METH:
    vals = np.concatenate([np.array(R[c][k]) / np.mean(R[c]["in"]) for c in R])
    FL[k] = 100 * (1 - vals.mean()); lo, hi = boot(vals, lambda a: 100 * (1 - a.mean()))
    sh = np.mean([np.mean(R[c]["sh_" + k]) for c in R])
    print("  %-6s %5.1f  [%5.1f, %5.1f]   sharp %.3f" % (k, FL[k], lo, hi, sh))
from ultralytics import YOLO
yl = YOLO("yolov8n.pt")
def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.25)[0]
    return [(int(b.cls), float(b.conf), b.xyxy[0].cpu().numpy()) for b in r.boxes if int(b.cls) in (0, 1, 2, 3, 5, 7)]
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
SCENES = [("shop1", D / "cctv_shop1.mp4"), ("shop2", D / "cctv_shop2.mp4"), ("shop3", D / "cctv_shop3.mp4"),
          ("shop_before", D / "cctv_shop_before+rain.mp4"), ("shop_after", D / "cctv_shop_after+rain.mp4")]
Y = {}
for name, p in SCENES:
    if not p.exists(): print("missing", name, flush=True); continue
    c = cv2.VideoCapture(str(p)); N = int(c.get(7)); rows = []
    for i in np.linspace(26, N - 27, 30).astype(int):
        o = outs(c, name, int(i)); x = o["in"]; P_ = o["_P"]; dx = det(x)
        if not dx: continue
        r = {}
        for k in METH:
            for tag, yy in ((k, o[k]), (k + "_prot", np.where(P_[..., None], x, o[k]))):
                r[tag] = float(np.mean([any(d[0] == e[0] and iou(d[2], e[2]) > 0.5 for e in det(yy)) for d in dx]))
        rows.append(r)
    CA.clear()
    if len(rows) < 5: print("skip", name, flush=True); continue
    Y[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    print("%-12s" % name, {k: round(Y[name][k], 3) for k in METH}, flush=True)
print("\n=== the two axes of Table 9 ===")
print("  %-6s %-10s %-10s %-10s" % ("", "flicker%", "kept-rate", "kept+prot"))
for k in METH:
    kp = np.mean([Y[n][k] for n in Y]); kpp = np.mean([Y[n][k + "_prot"] for n in Y])
    print("  %-6s %-10.1f %-10.3f %-10.3f" % (k, FL[k], kp, kpp))
json.dump({"flicker": {k: FL[k] for k in METH}, "kept": Y}, open(OUT / "mg21_eval.json", "w"), indent=1)
print("DONE")
