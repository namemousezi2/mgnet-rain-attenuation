"""Place the retrained S2VD backbone (S2VD-ft) on the two axes of Table 9, under the protocol
used there: the six natural-rain clips of Table 8 for flicker and sharpness, twenty windows per
clip, the same windows for every method, bare output and no motion protection; the five CCTV
shop scenes of Table 6 for detector retention, reported bare and with the optional protection.
Compared: the 9-frame median, MG-strong (seed 1), S2VD as released, and S2VD-ft."""
import json, cv2, numpy as np, sys, os
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
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
for _k in [k for k in sys.modules if k == "networks" or k.startswith("networks.")]: del sys.modules[_k]
from networks.derain_net import DerainNet
sv = DerainNet(n_features=32, n_resblocks=8).to(dev)
sv.load_state_dict(torch.load(str(S2 / "model_states" / "derainer_rho05.pt"), map_location=dev)); sv.eval()
pft = ck / "s2vd_ft.pth"
if not pft.exists(): print("MISSING", pft, "- run run_s2vd_ft.bat first"); sys.exit(1)
svf = DerainNet(n_features=32, n_resblocks=8).to(dev); svf.load_state_dict(torch.load(str(pft), map_location=dev)); svf.eval()
METH = ["med9", "MGs", "S2VD", "S2VDft"]
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
@torch.no_grad()
def run_s2(model, fr7):
    t = torch.from_numpy(np.stack(fr7)).permute(3, 0, 1, 2)[None].to(dev)
    return model(t)[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()[3]
def outs(c, cid, i):
    f9 = [gf(c, cid, i + j) for j in range(-8, 9, 2)]; x = f9[4]; o = {"in": x}
    o["med9"] = np.median(np.stack(f9), 0)
    m15 = np.median(np.stack([gf(c, cid, i + j) for j in range(-14, 15, 2)]), 0)
    w9 = torch.from_numpy(np.stack(f9)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([w9[2], w9[3], w9[4], w9[5], w9[6], w9[4] - w9.median(0).values], 0)[None]
        o["MGs"] = MGS(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    f7 = [gf(c, cid, i + j) for j in range(-3, 4)]
    o["S2VD"] = run_s2(sv, f7); o["S2VDft"] = run_s2(svf, f7)
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
    ok = [i for i in [int(t * fps) for t in np.linspace(8, sec - 8, 4 * NWIN)] if drift(c, cid, i - 8) < 1.0][:NWIN]
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
    print("  %-7s %5.1f  [%5.1f, %5.1f]   sharp %.3f" % (k, FL[k], lo, hi, sh))
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
    for i in np.linspace(20, N - 21, 30).astype(int):
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
print("  %-7s %-10s %-10s %-10s" % ("", "flicker%", "kept-rate", "kept+prot"))
for k in METH:
    kp = np.mean([Y[n][k] for n in Y]); kpp = np.mean([Y[n][k + "_prot"] for n in Y])
    print("  %-7s %-10.1f %-10.3f %-10.3f" % (k, FL[k], kp, kpp))
json.dump({"flicker": {k: FL[k] for k in METH}, "kept": Y}, open(OUT / "s2vdft_eval.json", "w"), indent=1)
print("DONE")
