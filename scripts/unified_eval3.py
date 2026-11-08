"""One protocol for every number on natural rain: same window set, same models,
3 seeds for MG-strong, bootstrap 95% CI, dry correction only where it is valid."""
import json, cv2, numpy as np, sys, os
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
import baselines_v2 as B
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
import sys as _s
NWIN = int(_s.argv[1]) if len(_s.argv) > 1 else 20
NYOLO = int(_s.argv[2]) if len(_s.argv) > 2 else 45
RNG = np.random.RandomState(0); DRY_MAX = 0.5; NS = [3, 5, 9, 15, 21]
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
NET = {"MG": loadmg(ck / "student_mg.pth"), "MGs_s1": loadmg(ck / "student_mg_strong.pth")}
for sd in (2, 3):
    p = ck / f"student_mg_strong_s{sd}.pth"
    if p.exists(): NET[f"MGs_s{sd}"] = loadmg(p)
BL = {}
for n in ["prenet_ft", "restormer"]:
    try: m, mult = B.LOADERS[n](); BL[n] = (m.to(dev).eval(), mult); print("loaded", n, flush=True)
    except Exception as e: print("skip baseline", n, e, flush=True)
def basel(n, x):
    m, mult = BL[n]
    try:
        with torch.no_grad(): return np.clip(B.infer(m, mult, x), 0, 1)
    except Exception:
        torch.cuda.empty_cache()
        with torch.no_grad(): return np.clip(B.infer(m, mult, x, tile=256), 0, 1)
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
for _k in [k for k in sys.modules if k == "networks" or k.startswith("networks.")]: del sys.modules[_k]
from networks.derain_net import DerainNet
sv = DerainNet(n_features=32, n_resblocks=8).to(dev); sv.load_state_dict(torch.load(str(S2 / "model_states" / "derainer_rho05.pt"), map_location=dev)); sv.eval()
SRC = Path(__file__).parent / "asf_src"; sys.path.insert(0, str(SRC)); cwd = os.getcwd(); os.chdir(SRC)
from Network import TOFlow
asf = TOFlow().to(dev); sd_ = torch.load(str(SRC / "toflow_models_mine" / "denoising_best.pkl"), map_location=dev)
asf.load_state_dict({k.replace("module.", "", 1): v for k, v in sd_.items()}, strict=False); asf.eval(); os.chdir(cwd)
@torch.no_grad()
def run_asf(fr):
    t = torch.from_numpy(np.stack([f[:, :, ::-1] for f in fr]).copy()).permute(0, 3, 1, 2)[None].to(dev)
    Hh = t.shape[-2]; ph = (32 - Hh % 32) % 32
    t = torch.nn.functional.pad(t.flatten(0, 1), (0, 0, 0, ph), mode="reflect").view(1, 3, 3, Hh + ph, -1)
    return np.ascontiguousarray(asf(t)[0, :, :Hh].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
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
        if len(CA) > 500: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]
def outs(c, cid, i, heavy=True):
    fr = [gf(c, cid, i + j) for j in range(-8, 9, 2)]; x = fr[4]; o = {"in": x}
    for N in NS:
        h = (N - 1) // 2; o["med%d" % N] = np.median(np.stack([gf(c, cid, i + j) for j in range(-2 * h, 2 * h + 1, 2)]), 0)
    win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
        for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf7"] = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        if heavy:
            t7 = torch.from_numpy(np.stack([gf(c, cid, i + j) for j in range(-3, 4)])).permute(3, 0, 1, 2)[None].to(dev)
            o["S2VD"] = sv(t7)[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()[3]
    o["_P"] = motion_mask(x, o["med15"])   # returned, NOT applied: protection is reported separately
    if heavy:
        for n, tag in (("prenet_ft", "PReNet"), ("restormer", "Restormer")):
            if n in BL: o[tag] = basel(n, x)
        o["ASF"] = run_asf([gf(c, cid, i + j) for j in (-1, 0, 1)])
    return o
def drift(c, cid, i, n=16):
    a = cv2.cvtColor(gf(c, cid, i), cv2.COLOR_BGR2GRAY); b = cv2.cvtColor(gf(c, cid, i + n), cv2.COLOR_BGR2GRAY)
    (dx, dy), _ = cv2.phaseCorrelate(a, b); return float(np.hypot(dx, dy))
PL = ROOT / "data" / "new_place_video_rain"; NC = ROOT / "data" / "new_clips"
CLIPS = [("front_road", PL / "front_road_rain.MOV", None), ("back_road", PL / "back_road_rain.MOV", None),
         ("house", PL / "already_rain.MOV", PL / "before_rain.MOV"),
         ("p4_1931", NC / "3 footage raining new place" / "IMG_1931.MOV", NC / "3 footage no rain new place" / "IMG_1869.MOV"),
         ("p4_1933", NC / "3 footage raining new place" / "IMG_1933.MOV", NC / "3 footage no rain new place" / "IMG_1872.MOV"),
         ("p4_1934", NC / "3 footage raining new place" / "IMG_1934.MOV", NC / "3 footage no rain new place" / "IMG_1873.MOV")]
def measure(p, cid, nwin=None):
    nwin = NWIN if nwin is None else nwin
    c = cv2.VideoCapture(str(p)); fps = c.get(5); N = int(c.get(7)); sec = N / fps
    ok = [i for i in [int(t * fps) for t in np.linspace(8, sec - 8, 4 * nwin)] if drift(c, cid, i - 8) < 1.0][:nwin]
    if len(ok) < 3: return None
    st = smask(c, ok[0] / fps - 3, ok[0] / fps + 3); s_ = st[sl]; per = {}
    for i in ok:
        o1, o2 = outs(c, cid, i), outs(c, cid, i + 1)
        for k in [k for k in o1 if k != "_P"]:
            per.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s_].mean()))
            per.setdefault("sh_" + k, []).append(shp(o1[k], st) / (shp(o1["in"], st) + 1e-9))
    CA.clear(); return {"per": {k: [float(x) for x in v] for k, v in per.items()}, "windows": [int(i) for i in ok]}
R = {}
for cid, p, dp in CLIPS:
    if not p.exists(): print("missing", cid, flush=True); continue
    m = measure(p, cid)
    if m is None: print("skip (camera not static)", cid, flush=True); continue
    R[cid] = m; print("done", cid, "n windows", len(m["windows"]), flush=True)
    if dp is not None and dp.exists():
        md = measure(dp, cid + "_dry")
        if md is not None: R[cid + "_dry"] = md; print("done", cid + "_dry", flush=True)
json.dump(R, open(OUT / "unified3_flicker.json", "w"), indent=1)
METH = ["med%d" % N for N in NS] + [k for k in NET] + ["mf7", "S2VD", "ASF", "PReNet", "Restormer"]
METH = [m for m in METH if m in R[list(R)[0]]["per"]]
print("\nNOTE: flicker and sharpness are for the BARE output of each method (no motion protection).")
def boot(v, f, n=2000):
    v = np.asarray(v); idx = RNG.randint(0, len(v), (n, len(v)))
    s = np.sort([f(v[j]) for j in idx]); return float(s[int(0.025 * n)]), float(s[int(0.975 * n)])
print("\n=== RAW flicker removed % [95% CI over windows] | sharpness ===")
for cid in [c for c in R if not c.endswith("_dry")]:
    fin = np.mean(R[cid]["per"]["in"]); print(cid, "n=%d  flicker_in=%.4f" % (len(R[cid]["windows"]), fin))
    for k in METH:
        v = np.array(R[cid]["per"][k]) / fin; lo, hi = boot(v, lambda a: 100 * (1 - a.mean()))
        print(f"   {k:10s} {100*(1-v.mean()):6.1f} [{lo:6.1f},{hi:6.1f}]  sharp {np.mean(R[cid]['per']['sh_'+k]):.3f}")
print("\n=== DRY-CORRECTED (only clips with f_dry/f_rain <= %.2f) ===" % DRY_MAX)
for cid in [c for c in R if not c.endswith("_dry") and c + "_dry" in R]:
    fr_, fd_ = np.mean(R[cid]["per"]["in"]), np.mean(R[cid + "_dry"]["per"]["in"]); rt = fd_ / fr_
    if rt > DRY_MAX: print(cid, "EXCLUDED  f_dry/f_rain=%.2f" % rt); continue
    print(cid, "f_dry/f_rain=%.2f" % rt)
    for k in METH:
        S = np.mean(R[cid]["per"][k]) / fr_; rho = np.mean(R[cid + "_dry"]["per"][k]) / fd_
        print(f"   {k:10s} {100*(1-(S*fr_-rho*fd_)/(fr_-fd_)):6.1f}")
seeds = [k for k in NET if k.startswith("MGs_")]
if len(seeds) > 1:
    print("\n=== MG-strong across %d seeds: raw flicker removed %% (mean +- SD) ===" % len(seeds))
    for cid in [c for c in R if not c.endswith("_dry")]:
        fin = np.mean(R[cid]["per"]["in"]); v = [100 * (1 - np.mean(R[cid]["per"][s]) / fin) for s in seeds]
        print(f"   {cid:12s} {np.mean(v):5.1f} +- {np.std(v):.1f}")
from ultralytics import YOLO
yl = YOLO("yolov8n.pt")
def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.25)[0]
    return [(int(b.cls), float(b.conf), b.xyxy[0].cpu().numpy()) for b in r.boxes if int(b.cls) in (0, 1, 2, 3, 5, 7)]
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]); i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)
YM = ["med%d" % N for N in NS] + [k for k in NET] + ["mf7"]
Y = {}
for name in ["cctv_shop1.mp4", "cctv_shop2.mp4", "cctv_shop3.mp4", "cctv_shop_before+rain.mp4", "cctv_shop_after+rain.mp4"]:
    c = cv2.VideoCapture(str(D / name)); N = int(c.get(7)); rows = []
    for i in np.linspace(26, N - 27, NYOLO).astype(int):
        o = outs(c, name, int(i), heavy=False); x = o["in"]; P_ = o["_P"]; dx = det(x); r = {}
        for k in YM:
            for tag, yy in ((k, o[k]), (k + "_prot", np.where(P_[..., None], x, o[k]))):
                r[tag] = float(np.mean([any(d[0] == e[0] and iou(d[2], e[2]) > 0.5 for e in det(yy)) for d in dx])) if dx else np.nan
        rows.append(r)
    Y[name] = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}; CA.clear()
    print("YOLO", name, {k: round(Y[name][k], 3) for k in YM}, "| prot:", {k: round(Y[name][k + "_prot"], 3) for k in YM}, flush=True)
json.dump(Y, open(OUT / "unified3_yolo.json", "w"), indent=1)
print("\n=== PARETO (mean over rain clips) ===")
rains = [c for c in R if not c.endswith("_dry")]
print("  (kept-rate shown bare | with motion protection)")
for k in YM:
    fl = np.mean([100 * (1 - np.mean(R[c]["per"][k]) / np.mean(R[c]["per"]["in"])) for c in rains])
    sh = np.mean([np.mean(R[c]["per"]["sh_" + k]) for c in rains]); kp = np.mean([Y[n][k] for n in Y])
    print(f"  {k:10s} flicker {fl:6.1f}   kept-rate {kp:.3f}   sharp {sh:.3f}")
if len(seeds) > 1:
    fl = [np.mean([100 * (1 - np.mean(R[c]["per"][s]) / np.mean(R[c]["per"]["in"])) for c in rains]) for s in seeds]
    kp = [np.mean([Y[n][s] for n in Y]) for s in seeds]; kpp = [np.mean([Y[n][s + "_prot"] for n in Y]) for s in seeds]
    print("  MGs seeds: flicker %.1f +- %.1f   kept-rate %.3f +- %.3f | prot %.3f +- %.3f" % (np.mean(fl), np.std(fl), np.mean(kp), np.std(kp), np.mean(kpp), np.std(kpp)))
print("DONE")
