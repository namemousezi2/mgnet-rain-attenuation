"""Does attenuating rain actually help anything? A motion-trigger test.

The detector experiments in this paper show only that every method costs a detector something and
that MG-Net costs least. They never show a benefit. The use that the paper claims for rain
attenuation -- a camera watched by an operator, or feeding logic that reacts to frame-to-frame
change -- is tested here directly, with the oldest and most common such logic: an OpenCV MOG2
background subtractor driving a motion alarm.

Two measurements, because either alone can be gamed:

  (1) False alarms. On rain clips that contain no moving objects, every trigger is caused by rain.
      The matching rain-free clip of the same scene gives the trigger rate of that scene without
      rain, so the excess over it is the rain-induced false-alarm rate.
  (2) True triggers kept. On the five CCTV shop scenes, which do contain people and vehicles, we
      measure the fraction of the frames that trigger on the input and still trigger on the output.
      Without this, a method could win measurement (1) by flattening the image.

No model is trained or tuned here; MOG2 runs with its default parameters on every method's output.
"""
import json, cv2, numpy as np, sys
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 90       # scored frames per clip, after warm-up
WARM = 30
NS = [3, 9, 21]
AREA = 0.001          # a trigger is a foreground blob covering more than 0.1% of the frame
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
NET = {"MG": loadmg(ck / "student_mg.pth"), "MGs": loadmg(ck / "student_mg_strong.pth")}
METH = ["input"] + ["med%d" % n for n in NS] + list(NET) + ["mf7"]
CA = {}
def gf(c, cid, i):
    k = (cid, int(i))
    if k not in CA:
        if len(CA) > 300: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]
def outputs(c, cid, i):
    f21 = [gf(c, cid, i + j) for j in range(-20, 21, 2)]
    x = f21[10]; o = {"input": x}
    for n in NS:
        h = (n - 1) // 2; o["med%d" % n] = np.median(np.stack(f21[10 - h:10 + h + 1]), 0)
    w9 = torch.from_numpy(np.stack(f21[6:15])).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([w9[2], w9[3], w9[4], w9[5], w9[6], w9[4] - w9.median(0).values], 0)[None]
        for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf7"] = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return o
K3 = np.ones((3, 3), np.uint8)
def trigger_series(frames):
    """frames: list of float BGR in [0,1], in temporal order. Returns a 0/1 trigger per scored frame."""
    bs = cv2.createBackgroundSubtractorMOG2()
    out = []
    for n, f in enumerate(frames):
        fg = bs.apply((np.clip(f, 0, 1) * 255).astype(np.uint8))
        if n < WARM: continue
        fg = (fg == 255).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, K3)
        out.append(1 if fg.sum() > AREA * fg.size else 0)
    return np.array(out)
def run_clip(path, cid):
    c = cv2.VideoCapture(str(path)); N = int(c.get(7))
    if N < NFR + WARM + 50: print("  too short", cid, flush=True); return None
    start = max(24, (N - (NFR + WARM)) // 2)
    seq = {k: [] for k in METH}
    for i in range(start, start + NFR + WARM):
        o = outputs(c, cid, i)
        for k in METH: seq[k].append(o[k])
    CA.clear()
    return {k: trigger_series(v) for k, v in seq.items()}
PL = ROOT / "data" / "new_place_video_rain"; NC = ROOT / "data" / "new_clips"
PAIRS = [("house", PL / "already_rain.MOV", PL / "before_rain.MOV"),
         ("p4_1931", NC / "3 footage raining new place" / "IMG_1931.MOV", NC / "3 footage no rain new place" / "IMG_1869.MOV"),
         ("p4_1933", NC / "3 footage raining new place" / "IMG_1933.MOV", NC / "3 footage no rain new place" / "IMG_1872.MOV"),
         ("p4_1934", NC / "3 footage raining new place" / "IMG_1934.MOV", NC / "3 footage no rain new place" / "IMG_1873.MOV")]
FA = {}
for cid, rp, dp in PAIRS:
    if not rp.exists() or not dp.exists(): print("missing", cid, flush=True); continue
    tr = run_clip(rp, cid + "_rain"); td = run_clip(dp, cid + "_dry")
    if tr is None or td is None: continue
    FA[cid] = {k: {"rain": float(tr[k].mean()), "dry": float(td[k].mean())} for k in METH}
    print("%-9s rain-clip trigger rate" % cid, {k: round(FA[cid][k]["rain"], 3) for k in METH}, flush=True)
    print("%-9s dry-clip  trigger rate" % cid, {k: round(FA[cid][k]["dry"], 3) for k in METH}, flush=True)
SHOP = [("shop1", D / "cctv_shop1.mp4"), ("shop2", D / "cctv_shop2.mp4"), ("shop3", D / "cctv_shop3.mp4"),
        ("shop_before", D / "cctv_shop_before+rain.mp4"), ("shop_after", D / "cctv_shop_after+rain.mp4")]
TP = {}
for cid, p in SHOP:
    if not p.exists(): print("missing", cid, flush=True); continue
    t = run_clip(p, cid)
    if t is None: continue
    base = t["input"].astype(bool)
    if base.sum() < 5: print("  too few input triggers", cid, int(base.sum()), flush=True); continue
    TP[cid] = {k: float(t[k][base].mean()) for k in METH}
    print("%-12s input triggers %3d  kept" % (cid, int(base.sum())), {k: round(TP[cid][k], 3) for k in METH}, flush=True)
print("\n=== motion-trigger test (MOG2, default parameters) ===")
print("  %-7s %-22s %-20s %-20s" % ("", "rain-clip triggers", "rain-free triggers", "true triggers kept"))
for k in METH:
    r = np.mean([FA[c][k]["rain"] for c in FA]) if FA else float("nan")
    dd = np.mean([FA[c][k]["dry"] for c in FA]) if FA else float("nan")
    tp = np.mean([TP[c][k] for c in TP]) if TP else float("nan")
    print("  %-7s %-22.3f %-20.3f %-20.3f" % (k, r, dd, tp))
json.dump({"false_alarm": FA, "true_kept": TP, "n_frames": NFR, "area_frac": AREA},
          open(OUT / "motion_trigger.json", "w"), indent=1)
print("DONE")
