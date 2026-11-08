"""AAU RainSnow, second use: the kept-rate and a raw flicker reduction on all twenty
public sequences, so that neither axis of the paper rests on five scenes from one site.

The kept-rate needs no ground truth: it compares the detections of the input with the
detections of the output, exactly as Section 4.2 defines it (YOLOv8n, confidence 0.25,
six road-user classes, a detection is kept when the output has one of the same class with
IoU above 0.5). The dataset's own boxes are therefore not used here; aau_eval.py uses them
for AP50.

The flicker figure is RAW: AAU has no rain-free clip of the same view, so the dry-clip
correction of Section 4.2 cannot be applied and the static-pixel set is taken from the
rain clip itself (temporal standard deviation below 0.02 over the window, eroded 5x5).
A raw value counts sensor noise as rain and therefore underestimates the removal; it is a
lower bound, as in Section 4.9, and is comparable between methods but not with the
corrected percentages of Tables 3 to 5.

Protocol: twenty windows per sequence, evenly spaced, the same windows for every method,
no motion protection in the flicker figures. Bootstrap 95% CI over sequences.
"""
import json, re, sys, cv2, numpy as np
from pathlib import Path
from common import ROOT, OUT
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
from mg_multi import MGM

dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
AAU = ROOT / "data" / "new_clips" / "AAU"
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
NS = [3, 5, 9, 15, 21]
HALF = 40                              # widest offset needed: the causal 21-frame median reaches i-40
CLS = {0, 1, 2, 3, 5, 7}               # person bicycle car motorbike bus truck

def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
def loadmgm(p):
    m = MGM().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
# kind decides which stack the model is fed: "m9" = 18 ch with (x - m9), "m21" = 18 ch with the
# centred 21-frame median, "m21c" = 18 ch with the causal one (past only), "momc" = 24 ch with
# (x - m3, x - m9, x - momc)
MODELS = {"MG": ("m9", loadmg(ck / "student_mg.pth")),
          "MGs": ("m9", loadmg(ck / "student_mg_strong.pth"))}
for tag, kind, fn in (("MG21", "m21", "student_mg21.pth"),
                      ("MG21c", "m21c", "student_mg21c.pth"),
                      ("MG21c2", "m21c", "student_mg21c_s2.pth"),
                      ("MG21c3", "m21c", "student_mg21c_s3.pth"),
                      ("MGMmc", "momc", "student_mgm_momc.pth")):
    if (ck / fn).exists(): MODELS[tag] = (kind, loadmgm(ck / fn) if kind == "momc" else loadmg(ck / fn))
    else: print("skip", tag, "- no", fn, flush=True)

class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()

METH = ["med%d" % N for N in NS] + list(MODELS) + ["mf7"]
print("models:", {k: v[0] for k, v in MODELS.items()}, flush=True)

ann = json.load(open(AAU / "aauRainSnow-rgb.json"))
SEQ = {}
for im in ann["images"]:
    SEQ.setdefault("/".join(im["file_name"].split("/")[:2]), []).append(im)
probe = json.load(open(OUT / "aau_probe.json"))
USE = sorted(k for k, v in probe.items()
             if (v["drift"] is not None and v["drift"] == v["drift"]) and v["drift"] < 1.0 and v["bright"] > 0.25)
print("sequences used:", len(USE), flush=True)

from ultralytics import YOLO
yl = YOLO("yolov8n.pt")

def det(img):
    r = yl.predict((np.clip(img, 0, 1) * 255).astype(np.uint8), verbose=False, conf=0.25)[0]
    return [(int(b.cls), b.xyxy[0].cpu().numpy().tolist()) for b in r.boxes if int(b.cls) in CLS]

def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

def kept(din, dout):
    """fraction of input detections that survive in the output"""
    if not din: return None
    k = 0
    for c, b in din:
        if any(c2 == c and iou(b, b2) > 0.5 for c2, b2 in dout): k += 1
    return k / len(din)

def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8)
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mv)
    big = np.isin(lab, [k for k in range(1, n) if st[k, 4] > 120]).astype(np.uint8)
    return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)

def run_all(frames_by_off, roi):
    """frames_by_off: dict offset -> frame, offsets -HALF..+HALF step 2. Returns outputs at offset 0."""
    x = frames_by_off[0]; o = {"in": x}
    for Nw in NS:
        h = (Nw - 1) // 2
        o["med%d" % Nw] = np.median(np.stack([frames_by_off[2 * k] for k in range(-h, h + 1)]), 0)
    m9at = lambda j: np.median(np.stack([frames_by_off[j + k] for k in range(-8, 9, 2)]), 0)
    momc = np.median(np.stack([o["med9"], m9at(-8), m9at(-16)]), 0)
    m21c = np.median(np.stack([frames_by_off[-j] for j in range(0, 41, 2)]), 0)   # past only
    fr = [frames_by_off[j] for j in range(-8, 9, 2)]
    win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).to(dev)
    with torch.no_grad():
        xt = win[4]; fr5 = [win[2], win[3], win[4], win[5], win[6]]
        md9 = t(o["med9"]); md3 = t(o["med3"]); md21 = t(o["med21"]); mdmc = t(momc); md21c = t(m21c)
        ST = {"m9": torch.cat(fr5 + [xt - md9], 0)[None],
              "m21": torch.cat(fr5 + [xt - md21], 0)[None],
              "m21c": torch.cat(fr5 + [xt - md21c], 0)[None],
              "momc": torch.cat(fr5 + [xt - md3, xt - md9, xt - mdmc], 0)[None]}
        for k, (kind, m_) in MODELS.items():
            o[k] = m_(ST[kind])[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf7"] = mf(ST["m9"])[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    P = motion_mask(x, o["med15"])
    for k in METH: o[k + "_prot"] = np.where(P[..., None], x, o[k])
    return o

per_seq = {}
for seq in USE:
    loc, name = seq.split("/")
    cap = cv2.VideoCapture(str(AAU / seq / "cam1.mkv")); N = int(cap.get(7))
    mpath = AAU / loc / (name + "-mask.png"); roi = None
    if mpath.exists():
        mm = cv2.imread(str(mpath), 0)
        if mm is not None: roi = mm > 127
    inside = lambda b: roi is None or roi[int(min(max((b[1] + b[3]) / 2, 0), roi.shape[0] - 1)),
                                         int(min(max((b[0] + b[2]) / 2, 0), roi.shape[1] - 1))]
    cache = {}
    def gf(i):
        i = int(min(max(i, 0), N - 1))
        if i not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
            cache[i] = f.astype(np.float32) / 255 if ok else None
        return cache[i]
    lo, hi = HALF + 1, N - HALF - 3
    if hi <= lo: print("  %-28s too short (%d frames)" % (seq, N), flush=True); continue
    centers = np.linspace(lo, hi, NWIN).astype(int)
    K = {m: [] for m in METH + [m + "_prot" for m in METH]}
    F = {m: [] for m in ["in"] + METH}
    nk = nf = 0
    for c0 in centers:
        wins = {}
        ok = True
        for t in (c0, c0 + 1):
            fb = {}
            for j in range(-HALF, HALF + 1, 2):
                f = gf(t + j)
                if f is None: ok = False; break
                fb[j] = f
            if not ok: break
            wins[t] = fb
        if not ok: continue
        out = {t: run_all(wins[t], roi) for t in wins}
        # --- kept-rate at t ---
        din = [(c, b) for c, b in det(out[c0]["in"]) if inside(b)]
        if din:
            for m in K:
                r = kept(din, [(c, b) for c, b in det(out[c0][m]) if inside(b)])
                if r is not None: K[m].append(r)
            nk += 1
        # --- raw flicker between t and t+1 on static pixels of the input ---
        stack = np.stack([wins[c0][j] for j in range(-8, 9, 2)])
        sd = stack.std(0).mean(2)
        S = cv2.erode((sd < 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        if roi is not None: S &= roi
        if S.sum() > 0.01 * S.size:
            for m in F:
                a, b = out[c0][m], out[c0 + 1][m]
                F[m].append(float(np.abs(a - b).mean(2)[S].mean()))
            nf += 1
        for t in list(wins): wins[t] = None
        out = None; cache.clear()      # a window needs 42 decoded frames; do not accumulate
    cache.clear()
    kr = {m: (float(np.mean(v)) if v else None) for m, v in K.items()}
    fl = {m: (float(np.mean(v)) if v else None) for m, v in F.items()}
    red = {m: (None if (fl[m] is None or not fl["in"]) else 100 * (1 - fl[m] / fl["in"])) for m in METH}
    per_seq[seq] = {"n_kept_windows": nk, "n_flicker_windows": nf, "kept": kr, "flicker": fl, "flicker_removed_raw_pct": red}
    print("  %-28s kept %2d win  flicker %2d win | MG kept %.3f  MG raw flicker %s"
          % (seq, nk, nf, kr["MG"] if kr["MG"] is not None else float("nan"),
             ("%.1f%%" % red["MG"]) if red["MG"] is not None else "n/a"), flush=True)

def boot(vals, B=2000, seed=0):
    v = [x for x in vals if x is not None]
    if len(v) < 2: return (None, None)
    rng = np.random.default_rng(seed); v = np.array(v)
    s = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)))

print("\n=== kept-rate over %d AAU sequences (mean over sequences, 95%% CI) ===" % len(per_seq))
summary = {}
for m in METH:
    bare = [per_seq[s]["kept"][m] for s in per_seq]
    prot = [per_seq[s]["kept"][m + "_prot"] for s in per_seq]
    rawf = [per_seq[s]["flicker_removed_raw_pct"][m] for s in per_seq]
    cb, cp, cf = boot(bare), boot(prot), boot(rawf)
    mb = float(np.nanmean([x for x in bare if x is not None]))
    mp = float(np.nanmean([x for x in prot if x is not None]))
    mfl = float(np.nanmean([x for x in rawf if x is not None]))
    summary[m] = {"kept": mb, "kept_ci": cb, "kept_prot": mp, "kept_prot_ci": cp,
                  "flicker_removed_raw_pct": mfl, "flicker_ci": cf}
    print("  %-6s kept %.3f [%.3f, %.3f]   with prot %.3f   raw flicker removed %5.1f%% [%.1f, %.1f]"
          % (m, mb, cb[0] or float("nan"), cb[1] or float("nan"), mp, mfl,
             cf[0] if cf[0] is not None else float("nan"), cf[1] if cf[1] is not None else float("nan")))

json.dump({"n_sequences": len(per_seq), "n_windows_per_sequence": NWIN,
           "note": "kept-rate uses no ground truth; flicker is RAW (no dry-clip correction, "
                   "static pixels taken from the rain clip itself) and is a lower bound",
           "per_sequence": per_seq, "summary": summary},
          open(OUT / "aau_kept.json", "w"), indent=1)
print("\nwrote", OUT / "aau_kept.json")
print("DONE")
