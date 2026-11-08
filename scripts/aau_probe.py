"""Which AAU RainSnow sequences are usable for a static-camera method?
Checks camera drift, brightness (day vs night) and that the annotated PNG matches the video frame."""
import json, cv2, numpy as np, re
from pathlib import Path
from common import ROOT, OUT
AAU = ROOT / "data" / "new_clips" / "AAU"
ann = json.load(open(AAU / "aauRainSnow-rgb.json"))
by = {}
for im in ann["images"]:
    seq = "/".join(im["file_name"].split("/")[:2]); by.setdefault(seq, []).append(im["file_name"])
def idx(fn): return int(re.search(r"cam1-(\d+)\.png", fn).group(1))
R = {}
print("%-28s %7s %6s %6s %9s %8s %8s" % ("sequence", "frames", "fps", "n_ann", "drift_px", "bright", "png_ok"))
for seq, files in sorted(by.items()):
    v = AAU / seq / "cam1.mkv"
    if not v.exists(): print("%-28s MISSING VIDEO" % seq); continue
    c = cv2.VideoCapture(str(v)); N = int(c.get(7)); fps = c.get(5)
    fr = sorted(idx(f) for f in files)
    def get(i):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = c.read(); return f if ok else None
    # drift over the annotated span
    a, b = get(fr[0]), get(fr[-1])
    dr = np.nan
    if a is not None and b is not None:
        ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32); gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(ga, gb); dr = float(np.hypot(dx, dy))
    br = float(np.mean([get(i).mean() for i in fr[::25] if get(i) is not None])) / 255
    # does the stored PNG equal the decoded frame?
    p = AAU / files[0]; ok = "n/a"
    if p.exists():
        png = cv2.imread(str(p)); f0 = get(fr[0])
        if png is not None and f0 is not None and png.shape == f0.shape:
            ok = "%.1f" % float(np.abs(png.astype(np.float32) - f0.astype(np.float32)).mean())
    R[seq] = dict(N=N, fps=fps, n=len(files), drift=dr, bright=br, png_mae=ok, size=list(a.shape[:2]) if a is not None else None)
    print("%-28s %7d %6.1f %6d %9.2f %8.3f %8s" % (seq, N, fps, len(files), dr, br, ok), flush=True)
json.dump(R, open(OUT / "aau_probe.json", "w"), indent=1)
good = [k for k, v in R.items() if v["drift"] < 1.0 and v["bright"] > 0.25]
print("\nstatic (drift < 1 px over the annotated span) AND bright (> 0.25):", len(good))
for k in good: print("   ", k, "drift %.2f bright %.3f" % (R[k]["drift"], R[k]["bright"]))
print("\nstatic only:", [k for k, v in R.items() if v["drift"] < 1.0])
print("DONE")
