import cv2, numpy as np, json, glob, os
from common import ROOT, OUT
D = ROOT / "data" / "new_clips"; W, H = 640, 360
names = sorted(os.path.basename(p) for p in glob.glob(str(D / "cctv_*.mp4")) + glob.glob(str(D / "CCTV_*.mp4")))
M = {}; info = {}
for n in names:
    c = cv2.VideoCapture(str(D / n)); N = int(c.get(7)); fr = []
    for i in np.linspace(0, N - 2, 20).astype(int):
        c.set(1, int(i)); ok, f = c.read()
        if ok: fr.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    if not fr: continue
    a = np.stack(fr); M[n] = np.median(a, 0).astype(np.uint8)
    g = a.mean(3); info[n] = dict(frames=N, fps=c.get(5), lum=float(a.mean()), drift=float(np.mean([np.abs(g[i] - g[0]).mean() for i in range(1, len(g))])))
def ecc(a, b):
    A = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32); B = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32); w = np.eye(2, 3, dtype=np.float32)
    try: c, w = cv2.findTransformECC(A, B, w, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5); return float(c), float(np.hypot(w[0, 2], w[1, 2]))
    except Exception: return None, None
dry = [n for n in M if "before" in n.lower() or "after" in n.lower() or "norain" in n.lower()]
rainy = [n for n in M if n not in dry and "shop" not in n.lower()]
res = {}
for r in rainy:
    best = None
    for d in dry:
        c, s = ecc(M[r], M[d])
        if c is not None and (best is None or c > best[1]): best = (d, c, s)
    res[r] = dict(info[r], best_dry=best[0] if best else None, ecc=best[1] if best else None, shift=best[2] if best else None)
    print(r, res[r], flush=True)
json.dump(dict(pairs=res, info=info), open(OUT / "cctv_pairs_check.json", "w"), indent=1); print("DONE")
