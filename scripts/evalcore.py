import json, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
ND = ROOT / "data" / "new_clips" / "New rain GT and light medium and heavy"
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8); mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mv); big = np.isin(lab, [k for k in range(1, n) if stats[k, 4] > 120]).astype(np.uint8); return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
def hf(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
def one(c, i, fn):
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0)
    o = {"in": x}; o.update(fn(c, i, x)); nr = cv2.dilate((np.abs(x - m9).mean(2) > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    return o, nr
def evaluate(fn, prefix):
    dm, cd, Nd = med(ND / "IMG_1768.MOV"); idx = lambda N: list(np.linspace(40, N - 41, 8).astype(int))
    def scene(c, st, N):
        s = st[sl]; fl = {}; sh = {}
        for i in idx(N):
            o1, nr = one(c, i, fn); o2, _ = one(c, i + 1, fn)
            for k in o1:
                fl.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s].mean())); sh.setdefault(k, []).append(hf(o1[k], nr))
        return {k: float(np.mean(v)) for k, v in fl.items()}, {k: float(np.mean(v)) for k, v in sh.items()}
    sd = smask(cd, 0, Nd - 2); fd, shd = scene(cd, sd, Nd); R = {"dry_flicker_in": fd["in"]}; print("DRY", fd, flush=True)
    for tag in ("IMG_1772", "IMG_1773", "IMG_1771", "IMG_1769"):
        rm, cr, N = med(ND / (tag + ".MOV")); GT, w = align(rm, dm); st = smask(cd, 0, Nd - 2, w)
        f, sh = scene(cr, st, N); fin = f["in"]; out = {}
        for k in f:
            if k == "in": continue
            S = f[k] / fin; rho = fd[k] / fd["in"]
            out["removed_" + k] = float(1 - (S * fin - rho * fd["in"]) / (fin - fd["in"])); out["sharp_nonrain_" + k] = sh[k] / sh["in"]
        R[tag] = out; print(tag, {k: round(v, 3) for k, v in out.items()}, flush=True)
    json.dump(R, open(OUT / (prefix + ".json"), "w"), indent=1); print("EVALDONE", flush=True)
def make_videos(fn, prefix):
    for tag, name in (("IMG_1769", "heavy"), ("IMG_1773", "medium"), ("IMG_1772", "light")):
        c = cv2.VideoCapture(str(ND / (tag + ".MOV"))); N = int(c.get(7)); fps = c.get(5) or 30; s0 = int(N * 0.4); vw = {}
        for k in range(8, 248):
            x = getf(c, s0 + k - 8); o = fn(c, s0 + k - 8, x)
            for m, im in o.items():
                if m not in vw: vw[m] = cv2.VideoWriter(str(OUT / f"{prefix}_{m}_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
                vw[m].write((np.clip(im, 0, 1) * 255).astype(np.uint8))
        for v in vw.values(): v.release()
        print("video", name, flush=True)
