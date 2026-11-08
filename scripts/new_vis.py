import json, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "new_eval.py")).read().split("ND = ROOT")[0])
ND = ROOT / "data" / "new_clips" / "New rain GT and light medium and heavy"
rows = []; R = {}
dm, cd, Nd = med(ND / "IMG_1768.MOV")
def nonrain_sharp(o, x, m9):
    dil = cv2.dilate(((np.abs(x - m9).mean(2) > 0.02)).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    return float(np.abs(o - cv2.GaussianBlur(o, (0, 0), 1.5)).mean(2)[sl][dil[sl]].mean()), float(dil[sl].mean())
for tag in ("IMG_1772", "IMG_1773", "IMG_1771", "IMG_1769", "IMG_1768"):
    c = cd if tag == "IMG_1768" else cv2.VideoCapture(str(ND / (tag + ".MOV"))); N = Nd if tag == "IMG_1768" else int(c.get(7)); i = N // 2
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; m9 = np.median(np.stack(fr), 0); o = outs_at(c, i)
    r = {}
    for k in ("in", "mf7", "MG", "full_m9"):
        s, fracs = nonrain_sharp(o[k], o["in"], m9); r[k] = s
    R[tag] = {k: r[k] / r["in"] for k in r if k != "in"}; R[tag]["nonrain_frac"] = fracs; print(tag, R[tag], flush=True)
    t = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
    d = t(np.abs(o["MG"] - o["in"]) * 8)
    rows.append(np.hstack([t(o["in"]), t(o["MG"]), t(o["mf7"]), d]))
cv2.imwrite(str(OUT / "new_vis.png"), np.vstack(rows)); json.dump(R, open(OUT / "new_vis.json", "w"), indent=1); print("DONE")
