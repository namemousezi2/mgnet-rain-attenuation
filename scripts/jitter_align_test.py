import json, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
dev = "cuda"
mg = MG().to(dev); mg.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); mg.eval()
def shp(a): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl].mean())
def align(fr, ref):
    g = lambda a: cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    r = g(ref); w = cv2.createHanningWindow((r.shape[1], r.shape[0]), cv2.CV_32F); out = []
    for f in fr:
        (dx, dy), _ = cv2.phaseCorrelate(r, g(f), w); out.append(shift(f, -dx, -dy))
    return out
def shift(f, dx, dy): return cv2.warpAffine(f, np.float32([[1, 0, dx], [0, 1, dy]]), (f.shape[1], f.shape[0]), borderMode=cv2.BORDER_REFLECT)
PL = ROOT / "data" / "new_place_video_rain"
jobs = [(PL / "front_road_rain.MOV", 10, "front_road"), (D / "cctv_heavy_rain_1-11_1.mp4", 20, "cctv_heavy"), (D / "cctv_medium_rain_1-11_4.mp4", 20, "cctv_medium")]
res = {}
for p, t0, name in jobs:
    c = cv2.VideoCapture(str(p)); fps = c.get(5) or 30; s0 = int(t0 * fps)
    if not c.isOpened() or int(c.get(7)) < s0 + 300: print("skip", name, flush=True); continue
    for s in (0, 1, 2):
        rng = np.random.RandomState(0); R = {"input": [], "median9": [], "mgnet": [], "median9_al": [], "mgnet_al": []}
        for k in range(8, 8 + 30):
            i = s0 + k
            fr = [shift(getf(c, i + j), *(rng.uniform(-s, s, 2) if s else (0, 0))) for j in range(-8, 9, 2)]
            for tag, F in (("", fr), ("_al", align(fr, fr[4]) if s else fr)):
                x = F[4]; m9 = np.median(np.stack(F), 0)
                win = torch.from_numpy(np.stack(F)).permute(0, 3, 1, 2).to(dev)
                with torch.no_grad():
                    X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
                    y = mg(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                if tag == "": R["input"].append(shp(x))
                R["median9" + tag].append(shp(m9)); R["mgnet" + tag].append(shp(y))
        r = {k: float(np.mean(R[k]) / np.mean(R["input"])) for k in R if k != "input"}
        res[f"{name}_shift{s}"] = r; print(name, "jitter +-%dpx" % s, "unaligned: median9 %.2f MG-Net %.2f | aligned: median9 %.2f MG-Net %.2f" % (r["median9"], r["mgnet"], r["median9_al"], r["mgnet_al"]), flush=True)
json.dump(res, open(str(ROOT / "experiments_v2" / "results" / "jitter_align_test.json"), "w"), indent=1)
print("DONE")
