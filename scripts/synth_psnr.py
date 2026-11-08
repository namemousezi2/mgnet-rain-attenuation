"""PSNR/SSIM against an exact reference, by rendering synthetic rain on our own rain-free clips.

Real rain has no pixel-aligned clean frame, which is why the rest of the paper uses reference-free
measures. This script closes that gap in the only way that keeps the reference exact: it takes
rain-free clips from static cameras, renders rain streaks onto every frame, and compares each
method's output with the untouched source frame.

The rain model is deliberately simple and is released with this script: per frame, N straight
streaks of random position, length and brightness are drawn at a common angle with a few degrees of
jitter, blurred, and composited as x + L*(1 - x) so that bright areas saturate rather than clip.
Streaks are drawn independently for every frame, which is the property a temporal median exploits.

Fairness note for the paper: MG-Net and its teacher were never trained on this rain model, and
neither were the medians. PReNet, Restormer, S2VD and ASF-Net were trained on other synthetic rain,
so this test is closer to their training distribution than to ours.
"""
import json, cv2, numpy as np, sys, os
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
from common import psnr, ssim
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
import baselines_v2 as B
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 24      # evaluated frames per clip per intensity
NS = [3, 9, 21]
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
BL = {}
for n in ["prenet_ft", "restormer"]:
    try: m, mult = B.LOADERS[n](); BL[n] = (m.to(dev).eval(), mult); print("loaded", n, flush=True)
    except Exception as e: print("skip baseline", n, e, flush=True)
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
for _k in [k for k in sys.modules if k == "networks" or k.startswith("networks.")]: del sys.modules[_k]
from networks.derain_net import DerainNet
SV = {}
for tag, w in (("S2VD", S2 / "model_states" / "derainer_rho05.pt"), ("S2VDft", ck / "s2vd_ft.pth")):
    if Path(w).exists():
        m = DerainNet(n_features=32, n_resblocks=8).to(dev); m.load_state_dict(torch.load(str(w), map_location=dev)); SV[tag] = m.eval()
SRC = Path(__file__).parent / "asf_src"; sys.path.insert(0, str(SRC)); cwd = os.getcwd(); os.chdir(SRC)
from Network import TOFlow
AS = {}
for tag, w in (("ASF", SRC / "toflow_models_mine" / "denoising_best.pkl"), ("ASFft", ck / "asf_ft.pth")):
    if Path(w).exists():
        a = TOFlow().to(dev); sd = torch.load(str(w), map_location=dev)
        a.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False); AS[tag] = a.eval()
os.chdir(cwd)
METH = ["input"] + ["med%d" % n for n in NS] + list(NET) + ["mf7"] + list(SV) + list(AS) + \
       (["PReNet"] if "prenet_ft" in BL else []) + (["Restormer"] if "restormer" in BL else [])
# ---------------- rain model ----------------
PRESET = {"light": (380, 11, 0.45), "medium": (850, 17, 0.60), "heavy": (1500, 23, 0.75)}
def rain_layer(h, w, rng, n, ln, br, angle=12.0):
    L = np.zeros((h, w), np.float32)
    for _ in range(n):
        x0 = int(rng.integers(0, w)); y0 = int(rng.integers(-ln, h))
        a = np.deg2rad(angle + rng.normal(0, 3.0)); d = ln * (0.7 + 0.6 * rng.random())
        cv2.line(L, (x0, y0), (int(x0 + d * np.sin(a)), int(y0 + d * np.cos(a))),
                 float(br * (0.5 + 0.5 * rng.random())), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(L, (0, 0), 0.8)
def add_rain(x, rng, preset):
    n, ln, br = PRESET[preset]
    L = rain_layer(x.shape[0], x.shape[1], rng, n, ln, br)[..., None]
    return np.clip(x + L * (1.0 - x), 0, 1).astype(np.float32)
# ---------------- clips: rain-free, static, none used to train MG-Net ----------------
PL = ROOT / "data" / "new_place_video_rain"; NC = ROOT / "data" / "new_clips"
CLIPS = [("house_dry", PL / "before_rain.MOV"),
         ("p4_dry_1869", NC / "3 footage no rain new place" / "IMG_1869.MOV"),
         ("p4_dry_1872", NC / "3 footage no rain new place" / "IMG_1872.MOV"),
         ("p4_dry_1873", NC / "3 footage no rain new place" / "IMG_1873.MOV")]
R = {}
for cid, p in CLIPS:
    if not p.exists(): print("missing", cid, flush=True); continue
    cap = cv2.VideoCapture(str(p)); N = int(cap.get(7))
    for preset in PRESET:
        rows = {k: {"psnr": [], "ssim": []} for k in METH}
        centres = np.linspace(24, N - 25, NFR).astype(int)
        for ci in centres:
            span = list(range(int(ci) - 20, int(ci) + 21))          # 41 frames, enough for med21
            clean = [getf(cap, i) for i in span]
            rng = np.random.default_rng(int(ci))
            rainy = [add_rain(f, np.random.default_rng(int(ci) * 1000 + j), preset) for j, f in enumerate(clean)]
            gt = clean[20]; x = rainy[20]
            out = {"input": x}
            for n in NS:
                h = (n - 1) // 2
                out["med%d" % n] = np.median(np.stack(rainy[20 - 2 * h:20 + 2 * h + 1:2]), 0)
            w9 = torch.from_numpy(np.stack(rainy[12:29:2])).permute(0, 3, 1, 2).to(dev)
            with torch.no_grad():
                X = torch.cat([w9[2], w9[3], w9[4], w9[5], w9[6], w9[4] - w9.median(0).values], 0)[None]
                for k, m_ in NET.items(): out[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                out["mf7"] = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                for k, m_ in SV.items():
                    t = torch.from_numpy(np.stack(rainy[17:24])).permute(3, 0, 1, 2)[None].to(dev)
                    out[k] = m_(t)[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()[3]
                for k, m_ in AS.items():
                    fr = [f[:, :, ::-1] for f in rainy[19:22]]
                    t = torch.from_numpy(np.stack(fr).copy()).permute(0, 3, 1, 2)[None].to(dev)
                    Hh = t.shape[-2]; ph = (32 - Hh % 32) % 32
                    t = torch.nn.functional.pad(t.flatten(0, 1), (0, 0, 0, ph), mode="reflect").view(1, 3, 3, Hh + ph, -1)
                    oo = m_(t)
                    if oo.dim() == 5: oo = oo[:, 1]
                    out[k] = np.ascontiguousarray(oo[0, :, :Hh].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
            for n, tag in (("prenet_ft", "PReNet"), ("restormer", "Restormer")):
                if n in BL:
                    m_, mult = BL[n]
                    with torch.no_grad():
                        try: out[tag] = np.clip(B.infer(m_, mult, x), 0, 1)
                        except Exception:
                            torch.cuda.empty_cache()
                            out[tag] = np.clip(B.infer(m_, mult, x, tile=256), 0, 1)
            for k in METH:
                rows[k]["psnr"].append(psnr(out[k], gt)); rows[k]["ssim"].append(ssim(out[k], gt))
        R[f"{cid}/{preset}"] = {k: {m: float(np.mean(v[m])) for m in ("psnr", "ssim")} for k, v in rows.items()}
        print(f"done {cid} {preset}", {k: round(R[f'{cid}/{preset}'][k]['psnr'], 2) for k in METH}, flush=True)
json.dump(R, open(OUT / "synth_psnr.json", "w"), indent=1)
print("\n=== synthetic rain on our own rain-free clips: PSNR (dB) / SSIM, mean over %d clips x %d frames ===" % (len(set(k.split('/')[0] for k in R)), NFR))
print("  %-10s %-16s %-16s %-16s %-16s" % ("", "light", "medium", "heavy", "mean"))
for k in METH:
    v = {p: np.mean([R[c][k]["psnr"] for c in R if c.endswith("/" + p)]) for p in PRESET}
    s = {p: np.mean([R[c][k]["ssim"] for c in R if c.endswith("/" + p)]) for p in PRESET}
    print("  %-10s %6.2f / %.4f  %6.2f / %.4f  %6.2f / %.4f  %6.2f / %.4f" %
          (k, v["light"], s["light"], v["medium"], s["medium"], v["heavy"], s["heavy"],
           np.mean(list(v.values())), np.mean(list(s.values()))))
print("DONE")
