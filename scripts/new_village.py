# New unseen scenes (Jungle light/medium/heavy, Road) with dry GT clips. Same metrics as evalcore.evaluate.
from evalcore import *
import torch, sys, os, time, torchvision; torchvision.__version__ = "0.99.0"
import baselines_v2 as B
dev = "cuda"; JD = ROOT / "data" / "new_clips"
CLIPS = ["IMG_1393", "IMG_1394", "IMG_1395", "IMG_1396", "IMG_1397"]
class MF(torch.nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [torch.nn.Conv2d(18, c, 3, padding=1), torch.nn.ReLU(True)]
        for _ in range(n): L += [torch.nn.Conv2d(c, c, 3, padding=1), torch.nn.ReLU(True)]
        L += [torch.nn.Conv2d(c, 3, 3, padding=1)]; s.net = torch.nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
from mg_net import MG
ck = ROOT / "experiments_v2" / "checkpoints"
mf = MF().to(dev); mf.load_state_dict(torch.load(ck / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf.eval()
mg = MG().to(dev); mg.load_state_dict(torch.load(ck / "student_mg.pth", map_location=dev)["model_state_dict"]); mg.eval()
M = {}
for n in ["prenet_ft", "restormer"]:
    m, mult = B.LOADERS[n](); M[n] = (m.to(dev).eval(), mult); print("loaded", n, flush=True)
def base(n, x):
    m, mult = M[n]
    try:
        with torch.no_grad(): return np.clip(B.infer(m, mult, x), 0, 1)
    except Exception:
        torch.cuda.empty_cache()
        with torch.no_grad(): return np.clip(B.infer(m, mult, x, tile=256), 0, 1)
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
for _k in [k for k in sys.modules if k == "networks" or k.startswith("networks.")]: del sys.modules[_k]  # PReNet and S2VD both use a package called networks
from networks.derain_net import DerainNet
sv = DerainNet(n_features=32, n_resblocks=8).to(dev); sv.load_state_dict(torch.load(str(S2 / "model_states" / "derainer_rho05.pt"), map_location=dev)); sv.eval()
SRC = Path(__file__).parent / "asf_src"; sys.path.insert(0, str(SRC)); cwd = os.getcwd(); os.chdir(SRC)
from Network import TOFlow
asf = TOFlow().to(dev); sd = torch.load(str(SRC / "toflow_models_mine" / "denoising_best.pkl"), map_location=dev); asf.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False); asf.eval(); os.chdir(cwd)
@torch.no_grad()
def run_asf(fr):
    t = torch.from_numpy(np.stack([f[:, :, ::-1] for f in fr]).copy()).permute(0, 3, 1, 2)[None].to(dev)
    Hh = t.shape[-2]; ph = (32 - Hh % 32) % 32; t = torch.nn.functional.pad(t.flatten(0, 1), (0, 0, 0, ph), mode="reflect").view(1, 3, 3, Hh + ph, -1)
    o = asf(t)[0, :, :Hh].clamp(0, 1).permute(1, 2, 0).cpu().numpy(); return np.ascontiguousarray(o[:, :, ::-1])
def fn(c, i, x):
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; m9n = np.median(np.stack(fr), 0); win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
        o = {"MG-Net": mg(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy(), "mf7": mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()}
        t7 = torch.from_numpy(np.stack([getf(c, i + j) for j in range(-3, 4)])).permute(3, 0, 1, 2)[None].to(dev)
        o["S2VD"] = sv(t7)[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()[3]
    o["ASF-Net"] = run_asf([getf(c, i + j) for j in (-1, 0, 1)]); o["PReNet"] = base("prenet_ft", x); o["Restormer"] = base("restormer", x)
    o["Median9"] = m9n
    return o
def hf(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
def one(c, i):
    fr = [getf(c, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0)
    o = {"in": x}; o.update(fn(c, i, x)); nr = cv2.dilate((np.abs(x - m9).mean(2) > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0
    return o, nr
def scene(c, st, N):
    s = st[sl]; fl = {}; sh = {}
    for i in list(np.linspace(40, N - 41, 6).astype(int)):
        o1, nr = one(c, i); o2, _ = one(c, i + 1)
        for k in o1:
            fl.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s].mean())); sh.setdefault(k, []).append(hf(o1[k], nr))
    return {k: float(np.mean(v)) for k, v in fl.items()}, {k: float(np.mean(v)) for k, v in sh.items()}
def stat_mask(c, N):
    ms = [np.median(np.stack([getf(c, t + j) for j in range(-8, 9, 2)]), 0) for t in np.linspace(40, N - 41, 8).astype(int)]
    d = np.max(np.stack(ms), 0) - np.min(np.stack(ms), 0)
    return cv2.erode((d.max(2) < 0.04).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
R = {}
for cl in CLIPS:
    c = cv2.VideoCapture(str(JD / (cl + ".MOV"))); N = int(c.get(7)); st = stat_mask(c, N); print(cl, "static frac", float(st[sl].mean()), flush=True)
    f, sh = scene(c, st, N); fin = f["in"]; out = {"flicker_in": fin, "static_frac": float(st[sl].mean())}
    for k in f:
        if k == "in": continue
        out["flick_ratio_" + k] = f[k] / fin; out["sharp_nonrain_" + k] = sh[k] / sh["in"]
    R[cl] = out; print(cl, {k: round(v, 4) for k, v in out.items()}, flush=True)
json.dump(R, open(OUT / "new_village.json", "w"), indent=1); print("DONE")
