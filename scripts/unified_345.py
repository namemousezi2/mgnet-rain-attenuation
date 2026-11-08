"""Tables 3-5 under the protocol of Tables 8-11: NWIN windows per scene, no motion
protection applied to any method, bootstrap CIs, three seeds for the strong target."""
import json, cv2, numpy as np, sys, os
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
HELD = ["cctv_heavy_rain_1-11_1.mp4", "cctv_medium_rain_1-11_4.mp4", "cctv_rain_1-11_3.mp4"]
NWIN = int(sys.argv[1]) if len(sys.argv) > 1 else 20
RNG = np.random.RandomState(0); NS = [3, 5, 9, 15, 21]; DRY_MAX = 0.5
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
import baselines_v2 as B
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
def loadmf(p):
    m = MF().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
mf7 = loadmf(ck / "student_mf7.pth"); mf6 = loadmf(ck / "student_mf6.pth")
NET = {"MG": loadmg(ck / "student_mg.pth"), "MGs_s1": loadmg(ck / "student_mg_strong.pth")}
for sd in (2, 3):
    p = ck / f"student_mg_strong_s{sd}.pth"
    if p.exists(): NET[f"MGs_s{sd}"] = loadmg(p)
BL = {}
for n in ["prenet_ft", "restormer"]:
    try: m, mult = B.LOADERS[n](); BL[n] = (m.to(dev).eval(), mult); print("loaded", n, flush=True)
    except Exception as e: print("skip", n, e, flush=True)
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
asf = TOFlow().to(dev); _sd = torch.load(str(SRC / "toflow_models_mine" / "denoising_best.pkl"), map_location=dev)
asf.load_state_dict({k.replace("module.", "", 1): v for k, v in _sd.items()}, strict=False); asf.eval(); os.chdir(cwd)
@torch.no_grad()
def run_asf(fr):
    t = torch.from_numpy(np.stack([f[:, :, ::-1] for f in fr]).copy()).permute(0, 3, 1, 2)[None].to(dev)
    Hh = t.shape[-2]; ph = (32 - Hh % 32) % 32
    t = torch.nn.functional.pad(t.flatten(0, 1), (0, 0, 0, ph), mode="reflect").view(1, 3, 3, Hh + ph, -1)
    return np.ascontiguousarray(asf(t)[0, :, :Hh].clamp(0, 1).permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
def repl(x, m, th):
    d = (x - m).mean(2); mk = (d > th / 255).astype(np.uint8); mk = mk & (1 - cv2.morphologyEx(mk, cv2.MORPH_OPEN, K5))
    mk = cv2.dilate(mk, K3).astype(bool); y = x.copy(); y[mk] = m[mk]; return y
def hf(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
CA = {}
def gf(c, cid, i):
    k = (cid, int(i))
    if k not in CA:
        if len(CA) > 500: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]
METH = ["med%d" % N for N in NS] + list(NET) + ["mf7", "mf6", "A4", "S2VD", "ASF", "PReNet", "Restormer"]
def outs(c, cid, i):
    fr = [gf(c, cid, i + j) for j in range(-8, 9, 2)]; x = fr[4]; o = {"in": x}
    for N in NS:
        h = (N - 1) // 2; o["med%d" % N] = np.median(np.stack([gf(c, cid, i + j) for j in range(-2 * h, 2 * h + 1, 2)]), 0)
    win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]
        for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf7"] = mf7(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf6"] = mf6(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        t7 = torch.from_numpy(np.stack([gf(c, cid, i + j) for j in range(-3, 4)])).permute(3, 0, 1, 2)[None].to(dev)
        o["S2VD"] = sv(t7)[0].clamp(0, 1).permute(1, 2, 3, 0).cpu().numpy()[3]
    o["A4"] = repl(x, o["med9"], 4)
    o["ASF"] = run_asf([gf(c, cid, i + j) for j in (-1, 0, 1)])
    if "prenet_ft" in BL: o["PReNet"] = basel("prenet_ft", x)
    if "restormer" in BL: o["Restormer"] = basel("restormer", x)
    o["_nonrain"] = (cv2.dilate((np.abs(x - o["med9"]).mean(2) > 0.02).astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)
    return o
def scene(cid, c, st, idxs, sharp_on_nonrain=False):
    s = st[sl]; per = {}
    for i in idxs:
        o1, o2 = outs(c, cid, i), outs(c, cid, i + 1)
        q = o1["_nonrain"] if sharp_on_nonrain else st
        for k in [k for k in o1 if k != "_nonrain"]:
            per.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s].mean()))
            per.setdefault("sh_" + k, []).append(hf(o1[k], q) / (hf(o1["in"], q) + 1e-9))
    CA.clear(); return {k: [float(x) for x in v] for k, v in per.items()}
def boot(v, n=2000):
    v = np.asarray(v); idx = RNG.randint(0, len(v), (n, len(v)))
    s = np.sort([100 * (1 - v[j].mean()) for j in idx]); return float(s[int(.025 * n)]), float(s[int(.975 * n)])
def report(tag, rain, dry):
    fr_, fd_ = np.mean(rain["in"]), np.mean(dry["in"]); rt = fd_ / fr_
    print(f"\n--- {tag}  flicker_in={fr_:.4f}  flicker_dry={fd_:.4f}  ratio={rt:.2f}" + ("" if rt <= DRY_MAX else "  [DRY REF TOO NOISY]"))
    for k in [m for m in METH if m in rain]:
        v = np.array(rain[k]) / fr_; lo, hi = boot(v); raw = 100 * (1 - v.mean())
        S = np.mean(rain[k]) / fr_; rho = np.mean(dry[k]) / fd_
        cor = 100 * (1 - (S * fr_ - rho * fd_) / (fr_ - fd_)) if rt <= DRY_MAX else float("nan")
        print(f"   {k:10s} raw {raw:6.1f} [{lo:6.1f},{hi:6.1f}]   corrected {cor:6.1f}   sharp {np.mean(rain['sh_'+k]):.3f}")
OUTJ = {}
# ===================== Table 3: natural rain (iPhone) =====================
print("\n================ TABLE 3: natural rain, iPhone ================", flush=True)
c = cv2.VideoCapture(str(D / "iphon_raining.MOV")); fps = c.get(5)
st = smask(c, 5 * fps, 10 * fps)
mid = scene("iph_mid", c, st, [int(t * fps) for t in np.linspace(55, 120, NWIN)])
hvy = scene("iph_hvy", c, st, [int(t * fps) for t in np.linspace(125, 240, NWIN)])
dry = scene("iph_dry", c, smask(c, 1.0 * fps, 2.3 * fps), [int(t * fps) for t in np.linspace(1.3, 1.8, NWIN)])
OUTJ["T3"] = {"mid": mid, "heavy": hvy, "dry": dry}
report("iPhone medium rain", mid, dry); report("iPhone heavy rain", hvy, dry)
print("\nTABLE 3 ROW (mean of medium and heavy, dry-corrected) | sharpness iPhone")
fr1, fr2, fd_ = np.mean(mid["in"]), np.mean(hvy["in"]), np.mean(dry["in"])
for k in [m for m in METH if m in mid]:
    v = []
    for sc, fr_ in ((mid, fr1), (hvy, fr2)):
        S = np.mean(sc[k]) / fr_; rho = np.mean(dry[k]) / fd_
        v.append(100 * (1 - (S * fr_ - rho * fd_) / (fr_ - fd_)))
    print(f"   {k:10s} removed {np.mean(v):6.1f}   sharp {np.mean([np.mean(mid['sh_'+k]), np.mean(hvy['sh_'+k])]):.3f}")
# sharpness on the three held-out CCTV clips
print("\nSharpness on the three held-out CCTV clips")
acc = {}
for nm in HELD:
    cc = cv2.VideoCapture(str(D / nm)); N = int(cc.get(7)); s2 = smask(cc, 0, N - 2)
    r = scene("cctv_" + nm, cc, s2, list(np.linspace(30, N - 31, max(6, NWIN // 2)).astype(int)))
    for k in [m for m in METH if m in r]: acc.setdefault(k, []).append(np.mean(r["sh_" + k]))
    print("  done", nm, flush=True)
for k in acc: print(f"   {k:10s} sharp_CCTV {np.mean(acc[k]):.3f}")
OUTJ["T3_cctv_sharp"] = {k: float(np.mean(v)) for k, v in acc.items()}
# ===================== Table 4: H01 artificial rain =====================
print("\n================ TABLE 4: H01 artificial rain ================", flush=True)
ND = ROOT / "data" / "new_clips" / "New rain GT and light medium and heavy"
dm, cd, Nd = med(ND / "IMG_1768.MOV")
sd0 = smask(cd, 0, Nd - 2)
dryH = scene("H01_dry", cd, sd0, list(np.linspace(40, Nd - 41, NWIN).astype(int)), True)
OUTJ["T4"] = {"dry": dryH}
for tag, lab in (("IMG_1772", "light"), ("IMG_1773", "medium"), ("IMG_1771", "medium_heavy"), ("IMG_1769", "heavy")):
    rm, cr, N = med(ND / (tag + ".MOV")); GT, w = align(rm, dm); stt = smask(cd, 0, Nd - 2, w)
    r = scene("H01_" + tag, cr, stt, list(np.linspace(40, N - 41, NWIN).astype(int)), True)
    OUTJ["T4"][lab] = r; report("H01 " + lab, r, dryH)
# ===================== Table 5: J01 field scene =====================
print("\n================ TABLE 5: J01 field scene (hose rain) ================", flush=True)
JD = ROOT / "data" / "new_clips" / "backyard jungle"
OUTJ["T5"] = {}; dryc = {}
for rain, gt in [("Jungle Light", "GT Jungle"), ("Jungle Medium", "GT Jungle"), ("Jungle Heavy", "GT Jungle"), ("ROAD RAIN", "GT ROAD")]:
    if gt not in dryc:
        dmj, cdj, Ndj = med(JD / (gt + ".MOV")); sdj = smask(cdj, 0, Ndj - 2)
        dryc[gt] = (dmj, cdj, Ndj, scene("J_" + gt, cdj, sdj, list(np.linspace(40, Ndj - 41, NWIN).astype(int)), True))
    dmj, cdj, Ndj, fdj = dryc[gt]
    rmj, crj, Nj = med(JD / (rain + ".MOV")); GTj, wj = align(rmj, dmj); stj = smask(cdj, 0, Ndj - 2, wj)
    r = scene("J_" + rain, crj, stj, list(np.linspace(40, Nj - 41, NWIN).astype(int)), True)
    OUTJ["T5"][rain] = r; report(rain, r, fdj)
json.dump({a: {b: {k: v for k, v in c_.items()} for b, c_ in d_.items()} if isinstance(d_, dict) and isinstance(list(d_.values())[0], dict) else d_ for a, d_ in OUTJ.items()}, open(OUT / "unified_345.json", "w"), indent=1)
print("\nDONE")
