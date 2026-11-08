"""Cross-camera generalisation: a model trained on one camera only, tested on the other."""
import json, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
def loadmg(p):
    m = MG().to(dev); m.load_state_dict(torch.load(p, map_location=dev)["model_state_dict"]); return m.eval()
NET = {"MG_both_orig": loadmg(ck / "student_mg.pth")}
# matched reference: the xcam models use the STRONG target, so the fair both-camera
# reference is the strong-target model, not the original MG-Net.
if (ck / "student_mg_strong.pth").exists(): NET["MGs_both"] = loadmg(ck / "student_mg_strong.pth")
for tag, f in (("MG_cctv_only", "student_mg_xcam_cctv.pth"), ("MG_iphone_only", "student_mg_xcam_iphone.pth")):
    if (ck / f).exists(): NET[tag] = loadmg(ck / f)
    else: print("MISSING", f, flush=True)
print("models:", list(NET), flush=True)
METH = ["median9"] + list(NET)
def shp(a, mk): return float(np.abs(a - cv2.GaussianBlur(a, (0, 0), 1.5)).mean(2)[sl][mk[sl]].mean())
def motion_mask(x, m15):
    mv = (np.abs(x - m15).mean(2) > 0.05).astype(np.uint8)
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st_, _ = cv2.connectedComponentsWithStats(mv)
    big = np.isin(lab, [k for k in range(1, n) if st_[k, 4] > 120]).astype(np.uint8)
    return cv2.dilate(big, np.ones((11, 11), np.uint8)).astype(bool)
CA = {}
def gf(c, cid, i):
    k = (cid, int(i))
    if k not in CA:
        if len(CA) > 500: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]
def outs(c, cid, i):
    fr = [gf(c, cid, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0)
    o = {"in": x, "median9": m9}
    X = torch.from_numpy(np.concatenate(fr[2:7] + [x - m9], 2)).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    P = motion_mask(x, np.median(np.stack([gf(c, cid, i + j) for j in range(-21, 22, 6)]), 0))
    for k in NET: o[k] = np.where(P[..., None], x, o[k])
    return o
def drift(c, cid, i, n=16):
    a = cv2.cvtColor(gf(c, cid, i), cv2.COLOR_BGR2GRAY); b = cv2.cvtColor(gf(c, cid, i + n), cv2.COLOR_BGR2GRAY)
    (dx, dy), _ = cv2.phaseCorrelate(a, b); return float(np.hypot(dx, dy))
PL = ROOT / "data" / "new_place_video_rain"; NC = ROOT / "data" / "new_clips"
IPHONE = [("front_road", PL / "front_road_rain.MOV"), ("back_road", PL / "back_road_rain.MOV"), ("house", PL / "already_rain.MOV"),
          ("p4_1931", NC / "3 footage raining new place" / "IMG_1931.MOV"),
          ("p4_1933", NC / "3 footage raining new place" / "IMG_1933.MOV"),
          ("p4_1934", NC / "3 footage raining new place" / "IMG_1934.MOV"),
          ("iphone_light2", NC / "iphone_light rain2.MOV")]
CCTV_T = [("cctv_heavy", NC / "cctv_heavy_rain_1-11_1.mp4"), ("cctv_medium", NC / "cctv_medium_rain_1-11_4.mp4"), ("cctv_rain", NC / "cctv_rain_1-11_3.mp4")]
def measure(cid, p):
    if not p.exists(): print("missing", cid, flush=True); return None
    c = cv2.VideoCapture(str(p)); fps = c.get(5) or 30; N = int(c.get(7)); sec = N / fps
    ok = [i for i in [int(t * fps) for t in np.linspace(8, sec - 8, 24)] if drift(c, cid, i - 8) < 1.0][:8]
    if len(ok) < 3: print("skip", cid, "stable", len(ok), flush=True); return None
    st = smask(c, ok[0] / fps - 3, ok[0] / fps + 3); s_ = st[sl]; acc = {}
    for i in ok:
        o1, o2 = outs(c, cid, i), outs(c, cid, i + 1)
        for k in o1:
            acc.setdefault(k, []).append(float(np.abs(o1[k] - o2[k]).mean(2)[sl][s_].mean()))
            acc.setdefault("sh_" + k, []).append(shp(o1[k], st) / (shp(o1["in"], st) + 1e-9))
    CA.clear(); return {k: float(np.mean(v)) for k, v in acc.items()}
R = {}
for grp, lst in (("iphone", IPHONE), ("cctv", CCTV_T)):
    for cid, p in lst:
        m = measure(cid, p)
        if m: R[grp + "/" + cid] = m; print("done", grp, cid, flush=True)
json.dump(R, open(OUT / "xcam_eval2.json", "w"), indent=1)
def table(grp, note):
    ks = [k for k in R if k.startswith(grp + "/")]
    if not ks: return
    print("\n=== test on %s clips (%s) ===" % (grp.upper(), note))
    print("   %-16s %s" % ("method", "  ".join("%9s" % k.split("/")[1][:9] for k in ks) + "     MEAN"))
    for m in METH:
        v = [100 * (1 - R[k][m] / R[k]["in"]) for k in ks]
        print("   %-16s %s  %7.1f" % (m, "  ".join("%9.1f" % x for x in v), np.mean(v)))
    print("   %-16s %s" % ("sharpness:", "  ".join("%s %.3f" % (m, np.mean([R[k]["sh_" + m] for k in ks])) for m in METH)))
table("iphone", "MG_cctv_only has NEVER seen an iPhone frame")
table("cctv", "MG_iphone_only has NEVER seen a CCTV frame")
print("\n=== cross-camera gap (flicker removed %, vs the model trained on both) ===")
for grp, held in (("iphone", "MG_cctv_only"), ("cctv", "MG_iphone_only")):
    ks = [k for k in R if k.startswith(grp + "/")]
    if not ks or held not in NET: continue
    ref = "MGs_both" if "MGs_both" in NET else "MG_both_orig"
    a = np.mean([100 * (1 - R[k][ref] / R[k]["in"]) for k in ks])
    b = np.mean([100 * (1 - R[k][held] / R[k]["in"]) for k in ks])
    o = np.mean([100 * (1 - R[k]["MG_cctv_only" if held == "MG_iphone_only" else "MG_iphone_only"] / R[k]["in"]) for k in ks])
    md = np.mean([100 * (1 - R[k]["median9"] / R[k]["in"]) for k in ks])
    print("  %-7s  both-camera(%s) %.1f   held-out-camera %.1f   gap %+.1f   other-single-camera %.1f   (median9 %.1f)" % (grp, ref, a, b, b - a, o, md))
print("DONE")
