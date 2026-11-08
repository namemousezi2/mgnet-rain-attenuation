"""Does attenuating rain buy anything measurable? A storage and bandwidth test.

The motion-trigger test (motion_trigger.py) found no benefit, because the triggers on our
footage come from wind-moved vegetation rather than from rain. This test measures a different
consequence of rain that is not a matter of judgement: rain changes almost every pixel in
almost every frame, so a video encoder has to spend bits on it. A surveillance site that
records continuously pays for those bits in disk and uplink.

The primary measurement needs no encoder installed. For a quantisation step q, both frames of
each consecutive pair are quantised to multiples of q, and we report the zeroth-order entropy
of their difference in bits per pixel. Rate-distortion theory makes this a bound on what an
inter-frame codec must spend on that pair at that quality; it is not a bitrate, because a real
encoder also models spatial context, and it is reported as a bound and compared only between
methods at the same q.

Quantising first is not cosmetic. A difference taken at full precision is dominated by one-LSB
dither: in the rainy input, static background pixels are frequently bit-identical between
frames, while any method that writes a small residual moves them by one level, so an
unquantised proxy reports MORE bits for a visibly steadier output. Any real encoder discards
that dither at its first quantisation step, and so does this measurement.

Reported for q = 4, 8 and 16, plus H.264 (libx264) at CRF 23 and 28 when an ffmpeg binary
happens to be available. The script begins with a synthetic self-test, printed into the log, on
two sequences it builds itself: one with sparse moving bright streaks and one that is static
apart from one-LSB dither. The measure must report the second as cheaper. It does so from q = 8
upward; at q = 1 it reports it as 149% more expensive, and at q = 4 the dither still partly
survives, so q = 8 and q = 16 are the columns to read.

This axis is read together with the kept-rate, not alone: a long median also removes rain
from the residual, and it removes the moving objects with it. The pairing of the two is the
result; neither number on its own is.
"""
import json, shutil, subprocess, sys, tempfile, os, cv2, numpy as np
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
import torch, torch.nn as nn, torchvision; torchvision.__version__ = "0.99.0"
from mg_net import MG

dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
NFR = int(sys.argv[1]) if len(sys.argv) > 1 else 120
NS = [3, 9, 21]
CRFS = [23, 28]
QS = [4, 8, 16]                 # quantisation steps for the entropy bound

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
METH = ["input"] + ["med%d" % n for n in NS] + list(NET) + ["mf7"]

def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p: return p, "PATH"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe(), "imageio_ffmpeg"
    except Exception:
        return None, None
FFM, FFSRC = find_ffmpeg()
print("ffmpeg:", FFM or "NOT FOUND (CRF numbers will be skipped)", "via", FFSRC, flush=True)
if FFM is None:
    print("  to get the real H.264 numbers: pip install imageio-ffmpeg, then run this again", flush=True)

def selftest():
    rng = np.random.default_rng(0)
    bg = rng.random((90, 120, 3)).astype(np.float32) * 0.5 + 0.25
    rainy, steady = [], []
    for _ in range(12):
        r = bg.copy(); m = rng.random((90, 120)) < 0.03
        r[m] = np.clip(r[m] + 0.35, 0, 1); rainy.append(r)
        steady.append(bg + (rng.random((90, 120, 3)).astype(np.float32) - 0.5) * (1 / 255))
    print("self-test: streaks that move vs static frames with +-1 LSB dither", flush=True)
    out = {}
    for q in [1] + QS:
        a, b = residual_bits(rainy, q), residual_bits(steady, q)
        out[q] = 100 * (1 - b / a)
        print("  q%-3d  streaky %9.0f  steady %9.0f  steady is %+6.1f%% cheaper"
              % (q, a, b, out[q]), flush=True)
    return out

CA = {}
def gf(c, cid, i):
    k = (cid, int(i))
    if k not in CA:
        if len(CA) > 300: CA.clear()
        CA[k] = getf(c, i)
    return CA[k]

def outputs(c, cid, i):
    f21 = [gf(c, cid, i + j) for j in range(-20, 21, 2)]
    x = f21[10]; o = {"input": x}
    for n in NS:
        h = (n - 1) // 2; o["med%d" % n] = np.median(np.stack(f21[10 - h:10 + h + 1]), 0)
    w9 = torch.from_numpy(np.stack(f21[6:15])).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        X = torch.cat([w9[2], w9[3], w9[4], w9[5], w9[6], w9[4] - w9.median(0).values], 0)[None]
        for k, m_ in NET.items(): o[k] = m_(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        o["mf7"] = mf(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return o

def u8(f): return (np.clip(f, 0, 1) * 255).astype(np.uint8)

def residual_bits(frames, q):
    """zeroth-order entropy of the quantised frame-to-frame difference, in bits per frame"""
    tot = 0.0
    for a, b in zip(frames[:-1], frames[1:]):
        A = (u8(a).astype(np.int16) // q) * q
        B = (u8(b).astype(np.int16) // q) * q
        d = (B - A + 255).ravel()
        h = np.bincount(d, minlength=511).astype(np.float64)
        pr = h / h.sum(); pr = pr[pr > 0]
        tot += float(-(pr * np.log2(pr)).sum()) * d.size
    return tot / max(len(frames) - 1, 1)

def residual_bytes(frames, q=1):
    """lossless PNG size of the signed frame-to-frame difference, summed over the clip.

    q > 1 quantises both inputs to multiples of q first. This matters: a plain difference is
    dominated by one-LSB dither. In the rainy input, static background pixels are often
    bit-identical between frames, which PNG stores almost for free, while any method that
    writes a small residual moves them by one level and breaks those runs, so the raw proxy
    reports MORE bytes for a visibly steadier output. A real encoder quantises that dither
    away, and q = 8 imitates that step; the q = 1 column is kept only to show the artefact."""
    tot = 0
    for a, b in zip(frames[:-1], frames[1:]):
        A, B = u8(a).astype(np.int16), u8(b).astype(np.int16)
        if q > 1: A = (A // q) * q; B = (B // q) * q
        d = (B - A + 255) // 2                                               # centre at 127, keep 8 bit
        ok, buf = cv2.imencode(".png", d.astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        tot += int(buf.nbytes) if ok else 0
    return tot

def x264_bytes(frames, crf):
    if FFM is None: return None
    td = tempfile.mkdtemp()
    try:
        for n, f in enumerate(frames): cv2.imwrite(os.path.join(td, "%05d.png" % n), u8(f))
        out = os.path.join(td, "o.mp4")
        cmd = [FFM, "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(td, "%05d.png"),
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", out]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print("    ffmpeg failed:", r.stderr.decode(errors="replace")[:200], flush=True); return None
        return os.path.getsize(out)
    finally:
        shutil.rmtree(td, ignore_errors=True)

def run_clip(path, cid):
    c = cv2.VideoCapture(str(path)); N = int(c.get(7))
    if N < NFR + 60: print("  too short", cid, flush=True); return None
    start = max(24, (N - NFR) // 2)
    seq = {k: [] for k in METH}
    for i in range(start, start + NFR):
        o = outputs(c, cid, i)
        for k in METH: seq[k].append(o[k])
    CA.clear()
    r = {}
    for k in METH:
        e = {"bits_q%d" % q: residual_bits(seq[k], q) for q in QS}
        e["residual_bytes"] = residual_bytes(seq[k])          # kept only to show the dither artefact
        for crf in CRFS: e["crf%d_bytes" % crf] = x264_bytes(seq[k], crf)
        r[k] = e
        print("  %-7s" % k, "  ".join("q%-2d %9.0f bits/frame" % (q, e["bits_q%d" % q]) for q in QS),
              "  ".join(" crf%d %8s B" % (crf, e["crf%d_bytes" % crf]) for crf in CRFS), flush=True)
    return r

PL = ROOT / "data" / "new_place_video_rain"; NC = ROOT / "data" / "new_clips"
ST = selftest()
CLIPS = [("house_rain", PL / "already_rain.MOV"),
         ("p4_1931", NC / "3 footage raining new place" / "IMG_1931.MOV"),
         ("p4_1933", NC / "3 footage raining new place" / "IMG_1933.MOV"),
         ("p4_1934", NC / "3 footage raining new place" / "IMG_1934.MOV"),
         ("shop1", D / "cctv_shop1.mp4"), ("shop2", D / "cctv_shop2.mp4"), ("shop3", D / "cctv_shop3.mp4"),
         ("shop_before", D / "cctv_shop_before+rain.mp4"), ("shop_after", D / "cctv_shop_after+rain.mp4")]

RES = {}
for cid, p in CLIPS:
    if not p.exists(): print("missing", cid, p, flush=True); continue
    print(cid, flush=True)
    r = run_clip(p, cid)
    if r: RES[cid] = r

def pct(cid, k, field):
    a = RES[cid]["input"][field]; b = RES[cid][k][field]
    if a in (None, 0) or b is None: return None
    return 100 * (1 - b / a)

print("\n=== bits saved relative to the rainy input (mean over %d clips) ===" % len(RES))
print("    q4/q8/q16 are the entropy bound at that quantisation step; residual is the unquantised")
print("    PNG proxy, kept only to show that it inverts on one-LSB dither and must not be used.")
summary = {}
fields = ["bits_q%d" % q for q in QS] + ["residual_bytes"] + ["crf%d_bytes" % c for c in CRFS]
for k in METH:
    if k == "input": continue
    e = {}
    for fl in fields:
        v = [pct(c, k, fl) for c in RES]; v = [x for x in v if x is not None]
        e[fl] = float(np.mean(v)) if v else None
    summary[k] = e
    print("  %-7s" % k, "  ".join("%s %s" % (fl.replace("_bytes", ""),
          ("%+5.1f%%" % e[fl]) if e[fl] is not None else "n/a") for fl in fields), flush=True)

json.dump({"n_frames": NFR, "ffmpeg": FFM, "crfs": CRFS, "selftest_pct_cheaper": ST, "per_clip": RES, "summary_pct_saved": summary,
           "quant_steps": QS,
           "note": "bits_q* is the zeroth-order entropy of the frame-to-frame difference after "
                   "quantising both frames to multiples of q: a bound on inter-frame rate at that "
                   "quality, not a bitrate. residual_bytes is an unquantised PNG proxy and is "
                   "dominated by one-LSB dither; it is reported only to document that. Read any of "
                   "these with the kept-rate: a long median saves the most bits because it also "
                   "removes the moving objects."},
          open(OUT / "bitrate_eval.json", "w"), indent=1)
print("\nwrote", OUT / "bitrate_eval.json")
print("DONE")
