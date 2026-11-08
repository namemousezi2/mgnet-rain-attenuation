"""Does 'restore first, then derain' beat 'derain only' on CCTV-like degraded rainy images?  (synthetic degradation of REAL rain, exact ground truth)
Input = SPA-Data real rainy image -> CCTV-like degradation D (Gaussian blur + sensor noise + JPEG; 3 levels). Ground truth = SPA clean image.
Pipelines (restorer R in {usm, nlm_usm, wiener(oracle sigma)}; derainer in {restormer, prenet_ft_nv, prenet}):
   input | R(x) | derain(x)  [A] | derain(R(x)) [B restore->derain] | R(derain(x)) [C derain->restore]
    python experiments_v2\\order_synth.py --n 100       (about 10-20 min)   -> results\\order_synth.csv / order_synth.json
"""
import argparse, numpy as np, pandas as pd, cv2, torch, sys
from common import OUT, imread01, psnr, ssim, save_json
import baselines_nv as BN
from eval_nv import spa_rows
LEVELS = {"clean": dict(sig=0.0, noise=0.0, q=100), "mild": dict(sig=0.8, noise=0.008, q=75), "medium": dict(sig=1.5, noise=0.018, q=50), "strong": dict(sig=2.5, noise=0.03, q=30)}


@torch.no_grad()
def infer(m, mult, bgr01):
    x = torch.from_numpy(np.ascontiguousarray(bgr01[..., ::-1].transpose(2, 0, 1)))[None].to(BN.dev); h, w = x.shape[2:]
    x = torch.nn.functional.pad(x, (0, (mult - w % mult) % mult, 0, (mult - h % mult) % mult), mode="reflect")
    try: y = m(x)
    except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); y = BN.tiled(m, x, 256, 32)
    y = y[0] if isinstance(y, (tuple, list)) else y
    y = y.float()[..., :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()[..., ::-1]
    return np.ascontiguousarray(y)


def degrade(x, sig, noise, q, rng):
    if sig == 0: return x
    y = cv2.GaussianBlur(x, (0, 0), sig) + rng.normal(0, noise, x.shape).astype(np.float32)
    ok, enc = cv2.imencode(".jpg", (np.clip(y, 0, 1) * 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q]); return cv2.imdecode(enc, 1).astype(np.float32) / 255


def usm(x, sig=1.5, amt=1.0): return np.clip(x + amt * (x - cv2.GaussianBlur(x, (0, 0), sig)), 0, 1)


def nlm_usm(x, sig=1.5):
    d = cv2.fastNlMeansDenoisingColored((x * 255).astype(np.uint8), None, 5, 5, 7, 21).astype(np.float32) / 255; return usm(d, sig, 0.8)


def wiener(x, sig, nsr=0.01):
    h, w = x.shape[:2]; fy = np.fft.fftfreq(h)[:, None]; fx = np.fft.fftfreq(w)[None]
    H = np.exp(-2 * (np.pi * sig) ** 2 * (fx ** 2 + fy ** 2)); G = H / (H ** 2 + nsr)
    return np.clip(np.stack([np.real(np.fft.ifft2(np.fft.fft2(x[..., c]) * G)) for c in range(3)], -1), 0, 1).astype(np.float32)


def load_ckpt(name):
    sys.path.insert(0, str(BN.TP / "PReNet")); from networks import PReNet
    m = PReNet(recurrent_iter=6, use_GPU=torch.cuda.is_available())
    m.load_state_dict(torch.load(BN.ROOT / "experiments_v2" / "checkpoints" / (name + ".pth"), map_location="cpu")["model_state_dict"]); return m, 1


import functools
for _n in ("prenet_ft_degaug", "prenet_ft_degaug_s1", "prenet_ft_degaug_s2", "prenet_ft_nv_s1", "prenet_ft_nv_s2"): BN.LOADERS[_n] = functools.partial(load_ckpt, _n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=100); ap.add_argument("--derainers", nargs="+", default=["restormer", "prenet_ft_nv", "prenet"])
    ap.add_argument("--no_restore", action="store_true"); ap.add_argument("--tile", type=int, default=0); a = ap.parse_args(); BN.TILE = a.tile
    rows = spa_rows(max(1, 1000 // a.n))[:a.n]; rng = np.random.default_rng(0); dev = BN.dev
    nets = {}
    for d in a.derainers:
        try: m, mult = BN.LOADERS[d](); nets[d] = (m.to(dev).eval(), mult)
        except Exception as e: print("!! cannot load", d, e)
    rec = []
    for i, r in enumerate(rows):
        x0 = imread01(r["curr"]); gt = imread01(r["clean"]); h = min(x0.shape[0], gt.shape[0]); w = min(x0.shape[1], gt.shape[1]); x0, gt = x0[:h, :w], gt[:h, :w]
        for lv, p in LEVELS.items():
            x = degrade(x0, p["sig"], p["noise"], p["q"], rng); R = {} if a.no_restore else dict(usm=lambda z: usm(z), nlm_usm=lambda z: nlm_usm(z), wiener=lambda z: wiener(z, p["sig"]))
            base = dict(img=i, level=lv); rec.append(dict(**base, derainer="-", pipeline="input", psnr=psnr(x, gt), ssim=ssim(x, gt)))
            for rn, rf in R.items(): y = rf(x); rec.append(dict(**base, derainer="-", pipeline=f"restore_only:{rn}", psnr=psnr(y, gt), ssim=ssim(y, gt)))
            for dn, (m, mult) in nets.items():
                A = infer(m, mult, x); rec.append(dict(**base, derainer=dn, pipeline="A_derain", psnr=psnr(A, gt), ssim=ssim(A, gt)))
                for rn, rf in R.items():
                    B = infer(m, mult, rf(x)); C = rf(A)
                    rec.append(dict(**base, derainer=dn, pipeline=f"B_{rn}_then_derain", psnr=psnr(B, gt), ssim=ssim(B, gt)))
                    rec.append(dict(**base, derainer=dn, pipeline=f"C_derain_then_{rn}", psnr=psnr(C, gt), ssim=ssim(C, gt)))
        if i % 10 == 0: print(i, "/", len(rows), flush=True)
    df = pd.DataFrame(rec); df.to_csv(OUT / "order_synth.csv", index=False); out = {}
    for (lv, dn, pl), g in df.groupby(["level", "derainer", "pipeline"]): out[f"{lv}|{dn}|{pl}"] = dict(psnr=float(g.psnr.mean()), ssim=float(g.ssim.mean()), n=len(g))
    for lv in LEVELS:                                           # paired gains over A_derain with bootstrap CI
        for dn in nets:
            A = df[(df.level == lv) & (df.derainer == dn) & (df.pipeline == "A_derain")].set_index("img").psnr
            for pl in df[(df.level == lv) & (df.derainer == dn)].pipeline.unique():
                if pl == "A_derain": continue
                d = (df[(df.level == lv) & (df.derainer == dn) & (df.pipeline == pl)].set_index("img").psnr - A).dropna().values
                bs = [rng.choice(d, len(d)).mean() for _ in range(1000)]
                out[f"GAIN_vs_A|{lv}|{dn}|{pl}"] = dict(mean_dB=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
                print(f"{lv:7s} {dn:13s} {pl:28s} gain vs A: {d.mean():+.2f} dB  CI [{np.percentile(bs,2.5):+.2f},{np.percentile(bs,97.5):+.2f}]", flush=True)
    save_json(out, "order_synth.json")
