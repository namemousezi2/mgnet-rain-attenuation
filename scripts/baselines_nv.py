"""Run published baselines with their OFFICIAL code + OFFICIAL pretrained weights on exactly
the same test data, and time them on the same GPU. (Zero-shot: no retraining; state this in the paper.)

Setup once (see README_TH.md):
  experiments_v2/third_party/Restormer   git clone https://github.com/swz30/Restormer
      weights: Deraining/pretrained_models/deraining.pth
  experiments_v2/third_party/PromptIR    git clone https://github.com/va1shn9v/PromptIR
      weights: ckpt/model.ckpt
  experiments_v2/third_party/PReNet      git clone https://github.com/csdwren/PReNet
      weights are inside the repo: logs/Rain100H/PReNet6/net_latest.pth and logs/real/PReNet1.pth

    python experiments_v2/baselines_v2.py --methods prenet prenet_real restormer
"""
import argparse, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from common import ROOT, OUT, imread01, psnr, ssim, psnr_y, save_json, CLASSES
from eval_v2 import spa_rows, qual_indices, QDIR  # (functions do not depend on the split dir)
import cv2

TP = ROOT / "experiments_v2" / "third_party"; SPL = ROOT / "experiments_v2" / "splits_nv"
dev = "cuda" if torch.cuda.is_available() else "cpu"
TILE = 0; HALF = False


def _assemble(path):
    """Rebuild a large weights file that was copied in parts (<name>.part0, .part1, ...) and check its SHA-256."""
    if path.exists(): return
    parts = sorted(path.parent.glob(path.name + ".part*"), key=lambda p: int(p.suffix[5:]))
    if not parts: raise FileNotFoundError(path)
    import hashlib
    with open(path, "wb") as f:
        for p in parts: f.write(p.read_bytes())
    ref = path.with_name(path.name + ".sha256")
    if ref.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != ref.read_text().strip():
        path.unlink(); raise RuntimeError("reassembled weights failed the SHA-256 check")
    print("  assembled", path)


def load_restormer():
    _assemble(TP / "Restormer/Deraining/pretrained_models/deraining.pth")
    sys.path.insert(0, str(TP / "Restormer" / "basicsr" / "models" / "archs"))
    from restormer_arch import Restormer          # needs: pip install einops
    m = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                  heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
                  dual_pixel_task=False)
    m.load_state_dict(torch.load(TP / "Restormer/Deraining/pretrained_models/deraining.pth", map_location="cpu")["params"])
    return m, 8


def load_promptir():
    sys.path.insert(0, str(TP / "PromptIR"))
    from net.model import PromptIR
    m = PromptIR(decoder=True)
    sd = torch.load(TP / "PromptIR/ckpt/model.ckpt", map_location="cpu")
    sd = sd.get("state_dict", sd); sd = {k.replace("net.", "", 1): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m, 16


def _prenet(rel, iters=6):
    sys.path.insert(0, str(TP / "PReNet"))
    from networks import PReNet
    m = PReNet(recurrent_iter=iters, use_GPU=torch.cuda.is_available())
    sd = torch.load(TP / "PReNet" / rel, map_location="cpu")
    m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    return m, 1


def load_prenet():        # official weights trained on Rain100H (synthetic)
    return _prenet("logs/Rain100H/PReNet6/net_latest.pth")


def load_prenet_real():   # official weights released for real rainy images
    f = OUT / "prenet_real_iter.txt"          # written by diag_prenet_real.py (best number of recurrent iterations)
    return _prenet("logs/real/PReNet1.pth", int(f.read_text()) if f.exists() else 6)


def load_prenet_ft():     # PReNet re-trained by us on exactly the same data/schedule as our models (train_prenet.py)
    sys.path.insert(0, str(TP / "PReNet"))
    from networks import PReNet
    m = PReNet(recurrent_iter=6, use_GPU=torch.cuda.is_available())
    m.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "prenet_ft_nv.pth", map_location="cpu")["model_state_dict"])
    return m, 1


LOADERS = dict(restormer=load_restormer, promptir=load_promptir, prenet=load_prenet, prenet_real=load_prenet_real, prenet_ft_nv=load_prenet_ft)


@torch.no_grad()
def infer(m, mult, bgr01, tile=None):
    x = torch.from_numpy(np.ascontiguousarray(bgr01[..., ::-1].transpose(2, 0, 1)))[None].to(dev)  # -> RGB
    h, w = x.shape[2:]; ph, pw = (mult - h % mult) % mult, (mult - w % mult) % mult
    x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast("cuda", dtype=torch.float16, enabled=HALF):
        if TILE:
            y = tiled(m, x, TILE, 32).float()
        else:
            try:
                y = m(x).float()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); y = tiled(m, x, 256, 32).float()
    y = y[0] if isinstance(y, (tuple, list)) else y
    y = y.float()[..., :h, :w].clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()[..., ::-1]
    return np.ascontiguousarray(y)


def tiled(m, x, t, o):
    _, _, H, W = x.shape; out = torch.zeros_like(x); wgt = torch.zeros_like(x)
    for i in list(range(0, H - t, t - o)) + [H - t]:
        for j in list(range(0, W - t, t - o)) + [W - t]:
            y = m(x[..., i:i + t, j:j + t]); y = (y[0] if isinstance(y, (tuple, list)) else y).float()
            out[..., i:i + t, j:j + t] += y; wgt[..., i:i + t, j:j + t] += 1
    return out / wgt


def evaluate(name, m, mult, rows, tag, q):
    rec = []
    t0 = time.time()
    for k, r in enumerate(rows):
        if k % 50 == 0: print(f"  {tag} {name} {k}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
        x = imread01(r["curr"]); gt = imread01(r["clean"]); h = min(x.shape[0], gt.shape[0]); w = min(x.shape[1], gt.shape[1])
        x, gt = x[:h, :w], gt[:h, :w]; y = infer(m, mult, x)
        rec.append(dict(idx=k, kind=r.get("kind", "spa"), label=r.get("label", -1), psnr=psnr(y, gt), ssim=ssim(y, gt),
                        psnr_y=psnr_y(y, gt), in_psnr=psnr(x, gt), in_ssim=ssim(x, gt)))
        if k in q: cv2.imwrite(str(QDIR / f"{tag}_{k:04d}_{name}.png"), (y * 255).round().astype(np.uint8))
    df = pd.DataFrame(rec); df.to_csv(OUT / f"{tag}_{name}.csv", index=False); return df


@torch.no_grad()
def speed_one(m, H, W, half):
    mm = m.float()
    x = torch.rand(1, 3, H, W, device=dev)
    ac = torch.autocast("cuda", dtype=torch.float16, enabled=half)
    try:
        if TILE: raise torch.cuda.OutOfMemoryError("forced tiling")
        with ac:
            for _ in range(5): mm(x)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(20): mm(x)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 20 * 1000; mode = "full-frame"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); torch.cuda.synchronize(); t0 = time.perf_counter()
        with ac:
            for _ in range(3): tiled(mm, x, 256, 32)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) / 3 * 1000; mode = "tiled-256" + ("" if TILE else " (OOM at full frame)")
    m.float(); torch.cuda.empty_cache()
    return dict(ms=ms, fps=1000 / ms, mode=mode)


def speed(m, mult):
    """Network-only latency (no pre/post-processing) at 640x360 and 1280x720, FP32 and FP16."""
    res = dict(params_M=sum(p.numel() for p in m.parameters()) / 1e6)
    for (W, H) in ((640, 360), (1280, 720)):
        Hp, Wp = H + (mult - H % mult) % mult, W + (mult - W % mult) % mult
        for half in (False, True):
            key = f"{W}x{H}_{'fp16' if half else 'fp32'}"
            try: res[key] = speed_one(m, Hp, Wp, half)
            except Exception as e: res[key] = dict(error=str(e)[:200])
            print(" ", key, res[key])
    return res


def main(a):
    test = pd.read_csv(SPL / "derain_test.csv")
    if a.quick: test = test.iloc[::5].reset_index(drop=True)
    rows = test.to_dict("records"); q = qual_indices(test); res = {}
    for name in a.methods:
        try:
            m, mult = LOADERS[name]()
        except Exception as e:
            print(f"!! {name}: could not load ({e}) -- see README_TH.md"); continue
        m = m.to(dev).eval(); print("==", name)
        from eval_nv import summarize
        if a.speed_only:
            d1 = pd.read_csv(OUT / f"derain_test_{name}.csv"); d2 = pd.read_csv(OUT / f"spa_test_{name}.csv")
        else:
            d1 = evaluate(name, m, mult, rows, "derain_test", q)
            d2 = evaluate(name, m, mult, spa_rows(5 if a.quick else 1), "spa_test", set(range(0, 1000, 100)))
        res[name] = dict(grouped_test=summarize(d1), spa_real_test=summarize(d2), speed=speed(m, mult))
        print(json.dumps(res[name], indent=1, default=float)); del m; torch.cuda.empty_cache()
    out = OUT / "summary_baselines.json"
    old = json.loads(out.read_text()) if out.exists() else {}
    old.update(json.loads(json.dumps(res, default=float)))
    save_json(old, "summary_baselines.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--methods", nargs="+", default=["prenet", "prenet_real", "restormer"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tile", type=int, default=0, help="force tiled inference with this tile size (e.g. 256) to avoid VRAM overflow")
    ap.add_argument("--speed_only", action="store_true", help="reuse the saved CSVs, only measure speed and write summary_baselines.json")
    ap.add_argument("--half", action="store_true", help="FP16 autocast for evaluation")
    a = ap.parse_args(); TILE = a.tile; HALF = a.half; main(a)
