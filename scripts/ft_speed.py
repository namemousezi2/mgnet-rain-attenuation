"""Throughput at 640x360 of the retrained baselines, measured the same way as Table 7:
frames already on the GPU, no video decoding, 100 timed forward passes after 10 warm-ups."""
import numpy as np, torch, time, sys, os, json
import torchvision; torchvision.__version__ = "0.99.0"
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"; Hh, Ww = 360, 640
def timeit(fn, n=100):
    for _ in range(10): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return n / (time.time() - t0)
R = {}
# S2VD and S2VD-ft : 7 frames in, [B,3,T,H,W]
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
from networks.derain_net import DerainNet
x7 = torch.rand(1, 3, 7, Hh, Ww, device=dev)
for tag, w in (("S2VD", S2 / "model_states" / "derainer_rho05.pt"), ("S2VDft", ck / "s2vd_ft.pth")):
    if not Path(w).exists(): print("skip", tag, "missing", w, flush=True); continue
    m = DerainNet(n_features=32, n_resblocks=8).to(dev); m.load_state_dict(torch.load(str(w), map_location=dev)); m.eval()
    with torch.no_grad(): R[tag] = timeit(lambda: m(x7))
    del m; torch.cuda.empty_cache()
# ASF and ASF-ft : 3 frames in, [B,T,C,H,W], height padded to a multiple of 32
SRC = Path(__file__).parent / "asf_src"; sys.path.insert(0, str(SRC)); cwd = os.getcwd(); os.chdir(SRC)
from Network import TOFlow
ph = (32 - Hh % 32) % 32
x3 = torch.rand(1, 3, 3, Hh + ph, Ww, device=dev)
for tag, w in (("ASF", SRC / "toflow_models_mine" / "denoising_best.pkl"), ("ASFft", ck / "asf_ft.pth")):
    if not Path(w).exists(): print("skip", tag, "missing", w, flush=True); continue
    sd = torch.load(str(w), map_location=dev)
    m = TOFlow().to(dev); m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}, strict=False); m.eval()
    with torch.no_grad(): R[tag] = timeit(lambda: m(x3), 30)
    del m; torch.cuda.empty_cache()
os.chdir(cwd)
print("\n=== fps at 640x360, network only, fp32 ===")
for k, v in R.items(): print("  %-8s %6.1f" % (k, v))
json.dump(R, open(OUT / "ft_speed.json", "w"), indent=1); print("DONE")
