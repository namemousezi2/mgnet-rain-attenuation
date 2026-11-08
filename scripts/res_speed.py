"""Throughput of the MG-Net pipeline at three surveillance resolutions.

Table 7 reports 640x360 because every quality measurement in the paper is made there. A referee may
reasonably ask whether the method is still real time on the cameras people actually install, so this
script times the same pipeline at 720p and 1080p: the nine-frame median on the GPU, the 18-channel
stack, and the network, with frames already resident on the GPU and no video decoding."""
import numpy as np, torch, time, json
from pathlib import Path
exec(open(str(Path(__file__).parent / "cctv_stA.py")).read().split("HELD =")[0])
from mg_net import MG
dev = "cuda"; ck = ROOT / "experiments_v2" / "checkpoints"
m = MG().to(dev); m.load_state_dict(torch.load(ck / "student_mg.pth", map_location=dev)["model_state_dict"]); m.eval()
RES = [(640, 360, "640x360"), (1280, 720, "1280x720 (720p)"), (1920, 1080, "1920x1080 (1080p)")]
def timeit(fn, n):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return n / (time.time() - t0)
R = {}
for W_, H_, tag in RES:
    win = torch.rand(9, 3, H_, W_, device=dev)          # the 9-frame window already on the GPU
    def pipeline():
        med = win.median(0).values
        x = win[4]
        X = torch.cat([win[2], win[3], x, win[5], win[6], x - med], 0)[None]
        return m(X)
    def net_only(X=torch.rand(1, 18, H_, W_, device=dev)):
        return m(X)
    n = 100 if H_ <= 720 else 40
    with torch.no_grad():
        try:
            R[tag] = {"pipeline_fp32": timeit(pipeline, n), "network_fp32": timeit(net_only, n)}
            with torch.autocast("cuda"):
                R[tag]["pipeline_amp"] = timeit(pipeline, n)
        except torch.cuda.OutOfMemoryError:
            R[tag] = {"error": "out of memory"}; torch.cuda.empty_cache(); continue
    torch.cuda.empty_cache()
    print(tag, {k: round(v, 1) for k, v in R[tag].items()}, flush=True)
print("\n=== MG-Net throughput by resolution (fps) ===")
print("  %-20s %-16s %-16s %-16s" % ("resolution", "pipeline fp32", "pipeline amp", "network only"))
for _, _, tag in RES:
    r = R.get(tag, {})
    if "error" in r: print("  %-20s %s" % (tag, r["error"])); continue
    print("  %-20s %-16.1f %-16.1f %-16.1f" % (tag, r["pipeline_fp32"], r["pipeline_amp"], r["network_fp32"]))
json.dump(R, open(OUT / "res_speed.json", "w"), indent=1); print("DONE")
