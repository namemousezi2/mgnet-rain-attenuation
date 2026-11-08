import torch, time, json, torch.nn as nn
from common import ROOT, OUT
from mg_net import MG
dev = "cuda"; H, W = 360, 640
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
mf7 = MF().to(dev); mf7.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mf7.pth", map_location=dev)["model_state_dict"]); mf7.eval()
mg = MG().to(dev); mg.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); mg.eval()
win = torch.rand(9, 3, H, W, device=dev)
def build(w): x = w[4]; md = w.median(0).values; return torch.cat([w[2], w[3], w[4], w[5], w[6], x - md], 0)[None]
def bench(f, n=100):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return n / (time.time() - t)
R = {}
with torch.no_grad():
    R["median_only_fps"] = bench(lambda: win.median(0).values)
    for amp in (False, True):
        k = "amp" if amp else "fp32"
        def full(m):
            def g():
                with torch.autocast("cuda", enabled=amp): return m(build(win))
            return g
        R["pipeline_mf7_" + k] = bench(full(mf7)); R["pipeline_mg_" + k] = bench(full(mg))
print(R); json.dump(R, open(OUT / "mg_speed.json", "w"), indent=1); print("DONE")
