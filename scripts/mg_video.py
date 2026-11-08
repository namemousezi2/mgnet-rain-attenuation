import cv2, numpy as np, json, time, torch, torch.nn as nn
from common import ROOT, OUT
dev = "cuda"; W, H = 640, 360; D = ROOT / "data" / "new_clips"
class MF(nn.Module):
    def __init__(s, c=64, n=7):
        super().__init__(); L = [nn.Conv2d(18, c, 3, padding=1), nn.ReLU(True)]
        for _ in range(n): L += [nn.Conv2d(c, c, 3, padding=1), nn.ReLU(True)]
        L += [nn.Conv2d(c, 3, 3, padding=1)]; s.net = nn.Sequential(*L)
    def forward(s, X): return X[:, 6:9] + s.net(X)
def L(n):
    q = MF().to(dev); q.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / n, map_location=dev)["model_state_dict"]); return q.eval()
from mg_net import MG
m6 = MG().to(dev); m6.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); m6.eval(); m = L("student_mf7.pth")
def getf(c, i):
    c.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = c.read(); return cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
@torch.no_grad()
def step(win, amp=False, mm=None):
    mm = mm or m  # win: (9,3,H,W) float on gpu
    x = win[4]; md = win.median(0).values
    X = torch.cat([win[2], win[3], win[4], win[5], win[6], x - md], 0)[None]
    with torch.autocast("cuda", enabled=amp): y = mm(X)
    return y[0].float().clamp(0, 1)
jobs = {"iphon_raining_heavy": ("iphon_raining.MOV", 130), "cctv_heavy_rain_1": ("cctv_heavy_rain_1-11_1.mp4", 1.0), "iphone_light_rain2": ("iphone_light rain2.MOV", 1.0)}
res = {}
for tag, (fn, st) in jobs.items():
    c = cv2.VideoCapture(str(D / fn)); fps = c.get(5) or 30; N = int(c.get(7)); s0 = int(st * fps) if st > 5 else 10; n = min(150, N - s0 - 12)
    fr = [getf(c, s0 + k) for k in range(-8, n + 8)]; T = [torch.from_numpy(f[:, :, ::1].astype(np.float32) / 255).permute(2, 0, 1).to(dev) for f in fr]
    vw = cv2.VideoWriter(str(OUT / ("mg_demo_" + tag + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 4, H))
    for k in range(8, 8 + n):
        win = torch.stack([T[k + j] for j in range(-8, 9, 2)]); y = step(win); y6 = step(win, False, m6); x = T[k]
        tonp = lambda t: (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lab = lambda im, t: cv2.putText(np.ascontiguousarray(im), t, (8, 24), 0, 0.7, (0, 255, 255), 2)
        vw.write(np.hstack([lab(tonp(x), "input"), lab(tonp(y6), "MG-Net"), lab(tonp(y), "mf7 (stronger)"), lab(tonp((y - x).abs().mul(8).clamp(0, 1)), "change x8")]))
    vw.release()
    wins = [torch.stack([T[k + j] for j in range(-8, 9, 2)]) for k in range(8, 8 + 60)]
    for amp in (False, True):
        for w in wins[:10]: step(w, amp)
        torch.cuda.synchronize(); t0 = time.time()
        for w in wins: step(w, amp)
        torch.cuda.synchronize(); res.setdefault(tag, {})["fps_amp" if amp else "fps_fp32"] = 60 / (time.time() - t0)
    print(tag, res[tag], flush=True)
json.dump(res, open(OUT / "mg_demo.json", "w"), indent=1); print("DONE")
