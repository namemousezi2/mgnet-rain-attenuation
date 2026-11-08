"""Cost of the motion-protection step: MG-Net vs a plain median, with and without it.
Everything at 640x360 on the same GPU, frames already decoded and resized."""
import torch, time, json, numpy as np, cv2
from common import ROOT, OUT
from mg_net import MG
dev = "cuda"; H, W = 360, 640
mg = MG().to(dev); mg.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); mg.eval()
win9 = torch.rand(9, 3, H, W, device=dev)          # 9-frame window, stride 2
win15 = torch.rand(7, 3, H, W, device=dev)         # 7-frame window, stride 6 -> m15 for the motion mask
K5 = np.ones((5, 5), np.uint8); K9 = np.ones((9, 9), np.uint8); K11 = np.ones((11, 11), np.uint8)
def build(w):
    x = w[4]; md = w.median(0).values
    return torch.cat([w[2], w[3], w[4], w[5], w[6], x - md], 0)[None], md
def protection(x_t, m15_t):
    """Section 3.6: motion mask from the 15-frame median, on the CPU with OpenCV."""
    x = x_t.permute(1, 2, 0).cpu().numpy(); m = m15_t.permute(1, 2, 0).cpu().numpy()
    mv = (np.abs(x - m).mean(2) > 0.05).astype(np.uint8)
    mv = cv2.morphologyEx(mv, cv2.MORPH_OPEN, K5); mv = cv2.morphologyEx(mv, cv2.MORPH_CLOSE, K9)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mv)
    big = np.isin(lab, [k for k in range(1, n) if st[k, 4] > 120]).astype(np.uint8)
    return cv2.dilate(big, K11).astype(bool)
def bench(f, n=100):
    for _ in range(10): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return n / (time.time() - t)
R = {}
with torch.no_grad():
    def med9():
        return win9.median(0).values
    def med9_prot():
        m9 = win9.median(0).values; m15 = win15.median(0).values
        P = protection(win9[4], m15); y = m9.clone()
        Pt = torch.from_numpy(P).to(dev)[None]; return torch.where(Pt, win9[4], y)
    def mgnet():
        X, _ = build(win9); return mg(X)
    def mgnet_prot():
        X, _ = build(win9); y = mg(X)[0]; m15 = win15.median(0).values
        P = protection(win9[4], m15); Pt = torch.from_numpy(P).to(dev)[None]
        return torch.where(Pt, win9[4], y)
    for k, f in (("median9", med9), ("median9_plus_protection", med9_prot),
                 ("MGNet", mgnet), ("MGNet_plus_protection", mgnet_prot)):
        R[k] = bench(f); print("%-26s %7.1f fps  (%.2f ms)" % (k, R[k], 1000 / R[k]), flush=True)
R["protection_overhead_ms"] = 1000 / R["median9_plus_protection"] - 1000 / R["median9"]
print("\nprotection step costs %.2f ms/frame" % R["protection_overhead_ms"])
print("median9+protection is %.2fx slower than MG-Net alone" % (R["MGNet"] / R["median9_plus_protection"]))
json.dump(R, open(OUT / "prot_speed.json", "w"), indent=1); print("DONE")
