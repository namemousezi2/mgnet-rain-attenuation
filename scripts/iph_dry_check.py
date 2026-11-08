# Re-measure the dry-clip response of iphon_raining with the first seconds (user: rain-free) instead of 8.3-9.7 s.
from evalcore import *
import torch
dev = "cuda"
from mg_net import MG
mg = MG().to(dev); mg.load_state_dict(torch.load(ROOT / "experiments_v2" / "checkpoints" / "student_mg.pth", map_location=dev)["model_state_dict"]); mg.eval()
c = cv2.VideoCapture(str(D / "iphon_raining.MOV")); fps = c.get(5); N = int(c.get(7))
def seg(tag, mask_from, mask_to, ts):
    st = smask(c, mask_from * fps, mask_to * fps)[sl]; fi, fo, fm = [], [], []
    for t in ts:
        res = []
        for i in (int(t * fps), int(t * fps) + 1):
            fr = [getf(c, i + j) for j in range(-8, 9, 2)]; x = fr[4]; m9 = np.median(np.stack(fr), 0)
            win = torch.from_numpy(np.stack(fr)).permute(0, 3, 1, 2).to(dev)
            with torch.no_grad(): X = torch.cat([win[2], win[3], win[4], win[5], win[6], win[4] - win.median(0).values], 0)[None]; y = mg(X)[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            res.append((x, y, m9))
        d = lambda k: float(np.abs(res[0][k] - res[1][k]).mean(2)[sl][st].mean())
        fi.append(d(0)); fo.append(d(1)); fm.append(d(2))
    print(tag, "flicker_in", np.mean(fi), "MG ratio", np.mean(fo) / np.mean(fi), "median ratio", np.mean(fm) / np.mean(fi), "static frac", float(st.mean()), flush=True)
seg("DRY_first_1.3to1.8s", 1.0, 2.3, np.linspace(1.3, 1.8, 6))
seg("DRY_old_8.3to9.7s", 5.0, 7.5, np.linspace(8.3, 9.7, 6))
print("DONE")
