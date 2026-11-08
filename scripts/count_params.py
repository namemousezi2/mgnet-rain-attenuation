import sys, os, torch, torchvision; torchvision.__version__ = "0.99.0"
from pathlib import Path
import baselines_v2 as B
dev = "cuda" if torch.cuda.is_available() else "cpu"
def n(m): return sum(p.numel() for p in m.parameters())
out = []
for k in B.LOADERS:
    try:
        m, _ = B.LOADERS[k](); out.append((k, n(m)))
    except Exception as e: out.append((k, "error: " + str(e)[:80]))
S2 = Path(__file__).parent / "s2vd_src"; sys.path.insert(0, str(S2))
try:
    from networks.derain_net import DerainNet
    out.append(("S2VD", n(DerainNet(n_features=32, n_resblocks=8))))
except Exception as e: out.append(("S2VD", "error: " + str(e)[:80]))
with open("params_log.txt", "w") as f:
    for k, v in out:
        s = f"{k}: {v:,}" if isinstance(v, int) else f"{k}: {v}"; print(s); f.write(s + "\n")
