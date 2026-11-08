import torch, torch.nn as nn, torch.nn.functional as F
def cb(i, o, s=1, d=1): return nn.Sequential(nn.Conv2d(i, o, 3, s, padding=d, dilation=d), nn.ReLU(True))
class MG(nn.Module):
    """Mask-gated median blend: out = x + a*(m9-x) + 0.05*tanh(r). a in [0,1] is a learned streak gate (U-Net, half/quarter-res context), so output is a convex mix of input and temporal median (sharp by construction) plus a small bounded residual."""
    def __init__(s, c=(24, 32, 48)):
        super().__init__()
        s.e0 = nn.Sequential(cb(18, c[0]), cb(c[0], c[0])); s.e1 = nn.Sequential(cb(c[0], c[1], 2), cb(c[1], c[1])); s.e2 = nn.Sequential(cb(c[1], c[2], 2), cb(c[2], c[2], 1, 2), cb(c[2], c[2], 1, 4))
        s.d1 = nn.Sequential(cb(c[2] + c[1], c[1]), cb(c[1], c[1])); s.d0 = nn.Sequential(cb(c[1] + c[0], c[0]), cb(c[0], c[0])); s.head = nn.Conv2d(c[0], 4, 3, padding=1)
    def forward(s, X):
        x = X[:, 6:9]; m9 = x - X[:, 15:18]
        a0 = s.e0(X); a1 = s.e1(a0); a2 = s.e2(a1)
        u1 = s.d1(torch.cat([F.interpolate(a2, size=a1.shape[-2:], mode="bilinear", align_corners=False), a1], 1))
        u0 = s.d0(torch.cat([F.interpolate(u1, size=a0.shape[-2:], mode="bilinear", align_corners=False), a0], 1))
        h = s.head(u0); a = torch.sigmoid(h[:, :1]); r = 0.05 * torch.tanh(h[:, 1:])
        return x + a * (m9 - x) + r
if __name__ == "__main__":
    m = MG(); print(sum(p.numel() for p in m.parameters()) / 1e3, "K params", m(torch.rand(1, 18, 360, 640)).shape)
