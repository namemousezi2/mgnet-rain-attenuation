"""Light real-time deraining student: PixelUnshuffle(2) -> conv trunk at half resolution -> PixelShuffle -> residual.  out = x - predicted_rain.  Input H,W must be even (infer pads to multiple of 2)."""
import torch, torch.nn as nn
from pathlib import Path


class Res(nn.Module):
    def __init__(s, c):
        super().__init__(); s.a = nn.Conv2d(c, c, 3, padding=1); s.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(s, x): return x + s.b(torch.relu(s.a(x)))


class Student(nn.Module):
    def __init__(s, c=64, n=8):
        super().__init__(); s.down = nn.PixelUnshuffle(2); s.head = nn.Conv2d(12, c, 3, padding=1); s.body = nn.Sequential(*[Res(c) for _ in range(n)])
        s.tail = nn.Conv2d(c, 12, 3, padding=1); s.up = nn.PixelShuffle(2)

    def forward(s, x):
        f = s.head(s.down(x)); f = s.body(f) + f; return x - s.up(s.tail(f))


class StudentX(Student):
    """Student + a small FULL-resolution refinement (3x3 convs, 16 ch, zero-initialised) so thin high-contrast lines are not damaged by the half-resolution trunk."""
    def __init__(s, c=64, n=8):
        super().__init__(c, n); s.ref = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 3, 3, padding=1)); nn.init.zeros_(s.ref[2].weight); nn.init.zeros_(s.ref[2].bias)

    def forward(s, x):
        y = super().forward(x); return y + s.ref(torch.cat([y, x], 1))


def load_student(name="student_distill"):
    from common import ROOT
    ck = torch.load(ROOT / "experiments_v2" / "checkpoints" / (name + ".pth"), map_location="cpu"); cf = ck["cfg"]; m = StudentX(cf[0], cf[1]) if len(cf) > 2 else Student(*cf); m.load_state_dict(ck["model_state_dict"]); return m, 2
