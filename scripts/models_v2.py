"""Ablatable version of models/derainer.py (identical parameter names, so the original
checkpoint loads into the full configuration)."""
import torch, torch.nn.functional as F
from models.derainer import ConditionalVideoDerainer
from models.classifier import RainIntensityClassifier
import os
from pathlib import Path
import torch.nn as nn
import models.classifier as _C

CK_DIR = Path(__file__).resolve().parent / "checkpoints"
# which classifier checkpoint every script uses: experiments_v2/checkpoints/<CLS_CKPT>.pth
CLS_NAME = os.environ.get("CLS_CKPT", "classifier")


def make_classifier(arch="mobilenet", pretrained=False):
    """arch = 'mobilenet' (MobileNetV3-Small, as described in the paper) or 'lightcnn' (the small fallback CNN
    in models/classifier.py). Raises instead of silently falling back."""
    if arch == "lightcnn":
        old = _C.HAS_TORCHVISION; _C.HAS_TORCHVISION = False
        try: return RainIntensityClassifier(3, pretrained=False)
        finally: _C.HAS_TORCHVISION = old
    m = RainIntensityClassifier(3, pretrained=pretrained)
    if isinstance(m.feature_extractor[0], nn.Conv2d):
        raise RuntimeError("MobileNetV3-Small could not be built (torchvision import failed or ImageNet weights "
                           "could not be downloaded). Check: python -c \"import torchvision; print(torchvision.__version__)\"")
    return m


def load_classifier_ckpt(dev, name=None):
    """Load experiments_v2/checkpoints/<name>.pth, detecting its backbone from the weights."""
    path = CK_DIR / f"{name or CLS_NAME}.pth"
    sd = torch.load(path, map_location=dev, weights_only=False)["model_state_dict"]
    arch = "lightcnn" if "feature_extractor.0.weight" in sd else "mobilenet"
    m = make_classifier(arch); m.load_state_dict(sd)
    print(f"  classifier: {path.name} ({arch})")
    return m.to(dev).eval()


class AblatableDerainer(ConditionalVideoDerainer):
    def __init__(self, use_temporal=True, use_film=True, use_cbam=True):
        super().__init__(in_channels=3, guidance_dim=3, use_temporal=use_temporal)
        self.use_film, self.use_cbam = use_film, use_cbam

    def forward(self, frames, guidance):
        frames = frames.contiguous()
        if self.use_temporal:
            target = frames[:, 1]
            x = torch.cat([self.temporal_fusion(frames), target], 1)
        else:
            target = frames[:, 1] if frames.dim() == 5 else frames
            x = target
        H, W = x.shape[2:]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        if ph or pw: x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.enc1(x); e2 = self.enc2(self.down1(e1)); e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        if self.use_film: b = self.film(b, guidance)
        a = (lambda m, t: m(t)) if self.use_cbam else (lambda m, t: t)
        d = self.dec3(torch.cat([self.up3(b), a(self.att3, e3)], 1))
        d = self.dec2(torch.cat([self.up2(d), a(self.att2, e2)], 1))
        d = self.dec1(torch.cat([self.up1(d), a(self.att1, e1)], 1))
        r = self.out_conv(d)[:, :, :H, :W]
        return torch.clamp(target - r, 0, 1), r


VARIANTS = {
    # name: (temporal, film, cbam, guidance used for training)
    "baseline_unet":   (False, False, False, "none"),
    "unet_film":       (False, True,  False, "soft"),
    "unet_temporal":   (True,  False, False, "none"),
    "temporal_film":   (True,  True,  False, "soft"),
    "full_soft":       (True,  True,  True,  "soft"),     # proposed
    "full_onehot":     (True,  True,  True,  "onehot"),   # hard-decision guidance (key comparison)
    "full_noguid":     (True,  True,  True,  "none"),     # FiLM receives a constant vector
}


def build(name):
    t, f, c, _ = VARIANTS[name]
    return AblatableDerainer(t, f, c)
