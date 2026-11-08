# MG-Net: code, weights and result logs

Companion material for "Real-Time Rain Attenuation for Static Surveillance Cameras via a
Sharpness-Preserving Masked Temporal Median".

Every number in the manuscript is traceable to a file in `results/`. The mapping from table or
claim to result file is in `TABLE_TO_LOG.md`.

## Weights

| File | What it is |
|---|---|
| `weights/student_mg.pth` | MG-Net, 131,268 parameters, original target (theta = 4/255) |
| `weights/student_mg_strong.pth`, `student_mg_strong_s2.pth`, `student_mg_strong_s3.pth` | the strong-target variant (theta = 2/255, no teacher term), three training seeds |
| `weights/student_mf7.pth` | the teacher, mf7, 270,659 parameters |
| `weights/student_mf6.pth` | the stage-1 model, mf6 |
| `weights/student_mg_xcam_cctv.pth`, `student_mg_xcam_iphone.pth` | the cross-camera models of Supplementary Table 7, each trained on one camera only |
| `weights/student_mg21.pth` | MG-21, a twenty-one-frame median as target and blend operand (Section 4.6) |
| `weights/s2vd_ft.pth` | the S2VD backbone fine-tuned on our clips and our target |
| `weights/asf_ft.pth` | the ASF-Net backbone fine-tuned on our clips and our target, SpyNet frozen |

## Code

**Model and training.** `scripts/mg_net.py` (MG-Net definition), `cctv_mg.py` (original target),
`cctv_mg_strong.py` and `cctv_mg_strong_seed.py` (strong target, three seeds), `cctv_mf6.py` and
`cctv_mf7.py` (teacher stages), `train_mg_xcam.py` (cross-camera), `train_mg21.py` (MG-21),
`new_nodistill.py` and `new_ablation_seeds.py` (distillation ablation).

**Retrained baselines.** `scripts/train_s2vd_ft.py` and `train_asf_ft.py` fine-tune the S2VD and
ASF-Net backbones on our clips, our target and our budget.

**Evaluation.**

| Script | Produces |
|---|---|
| `unified_345.py` | Tables 1 and 2, Supplementary Tables 3, 4 and 5 |
| `unified_eval3.py` | Tables 3, 5 and 6 |
| `aau_eval.py` | Table 7 |
| `aau_kept.py` | Table 8 |
| `bitrate_eval.py` | Table 9 |
| `aau_probe.py` | Supplementary Table 8 |
| `kept_wide.py` | Supplementary Table 6 |
| `xcam_eval2.py` | Supplementary Table 7 |
| `jitter_test.py`, `jitter_align_test.py` | Supplementary Tables 9 and 10 |
| `synth_psnr.py` | PSNR and SSIM on synthetic rain (Section 4.8) |
| `motion_trigger.py` | background-subtractor test (Section 4.8) |
| `mg_dry.py`, `iph_dry_check.py` | behaviour on rain-free video (Section 4.8) |
| `nturain_eval.py` | the single static NTURain clip (Section 4.8) |
| `new_mg5.py` | the five-frame window (Section 4.3) |
| `s2vdft_eval.py`, `asfft_eval.py`, `mg21_eval.py` | the retrained baselines and MG-21 (Section 4.6) |

**Timing.** `mg_speed.py`, `count_params.py`, `prot_speed.py` (cost of the protection step),
`ft_speed.py` (retrained baselines), `res_speed.py` (720p and 1080p).

## Running a script

Set `ROOT` in `scripts/common.py` to your own copy of the video folder first, then, from the
repository root:

```
set PYTHONPATH=%CD%\scripts
python scripts\unified_345.py > results\unified_345_log.txt 2>&1
```

on Windows, or

```
PYTHONPATH=$PWD/scripts python scripts/unified_345.py > results/unified_345_log.txt 2>&1
```

elsewhere. Every evaluation script is run the same way. `aau_eval.py`, `aau_kept.py` and
`aau_probe.py` additionally need the AAU RainSnow dataset, and `nturain_eval.py` needs NTURain.

## Results

`results/` holds the output of every run. Most are plain text logs; `aau_kept.json`,
`bitrate_eval.json` and `xcam_eval2.json` are JSON because those scripts write their summary to a
file rather than to stdout. `TABLE_TO_LOG.md` describes their structure.

Numbers in the published tables come from the unified protocol (`unified_345.py`,
`unified_eval3.py`): twenty windows per clip, the same windows for every method, and no
motion-protection step applied to any method for the flicker and sharpness measures. Logs from
earlier evaluation rounds are kept for transparency but no longer back any published number.

## Results that do not favour the method

These are in the manuscript and are repeated here so that nobody has to find them.

- On synthetic rain with an exact reference, MG-Net is mid-field at 34.10 dB PSNR, below the
  nine-frame median (34.45 dB) and well below ASF-Net as released (42.68 dB). The rain model is
  ours; fine-tuning ASF-Net on our real clips lowers its score here to 36.15 dB.
- MG-Net never exceeds the temporal median it is distilled from on flicker removal, on any scene.
  On the hose-rain site J01 the median removes about three times as much.
- On the public AAU RainSnow benchmark, no method we tested improves detection. Every output
  scores below the rainy input.
- The motion-trigger test did not show that attenuating rain reduces false alarms: on our clips the
  rainy input triggers on 0.367 of frames and the rain-free input on 0.364, because the triggers
  come from wind-moved vegetation.
- MG-21 removes more flicker than the strong-target variant on the public sequences (58.0% against
  47.2%) but retains fewer detections (0.691 against 0.729) and lags the output by twenty frames
  instead of eight.
- With a random shift of one to two pixels per frame, static-pixel sharpness falls to about half.
  A plain temporal median falls just as far, so this is a property of the approach rather than of
  the network, but the method does need a rigidly mounted camera.

## Not included

- The video recordings. The CCTV videos can be released only after faces and number plates are
  blurred and with the camera owner's agreement; the iPhone recordings can be shared for research
  purposes on request.
- Third-party baselines (PReNet, Restormer, S2VD, ASF-Net): use the authors' official repositories.
  The ASF-Net release contains testing code and weights only, with no training script.
- Logs for the distillation ablation of Section 4.2, which was run before the evaluation protocol
  was fixed and whose logs were not kept.

## Notes

- Run on Windows with one NVIDIA RTX 3070, Python 3.11 and PyTorch.
- Licence: MIT (see LICENSE).
