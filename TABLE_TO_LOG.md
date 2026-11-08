# Which table comes from which log

Every number in the manuscript is traceable to one of the files in `results/`.
Table and section numbers follow the submitted version of the paper.

## Main text

| Manuscript item | Result file | Script |
|---|---|---|
| Table 1 (natural rain) | `results/unified_345_log.txt` | `scripts/unified_345.py` |
| Table 2 (artificial rain, H01) | `results/unified_345_log.txt` | `scripts/unified_345.py` |
| Table 3 (detector retention, five shop scenes) | `results/unified_eval3_log.txt` | `scripts/unified_eval3.py` |
| Table 4 (parameters and throughput) | `results/mg_speed_log.txt`, `results/params_log.txt` | `scripts/mg_speed.py`, `count_params.py` |
| Table 4 (720p and 1080p rows) | `results/res_speed_log.txt` | `scripts/res_speed.py` |
| Table 4 (retrained baselines: S2VD 12.5 fps, ASF-Net 4.2 fps) | `results/ft_speed_log.txt` | `scripts/ft_speed.py` |
| Table 5 (four held-out viewpoints) | `results/unified_eval3_log.txt` | `scripts/unified_eval3.py` |
| Table 6 (flicker-detection trade-off) | `results/unified_eval3_log.txt` | `scripts/unified_eval3.py` |
| Table 7 (AAU RainSnow, AP50 against ground truth) | `results/aau_eval_log.txt` | `scripts/aau_eval.py` |
| Table 8 (AAU RainSnow, kept-rate and raw flicker) | `results/aau_kept.json` | `scripts/aau_kept.py` |
| Table 9 (encoded size) | `results/bitrate_eval.json` | `scripts/bitrate_eval.py` |

## Sections

| Claim | Result file | Script |
|---|---|---|
| Section 4.3, five-frame window | `results/new_mg5_log.txt` | `scripts/new_mg5.py` |
| Section 4.4, cost of the protection step (12.6 ms; median 223 to 59 fps; MG-Net 110 fps) | `results/prot_speed_log.txt` | `scripts/prot_speed.py` |
| Section 4.6, S2VD retrained on our data (33.2% flicker, 0.871 kept-rate, sharpness 0.971) | `results/s2vdft_eval_log.txt` | `scripts/train_s2vd_ft.py`, `s2vdft_eval.py` |
| Section 4.6, ASF-Net retrained on our data (16.3% flicker, 0.955 kept-rate) | `results/asfft_eval_log.txt` | `scripts/train_asf_ft.py`, `asfft_eval.py` |
| Section 4.6, MG-21 on our own clips | `results/mg21_eval_log.txt` | `scripts/train_mg21.py`, `mg21_eval.py` |
| Section 4.6, MG-21 on the public sequences (58.0% flicker, 0.691 kept-rate) | `results/aau_kept.json` | `scripts/aau_kept.py` |
| Section 4.6, cross-camera training | `results/xcam_eval2.json` | `scripts/train_mg_xcam.py`, `xcam_eval2.py` |
| Section 4.8, PSNR and SSIM on synthetic rain (MG-Net 34.10 dB, nine-frame median 34.45 dB, ASF-Net 42.68 dB, rainy input 29.55 dB) | `results/synth_psnr_log.txt` | `scripts/synth_psnr.py` |
| Section 4.8, motion-trigger test (rain 0.367 against rain-free 0.364; true triggers kept 0.937 for MG-Net, 0.617 for the nine-frame median) | `results/motion_trigger_log.txt` | `scripts/motion_trigger.py` |
| Section 4.8, behaviour on rain-free video | `results/mg_dry_log.txt`, `results/iph_dry_check_log.txt` | `scripts/mg_dry.py`, `iph_dry_check.py` |
| Section 4.8, NTURain single static clip | `results/nturain_eval_log.txt` | `scripts/nturain_eval.py` |
| Training runs behind the released weights | `results/cctv_mg_log.txt`, `cctv_mf7_log.txt`, `mg21_train_log.txt`, `asf_ft_train_log.txt`, `s2vd_ft_train_log.txt` | `scripts/cctv_mf6.py`, `cctv_mf7.py`, `cctv_mg.py`, `cctv_mg_strong.py`, `cctv_mg_strong_seed.py` |
| Clip pairing used by every training and evaluation script | `results/cctv_pairs_check.json` | `scripts/cctv_pairs_check.py` |
| Section 4.8, the earlier dry reference (8.3-9.7 s of the same clip: 59% MG-Net, 60% full median, 50% mf7, 31% S2VD, 26% ASF-Net) | `results/new_iph_all_log.txt` | `scripts/new_iph_all.py` |
| Supplementary Data, the five excluded night recordings (input flicker 0.0002-0.0007; N01_village_night_rain_4 dominated by moving objects) | `results/new_village_log.txt` | `scripts/new_village.py` |
| Supplementary Data, the excluded shop-front scene S02 (rain clip 0.0023 against dry clip 0.0026) | `results/flicker_mg_log.txt` | `scripts/flicker_mg.py` |

## Supplementary Material

| Supplementary item | Result file | Script |
|---|---|---|
| Table 1 (training settings) | the training logs listed above | the matching `cctv_*.py` |
| Table 2 (recording sites) | from the recording log and the video timestamps, not from a script | - |
| Table 3 (per-scene values, bootstrap intervals) | `results/unified_345_log.txt` | `scripts/unified_345.py` |
| Table 4 (seed-to-seed spread) | `results/unified_345_log.txt` | `scripts/unified_345.py` |
| Table 5 (site J01, hose rain) | `results/unified_345_log.txt` | `scripts/unified_345.py` |
| Table 6 (detector retention, extended set) | `results/kept_wide_log.txt` | `scripts/kept_wide.py` |
| Table 7 (cross-camera) | `results/xcam_eval2.json` | `scripts/xcam_eval2.py` |
| Table 8 (AAU sequences that pass the static-camera test) | `results/aau_probe_log.txt` | `scripts/aau_probe.py` |
| Table 9 (camera jitter, integer shifts) | `results/jitter_test_log.txt` | `scripts/jitter_test.py` |
| Table 10 (camera jitter, sub-pixel shifts) | `results/jitter_align_test_log.txt` | `scripts/jitter_align_test.py` |
| Algorithms 1 and 2 (training loop, inference loop) | - | `scripts/cctv_mg_strong.py`, `mg_video.py` |

## Reading the result files

Most results are plain text logs. Three are JSON because the script writes its summary there
rather than to stdout:

- `aau_kept.json` holds `summary` (one entry per method: `kept`, `kept_ci`, `kept_prot`,
  `flicker_removed_raw_pct`, `flicker_ci`) and `per_sequence` for all twenty sequences.
- `bitrate_eval.json` holds `summary_pct_saved` (percentage saved at CRF 23 and CRF 28 and
  for the entropy bound at each quantisation step), `per_clip` for the nine clips, and
  `selftest_pct_cheaper` for the synthetic control.
- `xcam_eval2.json` holds the per-clip flicker and sharpness for each training set.
  `xcam_eval2_log.txt` records the run but its summary section is empty, so the numbers in
  Supplementary Table 7 come from the JSON.

## Protocol notes

Every number in Tables 1, 2, 3, 5 and 6 comes from the unified protocol (`unified_345.py`,
`unified_eval3.py`): twenty windows per clip, the same windows for every method, and no
motion-protection step applied to any method for the flicker and sharpness measures. Logs from
earlier evaluation rounds remain in `results/` for transparency but no longer back any published
number.

Table 3 and Table 6 report the same quantity on the same five scenes and come from the same run,
so their values agree: the per-scene kept-rates of Table 3 average to the kept-rate column of
Table 6.

The three retraining runs of Section 4.6 use the training clips, the masked temporal median target
(theta = 2/255, thin-structure test, motion protection), the mask-weighted L1 loss with lambda = 20
and the 0.4 h budget of the MG-Net strong-target model, on the same GPU. The two baselines start
from their authors' released weights at a learning rate of 1e-4.

They do not reproduce the published training procedures. S2VD is semi-supervised and trains its own
rain generator; ASF-Net adds online adaptive learning at test time, and its public release contains
testing code and weights only, with no training script. The runs answer one question: how those
architectures compare with the gated blend when all three see the same data, target and budget.

The flicker and kept-rate values in those three logs come from runs separate from the one behind
Table 6. The MG-Net and median values reproduce Table 6 to within 0.2 points of flicker and 0.02 of
kept-rate, but the window sets are not identical, so the two should not be read as one table.

Two claims in the paper are not backed by a log in this repository. The distillation ablation of
Section 4.2 ("differences smaller than the spread between seeds") was run before the protocol was
fixed, with `new_ablation_seeds.py` and `new_nodistill.py`; its logs were not kept. The recording
sites of Supplementary Table 2 come from the recording log and the video timestamps.

`results/` also holds logs from evaluation rounds that the unified protocol replaced
(`new_jungle*`, `new_eval`, `new_base1`, `new_asf*`, `new_s2vd`, `new_clips3_eval`,
`baselines_cmp`, `s2vd_cmp`, `prot3`, `flicker7`, `probe_*`). They are kept so that the history of
the measurements is visible, their scripts were removed with the clean-up, and no published number
depends on them.
