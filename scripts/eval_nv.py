"""Evaluate everything on the HELD-OUT test data and write all numbers used in the paper.

    python experiments_v2/eval_v2.py                # all trained variants
    python experiments_v2/eval_v2.py --quick        # 1 in 5 test samples (smoke test)

Outputs (experiments_v2/results/):
    classifier_test.json        accuracy, macro-F1, confusion matrix, per-source accuracy
    derain_test_<variant>.csv   per-image PSNR/SSIM on the grouped test split (native resolution)
    spa_test_<variant>.csv      per-image PSNR/SSIM on the official SPA-Data real test set
    summary_derain.json         aggregated numbers (Table II / V)
    video_temporal.json         flicker, guidance switches and accuracy vs. stride K (Table IV)
    qualitative/                PNGs for the qualitative figure
"""
import argparse, json, glob
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch
from common import ROOT, OUT, SPA_TEST, REAL_DIR, DATA, imread01, psnr, ssim, psnr_y, save_json, CLASSES
from models_v2 import VARIANTS, build, AblatableDerainer, RainIntensityClassifier, load_classifier_ckpt, CLS_NAME
from data_v2 import ClsDataset, cls_input_from_bgr01

SPL = ROOT / "experiments_v2" / "splits_nv"; CK = ROOT / "experiments_v2" / "checkpoints"
QDIR = OUT / "qualitative"; QDIR.mkdir(exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"


def f1_macro(cm):
    f = []
    for k in range(3):
        tp = cm[k, k]; p = tp / max(cm[:, k].sum(), 1); r = tp / max(cm[k].sum(), 1)
        f.append(0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(f)), [float(x) for x in f]


@torch.no_grad()
def eval_classifier(clf):
    ds = ClsDataset(SPL / "cls_test.csv"); preds = []
    for i in range(0, len(ds), 64):
        xs = torch.stack([ds[j][0] for j in range(i, min(i + 64, len(ds)))]).to(dev)
        preds += clf(xs)[0].argmax(1).cpu().tolist()
    df = ds.df.copy(); df["pred"] = preds
    cm = np.zeros((3, 3), int)
    for t, p_ in zip(df.label, df.pred): cm[t, p_] += 1
    mf1, f1s = f1_macro(cm)
    res = dict(n=len(df), accuracy=float((df.label == df.pred).mean() * 100), macro_f1=mf1, f1_per_class=f1s,
               confusion_matrix=cm.tolist(), classes=CLASSES,
               per_kind_accuracy={k: float((g.label == g.pred).mean() * 100) for k, g in df.groupby("kind")})
    save_json(res, "classifier_test.json" if CLS_NAME == "classifier" else f"{CLS_NAME}_test.json"); print(json.dumps(res, indent=1))
    return res


def load_derainer(path, name=None):
    ck = torch.load(path, map_location=dev)
    name = name or ck.get("variant", "full_soft")
    m = build(name) if name in VARIANTS else AblatableDerainer()
    m.load_state_dict(ck["model_state_dict"]); return m.to(dev).eval(), name


@torch.no_grad()
def guidance(mode, frame_bgr01, clf, label=None):
    if mode == "none": return torch.full((1, 3), 1 / 3, device=dev)
    _, g = clf(cls_input_from_bgr01(frame_bgr01).to(dev))
    if mode == "onehot":  # realistic hard decision: argmax of the classifier, NOT the true label
        return torch.nn.functional.one_hot(g.argmax(1), 3).float()
    return g


@torch.no_grad()
def run(m, frames, g):
    x = torch.from_numpy(np.ascontiguousarray(np.stack([f.transpose(2, 0, 1) for f in frames])))[None].to(dev).contiguous()
    out, _ = m(x, g)
    return out[0].permute(1, 2, 0).cpu().numpy()


def eval_pairs(m, gmode, clf, rows, tag, name, save_q=()):
    rec = []
    for k, r in enumerate(rows):
        fr = [imread01(r["prev"]), imread01(r["curr"]), imread01(r["next"])]; gt = imread01(r["clean"])
        h = min(x.shape[0] for x in fr + [gt]); w = min(x.shape[1] for x in fr + [gt])
        fr = [x[:h, :w] for x in fr]; gt = gt[:h, :w]
        out = run(m, fr, guidance(gmode, fr[1], clf))
        rec.append(dict(idx=k, kind=r.get("kind", "spa"), label=r.get("label", -1),
                        psnr=psnr(out, gt), ssim=ssim(out, gt), psnr_y=psnr_y(out, gt),
                        in_psnr=psnr(fr[1], gt), in_ssim=ssim(fr[1], gt)))
        if k in save_q:
            cv2.imwrite(str(QDIR / f"{tag}_{k:04d}_{name}.png"), (out * 255).round().astype(np.uint8))
            cv2.imwrite(str(QDIR / f"{tag}_{k:04d}_input.png"), (fr[1] * 255).round().astype(np.uint8))
            cv2.imwrite(str(QDIR / f"{tag}_{k:04d}_gt.png"), (gt * 255).round().astype(np.uint8))
    df = pd.DataFrame(rec); df.to_csv(OUT / f"{tag}_{name}.csv", index=False); return df


def spa_rows(step):
    rows = []
    for g in sorted(glob.glob(str(SPA_TEST / "gt" / "*gt.png")))[::step]:
        i = Path(g).name.replace("gt.png", "")
        r = str(SPA_TEST / "rain" / f"{i}.png"); rows.append(dict(prev=r, curr=r, next=r, clean=g))
    return rows


def qual_indices(df):
    pick = []
    for (kind, lab), g in df.groupby(["kind", "label"]):
        pick.append(int(g.index[len(g) // 2]))
    return set(pick)


@torch.no_grad()
def video_temporal(models, clf, n_frames, Ks):
    """Consecutive frames from the TEST time-segment of each real video (640x360)."""
    tri = pd.read_csv(REAL_DIR / "triplets.csv"); res = {}
    for vname, g in tri[tri.split == "test"].groupby("video"):
        src = next((p for p in DATA.rglob(vname)), None)
        if src is None: continue
        cap = cv2.VideoCapture(str(src)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(g.frame_start.min()))
        frames = []
        for _ in range(n_frames + 2):
            ok, f = cap.read()
            if not ok: break
            frames.append(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
        cap.release(); label = int(g.label.iloc[0]); vres = {}
        inp = np.mean([np.abs(frames[t] - frames[t - 1]).mean() for t in range(2, len(frames) - 1)])
        vres["input_flicker"] = float(inp)
        for name, (m, gmode) in models.items():
            for K in (Ks if gmode == "soft" else [1]):
                outs, gs, preds = [], [], []; cache = None
                for t in range(1, len(frames) - 1):
                    if cache is None or (t - 1) % K == 0:
                        cache = guidance(gmode, frames[t], clf)
                    gs.append(cache[0].cpu().numpy()); preds.append(int(cache.argmax()))
                    outs.append(run(m, frames[t - 1:t + 2], cache))
                fl = np.mean([np.abs(outs[t] - outs[t - 1]).mean() for t in range(1, len(outs))])
                gdiff = np.mean([np.abs(gs[t] - gs[t - 1]).sum() for t in range(1, len(gs))])
                vres[f"{name}_K{K}"] = dict(flicker=float(fl), flicker_ratio=float(fl / inp),
                                            guidance_change=float(gdiff), frame_acc=float(np.mean(np.array(preds) == label) * 100),
                                            switches=int(np.sum(np.diff(preds) != 0)))
                if K == 1 or K == 10:
                    for t in (len(outs) // 2,):
                        cv2.imwrite(str(QDIR / f"video_{Path(vname).stem[:20]}_{name}_K{K}.png"),
                                    (outs[t] * 255).round().astype(np.uint8))
        res[vname] = vres; print(vname, json.dumps(vres, indent=1))
    save_json(res, "video_temporal.json"); return res


def summarize(df):
    out = dict(n=len(df), psnr=df.psnr.mean(), ssim=df.ssim.mean(), psnr_y=df.psnr_y.mean(),
               input_psnr=df.in_psnr.mean(), input_ssim=df.in_ssim.mean())
    if "label" in df and (df.label >= 0).any():
        for k, c in enumerate(CLASSES):
            s = df[df.label == k]
            if len(s): out[c] = dict(n=len(s), psnr=s.psnr.mean(), ssim=s.ssim.mean())
        out["per_kind"] = {k: dict(n=len(s), psnr=s.psnr.mean(), ssim=s.ssim.mean()) for k, s in df.groupby("kind")}
    return out


def main(a):
    clf = load_classifier_ckpt(dev)
    eval_classifier(clf)
    if a.cls_only: return
    test = pd.read_csv(SPL / "derain_test.csv")
    if a.quick: test = test.iloc[::5].reset_index(drop=True)
    rows = test.to_dict("records"); q = qual_indices(test)
    ckpts = sorted(CK.glob("derainer_*.pth"))
    if (ROOT / "checkpoints" / "derainer_best.pth").exists() and a.include_original:
        ckpts.append(ROOT / "checkpoints" / "derainer_best.pth")
    summary, models = {}, {}
    for cp in ckpts:
        if a.only and not any(k in cp.stem for k in a.only): continue
        m, var = load_derainer(cp)
        name = "original_checkpoint" if cp.parent.parent == ROOT else cp.stem.replace("derainer_", "")
        gmode = VARIANTS.get(var, (0, 0, 0, "soft"))[3]
        print(f"== {name} (guidance at test: {gmode})")
        d1 = eval_pairs(m, gmode, clf, rows, "derain_test", name, q)
        d2 = eval_pairs(m, gmode, clf, spa_rows(5 if a.quick else 1), "spa_test", name, set(range(0, 1000, 100)))
        summary[name] = dict(guidance=gmode, grouped_test=summarize(d1), spa_real_test=summarize(d2))
        print(json.dumps(summary[name], indent=1, default=float))
        if name in ("full_soft", "full_onehot", "full_noguid"): models[name] = (m, gmode)
        else: del m
    save_json(json.loads(json.dumps(summary, default=float)), "summary_derain.json" if not a.only else "summary_derain_" + "_".join(a.only) + ".json")
    if models and not a.only: video_temporal(models, clf, 60 if a.quick else 300, [1, 5, 10, 15, 20, 30])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls_only", action="store_true", help="only evaluate the classifier")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", help="evaluate only checkpoints whose name contains one of these strings")
    ap.add_argument("--include_original", action="store_true",
                    help="also evaluate checkpoints/derainer_best.pth (NOTE: it was trained on a random split, "
                         "so its grouped-test numbers are optimistic; its SPA numbers are clean)")
    main(ap.parse_args())
