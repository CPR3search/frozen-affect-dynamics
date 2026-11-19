"""Pain direction AUC and the CLIP-video contrasts for all encoders."""
from __future__ import annotations
import json, os, sys
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP
from affectdyn.inference.aggregation_permutation_tests import perm_test, boot_ci, holm
from affectdyn.provenance import run_provenance, audit_loso

OUT = "runs/pain_encoder_maps"
KS = (2, 4, 8, 16)
VIDEO_SOURCES = {                       # Map models to cache paths and key templates.
    "vjepa":        ("runs/temporal_aggregation/vjepa_windows.npz", "K{k}_up"),
    "videomae":     ("runs/temporal_aggregation/videomae_windows_fixed.npz", "K{k}_up"),
    "videomaev2":   ("runs/temporal_aggregation/videomaev2_windows.npz", "K{k}_up"),
    "timesformer":  ("runs/pain_encoder_maps/timesformer_windows.npz", "K{k}_up"),
    "xclip":        ("runs/pain_encoder_maps/xclip_windows.npz", "K{k}_up"),
    "internvideo2": ("runs/pain_encoder_maps/internvideo2_windows.npz", "K{k}_up"),
}


def oof_reps(X, y, sub, mask, folds, n_rep=N_REP):
    """One seeded probe fit per rep; returns the per-rep out-of-fold score vectors."""
    import torch
    out = []
    for rep in range(n_rep):
        torch.manual_seed(1000 + rep)
        out.append(_oof_scores(X, y, sub, mask, folds))
    return out


def per_fold_mean(scores, y, sub, keep):
    vals = []
    for s in sorted(set(sub[keep].tolist())):
        m = keep & (sub == s)
        if m.sum() < 1 or len(set(y[m].tolist())) < 2:
            continue
        v = _auc(scores[m], y[m])
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)), float(np.std(vals))


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/map.log", "w")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    folds = _loso_folds(sub)
    audit_loso(sub, folds)

    cells = {}
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    for e in IMG_ENCODERS:
        for k in KS:
            cells[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for name, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            key = tmpl.format(k=k)
            if key in z.files:
                cells[f"{name}_K{k}"] = video_readout(z["full"], z[key])

    log(f"W2 map | {mask.sum()} direction-labelled clips, {len(set(sub.tolist()))} subjects, LOSO "
        f"| {len(cells)} cells | {N_REP} seeded probe fits each")
    log(f"video models: {', '.join(VIDEO_SOURCES)}")

    reps = {}
    for name, X in cells.items():
        reps[name] = oof_reps(X, y, sub, mask, folds)
        log(f"  scored {name}  (dim {X.shape[1]})")

    keep = mask.copy()
    for rs in reps.values():
        for s in rs:
            keep &= ~np.isnan(s)
    yk = y[keep]
    log(f"\nclips retained: {keep.sum()}")

    # Compute fold means over repeats and pooled AUC from mean out-of-fold scores.
    table, pooled = {}, {}
    for name, rs in reps.items():
        pf = [per_fold_mean(s, y, sub, keep)[0] for s in rs]
        avg = np.mean(rs, axis=0)
        pooled[name] = avg
        table[name] = {"per_fold_mean": float(np.mean(pf)), "per_fold_sd_over_reps": float(np.std(pf)),
                       "pooled_oof": float(_auc(avg[keep], yk))}

    log("\n=== direction AUC at matched readout (per-fold mean | pooled out-of-fold) ===")
    log(f"  {'encoder':16s} " + "".join(f"K={k:<14d}" for k in KS))
    for fam in list(IMG_ENCODERS) + list(VIDEO_SOURCES):
        row = f"  {fam:16s} "
        for k in KS:
            n = f"{fam}_K{k}"
            row += (f"{table[n]['per_fold_mean']:.3f}|{table[n]['pooled_oof']:.3f}   "
                    if n in table else " " * 16)
        log(row)

    def contrast(a, b, label):
        sa, sb = pooled[a][keep], pooled[b][keep]
        obs, p = perm_test(sa, sb, yk)
        lo, hi = boot_ci(sa, sb, yk)
        return {"label": label, "a": a, "b": b, "auc_a": _auc(sa, yk), "auc_b": _auc(sb, yk),
                "gap": obs, "p": p, "ci": [lo, hi]}

    rows = [contrast(f"clip_K{k}", f"{v}_K{k}", f"K={k:2d}: CLIP vs {v}")
            for k in KS for v in VIDEO_SOURCES if f"{v}_K{k}" in pooled]
    for r, a in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = a
    log(f"\n=== PRIMARY: CLIP vs each video model at matched K, Holm over {len(rows)} contrasts ===")
    for r in rows:
        sig = "*" if r["p_holm"] < 0.05 else " "
        log(f" {sig} {r['label']:28s} {r['auc_a']:.3f} vs {r['auc_b']:.3f}  gap {r['gap']:+.3f}  "
            f"95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  p={r['p']:.4f}  Holm={r['p_holm']:.4f}")

    nsig = sum(r["p_holm"] < 0.05 for r in rows)
    log(f"\n  significant after Holm: {nsig} of {len(rows)}")

    desc = []
    for k in KS:
        bi = max(IMG_ENCODERS, key=lambda e: _auc(pooled[f"{e}_K{k}"][keep], yk))
        vids = [v for v in VIDEO_SOURCES if f"{v}_K{k}" in pooled]
        bv = max(vids, key=lambda v: _auc(pooled[f"{v}_K{k}"][keep], yk))
        d = contrast(f"{bi}_K{k}", f"{bv}_K{k}", f"K={k:2d}: best image {bi} vs best video {bv}")
        desc.append(d)
    log("\n=== DESCRIPTIVE: best image vs best video at matched K (selection bias both sides) ===")
    for r in desc:
        log(f"   {r['label']:44s} {r['auc_a']:.3f} vs {r['auc_b']:.3f}  gap {r['gap']:+.3f}  "
            f"p={r['p']:.4f}")

    json.dump({"provenance": run_provenance(seeds=[1000 + r for r in range(N_REP)]),
               "table": table, "primary": rows, "descriptive": desc,
               "n_clips": int(keep.sum()), "n_rep": N_REP},
              open(f"{OUT}/map.json", "w"), indent=2)
    log(f"\nwrote {OUT}/map.json")
    log("Machine-side. Claims drawn from this go to GATE H3, not into prose unreviewed.")
    logf.close()


if __name__ == "__main__":
    main()
