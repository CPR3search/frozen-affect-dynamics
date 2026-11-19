"""Pain peak task for all encoders."""
from __future__ import annotations
import json, os, sys
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS, OUT
from affectdyn.provenance import run_provenance, audit_loso

NPERM = 20000
NBOOT = 5000


def label_perm_test(scores, y, nperm=NPERM, seed=0):
    """Null of no association: scores fixed, labels permuted."""
    rng = np.random.default_rng(seed)
    obs = _auc(scores, y)
    cnt = 0
    for _ in range(nperm):
        yp = rng.permutation(y)
        if abs(_auc(scores, yp) - 0.5) >= abs(obs - 0.5) - 1e-12:
            cnt += 1
    return float(obs), float((cnt + 1) / (nperm + 1))


def boot_auc_ci(scores, y, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(_auc(scores[idx], y[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/shape.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub, traj = lab["subject"], lab["traj"]
    y = (traj == 2).astype(int)
    mask = traj != -1
    folds = _loso_folds(sub)
    audit_loso(sub, folds)
    npos = int(y[mask].sum())
    log(f"W2 shape probe | peaked {npos} vs monotonic {int(mask.sum()) - npos} over "
        f"{len(set(sub[mask].tolist()))} subjects, peaked present in "
        f"{len(set(sub[mask & (y == 1)].tolist()))} of them | LOSO, pooled out-of-fold primary")

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
    log(f"cells: {len(cells)} ({len(IMG_ENCODERS)} image + {len(VIDEO_SOURCES)} video encoders "
        f"x {len(KS)} readout densities)")

    import torch
    scores = {}
    for name, X in cells.items():
        acc = None
        for rep in range(N_REP):
            torch.manual_seed(2000 + rep)
            s = _oof_scores(X, y, sub, mask, folds)
            acc = s if acc is None else acc + s
        scores[name] = acc / N_REP
        log(f"  scored {name}")

    keep = mask.copy()
    for s in scores.values():
        keep &= ~np.isnan(s)
    yk = y[keep]
    log(f"\nclips retained: {keep.sum()} ({int(yk.sum())} peaked)")

    rows = []
    for name, s in scores.items():
        auc, p = label_perm_test(s[keep], yk)
        lo, hi = boot_auc_ci(s[keep], yk)
        rows.append({"cell": name, "auc": auc, "p": p, "ci": [lo, hi]})
    for r, a in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = a

    log("\n=== trajectory-shape decodability, pooled out-of-fold AUC vs chance ===")
    log(f"  {'encoder':16s} " + "".join(f"K={k:<18d}" for k in KS))
    by = {r["cell"]: r for r in rows}
    for fam in list(IMG_ENCODERS) + list(VIDEO_SOURCES):
        line = f"  {fam:16s} "
        for k in KS:
            n = f"{fam}_K{k}"
            if n in by:
                r = by[n]
                star = "*" if r["p_holm"] < 0.05 else ("." if r["p"] < 0.05 else " ")
                line += f"{r['auc']:.3f} p={r['p']:.3f}{star}  "
            else:
                line += " " * 20
        log(line)
    log("  * significant after Holm    . nominally significant, does not survive correction")

    nsig = sum(r["p_holm"] < 0.05 for r in rows)
    nnom = sum(r["p"] < 0.05 for r in rows)
    best = max(rows, key=lambda r: r["auc"])
    log(f"\n  cells significant after Holm : {nsig} of {len(rows)}")
    log(f"  nominally significant          : {nnom} of {len(rows)}")
    log(f"  best cell                      : {best['cell']} AUC {best['auc']:.3f} "
        f"CI [{best['ci'][0]:.3f}, {best['ci'][1]:.3f}] p={best['p']:.4f} Holm={best['p_holm']:.4f}")

    # Use the null cell's bootstrap interval half-width to describe resolution.
    med_hw = float(np.median([(r["ci"][1] - r["ci"][0]) / 2 for r in rows]))
    log(f"\n  POWER: median bootstrap CI half-width {med_hw:.3f} AUC with {int(yk.sum())} peaked "
        f"clips. Effects smaller than roughly this are invisible here, so any null below is a "
        f"non-detection at the available power, not demonstrated absence.")

    json.dump({"provenance": run_provenance(seeds=[2000 + r for r in range(N_REP)]),
               "rows": rows, "n_clips": int(keep.sum()), "n_peaked": int(yk.sum()),
               "n_rep": N_REP, "median_ci_halfwidth": med_hw},
              open(f"{OUT}/shape.json", "w"), indent=2)
    log(f"\nwrote {OUT}/shape.json")
    log("Machine-side. Any reframing drawn from this is a GATE H3 decision.")
    logf.close()


if __name__ == "__main__":
    main()
