"""Reversal control for the image encoders."""
from __future__ import annotations
import json, os, sys

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, IMG_ENCODERS, KS, N_REP
from affectdyn.analyses.peak.pain_peak_encoder_map import boot_auc_ci
from affectdyn.provenance import run_provenance

OUT = "runs/cross_dataset_encoder_checks"


def oof_transfer(X_train, X_eval, y, sub, mask, folds):
    """Fit on the training fold of X_train, score the held-out fold of X_eval."""
    import torch, torch.nn as nn
    scores = np.full(len(y), np.nan, np.float32)
    for k in range(max(folds.values()) + 1):
        te = np.array([folds[s] == k for s in sub]) & mask
        tr = np.array([folds[s] != k for s in sub]) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        mu, sd = X_train[tr].mean(0), X_train[tr].std(0) + 1e-6
        xt = torch.tensor((X_train[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        clf = nn.Linear(X_train.shape[1], 1)
        opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores[te] = clf(torch.tensor((X_eval[te] - mu) / sd,
                                          dtype=torch.float32)).squeeze(-1).numpy()
    return scores


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/reversal.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub, direction = lab["subject"], lab["direction"]
    y, mask = (direction == 1).astype(int), direction != -1
    folds = _loso_folds(sub)
    frames = np.load("runs/temporal_aggregation/img_frames.npz")

    log("Reversal control on UNBC direction, image encoders (exact from cached per-frame features)")
    log(f"  {int(mask.sum())} direction-labelled clips, leave-one-subject-out, "
        f"{N_REP} seeded fits per cell")
    log("  the probe is fitted on ORDERED clips and applied to the REVERSED clips of the held-out")
    log("  subject; a signed direction inverts the probe, so the reversed AUC falls below 0.5\n")
    log(f"  {'encoder':10s} {'K':>3s} {'ordered':>9s} {'reversed':>9s} {'95% CI (reversed)':>22s}")

    rows = []
    for e in IMG_ENCODERS:
        f = frames[e]
        for k in KS:
            Xo = image_readout(f, k)
            Xr = image_readout(f[:, ::-1, :], k)             # Reversal preserves the mean and negates the endpoint difference.
            so = sr = None
            for r in range(N_REP):
                torch.manual_seed(1000 + r)
                a = oof_transfer(Xo, Xo, y, sub, mask, folds)
                torch.manual_seed(1000 + r)
                b = oof_transfer(Xo, Xr, y, sub, mask, folds)
                so = a if so is None else so + a
                sr = b if sr is None else sr + b
            so, sr = so / N_REP, sr / N_REP
            keep = mask & ~np.isnan(so) & ~np.isnan(sr)
            auc_o, auc_r = _auc(so[keep], y[keep]), _auc(sr[keep], y[keep])
            lo, hi = boot_auc_ci(sr[keep], y[keep])
            rows.append({"encoder": e, "K": k, "auc_ordered": auc_o, "auc_reversed": auc_r,
                         "ci_reversed": [lo, hi], "n": int(keep.sum())})
            log(f"  {e:10s} {k:3d} {auc_o:9.3f} {auc_r:9.3f}      [{lo:.3f}, {hi:.3f}]")

    rev = [r["auc_reversed"] for r in rows]
    log(f"\n  reversed AUC spans {min(rev):.3f} to {max(rev):.3f} over {len(rows)} cells; "
        f"{sum(1 for v in rev if v < 0.5)} of {len(rows)} fall below chance")
    log("  video encoders are NOT included: their appearance term is a clip embedding that reversal")
    log("  changes, and the reversed embedding is not cached. That arm needs re-extraction.")

    json.dump({"provenance": run_provenance(seeds=list(range(N_REP))),
               "scope": "image encoders only; exact from cached per-frame features",
               "rows": rows}, open(f"{OUT}/reversal.json", "w"), indent=2)
    log(f"\nwrote {OUT}/reversal.json")
    log("Machine-side. GATE H3 decides what is claimed.")
    logf.close()


if __name__ == "__main__":
    main()
