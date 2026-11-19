"""Aff-Wild2 peak task and direction for the image encoders."""
from __future__ import annotations
import json, os, sys

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.features.temporal_aggregation import image_readout, IMG_ENCODERS
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.peak.pain_peak_encoder_map import label_perm_test, boot_auc_ci
from affectdyn.provenance import run_provenance

OUT = "runs/affwild_clips_features"
KS = (2, 4, 8, 16)
N_SEED = 3
NFOLD = 5


def grouped_folds(sub, nfold=NFOLD, seed=0):
    """Assign whole clusters to folds. For Aff-Wild2, sub contains video-ID prefixes, not verified
    subject identities.
    """
    subs = sorted(set(sub.tolist()))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subs))
    fold_of = {subs[i]: int(k % nfold) for k, i in enumerate(order)}
    assign = np.array([fold_of[s] for s in sub])
    for k in range(nfold):
        tr, te = set(sub[assign != k].tolist()), set(sub[assign == k].tolist())
        if tr & te:
            raise RuntimeError(f"fold {k}: {len(tr & te)} video-ID groups on both sides")
    return assign


def oof(X, y, assign, mask, seed, dev):
    """Return out-of-fold logits for one seeded fixed-step fit using the supplied cluster-fold
    assignments and eligibility mask. Uses 300 full-batch Adam steps, learning rate 1e-2 and weight
    decay 1e-3, with training-fold standardization; unscored rows remain NaN.
    """
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    scores = np.full(len(y), np.nan, np.float32)
    for k in range(assign.max() + 1):
        te, tr = (assign == k) & mask, (assign != k) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device=dev)
        yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
        xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device=dev)
        clf = nn.Linear(X.shape[1], 1).to(dev)
        opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores[te] = clf(xe).squeeze(-1).cpu().numpy()
    return scores


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(f"{OUT}/probe.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    lab = np.load(f"{OUT}/labels.npz")
    sub, traj, direction = lab["subject"], lab["traj"], lab["direction"]
    frames = np.load(f"{OUT}/img_frames.npz")
    assign = grouped_folds(sub)

    tasks = {
        "shape": ((traj == 2).astype(int), traj != -1),
        "direction": ((direction == 1).astype(int), direction != -1),
    }
    log(f"W3 probe on Aff-Wild2 valence | {len(sub)} clips, {len(set(sub.tolist()))} video-ID groups, "
        f"video-ID-grouped {NFOLD}-fold, {N_SEED} seeded fits | device {dev}")
    for t, (y, m) in tasks.items():
        log(f"  {t}: {int(m.sum())} clips, {int(y[m].sum())} positive "
            f"({100 * y[m].mean():.1f}%), {len(set(sub[m].tolist()))} video-ID groups")
    log(f"  UNBC for comparison: shape 27 peaked of 345 over 24 subjects; direction 219 clips")

    # Historical provenance stores repeat indices here; actual Torch fit seeds are 100+r, as used below.
    res = {"provenance": run_provenance(seeds=list(range(N_SEED))),
           "n_clips": int(len(sub)), "n_subjects": int(len(set(sub.tolist()))), "nfold": NFOLD}

    for task, (y, mask) in tasks.items():
        log(f"\n=== {task} ===")
        log(f"  {'encoder':10s} " + "".join(f"K={k:<20d}" for k in KS))
        rows = []
        for e in IMG_ENCODERS:
            cells = []
            for k in KS:
                X = image_readout(frames[e], k)
                s = np.nanmean([oof(X, y, assign, mask, 100 + r, dev) for r in range(N_SEED)],
                               axis=0)
                keep = mask & ~np.isnan(s)
                auc, p = label_perm_test(s[keep], y[keep])
                lo, hi = boot_auc_ci(s[keep], y[keep])
                cells.append({"encoder": e, "K": k, "auc": auc, "p": p, "ci": [lo, hi],
                              "n": int(keep.sum()), "n_pos": int(y[keep].sum())})
            rows.extend(cells)
        for r, a in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = a
        by = {(r["encoder"], r["K"]): r for r in rows}
        for e in IMG_ENCODERS:
            line = f"  {e:10s} "
            for k in KS:
                r = by[(e, k)]
                star = "*" if r["p_holm"] < 0.05 else ("." if r["p"] < 0.05 else " ")
                line += f"{r['auc']:.3f} p={r['p']:.4f}{star}   "
            log(line)
        med_hw = float(np.median([(r["ci"][1] - r["ci"][0]) / 2 for r in rows]))
        log(f"  significant after Holm: {sum(r['p_holm'] < 0.05 for r in rows)} of {len(rows)}")
        log(f"  median bootstrap CI half-width {med_hw:.3f} AUC "
            f"(UNBC shape was 0.108 with 27 positives)")
        best = max(rows, key=lambda r: r["auc"])
        log(f"  best: {best['encoder']} K={best['K']} AUC {best['auc']:.3f} "
            f"CI [{best['ci'][0]:.3f}, {best['ci'][1]:.3f}] p={best['p']:.4f} "
            f"Holm={best['p_holm']:.4f}")
        res[task] = {"rows": rows, "median_ci_halfwidth": med_hw}

    log("\n=== CLIP peak-task results ===")
    c = {(r["encoder"], r["K"]): r for r in res["shape"]["rows"]}
    for k in KS:
        r = c[("clip", k)]
        log(f"  CLIP shape K={k:2d}: AUC {r['auc']:.3f}  p={r['p']:.4f}  Holm={r['p_holm']:.4f}  "
            f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]   (UNBC K=4 was 0.686, Holm 0.046)")

    json.dump(res, open(f"{OUT}/probe.json", "w"), indent=2)
    log(f"\nwrote {OUT}/probe.json")
    log("Wrote fixed-step probe results; see the cluster-aware inference modules for article decisions.")
    logf.close()


if __name__ == "__main__":
    main()
