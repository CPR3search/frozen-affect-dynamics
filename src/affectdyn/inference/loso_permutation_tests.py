"""Paired permutation tests and leave-one-subject-out fold helpers.

The reusable article helper is _oof_scores. The module's __main__ also runs a legacy emotion
comparison requiring runs/affwild_emotion_features/emotion_feats.npz, whose producer is not included
in this release; it is not the article's Aff-Wild2 valence pipeline.
"""
from __future__ import annotations
import os, sys
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc


def _oof_scores(X, y, sub, mask, folds):
    """Return float32 out-of-fold logits from one 300-step full-batch Adam fit per fold, with learning
    rate 1e-2 and coupled weight decay 1e-3 on weights and bias. Standardize using eligible training
    rows only. The caller sets the Torch seed and averages repeated fits; excluded or unscored rows
    remain NaN.
    """
    import torch, torch.nn as nn
    scores = np.full(len(y), np.nan, np.float32)
    nfold = max(folds.values()) + 1
    for k in range(nfold):
        te = np.array([folds[s] == k for s in sub]) & mask
        tr = np.array([folds[s] != k for s in sub]) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        clf = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores[te] = clf(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
    return scores


def _perm_test(sa, sb, y, nperm=20000, seed=0):
    """Paired permutation test on AUC(a) - AUC(b). sa, sb clip-aligned; y binary."""
    rng = np.random.default_rng(seed)
    obs = _auc(sa, y) - _auc(sb, y)
    n = len(y)
    cnt = 0
    for _ in range(nperm):
        flip = rng.random(n) < 0.5
        pa = np.where(flip, sb, sa); pb = np.where(flip, sa, sb)
        if abs(_auc(pa, y) - _auc(pb, y)) >= abs(obs) - 1e-12:
            cnt += 1
    p = (cnt + 1) / (nperm + 1)
    return obs, p


def _loso_folds(sub):
    subs = sorted(set(sub.tolist()))
    return {s: i for i, s in enumerate(subs)}              # Assign one held-out fold per subject.


def _kfold_folds(sub, k=5, seed=0):
    subs = sorted(set(sub.tolist())); rng = np.random.default_rng(seed)
    return {s: int(rng.integers(k)) for s in subs}


def run_unbc():
    cp = np.load("runs/unbc_pretraining_comparison/crosspretrain_feats.npz")
    vj = np.load("runs/unbc_vjepa_features/vjepa_fair.npz")
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int); mask = lab["direction"] != -1
    folds = _loso_folds(sub)
    clip = _oof_scores(cp["clip_tord"], y, sub, mask, folds)
    vjep = _oof_scores(vj["vord_temporal"], y, sub, mask, folds)
    keep = mask & ~np.isnan(clip) & ~np.isnan(vjep)
    a_clip, a_vj = _auc(clip[keep], y[keep]), _auc(vjep[keep], y[keep])
    obs, p = _perm_test(clip[keep], vjep[keep], y[keep])
    print(f"\n=== UNBC pain, P1 direction (LOSO, {keep.sum()} clips) ===")
    print(f"  CLIP temporal   OOF-AUC {a_clip:.3f}")
    print(f"  V-JEPA-2 (fair) OOF-AUC {a_vj:.3f}")
    print(f"  gap (CLIP - V-JEPA) = {obs:+.3f}   paired-permutation p = {p:.4f}")


def run_emotion():
    d = np.load("runs/affwild_emotion_features/emotion_feats.npz")
    sub = d["subject"]; y = d["y"].astype(int); mask = np.ones(len(y), bool)
    folds = _kfold_folds(sub, 5, seed=0)
    clip = _oof_scores(d["clip_temporal"], y, sub, mask, folds)
    vjep = _oof_scores(d["vjepa_temporal"], y, sub, mask, folds)
    keep = ~np.isnan(clip) & ~np.isnan(vjep)
    a_clip, a_vj = _auc(clip[keep], y[keep]), _auc(vjep[keep], y[keep])
    obs, p = _perm_test(clip[keep], vjep[keep], y[keep])
    print(f"\n=== Aff-Wild2 emotion, onset-vs-offset (5-fold, {keep.sum()} clips) ===")
    print(f"  CLIP temporal   OOF-AUC {a_clip:.3f}")
    print(f"  V-JEPA-2 temporal OOF-AUC {a_vj:.3f}")
    print(f"  gap (CLIP - V-JEPA) = {obs:+.3f}   paired-permutation p = {p:.4f}")


if __name__ == "__main__":
    run_unbc()
    run_emotion()
