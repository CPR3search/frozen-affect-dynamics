"""Paired permutation tests for the aggregation sweep; Holm helper."""
from __future__ import annotations
import json, os, sys
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, OUT, N_REP
from affectdyn.provenance import run_provenance, audit_loso

NPERM = 20000
NBOOT = 5000


def oof_averaged(X, y, sub, mask, folds, n_rep=N_REP):
    """Out-of-fold per-clip scores, averaged over seeded probe fits."""
    import torch
    acc = None
    for rep in range(n_rep):
        torch.manual_seed(1000 + rep)
        s = _oof_scores(X, y, sub, mask, folds)
        acc = s if acc is None else acc + s
    return acc / n_rep


def perm_test(sa, sb, y, nperm=NPERM, seed=0):
    rng = np.random.default_rng(seed)
    obs = _auc(sa, y) - _auc(sb, y)
    n = len(y)
    cnt = 0
    for _ in range(nperm):
        flip = rng.random(n) < 0.5
        pa = np.where(flip, sb, sa); pb = np.where(flip, sa, sb)
        if abs(_auc(pa, y) - _auc(pb, y)) >= abs(obs) - 1e-12:
            cnt += 1
    return float(obs), float((cnt + 1) / (nperm + 1))


def boot_ci(sa, sb, y, nboot=NBOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        if len(set(y[idx].tolist())) < 2:
            continue
        diffs.append(_auc(sa[idx], y[idx]) - _auc(sb[idx], y[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    """Return Holm-Bonferroni adjusted p values in input order. The caller supplies the complete
    comparison family, including any retained unreported tests; family membership is not inferred
    here.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def main():
    logf = open(f"{OUT}/significance.log", "w")

    def log(msg=""):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    folds = _loso_folds(sub)
    audit_loso(sub, folds)

    IMG = np.load(f"{OUT}/img_frames.npz")
    VJ = np.load(f"{OUT}/vjepa_windows.npz")
    VM = np.load(f"{OUT}/videomae_windows.npz")

    # Build each cell's representation once.
    cells = {}
    for e in IMG_ENCODERS:
        for k in (2, 4, 8, 16):
            cells[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for key in [k for k in VJ.files if k != "full"]:
        cells[f"vjepa_{key}"] = video_readout(VJ["full"], VJ[key])
    for key in [k for k in VM.files if k != "full"]:
        cells[f"videomae_{key}"] = video_readout(VM["full"], VM[key])

    log(f"GATE G0-R significance | {mask.sum()} direction-labelled clips, "
        f"{len(set(sub.tolist()))} subjects, LOSO | {NPERM} permutations, {NBOOT} bootstrap, "
        f"scores averaged over {N_REP} seeded probe fits")

    scores = {}
    for name, X in cells.items():
        scores[name] = oof_averaged(X, y, sub, mask, folds)
        log(f"  scored {name}")
    keep = mask.copy()
    for s in scores.values():
        keep &= ~np.isnan(s)
    yk = y[keep]
    log(f"\nclips retained for testing: {keep.sum()}")
    for name in scores:
        log(f"  pooled OOF AUC  {name:22s} {_auc(scores[name][keep], yk):.3f}")

    def contrast(a, b, label):
        sa, sb = scores[a][keep], scores[b][keep]
        obs, p = perm_test(sa, sb, yk)
        lo, hi = boot_ci(sa, sb, yk)
        return {"label": label, "a": a, "b": b, "auc_a": _auc(sa, yk), "auc_b": _auc(sb, yk),
                "gap": obs, "p": p, "ci": [lo, hi]}

    def report(rows, title, adjust=False):
        log(f"\n=== {title} ===")
        if adjust:
            adj = holm([r["p"] for r in rows])
            for r, a in zip(rows, adj):
                r["p_holm"] = a
        for r in rows:
            extra = f"  Holm-adj p={r['p_holm']:.4f}" if "p_holm" in r else ""
            sig = "*" if (r.get("p_holm", r["p"])) < 0.05 else " "
            log(f" {sig} {r['label']:34s} {r['auc_a']:.3f} vs {r['auc_b']:.3f}  "
                f"gap {r['gap']:+.3f}  95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]  "
                f"p={r['p']:.4f}{extra}")

    res = {"provenance": run_provenance(seeds=[1000 + r for r in range(N_REP)])}

    best = {}
    for k in (2, 4, 8, 16):
        bi = max(IMG_ENCODERS, key=lambda e: _auc(scores[f"{e}_K{k}"][keep], yk))
        vids = [n for n in scores if n.startswith(("vjepa_", "videomae_")) and f"_K{k}_" in n]
        bv = max(vids, key=lambda n: _auc(scores[n][keep], yk))
        best[k] = (f"{bi}_K{k}", bv)
    rows = [contrast(best[k][0], best[k][1], f"K={k}: {best[k][0]} vs {best[k][1]}")
            for k in (2, 4, 8, 16)]
    report(rows, "PRE-DECLARED, best image vs best video at matched K (selection bias both sides)")
    res["predeclared_matched_K"] = rows

    rows = []
    for k in (2, 4, 8, 16):
        for fam, tag in (("videomae", f"videomae_K{k}_up"), ("vjepa", f"vjepa_K{k}_up")):
            if tag in scores:
                rows.append(contrast(f"clip_K{k}", tag, f"K={k}: CLIP vs {fam}"))
    report(rows, "FIXED-ENCODER, CLIP vs each video model at matched K (Holm within family)",
           adjust=True)
    res["fixed_encoder_matched_K"] = rows

    rows = [
        contrast("clip_K16", "vjepa_K2_native", "as submitted: CLIP K=16 vs V-JEPA K=2"),
        contrast("clip_K2", "vjepa_K2_native", "matched coarse: CLIP K=2 vs V-JEPA K=2"),
        contrast("clip_K16", "videomae_K16_up", "matched dense: CLIP K=16 vs VideoMAE K=16"),
    ]
    report(rows, "SECONDARY, size of the published gap versus the matched gap")
    res["published_vs_matched"] = rows

    # Test readout-density differences within each family.
    rows = [
        contrast("clip_K8", "clip_K2", "CLIP: K=8 vs K=2"),
        contrast("clip_K8", "clip_K16", "CLIP: K=8 vs K=16 (endpoint)"),
        contrast("videomae_K8_up", "videomae_K2_up", "VideoMAE: K=8 vs K=2"),
        contrast("vjepa_K4_up", "vjepa_K4_native", "V-JEPA: K=4 upsampled vs native"),
    ]
    report(rows, "SECONDARY, readout density and temporal context within a family")
    res["within_family"] = rows

    json.dump(res, open(f"{OUT}/significance.json", "w"), indent=2)
    log(f"\nwrote {OUT}/significance.json")
    logf.close()


if __name__ == "__main__":
    main()
