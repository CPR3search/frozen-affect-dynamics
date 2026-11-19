"""Shuffled and appearance-only controls for CLIP on the pain peak task."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, N_REP
from affectdyn.inference.aggregation_permutation_tests import perm_test, boot_ci
from affectdyn.analyses.peak.pain_peak_encoder_map import label_perm_test, boot_auc_ci
from affectdyn.analyses.direction.pain_direction_encoder_map import KS, OUT
from affectdyn.provenance import run_provenance, audit_loso

ENCODER = "clip"
N_SHUFFLE = 5


def _scores(X, y, sub, mask, folds, seed0=3000):
    import torch
    acc = None
    for rep in range(N_REP):
        torch.manual_seed(seed0 + rep)
        s = _oof_scores(X, y, sub, mask, folds)
        acc = s if acc is None else acc + s
    return acc / N_REP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    out = ap.parse_args().out_dir
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/shape_control.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub, traj = lab["subject"], lab["traj"]
    y = (traj == 2).astype(int)
    mask = traj != -1
    folds = _loso_folds(sub)
    audit_loso(sub, folds)
    frames = np.load("runs/temporal_aggregation/img_frames.npz")[ENCODER]   # Frame features have shape [N, T, D].

    log(f"Order-destroying controls on the surviving shape cell | encoder {ENCODER} | "
        f"{int(y[mask].sum())} peaked vs {int(mask.sum() - y[mask].sum())} monotonic | "
        f"LOSO, pooled out-of-fold | {N_SHUFFLE} shuffle seeds, {N_REP} probe fits per cell")

    # Use only the temporal mean for the static control.
    static = frames.mean(1)
    s_static = _scores(static, y, sub, mask, folds)

    results = {}
    all_scores = {"static": s_static}
    for k in KS:
        ordered = image_readout(frames, k)
        s_ord = _scores(ordered, y, sub, mask, folds)
        s_shufs = []
        for sh in range(N_SHUFFLE):
            rng = np.random.default_rng(500 + sh)
            perm = np.stack([rng.permutation(frames.shape[1]) for _ in range(frames.shape[0])])
            shuffled = np.take_along_axis(frames, perm[:, :, None], axis=1)
            s_shufs.append(_scores(image_readout(shuffled, k), y, sub, mask, folds,
                                   seed0=4000 + 10 * sh))
        all_scores[f"ord_K{k}"] = s_ord
        all_scores[f"shuf_K{k}"] = np.mean(s_shufs, axis=0)
        results[k] = {"shuffle_seeds": N_SHUFFLE}
        log(f"  scored K={k}")

    keep = mask.copy()
    for s in all_scores.values():
        keep &= ~np.isnan(s)
    yk = y[keep]
    log(f"\nclips retained: {keep.sum()} ({int(yk.sum())} peaked)")
    np.savez(f"{out}/shape_control_scores.npz", keep=keep, y=y, **all_scores)

    a_static, p_static = label_perm_test(s_static[keep], yk)
    lo_s, hi_s = boot_auc_ci(s_static[keep], yk)
    log(f"\n=== static control: temporal mean only, endpoint term removed ===")
    log(f"  AUC {a_static:.3f}  95% CI [{lo_s:.3f}, {hi_s:.3f}]  vs chance p={p_static:.4f}")
    log("  If this were near the ordered value, the shape signal would be appearance, not order.")

    log(f"\n=== ordered vs shuffled, per readout density ===")
    for k in KS:
        so, ss = all_scores[f"ord_K{k}"][keep], all_scores[f"shuf_K{k}"][keep]
        ao, a_s = _auc(so, yk), _auc(ss, yk)
        gap, p = perm_test(so, ss, yk)
        lo, hi = boot_ci(so, ss, yk)
        _, p_ord = label_perm_test(so, yk)
        _, p_shuf = label_perm_test(ss, yk)
        results[k].update({"auc_ordered": ao, "auc_shuffled": a_s, "gap": gap, "p_gap": p,
                           "ci_gap": [lo, hi], "p_ordered_vs_chance": p_ord,
                           "p_shuffled_vs_chance": p_shuf})
        star = "*" if p < 0.05 else " "
        log(f" {star} K={k:2d}  ordered {ao:.3f} (vs chance p={p_ord:.4f})   "
            f"shuffled {a_s:.3f} (p={p_shuf:.4f})   gap {gap:+.3f}  "
            f"95% CI [{lo:+.3f}, {hi:+.3f}]  p={p:.4f}")

    k4 = results[4]
    log(f"\n=== reading, for the cell that survived Holm (K=4) ===")
    log(f"  ordered {k4['auc_ordered']:.3f}, shuffled {k4['auc_shuffled']:.3f}, "
        f"gap {k4['gap']:+.3f}, p={k4['p_gap']:.4f}; static {a_static:.3f}")
    if k4["p_gap"] < 0.05 and a_static < k4["auc_ordered"] - 0.05:
        log("  The signal is order-dependent: destroying frame order costs it, and the appearance-"
            "only readout does not recover it. Still consistent with a boundary cue rather than an "
            "interior-trajectory representation; the controls cannot separate those two.")
    else:
        log("  The signal is NOT established as order-dependent. It is consistent with static "
            "appearance, and the surviving shape cell must be reported as such rather than as "
            "evidence of temporal content.")

    json.dump({"provenance": run_provenance(seeds=[3000 + r for r in range(N_REP)], device="cpu"),
               "encoder": ENCODER, "static": {"auc": a_static, "p": p_static, "ci": [lo_s, hi_s]},
               "by_K": results, "n_clips": int(keep.sum()), "n_peaked": int(yk.sum())},
              open(f"{out}/shape_control.json", "w"), indent=2)
    log(f"\nwrote {out}/shape_control.json")
    log("Machine-side. GATE H3 decides what the manuscript claims from this.")
    logf.close()


if __name__ == "__main__":
    main()
