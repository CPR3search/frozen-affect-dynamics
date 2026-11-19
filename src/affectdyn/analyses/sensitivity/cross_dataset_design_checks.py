"""Pain direction under 5-fold partitions and other structural checks."""
from __future__ import annotations
import json, os, sys

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES
from affectdyn.analyses.peak.pain_peak_encoder_map import label_perm_test, boot_auc_ci
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof as oof_grouped
from affectdyn.provenance import run_provenance

OUT = "runs/cross_dataset_encoder_checks"
K = 16


def main():
    import torch
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(f"{OUT}/confounds.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    res = {"provenance": run_provenance(seeds=list(range(N_REP)))}

    log("=== R1.1  UNBC shape: leave-one-subject-out against subject-grouped five-fold ===")
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub_u, traj_u = lab["subject"], lab["traj"]
    y_u, m_u = (traj_u == 2).astype(int), traj_u != -1
    folds_loso = _loso_folds(sub_u)
    assign5 = grouped_folds(sub_u, nfold=5, seed=0)

    cells = {}
    IMG_U = np.load("runs/temporal_aggregation/img_frames.npz")
    for e in IMG_ENCODERS:
        cells[e] = image_readout(IMG_U[e], K)
    for name, (path, tmpl) in VIDEO_SOURCES.items():
        if os.path.exists(path):
            z = np.load(path)
            key = tmpl.format(k=K)
            if key in z.files:
                cells[name] = video_readout(z["full"], z[key])

    log(f"  {'encoder':14s} {'LOSO':>8s} {'grouped-5':>11s} {'delta':>8s}")
    rows = []
    for name, X in cells.items():
        a = None
        for r in range(N_REP):
            torch.manual_seed(1000 + r)
            s = _oof_scores(X, y_u, sub_u, m_u, folds_loso)
            a = s if a is None else a + s
        s_loso = a / N_REP
        s_g5 = np.nanmean([oof_grouped(X, y_u, assign5, m_u, 1000 + r, dev)
                           for r in range(N_REP)], axis=0)
        keep = m_u & ~np.isnan(s_loso) & ~np.isnan(s_g5)
        auc_l, auc_g = _auc(s_loso[keep], y_u[keep]), _auc(s_g5[keep], y_u[keep])
        rows.append({"encoder": name, "auc_loso": auc_l, "auc_grouped5": auc_g,
                     "delta": auc_g - auc_l})
        log(f"  {name:14s} {auc_l:8.3f} {auc_g:11.3f} {auc_g - auc_l:+8.3f}")
    d = [abs(r["delta"]) for r in rows]
    log(f"\n  median |delta| {np.median(d):.3f}, max {max(d):.3f}")
    log("  A small delta means the cross-corpus comparison is not an estimator artefact.")
    res["r1_1_split_protocol"] = rows

    log("\n=== R1.2  Aff-Wild2: image encoders re-probed on the SAME 2,500-clip subset as video ===")
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    labw = np.load("runs/affwild_clips_features/labels.npz")
    sub_w = labw["subject"][idx]
    traj_w, dir_w = labw["traj"][idx], labw["direction"][idx]
    assign_w = grouped_folds(sub_w)
    IMG_W = np.load("runs/affwild_clips_features/img_frames.npz")

    for task, (yw, mw) in {"shape": ((traj_w == 2).astype(int), traj_w != -1),
                           "direction": ((dir_w == 1).astype(int), dir_w != -1)}.items():
        log(f"\n  --- {task}: {int(mw.sum())} clips, {int(yw[mw].sum())} positive (matched to video) ---")
        out = []
        for e in IMG_ENCODERS:
            X = image_readout(IMG_W[e][idx], K)
            s = np.nanmean([oof_grouped(X, yw, assign_w, mw, 400 + r, dev)
                            for r in range(N_REP)], axis=0)
            keep = mw & ~np.isnan(s)
            auc, p = label_perm_test(s[keep], yw[keep])
            lo, hi = boot_auc_ci(s[keep], yw[keep])
            out.append({"encoder": e, "auc": auc, "p": p, "ci": [lo, hi],
                        "n": int(keep.sum()), "n_pos": int(yw[keep].sum())})
        for r, a in zip(out, holm([r["p"] for r in out])):
            r["p_holm"] = a
        for r in out:
            star = "*" if r["p_holm"] < 0.05 else " "
            log(f"   {star} {r['encoder']:10s} AUC {r['auc']:.3f}  "
                f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  p={r['p']:.4f}  Holm={r['p_holm']:.4f}")
        res[f"r1_2_matched_subset_{task}"] = out

    log("\n=== R3.1  detection floor beside every corpus, so nulls are interpretable ===")
    try:
        u = json.load(open("runs/pain_encoder_maps/shape.json"))
        w = json.load(open("runs/affwild_clips_features/probe.json"))
        fl_u = 0.5 + u["median_ci_halfwidth"]
        fl_w = 0.5 + w["shape"]["median_ci_halfwidth"]
        log(f"  UNBC      {u['n_peaked']} peaked, CI half-width {u['median_ci_halfwidth']:.3f}, "
            f"detects nothing below AUC {fl_u:.3f}")
        log(f"  Aff-Wild2 (full) CI half-width {w['shape']['median_ci_halfwidth']:.3f}, "
            f"floor {fl_w:.3f}")
        sub_floor = float(np.median([(r["ci"][1] - r["ci"][0]) / 2
                                     for r in res["r1_2_matched_subset_shape"]]))
        log(f"  Aff-Wild2 (2,500 subset) CI half-width {sub_floor:.3f}, "
            f"floor {0.5 + sub_floor:.3f}")
        res["detection_floors"] = {"unbc": fl_u, "affwild_full": fl_w,
                                   "affwild_subset": 0.5 + sub_floor}
    except Exception as e:                                             # noqa: BLE001
        log(f"  unavailable: {type(e).__name__}: {e}")

    json.dump(res, open(f"{OUT}/confounds.json", "w"), indent=2)
    log(f"\nwrote {OUT}/confounds.json")
    log("Machine-side. GATE H3 decides what is claimed.")
    logf.close()


if __name__ == "__main__":
    main()
