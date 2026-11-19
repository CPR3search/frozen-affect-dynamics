"""Aff-Wild2 peak task and direction for the video encoders."""
from __future__ import annotations
import json, os, sys

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.features.temporal_aggregation import video_readout, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.peak.pain_peak_encoder_map import label_perm_test, boot_auc_ci
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof
from affectdyn.provenance import run_provenance

OUT = "runs/affwild_clips_features"
MODELS = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")


def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(f"{OUT}/video_probe.log", "w")

    def log(m=""):
        print(m, flush=True); logf.write(m + "\n"); logf.flush()

    idx = np.load(f"{OUT}/video_subset_idx.npy")
    lab = np.load(f"{OUT}/labels.npz")
    sub = lab["subject"][idx]
    traj, direction = lab["traj"][idx], lab["direction"][idx]
    assign = grouped_folds(sub)
    tasks = {"shape": ((traj == 2).astype(int), traj != -1),
             "direction": ((direction == 1).astype(int), direction != -1)}

    log(f"W3 video probe | {len(idx)} clips, {len(set(sub.tolist()))} subjects, "
        f"subject-grouped 5-fold, K=16 matched readout, {N_REP} seeded fits | device {dev}")
    for t, (y, m) in tasks.items():
        log(f"  {t}: {int(m.sum())} clips, {int(y[m].sum())} positive")
    log("  UNBC comparison values come from runs/pain_encoder_maps/shape.json and runs/pain_encoder_maps/map.json\n")

    cells = {}
    for name in MODELS:
        p = f"{OUT}/{name}_windows.npz"
        if not os.path.exists(p):
            log(f"  [{name}] cache absent, skipped"); continue
        z = np.load(p)
        cells[name] = video_readout(z["full"], z["K16_up"])

    res = {"provenance": run_provenance(seeds=list(range(N_REP))), "n_clips": int(len(idx))}
    for task, (y, mask) in tasks.items():
        log(f"=== {task} ===")
        rows = []
        for name, X in cells.items():
            s = np.nanmean([oof(X, y, assign, mask, 300 + r, dev) for r in range(N_REP)], axis=0)
            keep = mask & ~np.isnan(s)
            auc, p = label_perm_test(s[keep], y[keep])
            lo, hi = boot_auc_ci(s[keep], y[keep])
            rows.append({"encoder": name, "auc": auc, "p": p, "ci": [lo, hi],
                         "n": int(keep.sum()), "n_pos": int(y[keep].sum())})
        for r, a in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = a
        for r in rows:
            star = "*" if r["p_holm"] < 0.05 else ("." if r["p"] < 0.05 else " ")
            log(f" {star} {r['encoder']:14s} AUC {r['auc']:.3f}  "
                f"CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  p={r['p']:.4f}  "
                f"Holm={r['p_holm']:.4f}")
        log(f"  significant after Holm: {sum(r['p_holm'] < 0.05 for r in rows)} of {len(rows)}\n")
        res[task] = rows

    log("=== the dissociation question ===")
    try:
        unbc = {r["cell"].rsplit("_K", 1)[0]: r for r in json.load(open("runs/pain_encoder_maps/shape.json"))["rows"]
                if r["cell"].endswith("_K16")}
        log(f"  {'encoder':14s} {'UNBC shape':>12s} {'Aff-Wild2 shape':>17s}   reading")
        for r in res["shape"]:
            e = r["encoder"]
            u = unbc.get(e)
            if u is None:
                log(f"  {e:14s} {'n/a':>12s} {r['auc']:17.3f}   (not in the UNBC sweep)")
                continue
            us, ws = u["p_holm"] < 0.05, r["p_holm"] < 0.05
            read = ("BOTH corpora" if us and ws else
                    "UNBC only" if us else "Aff-Wild2 only" if ws else "neither")
            log(f"  {e:14s} {u['auc']:12.3f} {r['auc']:17.3f}   {read}")
    except Exception as e:                                             # noqa: BLE001
        log(f"  UNBC comparison unavailable: {type(e).__name__}: {e}")

    json.dump(res, open(f"{OUT}/video_probe.json", "w"), indent=2)
    log(f"\nwrote {OUT}/video_probe.json")
    log("Machine-side. GATE H3 decides what is claimed.")
    logf.close()


if __name__ == "__main__":
    main()
