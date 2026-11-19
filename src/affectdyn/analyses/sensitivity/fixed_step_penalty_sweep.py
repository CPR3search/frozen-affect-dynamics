"""Direction contrasts across a fixed-step penalty grid.

Usage:  python -m affectdyn.analyses.sensitivity.fixed_step_penalty_sweep
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time, zlib

os.environ["CUDA_VISIBLE_DEVICES"] = ""                    # Run on the CPU.
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, video_readout, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS
from affectdyn.provenance import run_provenance, audit_loso

OUT = "runs/fixed_step_penalty_sweep"
SEED = 20260920
W9_SEED = 20260911
NBOOT = 10_000
ALPHA = 0.05
TOL = 0.005
LR = 1e-2
WDS = (1e-4, 1e-3, 1e-2, 1e-1)
STEPS = (300, 1000)
PUBLISHED = (1e-3, 300)
MASKED_AE = {"videomae", "videomaev2"}
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-16 — W32-W34 PRE-DECLARED CRITERIA"


def _members(groups):
    codes = np.unique(groups, return_inverse=True)[1]
    return [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]


def paired_bootstrap(sa, sb, y, groups, rng, nboot):
    mem = _members(groups); G = len(mem); vals = []
    for _ in range(nboot):
        i = np.concatenate([mem[g] for g in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(_auc(sa[i], y[i]) - _auc(sb[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"ci": [float(lo), float(hi)], "p_boot": float(min(1.0, max(p, 1.0 / len(v))))}


def oof_scores(X, y, sub, mask, folds, wd, steps, lr=LR):
    """stage1d_significance._oof_scores with weight decay and step count exposed."""
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
        opt = torch.optim.Adam(clf.parameters(), lr, weight_decay=wd)
        for _ in range(steps):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores[te] = clf(torch.tensor((X[te] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
    return scores


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    od = args.out_dir
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(od, exist_ok=True)
    out_json = f"{od}/operating_point.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{od}/operating_point.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(1)
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]
    y = (lab["direction"] == 1).astype(int)
    mask = lab["direction"] != -1
    folds = _loso_folds(sub); audit_loso(sub, folds)
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    X = {f"clip_K{k}": image_readout(IMG["clip"], k) for k in KS}
    for name, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            key = tmpl.format(k=k)
            if key in z.files:
                X[f"{name}_K{k}"] = video_readout(z["full"], z[key])
    M = json.load(open("runs/pain_encoder_maps/map.json"))
    logged_rows = [r for r in M["primary"] if r["a"] in X and r["b"] in X]
    log(f"W32 | {len(X)} cells, {len(logged_rows)} contrasts | grid {len(WDS)}x{len(STEPS)} | seed {SEED}")

    def rng_for(name):
        return np.random.default_rng([W9_SEED, zlib.crc32(name.encode())])

    settings, gate = {}, None
    for wd in WDS:
        for st in STEPS:
            tag = f"wd{wd:g}_steps{st}"
            t1 = time.time()
            pooled = {}
            for name, Xc in X.items():
                reps = []
                for rep in range(N_REP):
                    torch.manual_seed(1000 + rep)
                    reps.append(oof_scores(Xc, y, sub, mask, folds, wd, st))
                pooled[name] = np.mean(reps, axis=0)
            keep = mask.copy()
            for s in pooled.values():
                keep &= ~np.isnan(s)
            yk, gk = y[keep].astype(np.int64), sub[keep]
            cells = {n: float(_auc(s[keep], yk)) for n, s in pooled.items()}
            np.savez_compressed(f"{od}/scores_{tag}.npz", keep=keep, y=y, subject=sub, **pooled)
            rows = []
            for r in logged_rows:
                sa, sb = pooled[r["a"]][keep], pooled[r["b"]][keep]
                b = paired_bootstrap(sa, sb, yk, gk, rng_for(r["label"] + ":subject"), NBOOT)
                rows.append({"label": r["label"], "a": r["a"], "b": r["b"], "video": r["b"].rsplit("_K", 1)[0],
                             "gap": float(_auc(sa, yk) - _auc(sb, yk)), "p": b["p_boot"], "ci": b["ci"],
                             "logged_gap": float(r["gap"])})
            for r, a in zip(rows, holm([x["p"] for x in rows])):
                r["holm"] = a; r["survives"] = bool(a < ALPHA and r["gap"] > 0)
            surv = sorted({r["video"] for r in rows if r["survives"]})
            outside = [v for v in surv if v not in MASKED_AE]
            settings[tag] = {"weight_decay": wd, "steps": st, "cells": cells, "contrasts": rows,
                             "n_survivors": sum(r["survives"] for r in rows), "survivor_encoders": surv,
                             "survivors_outside_masked_ae": outside,
                             "survivor_labels": [r["label"] for r in rows if r["survives"]]}
            log(f"  {tag:18s} survivors {settings[tag]['n_survivors']} {surv} outside {outside} "
                f"({time.time() - t1:.0f}s)")
            if (wd, st) == PUBLISHED:
                dmax = max(abs(cells[n] - float(M["table"][n]["pooled_oof"])) for n in cells)
                gmax = max(abs(r["gap"] - r["logged_gap"]) for r in rows)
                pub = sorted(r["label"] for r in rows if r["survives"])
                W9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))["contrasts"]
                w9_surv = sorted(r["label"] for r in W9 if r["holm"]["boot_subject"] < ALPHA)
                gate = {"setting": tag, "max_abs_cell_delta": dmax, "max_abs_gap_delta": gmax,
                        "survivors": pub, "w9_survivors": w9_surv, "n_survivors": len(pub),
                        "rule": "AUCs and gaps within 0.005 of runs/pain_encoder_maps AND the survivor set equal to runs/pain_direction_cluster_inference",
                        "passes": bool(dmax <= TOL and gmax <= TOL and pub == w9_surv)}
                log(f"  gate: {gate}")

    counts = {k: v["n_survivors"] for k, v in settings.items()}
    any_outside = {k: v["survivors_outside_masked_ae"] for k, v in settings.items() if v["survivors_outside_masked_ae"]}
    robust = bool(gate and gate["passes"] and all(4 <= c <= 6 for c in counts.values()) and not any_outside)
    verdict = {"gate_passes": bool(gate and gate["passes"]), "survivor_counts": counts,
               "count_range": [min(counts.values()), max(counts.values())],
               "settings_with_survivors_outside_masked_ae": any_outside,
               "robust_to_operating_point": robust,
               "reporting_rule": ("robust -> the verdicts hold across the declared grid, which is named; "
                                  "otherwise the separation depends on the operating point and the range is given")}
    log(f"verdict: {verdict}")
    res = {"provenance": run_provenance(seeds=[SEED, W9_SEED], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "grid": {"lr": LR, "weight_decay": list(WDS), "steps": list(STEPS), "published": list(PUBLISHED)},
           "nboot": NBOOT, "gate": gate, "settings": settings, "verdict": verdict,
           "status": "COMPLETE" if (gate and gate["passes"]) else "HOLD: gate failed"}
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
