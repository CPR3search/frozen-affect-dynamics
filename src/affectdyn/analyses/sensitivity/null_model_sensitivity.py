"""Sensitivity of the clustered-test decisions to the null model."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import itertools, json, os, sys, time

import numpy as np

import affectdyn.inference.calibrated_clustered_auc as w12
from affectdyn.inference.calibrated_clustered_auc import ALPHA, Design, holm, sim_paired, sim_single, wilson
from affectdyn.provenance import run_provenance

OUT = "runs/null_model_sensitivity"
SEED = 20260912
N_NULL = 20_000

THETA_DIR = (0.80, 0.8724, 0.93)
RHO_DIR = (0.30, 0.5431, 0.75)
ICCS = (0.0, 0.3, 0.6)
W12_ICC = 0.3


def set_icc(icc):
    """Set calibrated_clustered_auc.ICC, which Design.draw reads at call time."""
    w12.ICC = icc


def calibrated_survivors(rows, nulls):
    """Calibrated p per test against the null of ITS OWN clip set, then Holm across the whole family."""
    ps = []
    for r in rows:
        a = np.abs(nulls[r["null"]])
        ps.append(float((1 + (a >= abs(r["z"])).sum()) / (1 + len(a))))
    return [r["test"] for r, adj in zip(rows, holm(ps)) if adj < ALPHA]


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/null_robustness.log", "w")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    W9 = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    C12 = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    rng = np.random.default_rng(SEED)

    dkeep = W9["keep"].astype(bool)
    d_dir = Design("unbc_direction", W9["y"][dkeep].astype(np.int64), W9["subject"][dkeep], 24)
    shape = {}
    for name, pre, mask, m in (("unbc_shape", "unbc", "keep", 36), ("aw2_full", "aw2img", "mask", 18),
                               ("aw2_subset", "aw2vid", "mask", 18)):
        k = W7[f"{pre}__{mask}"].astype(bool)
        shape[name] = Design(name, W7[f"{pre}__y"][k].astype(np.int64), W7[f"{pre}__subject"][k], m)

    res = {"provenance": run_provenance(seeds=[SEED]), "seed": SEED, "n_null": N_NULL,
           "grid": {"theta_direction": list(THETA_DIR), "rho_direction": list(RHO_DIR),
                    "icc": list(ICCS)},
           "w12_configuration": {"theta_direction": THETA_DIR[1], "rho_direction": RHO_DIR[1],
                                 "icc": W12_ICC},
           "families": {}}
    p(f"W19 null robustness | {N_NULL} null studies per configuration | seed {SEED}")

    # Evaluate 24 direction contrasts across theta, rho, and ICC.
    rows = C12["families"]["direction_contrasts_24"]["rows"]
    cells = []
    for r in rows:
        r["null"] = "dir"
    for th, rh, icc in itertools.product(THETA_DIR, RHO_DIR, ICCS):
        set_icc(icc)
        z0 = sim_paired(d_dir, th, rh, rng, N_NULL)
        surv = calibrated_survivors(rows, {"dir": z0})
        a_over_m = ALPHA / 24
        k = int((np.abs(z0) >= np.quantile(np.abs(z0), 1 - a_over_m)).sum())
        cells.append({"theta": th, "rho": rh, "icc": icc, "n_survivors": len(surv),
                      "survivors": surv, "null_rate_at_alpha_over_m": k / N_NULL,
                      "wilson": wilson(k, N_NULL),
                      "is_w12_configuration": bool(th == THETA_DIR[1] and rh == RHO_DIR[1] and icc == W12_ICC)})
        p(f"  contrasts theta {th:.4f} rho {rh:.4f} icc {icc:.1f} -> {len(surv)} survive")
    counts = sorted({c["n_survivors"] for c in cells})
    base = next(c for c in cells if c["is_w12_configuration"])
    res["families"]["direction_contrasts_24"] = {
        "n_tests": 24, "configurations": len(cells), "survivor_counts": counts,
        "robust": len(counts) == 1, "at_w12_configuration": base["n_survivors"],
        "survivors_at_w12": base["survivors"],
        "tests_that_ever_change": sorted({t for c in cells for t in set(c["survivors"]) ^ set(base["survivors"])}),
        "cells": cells}

    # Evaluate shape families with null AUC fixed at one half.
    for r in C12["families"]["unbc_shape_36"]["rows"]:
        r["null"] = "unbc_shape"
    for r in C12["families"]["aw2_shape_18"]["rows"]:
        r["null"] = "aw2_full" if r["test"].startswith("aw2_image:") else "aw2_subset"

    for fam, rowset, designs in (("unbc_shape_36", C12["families"]["unbc_shape_36"]["rows"],
                                  ["unbc_shape"]),
                                 ("aw2_shape_18", C12["families"]["aw2_shape_18"]["rows"],
                                  ["aw2_full", "aw2_subset"])):
        cells = []
        for icc in ICCS:
            set_icc(icc)
            nulls = {n: sim_single(shape[n], 0.5, rng, N_NULL)[0] for n in designs}
            surv = calibrated_survivors(rowset, nulls)
            probe = shape[designs[0]]
            a_over_m = ALPHA / probe.m
            z0 = nulls[designs[0]]
            k = int((np.abs(z0) >= np.quantile(np.abs(z0), 1 - a_over_m)).sum())
            cells.append({"icc": icc, "n_survivors": len(surv), "survivors": surv,
                          "null_rate_at_alpha_over_m": k / N_NULL, "wilson": wilson(k, N_NULL),
                          "is_w12_configuration": bool(icc == W12_ICC)})
            p(f"  {fam} icc {icc:.1f} -> {len(surv)} of {len(rowset)} survive")
        counts = sorted({c["n_survivors"] for c in cells})
        base = next(c for c in cells if c["is_w12_configuration"])
        res["families"][fam] = {"n_tests": len(rowset), "configurations": len(cells),
                                "survivor_counts": counts, "robust": len(counts) == 1,
                                "at_w12_configuration": base["n_survivors"],
                                "survivors_at_w12": base["survivors"],
                                "tests_that_ever_change": sorted({t for c in cells
                                                                  for t in set(c["survivors"]) ^ set(base["survivors"])}),
                                "cells": cells}
    set_icc(W12_ICC)

    expect = {"direction_contrasts_24": 4, "unbc_shape_36": 0, "aw2_shape_18": 5}
    got = {k: v["at_w12_configuration"] for k, v in res["families"].items()}
    res["gate"] = {"expected_at_w12_configuration": expect, "observed": got,
                   "passes": all(got[k] == expect[k] for k in expect if k in got)}
    res["summary"] = {k: {"counts": v["survivor_counts"], "robust": v["robust"]}
                      for k, v in res["families"].items()}
    res["status"] = "COMPLETE" if res["gate"]["passes"] else "COMPLETE WITH GATE MISMATCH"
    json.dump(res, open(f"{OUT}/null_robustness.json", "w"), indent=2)
    p(f"\ngate: expected {expect}, observed {got}, passes {res['gate']['passes']}")
    for k, v in res["summary"].items():
        p(f"  {k:22s} survivor counts across the grid {v['counts']}  robust {v['robust']}")
    p(f"wrote {OUT}/null_robustness.json | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
