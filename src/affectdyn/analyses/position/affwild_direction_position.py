"""Position diagnostics for the Aff-Wild2 direction labels.

Usage:  python -m affectdyn.analyses.position.affwild_direction_position
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import json, os, re, sys, time

import numpy as np

from affectdyn.analyses.sensitivity.auc_tie_audit import midrank                                        # noqa: E402
from affectdyn.provenance import run_provenance                                          # noqa: E402

OUT = "runs/affwild_transfer_test"
SEED = 20260927
NBOOT = 10_000
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-09-26 — W63 PRE-DECLARED CRITERIA"


def members(groups):
    codes = np.unique(groups, return_inverse=True)[1]
    return [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]


def midrank_boot(x, y, groups, rng, nboot=NBOOT):
    """Subject bootstrap of the MIDRANK AUC against one half, ties at half credit in every replicate."""
    mem = members(groups); G = len(mem)
    vals = []
    for _ in range(nboot):
        i = np.concatenate([mem[k] for k in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(midrank(x[i].astype(np.float64), y[i].astype(np.int64)))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"auc": midrank(x.astype(np.float64), y.astype(np.int64)),
            "ci": [float(lo), float(hi)], "ci_contains_half": bool(lo <= 0.5 <= hi),
            "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "resamples_skipped": int(nboot - len(v)),
            "estimator": "midrank AUC in the point estimate AND in every bootstrap replicate"}


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    def log(m=""):
        print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    lab = np.load("runs/affwild_clips_features/labels.npz", allow_pickle=True)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    man = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    subj = np.array([str(s) for s in lab["subject"]])
    d_all = lab["direction"]
    assert len(man) == len(subj)

    # Compute relative position by sorting parsed first-frame indices.
    start = np.array([float(re.findall(r"(\d+)", os.path.basename(c["start"]))[0]) for c in man])
    rel = np.empty(len(subj))
    for s in np.unique(subj):
        ix = np.flatnonzero(subj == s)
        o = np.argsort(start[ix], kind="mergesort")
        r = np.empty(len(ix)); r[o] = np.arange(len(ix))
        rel[ix] = r / max(1, len(ix) - 1)

    sub_m = np.zeros(len(subj), bool); sub_m[idx] = True
    lab_m = d_all != -1
    out = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRIT,
           "supersedes": "the position_diagnostics block of runs/affwild_transfer_test/transfer.json",
           "nboot": NBOOT, "diagnostics": {}}
    specs = [("subset_relative_position", rel, sub_m & lab_m, 11),
             ("subset_start_frame", start, sub_m & lab_m, 12),
             ("full_set_relative_position", rel, lab_m, 13),
             ("full_set_start_frame", start, lab_m, 14)]
    for name, x, m, off in specs:
        y = (d_all[m] == 1).astype(np.int64)
        r = midrank_boot(x[m], y, subj[m], np.random.default_rng(SEED + off))
        r["n"] = int(m.sum()); r["n_groups"] = int(len(np.unique(subj[m])))
        out["diagnostics"][name] = r
        log(f"{name:28s} n {r['n']:6d} auc {r['auc']:.4f} CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] "
            f"contains_half {r['ci_contains_half']} p {r['p_boot']:.4f}")

    # Compare estimators on the same variable and identical draws.
    from affectdyn.features.unbc_pretraining_comparison import _auc
    y_f = (d_all[lab_m] == 1).astype(np.int64)
    mem = members(subj[lab_m]); G = len(mem)
    rng = np.random.default_rng(SEED + 13)
    v_sort = []
    for _ in range(NBOOT):
        i = np.concatenate([mem[k] for k in rng.integers(0, G, G)])
        if y_f[i].min() == y_f[i].max():
            continue
        v_sort.append(_auc(rel[lab_m][i], y_f[i]))
    lo, hi = np.percentile(v_sort, [2.5, 97.5])
    out["full_set_relative_position_sort_rank"] = {
        "auc": float(_auc(rel[lab_m], y_f)), "ci": [float(lo), float(hi)],
        "ci_contains_half": bool(lo <= 0.5 <= hi),
        "note": ("the SAME variable and the same draws under the sort-rank estimator. Both conventions "
                 "contain one half, so the borderline verdict is NOT a tie-convention artefact. The "
                 "0.4874 reported earlier came from relative position built on MANIFEST ORDER rather "
                 "than on frame order, so the sensitivity is to the diagnostic's specification")}
    log(f"full_set_relative_position (sort rank)  auc {out['full_set_relative_position_sort_rank']['auc']:.4f} "
        f"CI [{out['full_set_relative_position_sort_rank']['ci'][0]:.4f}, "
        f"{out['full_set_relative_position_sort_rank']['ci'][1]:.4f}]")

    out["interpretation"] = (
        "The 2 diagnostics measure different variables and their pooled AUCs need not agree, relative "
        "position being normalised within each identifier. On the subset, relative position separates "
        "the direction labels BELOW one half while start frame shows no detected departure from it. On "
        "the full constructed set, start frame separates them ABOVE one half while relative position is "
        "borderline. BOTH CLIP SETS THEREFORE CARRY EVIDENCE OF POSITION ASSOCIATION, through different "
        "diagnostics. An interval containing one half establishes neither balance nor absence of "
        "association, and none of this identifies a cause or shows that position produced any encoder "
        "difference. Full-set association means not every association can be assigned to subset "
        "construction, but it does NOT exclude selection effects on the subset, and these linear "
        "diagnostics cannot exclude a nonlinear position association.")
    out["status"] = "COMPLETE"
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{OUT}/position.json", "w"), indent=1)
    log(f"wrote {OUT}/position.json")


if __name__ == "__main__":
    main()
