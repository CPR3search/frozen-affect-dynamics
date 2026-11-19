"""Recording-position diagnostics for the Aff-Wild2 clip sets.

Usage:  python -m affectdyn.analyses.position.affwild_recording_position
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time

import numpy as np

from affectdyn.provenance import run_provenance

OUT = "runs/affwild_recording_position"
SEED = 20260918
NBOOT = 10000
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-15 — W26-W30 PRE-DECLARED CRITERIA"
EXPECT = {"subset": (2500, 287, 2213, 388), "full": (13276, 287, 12989, 388)}


def _auc(s, y):                                            # Match the cross-pretraining AUC estimator.
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _members(groups):                                      # Group indices by subject.
    codes = np.unique(groups, return_inverse=True)[1]
    return [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]


def stratified_subset(clips, n, seed=0):                   # Match the extraction module's stratified sampling.
    rng = np.random.default_rng(seed)
    peaked = [i for i, c in enumerate(clips) if c["traj"] == 2]
    rest = [i for i, c in enumerate(clips) if c["traj"] != 2]
    by_subj = {}
    for i in rest:
        by_subj.setdefault(clips[i]["subject"], []).append(i)
    order = []
    subs = sorted(by_subj)
    rng.shuffle(subs)
    depth = 0
    while len(order) < max(0, n - len(peaked)):
        added = False
        for s in subs:
            if depth < len(by_subj[s]):
                order.append(by_subj[s][depth]); added = True
                if len(order) >= n - len(peaked):
                    break
        if not added:
            break
        depth += 1
    idx = sorted(set(peaked) | set(order))
    return idx


def subject_bootstrap_vs_half(s, y, groups, rng, nboot):
    """Percentile interval and two-sided p of an AUC against one half, resampling whole subjects."""
    mem = _members(groups); G = len(mem); vals = []
    for _ in range(nboot):
        i = np.concatenate([mem[k] for k in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(_auc(s[i], y[i]))
    v = np.asarray(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"ci": [float(lo), float(hi)], "p_boot": float(min(1.0, max(p, 1.0 / len(v)))),
            "resamples_skipped": int(nboot - len(v))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/aw2_position.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/aw2_position.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    clips = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    lab = np.load("runs/affwild_clips_features/labels.npz", allow_pickle=True)
    subset = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    subj = np.array([str(c["subject"]) for c in clips])
    traj = np.array([int(c["traj"]) for c in clips])
    assert all(str(a) == b for a, b in zip(lab["subject"], subj)) and np.array_equal(lab["traj"], traj)
    start = np.array([int(os.path.splitext(os.path.basename(c["start"]))[0]) for c in clips], dtype=float)
    rel = np.empty(len(clips)); singletons = 0
    for k, idx in enumerate(_members(subj)):
        n = len(idx)                                       # Ascending indices preserve manifest order.
        if n == 1:
            rel[idx] = 0.5; singletons += 1
        else:
            rel[idx] = np.arange(n) / (n - 1)
    y_all = (traj == 2).astype(np.int64)
    assert set(np.unique(traj)) <= {0, 1, 2}

    sets = {"subset": np.sort(subset), "full": np.arange(len(clips))}
    counts = {}
    for name, ix in sets.items():
        counts[name] = (len(ix), int(y_all[ix].sum()), int((y_all[ix] == 0).sum()), len(np.unique(subj[ix])))
    gate_i = all(counts[k] == EXPECT[k] for k in EXPECT)
    recon = stratified_subset([{"traj": int(t), "subject": s} for t, s in zip(traj, subj)], 2500, seed=0)
    gate_ii = bool(np.array_equal(np.array(recon), np.sort(subset)))
    gate = {"counts": counts, "expected": EXPECT, "gate_i_counts_match": bool(gate_i),
            "gate_ii_verbatim_reconstruction_matches_saved_subset": gate_ii, "passes": bool(gate_i),
            "singleton_subjects_given_0.5": singletons}
    log(f"gate: {gate}")
    if not gate_i:
        json.dump({"provenance": run_provenance(seeds=[SEED], device="cpu"), "gate": gate,
                   "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        log("HOLD"); return

    rng = np.random.default_rng(SEED)
    results = {}
    for name, ix in sets.items():
        yy, gg = y_all[ix], subj[ix]
        for q, arr in (("relative_position", rel), ("start_frame", start)):
            s = arr[ix]
            auc = _auc(s, yy)
            b = subject_bootstrap_vs_half(s, yy, gg, rng, NBOOT)
            med_pos, med_neg = float(np.median(s[yy == 1])), float(np.median(s[yy == 0]))
            results[f"{name}:{q}"] = {"auc": auc, **b, "median_peaked": med_pos, "median_monotonic": med_neg,
                                      "ci_excludes_half": bool(b["ci"][0] > 0.5 or b["ci"][1] < 0.5)}
            log(f"  {name:6s} {q:17s} AUC {auc:.4f} CI {b['ci'][0]:.3f} to {b['ci'][1]:.3f} p {b['p_boot']:.4f} "
                f"| median peaked {med_pos:.3f} monotonic {med_neg:.3f}")

    present = results["subset:relative_position"]["ci_excludes_half"]
    verdict = {"position_confound_present_on_subset": bool(present),
               "subset_relative_position_auc": results["subset:relative_position"]["auc"],
               "full_relative_position_auc": results["full:relative_position"]["auc"],
               "reporting_rule": ("present -> V-JEPA-2's subset result described as potentially confounded by position "
                                  "within a recording until a position-matched rerun; absent -> bounds the label-side "
                                  "confound only, feature drift unrelated to position not excluded")}
    log(f"verdict: {verdict}")
    res = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "nboot": NBOOT, "gate": gate, "results": results, "verdict": verdict, "status": "COMPLETE"}
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
