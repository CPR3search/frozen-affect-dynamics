"""Form-matched against mixed-granularity direction contrasts.

Usage:  python -m affectdyn.analyses.direction.mixed_granularity_contrasts
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time, zlib

import numpy as np

from affectdyn.provenance import run_provenance

OUT = "runs/mixed_granularity_contrasts"
NBOOT = 10000
W9_SEED = 20260911
ALPHA = 0.05
VIDEO = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")
MASKED_AE = {"videomae", "videomaev2"}
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-15 — W26-W30 PRE-DECLARED CRITERIA"


def _auc(s, y):                                            # Match the cross-pretraining AUC estimator.
    y = y.astype(bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _members(groups):                                      # Group indices by subject.
    codes = np.unique(groups, return_inverse=True)[1]
    return [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]


def paired_bootstrap(sa, sb, y, groups, unit, rng, nboot):  # Share resamples between both score vectors.
    """AUC(a) - AUC(b) on identical clips, resampling clips or whole subjects, one resample for both."""
    mem = _members(groups); G = len(mem); n = len(y); vals = []
    for _ in range(nboot):
        i = rng.integers(0, n, n) if unit == "clip" else np.concatenate([mem[g] for g in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        vals.append(_auc(sa[i], y[i]) - _auc(sb[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"ci": [float(lo), float(hi)], "ci_halfwidth": float((hi - lo) / 2),
            "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "resamples_skipped": int(nboot - len(v))}


def holm(pvals):                                           # Adjust p-values with Holm's procedure.
    """Local Holm adjustment; the shared implementation is
    inference.aggregation_permutation_tests.holm. Family membership and size are specified by the
    caller.
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


def rng_for(name):
    return np.random.default_rng([W9_SEED, zlib.crc32(name.encode())])


def family(rows, pkey="p"):
    adj = holm([r[pkey] for r in rows])
    out = []
    for r, a in zip(rows, adj):
        out.append({**r, "holm": a, "survives": bool(a < ALPHA)})
    return out


def survivors(rows):
    return sorted((r["video"], "clip_higher" if r["gap"] > 0 else "video_higher") for r in rows if r["survives"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/unmatched_contrasts.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/unmatched_contrasts.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    Z = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    keep = Z["keep"].astype(bool)
    y, g = Z["y"][keep].astype(np.int64), Z["subject"][keep]
    S = {k: Z[k][keep] for k in Z.files if k not in ("subject", "y", "keep")}
    W9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))
    rows9 = W9["contrasts"]
    log(f"W26 | {len(y)} clips, {int(y.sum())} positives, {len(np.unique(g))} subjects | B={NBOOT} | "
        f"{len(rows9)} logged contrasts")

    gate_rows, ok = [], True
    for r in rows9:
        sa, sb = S[r["a"]], S[r["b"]]
        gap = _auc(sa, y) - _auc(sb, y)
        b = paired_bootstrap(sa, sb, y, g, "subject", rng_for(r["label"] + ":subject"), NBOOT)
        same_gap = abs(gap - r["gap"]) <= 1e-12
        same_p = b["p_boot"] == r["subject_boot"]["p_boot"]
        same_ci = b["ci"] == r["subject_boot"]["ci"]
        ok &= same_gap and same_p and same_ci
        gate_rows.append({"label": r["label"], "gap_delta": abs(gap - r["gap"]), "p_boot": b["p_boot"],
                          "p_boot_logged": r["subject_boot"]["p_boot"], "p_equal": same_p, "ci_equal": same_ci})
    holm24 = holm([r["subject_boot"]["p_boot"] for r in rows9])
    holm_ok = all(abs(a - r["holm"]["boot_subject"]) <= 1e-12 for a, r in zip(holm24, rows9))
    ok &= holm_ok
    gate = {"rule": "every gap to 1e-12, every subject-bootstrap p and interval exactly, Holm to 1e-12",
            "n_contrasts": len(gate_rows), "max_gap_delta": max(x["gap_delta"] for x in gate_rows),
            "all_p_equal": all(x["p_equal"] for x in gate_rows), "all_ci_equal": all(x["ci_equal"] for x in gate_rows),
            "holm_reproduces": holm_ok, "passes": bool(ok)}
    log(f"gate: {gate}")
    if not ok:
        json.dump({"provenance": run_provenance(seeds=[W9_SEED], device="cpu"), "gate": gate,
                   "gate_rows": gate_rows, "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        log("HOLD: the stored scores do not reproduce W9; nothing is reported")
        return

    # Evaluate the unmatched family and matched comparisons.
    U = []
    for v in VIDEO:
        label = f"unmatched: clip_K16 vs {v}_K2"
        sa, sb = S["clip_K16"], S[f"{v}_K2"]
        b = paired_bootstrap(sa, sb, y, g, "subject", rng_for(label + ":subject"), NBOOT)
        U.append({"label": label, "video": v, "auc_clip": _auc(sa, y), "auc_video": _auc(sb, y),
                  "gap": _auc(sa, y) - _auc(sb, y), "p": b["p_boot"], "ci": b["ci"],
                  "resamples_skipped": b["resamples_skipped"]})
        log(f"  {label:36s} gap {U[-1]['gap']:+.4f} CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f} p {b['p_boot']:.4f}")
    U6 = family(U)

    def matched(k):
        rs = [r for r in rows9 if r["a"] == f"clip_K{k}" and r["b"].endswith(f"_K{k}")]
        return family([{"label": r["label"], "video": r["b"].rsplit("_K", 1)[0], "gap": r["gap"],
                        "p": r["subject_boot"]["p_boot"], "ci": r["subject_boot"]["ci"]} for r in rs])

    M2, M16 = matched(2), matched(16)
    union = [{"label": r["label"], "p": r["subject_boot"]["p_boot"]} for r in rows9] + [{"label": u["label"], "p": u["p"]} for u in U]
    adj30 = holm([x["p"] for x in union])
    u30 = {x["label"]: a for x, a in zip(union, adj30)}
    m24_changed = [r["label"] for r, a in zip(rows9, adj30[:24]) if (a < ALPHA) != (r["holm"]["boot_subject"] < ALPHA)]

    sU, sM2, sM16 = survivors(U6), survivors(M2), survivors(M16)
    changes = [v for v, _ in sU] != [v for v, _ in sM2]
    outside = [v for v, s in sU if v not in MASKED_AE]
    verdict = {
        "survivors_U6_unmatched": sU, "survivors_M6_K2": sM2, "survivors_M6_K16": sM16,
        "matching_changes_a_conclusion": bool(changes),
        "U6_survivors_outside_masked_autoencoding": outside,
        "unmatched_holm_within_30": {u["label"]: u30[u["label"]] for u in U},
        "matched_decisions_changed_within_30": m24_changed,
        "auc_clip_K16": _auc(S["clip_K16"], y), "auc_clip_K2": _auc(S["clip_K2"], y),
        "reporting_rule": ("sets differ -> recommendation 1 gains direct evidence and the change is reported; "
                           "identical -> matching was a precaution with no effect on these direction verdicts"),
    }
    log(f"survivors U6 {sU} | M6@K2 {sM2} | M6@K16 {sM16} | changes {changes} | outside MAE {outside}")

    res = {"provenance": run_provenance(seeds=[W9_SEED], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "nboot": NBOOT, "gate": gate, "gate_rows": gate_rows, "U6": U6, "M6_K2": M2, "M6_K16": M16,
           "verdict": verdict, "status": "COMPLETE"}
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
