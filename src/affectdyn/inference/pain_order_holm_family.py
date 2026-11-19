"""Correction family for CLIP's pain order controls.

Usage:  python -m affectdyn.inference.pain_order_holm_family
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, os, sys, time

from affectdyn.provenance import run_provenance

OUT = "runs/pain_order_holm_family"
ALPHA = 0.05
GATE_TOL = 1e-4
F5 = ("K2:ordered_minus_shuffled", "K4:ordered_minus_shuffled", "K8:ordered_minus_shuffled",
      "K16:ordered_minus_shuffled", "K4:ordered_minus_static")
SHUFFLED = F5[:4]
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-15 — W26-W30 PRE-DECLARED CRITERIA"


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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/pain_control_family.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/pain_control_family.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    W11 = {t["test"]: t for t in json.load(open("runs/pain_order_cluster_bootstrap/pain_controls_cluster.json"))["tests"]}
    W12 = {r["test"]: r for r in json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))["families"]["w11_exploratory_5"]["rows"]}
    assert set(F5) <= set(W11) and set(F5) <= set(W12), (sorted(W11), sorted(W12))

    p4 = [W11[t]["subject_boot"]["p_boot"] for t in SHUFFLED]
    adj4 = holm(p4)
    deltas = {t: abs(a - W11[t]["holm_subject_across_shuffle_gaps"]) for t, a in zip(SHUFFLED, adj4)}
    k4 = dict(zip(SHUFFLED, adj4))["K4:ordered_minus_shuffled"]
    gate = {"rule": "Holm across the 4 shuffled gaps reproduces the logged holm_subject_across_shuffle_gaps to 1e-4",
            "k4_recomputed": k4, "k4_logged": W11["K4:ordered_minus_shuffled"]["holm_subject_across_shuffle_gaps"],
            "max_delta": max(deltas.values()), "p_boot_matches_w12": all(W11[t]["subject_boot"]["p_boot"] == W12[t]["p_bootstrap"] for t in F5)}
    gate["passes"] = bool(gate["max_delta"] <= GATE_TOL and gate["p_boot_matches_w12"])
    log(f"gate: {gate}")
    if not gate["passes"]:
        json.dump({"provenance": run_provenance(device="cpu"), "gate": gate, "status": "HOLD: gate failed"},
                  open(out_json, "w"), indent=2)
        log("HOLD"); return

    pb = [W11[t]["subject_boot"]["p_boot"] for t in F5]
    pc = [W12[t]["p_calibrated"] for t in F5]
    ab, ac = holm(pb), holm(pc)
    rows = []
    for t, x, y_, a, c in zip(F5, pb, pc, ab, ac):
        rows.append({"test": t, "role": W11[t]["role"], "gap": W11[t]["gap"], "subject_ci": W11[t]["subject_boot"]["ci"],
                     "p_bootstrap": x, "p_calibrated": y_, "holm_F5_bootstrap": a, "holm_F5_calibrated": c})
        log(f"  {t:28s} gap {W11[t]['gap']:+.4f} p_boot {x:.4f} -> {a:.4f} | p_cal {y_:.4f} -> {c:.4f}")
    r4 = rows[1]
    verdict = {"family": list(F5), "label": "exploratory",
               "k4_holm_F5_bootstrap": r4["holm_F5_bootstrap"], "k4_holm_F5_calibrated": r4["holm_F5_calibrated"],
               "k4_significant_within_F5": bool(r4["holm_F5_bootstrap"] < ALPHA and r4["holm_F5_calibrated"] < ALPHA),
               "reporting_rule": ("significant only if both adjusted p < 0.05; otherwise the contribution does not reach "
                                  "significance under the calibrated procedure within the declared family, both values given")}
    log(f"verdict: {verdict}")
    res = {"provenance": run_provenance(device="cpu"), "criteria_heading": CRITERIA_HEADING, "gate": gate,
           "rows": rows, "verdict": verdict, "status": "COMPLETE"}
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
