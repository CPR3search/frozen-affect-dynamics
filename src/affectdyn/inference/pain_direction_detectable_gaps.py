"""Minimum detectable gap of the 24-test direction contrast family.

Usage:  python -m affectdyn.inference.pain_direction_detectable_gaps
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, json, math, os, sys, time
from statistics import NormalDist

import numpy as np

from affectdyn.inference.calibrated_clustered_auc import ALPHA, ICC, BATCH, Design, sim_paired, wilson
from affectdyn.inference.pooled_auc_detection_scale import pooled_auc, pair_counts
from affectdyn.provenance import run_provenance

OUT = "runs/pain_direction_detectable_gaps"
SEED = 20260917
M = 24
N_NULL = 100_000
N_POWER = 1000
N_FINAL = 2000
STEPS = 9
DELTA_HI = 2 * (0.995 - 0.8724)
POWER = 0.80
DELTA_OBSERVED = 0.062
GATE_RATE_TOL = 0.0005
GATE_W12_TOL = 0.0012
OTHER = ("vjepa", "timesformer", "xclip", "internvideo2")
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-15 — W26-W30 PRE-DECLARED CRITERIA"


def sim_paired_alt(d, theta_a, theta_b, rho, rng, n):
    zs = np.empty(n)
    for i in range(0, n, BATCH):
        b = min(BATCH, len(zs) - i)
        e1, e2 = rng.standard_normal((b, len(d.y))), rng.standard_normal((b, len(d.y)))
        u1, u2 = rng.standard_normal((b, d.K)), rng.standard_normal((b, d.K))
        sa = d.draw(rng, b, theta_a, e1, u1)
        sb = d.draw(rng, b, theta_b, rho * e1 + math.sqrt(1 - rho ** 2) * e2, rho * u1 + math.sqrt(1 - rho ** 2) * u2)
        A, Bc = d.components(sa), d.components(sb)
        v = d.var(A) + d.var(Bc) - 2 * d.var(A, Bc)
        zs[i:i + b] = (A[0] - Bc[0]) / np.sqrt(v)
    return zs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/contrast_mde.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/contrast_mde.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    Z = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    keep = Z["keep"].astype(bool)
    y, g = Z["y"][keep].astype(np.int64), Z["subject"][keep]
    d = Design("unbc_direction", y, g, M)
    C12 = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    theta = C12["calibration"]["parameters"]["theta_direction"]
    rho = C12["calibration"]["parameters"]["rho_direction"]
    assert abs(theta - 0.8724) < 1e-4 and abs(rho - 0.5431) < 1e-4, (theta, rho)
    w12_rate = C12["calibration"]["paired_unbc_direction"]["rate_at_alpha_over_m"]
    rng = np.random.default_rng(SEED)
    a_over_m = ALPHA / M
    log(f"W27 | design M={d.M} N={d.N} K={d.K} | theta {theta:.4f} rho {rho:.4f} ICC {ICC} | seed {SEED}")

    z_null = sim_paired(d, theta, rho, rng, N_NULL)
    az = np.abs(z_null)
    crit = float(np.quantile(az, 1 - a_over_m))
    rate_crit = float((az >= crit).mean())
    z_nom = NormalDist().inv_cdf(1 - a_over_m / 2)
    rate_nom = float((az >= z_nom).mean())
    gate = {"threshold_abs_z": crit, "nominal_abs_z": z_nom, "null_rate_at_threshold": rate_crit,
            "alpha_over_m": a_over_m, "gate_i_rate_reproduces_alpha_over_m": bool(abs(rate_crit - a_over_m) <= GATE_RATE_TOL),
            "nominal_rate_at_alpha_over_m": rate_nom, "w12_nominal_rate": w12_rate,
            "gate_ii_matches_w12": bool(abs(rate_nom - w12_rate) <= GATE_W12_TOL)}
    gate["passes"] = bool(gate["gate_i_rate_reproduces_alpha_over_m"] and gate["gate_ii_matches_w12"])
    log(f"gate: {gate}")
    if not gate["passes"]:
        json.dump({"provenance": run_provenance(seeds=[SEED], device="cpu"), "gate": gate,
                   "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        log("HOLD"); return

    # Bisect the contrast effect delta.
    def power(delta, n):
        z = sim_paired_alt(d, theta + delta / 2, theta - delta / 2, rho, rng, n)
        k = int((z >= crit).sum())
        return k / n, wilson(k, n)

    lo, hi, trace = 0.0, DELTA_HI, []
    top, _ = power(DELTA_HI, N_POWER)
    trace.append({"delta": round(DELTA_HI, 5), "power": top})
    exceeds = top < POWER
    if not exceeds:
        for _ in range(STEPS):
            mid = (lo + hi) / 2
            pw, _ = power(mid, N_POWER)
            trace.append({"delta": round(mid, 5), "power": pw})
            log(f"  delta {mid:.4f} power {pw:.3f}")
            if pw < POWER:
                lo = mid
            else:
                hi = mid
        p_mde, w_mde = power(hi, N_FINAL)
    else:
        p_mde, w_mde = top, None
        log(f"  power at the upper bound {DELTA_HI:.4f} is {top:.3f}: the MDE exceeds it")
    p_obs, w_obs = power(DELTA_OBSERVED, N_FINAL)

    w, tot = pair_counts(y, g)[:2]
    pooled_gap = (pooled_auc(theta + hi / 2, w, tot) - pooled_auc(theta - hi / 2, w, tot)) if not exceeds else None

    # Compare observed gaps with the MDE.
    W9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))
    nonsurv = [{"label": r["label"], "video": r["b"].rsplit("_K", 1)[0], "gap": r["gap"],
                "holm_subject": r["holm"]["boot_subject"]} for r in W9["contrasts"] if r["holm"]["boot_subject"] >= ALPHA]
    others = [r for r in nonsurv if r["video"] in OTHER]
    max_other = max(r["gap"] for r in others)
    mde = None if exceeds else hi
    verdict = {
        "mde_delta_theta_scale": mde, "mde_exceeds_upper_bound": bool(exceeds), "upper_bound": DELTA_HI,
        "mde_gap_pooled_scale": pooled_gap, "power_at_mde": p_mde, "power_at_mde_wilson": w_mde,
        "power_at_observed_0.062": p_obs, "power_at_observed_0.062_wilson": w_obs,
        "n_non_surviving": len(nonsurv), "max_non_surviving_gap_other_objectives": max_other,
        "all_other_objective_gaps_below_mde": bool(exceeds or all(r["gap"] < hi for r in others)),
        "other_objective_gaps_at_or_above_mde": [r for r in others if (not exceeds) and r["gap"] >= hi],
        "reporting_rule": ("all below -> restate as no contrast against the other 4 objectives reaching significance "
                           "at this sample's sensitivity, quoting the MDE, and drop 'rather than the modality boundary' "
                           "as a finding; any at or above -> report as such"),
    }
    log(f"MDE delta {mde} (pooled gap {pooled_gap}) power {p_mde} | power at 0.062 {p_obs:.3f} | "
        f"max other-objective gap {max_other:+.4f} | all below {verdict['all_other_objective_gaps_below_mde']}")
    res = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "design": {"M": d.M, "N": d.N, "K": d.K, "m": M, "theta": theta, "rho": rho, "icc": ICC,
                      "within_pairs": w, "total_pairs": tot},
           "n_null": N_NULL, "n_power_per_step": N_POWER, "n_final": N_FINAL, "gate": gate, "trace": trace,
           "non_surviving": nonsurv, "verdict": verdict,
           "conditional_on": "fixed out-of-fold predictions and this null family; not the paired subject bootstrap",
           "status": "COMPLETE"}
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
