"""Minimum detectable effect of the clustered AUC test."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import json, os, sys, time

import numpy as np

from affectdyn.inference.calibrated_clustered_auc import ALPHA, ICC, Design, sim_single, wilson
from affectdyn.provenance import run_provenance

OUT = "runs/calibrated_detection_limits"
SEED = 20260912
N_NULL = 100_000
N_POWER = 500
STEPS = 7
POWER = 0.80
GATE_TOL = 0.002


def mde(d, rng, p, n_null=N_NULL, n_power=N_POWER, steps=STEPS):
    """Bisect the AUC at which the calibrated rule reaches 80 percent power on this clip set."""
    a_over_m = ALPHA / d.m
    t0 = time.time()
    z_null = np.abs(sim_single(d, 0.5, rng, n_null)[0])
    crit = float(np.quantile(z_null, 1 - a_over_m))
    rate = float((z_null >= crit).mean())
    p(f"  {d.name}: m={d.m} alpha/m={100 * a_over_m:.3f}% threshold |z|>={crit:.3f} "
      f"(null rate at the threshold {100 * rate:.3f}%, {time.time() - t0:.0f}s)")

    lo, hi, trace = 0.5, 0.95, []
    for _ in range(steps):
        mid = (lo + hi) / 2
        z, auc, _ = sim_single(d, mid, rng, n_power)
        rej = int(((np.abs(z) >= crit) & (auc > 0.5)).sum())
        pw = rej / n_power
        trace.append({"theta": round(mid, 5), "power": pw,
                      "wilson": [round(x, 4) for x in wilson(rej, n_power)]})
        p(f"    theta {mid:.4f} power {pw:.3f}")
        if pw < POWER:
            lo = mid
        else:
            hi = mid
    z, auc, _ = sim_single(d, hi, rng, n_power)
    rej = int(((np.abs(z) >= crit) & (auc > 0.5)).sum())
    return {"clip_set": d.name, "holm_family": d.m, "alpha_over_m": a_over_m,
            "threshold_abs_z": crit, "null_rate_at_threshold": rate,
            "mde_auc_at_80pc_power": round(hi, 4), "bracket": [round(lo, 4), round(hi, 4)],
            "power_at_mde": rej / n_power, "power_at_mde_wilson": wilson(rej, n_power),
            "n_null": n_null, "n_power_per_step": n_power, "icc": ICC, "trace": trace}


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/calibrated_power.log", "w")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    C12 = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    rng = np.random.default_rng(SEED)

    designs = [Design("unbc_shape", W7["unbc__y"][W7["unbc__keep"].astype(bool)].astype(np.int64),
                      W7["unbc__subject"][W7["unbc__keep"].astype(bool)], 36),
               Design("aw2_full", W7["aw2img__y"][W7["aw2img__mask"].astype(bool)].astype(np.int64),
                      W7["aw2img__subject"][W7["aw2img__mask"].astype(bool)], 18),
               Design("aw2_subset", W7["aw2vid__y"][W7["aw2vid__mask"].astype(bool)].astype(np.int64),
                      W7["aw2vid__subject"][W7["aw2vid__mask"].astype(bool)], 18)]
    p(f"W18 calibrated power | {N_NULL} null studies, {N_POWER} per bisection step, {STEPS} steps, "
      f"ICC {ICC}, seed {SEED}")

    res = {"provenance": run_provenance(seeds=[SEED]), "seed": SEED, "n_null": N_NULL,
           "n_power_per_step": N_POWER, "power_target": POWER, "icc": ICC,
           "conditional_on": "fixed out-of-fold predictions, as the published tests are",
           "clip_sets": {}, "gate": {}}

    gate_ok = True
    for d in designs:
        r = mde(d, rng, p)
        w12 = C12["calibration"].get(f"single_{d.name}")
        r["w12_rate_at_alpha_over_m"] = w12["rate_at_alpha_over_m"] if w12 else None
        r["gate_threshold_reproduces_nominal"] = bool(abs(r["null_rate_at_threshold"] - r["alpha_over_m"])
                                                      <= GATE_TOL)
        gate_ok &= r["gate_threshold_reproduces_nominal"]
        res["clip_sets"][d.name] = r
        p(f"  {d.name}: MDE_calibrated {r['mde_auc_at_80pc_power']:.4f} "
          f"(power {r['power_at_mde']:.3f}, Wilson {[round(x, 3) for x in r['power_at_mde_wilson']]})\n")

    res["gate"] = {"tolerance": GATE_TOL, "passes": bool(gate_ok),
                   "note": "the threshold is the null's own quantile, so its rate must return alpha/m"}
    res["status"] = "COMPLETE" if gate_ok else "HOLD: threshold gate failed"
    json.dump(res, open(f"{OUT}/calibrated_power.json", "w"), indent=2)
    for n, r in res["clip_sets"].items():
        p(f"{n:12s} MDE_calibrated {r['mde_auc_at_80pc_power']:.3f} at alpha/m "
          f"{100 * r['alpha_over_m']:.3f}% over {r['holm_family']} tests")
    p(f"wrote {OUT}/calibrated_power.json | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
