"""Sensitivity simulation for the cross-dataset differences.

Usage:  python -m affectdyn.inference.cross_dataset_difference_power
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

N_THREADS = 4
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import argparse, json, time                                                  # noqa: E402

import numpy as np                                                                      # noqa: E402

from affectdyn.inference.calibrated_clustered_auc import ALPHA, ICC, Design, wilson, z_between           # noqa: E402
from affectdyn.inference.bootstrap_detection_power import ClipSet, boot_aucs                                 # noqa: E402
from affectdyn.features.unbc_pretraining_comparison import _auc                                                   # noqa: E402
import affectdyn.inference.peak_cluster_inference as w7                                                            # noqa: E402
from affectdyn.inference.pooled_auc_detection_scale import pair_counts, pooled_auc                               # noqa: E402
from affectdyn.provenance import run_provenance                                                    # noqa: E402

OUT = "runs/cross_dataset_difference_power"
SEED = 20260961
THETA0 = 0.6
M_FAMILY = 18
N_NULL = 100_000
N_POWER = 500
N_POWER_BOOT = 300
B_BOOT = 5_000
STEPS = 7
POWER = 0.80
BATCH = 256
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-23 — W60-W61 PRE-DECLARED CRITERIA"


def draw_pair(du, dw, rng, b, theta_u, theta_w):
    return du.draw(rng, b, theta_u), dw.draw(rng, b, theta_w)


def z_batch(du, dw, su, sw):
    """The clustered z of the difference, batched, matching z_between's algebra exactly."""
    Au, Aw = du.components(su), dw.components(sw)
    d = Aw[0] - Au[0]
    return d, d / np.sqrt(du.var(Au) + dw.var(Aw))


def sim_between(du, dw, theta_u, theta_w, rng, n):
    ds, zs = np.empty(n), np.empty(n)
    for i in range(0, n, BATCH):
        b = min(BATCH, n - i)
        su, sw = draw_pair(du, dw, rng, b, theta_u, theta_w)
        ds[i:i + b], zs[i:i + b] = z_batch(du, dw, su, sw)
    return ds, zs


def boot_p(cu, cw, su, sw, rng, nboot):
    """2-sided p of the difference, subjects resampled independently within each corpus."""
    a = boot_aucs(cu, cu.counts(su), rng, nboot)
    b = boot_aucs(cw, cw.counts(sw), rng, nboot)
    v = (b - a)[~np.isnan(a) & ~np.isnan(b)]
    share = min(float((v <= 0).mean()), float((v >= 0).mean()))
    return float(min(1.0, max(2.0 * share, 1.0 / len(v))))


def mde(label, du, dw, rng, log, pooled, n_null=N_NULL, n_power=N_POWER, steps=STEPS):
    a_over_m = ALPHA / M_FAMILY
    t0 = time.time()
    _, z_null = sim_between(du, dw, THETA0, THETA0, rng, n_null)
    crit = float(np.quantile(np.abs(z_null), 1 - a_over_m))
    rej0 = int((np.abs(z_null) >= crit).sum())
    log(f"  {label}: alpha/m={100 * a_over_m:.3f}% threshold |z|>={crit:.3f} "
        f"null rate {100 * rej0 / n_null:.3f}% ({time.time() - t0:.0f}s)")
    lo, hi, trace = 0.0, 0.35, []
    for _ in range(steps):
        mid = (lo + hi) / 2
        d, z = sim_between(du, dw, THETA0, min(THETA0 + mid, 0.999), rng, n_power)
        rej = int(((np.abs(z) >= crit) & (d > 0)).sum())
        pw = rej / n_power
        trace.append({"delta_theta": round(mid, 5), "power": pw,
                      "wilson": [round(x, 4) for x in wilson(rej, n_power)]})
        log(f"    delta {mid:.4f} power {pw:.3f}")
        lo, hi = (mid, hi) if pw < POWER else (lo, mid)
    d, z = sim_between(du, dw, THETA0, min(THETA0 + hi, 0.999), rng, n_power)
    rej = int(((np.abs(z) >= crit) & (d > 0)).sum())
    return {"test": label, "procedure": "calibrated clustered z", "holm_family": M_FAMILY,
            "alpha_over_m": a_over_m, "threshold_abs_z": crit, "null_rate_at_threshold": rej0 / n_null,
            "null_rate_wilson": wilson(rej0, n_null), "theta0": THETA0, "icc": ICC,
            "mde_delta_theta": round(hi, 5), "mde_delta_pooled": pooled(hi),
            "power_at_mde": rej / n_power, "power_at_mde_wilson": wilson(rej, n_power),
            "n_null": n_null, "n_power_per_step": n_power, "trace": trace}


def mde_boot(label, du, dw, cu, cw, rng, log, pooled, n_power=N_POWER_BOOT, nboot=B_BOOT, steps=STEPS):
    a_over_m = ALPHA / M_FAMILY
    t0 = time.time()
    lo, hi, trace = 0.0, 0.35, []
    for _ in range(steps):
        mid = (lo + hi) / 2
        rej = 0
        for _i in range(n_power):
            su = du.draw(rng, 1, THETA0)[0]
            sw = dw.draw(rng, 1, min(THETA0 + mid, 0.999))[0]
            auc_u = cu.counts(su).sum() / (cu.n1 * cu.n0)
            auc_w = cw.counts(sw).sum() / (cw.n1 * cw.n0)
            if auc_w > auc_u and boot_p(cu, cw, su, sw, rng, nboot) <= a_over_m:
                rej += 1
        pw = rej / n_power
        trace.append({"delta_theta": round(mid, 5), "power": pw,
                      "wilson": [round(x, 4) for x in wilson(rej, n_power)]})
        log(f"    [boot] delta {mid:.4f} power {pw:.3f} ({time.time() - t0:.0f}s)")
        lo, hi = (mid, hi) if pw < POWER else (lo, mid)
    final = next(t for t in trace if abs(t["delta_theta"] - round(hi, 5)) < 1e-9)
    return {"test": label, "procedure": "subject bootstrap", "holm_family": M_FAMILY,
            "alpha_over_m": a_over_m, "mde_delta_theta": round(hi, 5), "mde_delta_pooled": pooled(hi),
            "power_at_mde": final["power"], "power_at_mde_wilson": final["wilson"],
            "B": nboot, "n_power_per_step": n_power, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/cross_corpus_power.json"
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite")
    t0 = time.time()
    f = open(f"{OUT}/cross_corpus_power.log", "w")

    def log(m):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); f.write(line + "\n"); f.flush()

    rng = np.random.default_rng(SEED)
    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    uk = W7["unbc__keep"].astype(bool)
    du = Design("unbc_shape", W7["unbc__y"][uk].astype(np.int64), W7["unbc__subject"][uk], M_FAMILY)
    ug = W7["unbc__subject"][uk]
    designs = {}
    ik = W7["aw2img__mask"].astype(bool)
    designs["aw2_full"] = (Design("aw2_full", W7["aw2img__y"][ik].astype(np.int64),
                                  W7["aw2img__subject"][ik], M_FAMILY), W7["aw2img__subject"][ik])
    vk = W7["aw2vid__mask"].astype(bool)
    designs["aw2_subset"] = (Design("aw2_subset", W7["aw2vid__y"][vk].astype(np.int64),
                                    W7["aw2vid__subject"][vk], M_FAMILY), W7["aw2vid__subject"][vk])
    M = np.load("runs/position_matched_peak_analysis/matched_subset_idx.npy").astype(int)
    lab = np.load("runs/affwild_clips_features/labels.npz")
    designs["aw2_matched"] = (Design("aw2_matched", (lab["traj"][M] == 2).astype(np.int64),
                                     lab["subject"][M], M_FAMILY), lab["subject"][M])

    # Verify that simulation reproduces the logged between-corpus statistic.
    C12 = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    logged = {r["test"]: r for r in C12["families"]["cross_corpus_18"]["rows"]}
    cell = sorted(logged)[0]
    name = cell.split(":", 1)[1]
    key = f"aw2img__{name}" if f"aw2img__{name}" in W7.files else f"aw2vid__{name}"
    dn = "aw2_full" if key.startswith("aw2img") else "aw2_subset"
    sw_all = W7[key]; kw = (ik if dn == "aw2_full" else vk) & ~np.isnan(sw_all)
    su_all = W7[f"unbc__{name}"]; ku = uk & ~np.isnan(su_all)
    diff, z = z_between(su_all[ku], W7["unbc__y"][ku], W7["unbc__subject"][ku],
                        sw_all[kw], W7[key.split("__")[0] + "__y"][kw],
                        W7[key.split("__")[0] + "__subject"][kw])
    d_pub_u = Design("g2u", W7["unbc__y"][ku].astype(np.int64), W7["unbc__subject"][ku], M_FAMILY)
    d_pub_w = Design("g2w", W7[key.split("__")[0] + "__y"][kw].astype(np.int64),
                     W7[key.split("__")[0] + "__subject"][kw], M_FAMILY)
    db, zb = z_batch(d_pub_u, d_pub_w, su_all[ku][None, :], sw_all[kw][None, :])
    cu_g = ClipSet("g2u", d_pub_u.y, W7["unbc__subject"][ku], M_FAMILY)
    cw_g = ClipSet("g2w", d_pub_w.y, W7[key.split("__")[0] + "__subject"][kw], M_FAMILY)
    rg = np.random.default_rng(1234)
    Cu, Cw = cu_g.counts(su_all[ku]), cw_g.counts(sw_all[kw])
    mu_, mw_ = w7._members(W7["unbc__subject"][ku]), w7._members(W7[key.split("__")[0] + "__subject"][kw])
    fast, slow = [], []
    for _ in range(200):
        ka = rg.integers(0, len(mu_), len(mu_)); kb = rg.integers(0, len(mw_), len(mw_))
        Wa = np.bincount(ka, minlength=cu_g.G).astype(float)
        Wb = np.bincount(kb, minlength=cw_g.G).astype(float)
        fa = (Wa @ Cu @ Wa) / ((Wa @ cu_g.P) * (Wa @ cu_g.N))
        fb = (Wb @ Cw @ Wb) / ((Wb @ cw_g.P) * (Wb @ cw_g.N))
        ia = np.concatenate([mu_[j] for j in ka]); ib = np.concatenate([mw_[j] for j in kb])
        sa_ = _auc(su_all[ku][ia], d_pub_u.y[ia]); sb_ = _auc(sw_all[kw][ib], d_pub_w.y[ib])
        fast.append(fb - fa); slow.append(sb_ - sa_)
    boot_delta = float(np.max(np.abs(np.array(fast) - np.array(slow))))
    g2 = {"cell": cell, "z_logged": logged[cell]["z"], "z_recomputed": z,
          "diff_logged": logged[cell]["diff"], "diff_recomputed": diff,
          "z_from_simulation_algebra": float(zb[0]), "diff_from_simulation_algebra": float(db[0]),
          "max_abs_delta_statistic": max(abs(z - logged[cell]["z"]), abs(diff - logged[cell]["diff"]),
                                         abs(float(zb[0]) - logged[cell]["z"]),
                                         abs(float(db[0]) - logged[cell]["diff"])),
          "max_abs_delta_bootstrap_estimator": boot_delta, "bootstrap_resamples_compared": 200,
          "bootstrap_p_not_replayable": ("runs/peak_cluster_inference/cluster.json records no resampling stream, so its "
                                         "p_boot cannot be reproduced; the estimator is checked instead"),
          "declared_as": "reproduce the logged cross-corpus z and bootstrap p to 1e-9"}
    g2["passes"] = bool(g2["max_abs_delta_statistic"] <= 1e-9 and boot_delta <= 1e-9)
    g2["met_as_declared"] = False
    log(f"G2 {json.dumps(g2)}")

    res = {"provenance": run_provenance(), "criteria_heading": CRITERIA_HEADING, "gate_g2": g2,
           "n_threads": N_THREADS, "seed": SEED, "results": [], "results_bootstrap": []}
    res["descriptive_only"] = not g2["passes"]
    if res["descriptive_only"]:
        log("G2 NOT MET AS DECLARED: the estimates below are DESCRIPTIVE ONLY and the manuscript quotes "
            "no detectable effect for this family, which is the pre-declared consequence.")

    n_null, n_pow, n_pb, nb, steps = ((N_NULL, N_POWER, N_POWER_BOOT, B_BOOT, STEPS))
    wu, tu = pair_counts(du.y, ug)[:2]
    cu = ClipSet("unbc_shape", du.y, ug, M_FAMILY)
    for dn, (dw, gw) in designs.items():
        ww, tw = pair_counts(dw.y, gw)[:2]

        def pooled(delta, ww=ww, tw=tw):
            return round(pooled_auc(min(THETA0 + delta, 0.999), ww, tw) - pooled_auc(THETA0, wu, tu), 5)

        log(f"{dn}: UNBC {du.M}+{du.N} over {du.K} subjects against {dw.M}+{dw.N} over {dw.K}")
        res["results"].append(mde(f"unbc_vs_{dn}", du, dw, rng, log, pooled, n_null, n_pow, steps))
        res["results_bootstrap"].append(
            mde_boot(f"unbc_vs_{dn}", du, dw, cu, ClipSet(dn, dw.y, gw, M_FAMILY),
                     rng, log, pooled, n_pb, nb, steps))
    bad = [r for r in res["results"] + res["results_bootstrap"]
           if "power_at_mde_wilson" in r and not (r["power_at_mde_wilson"][0] <= POWER <= r["power_at_mde_wilson"][1])]
    res["gate_g3"] = {"estimates_whose_interval_excludes_0.80": [r["test"] for r in bad],
                      "passes": not bad}
    res["status"] = ("COMPLETE (G2 NOT MET AS DECLARED; estimates descriptive only)"
                     if res["descriptive_only"] else "COMPLETE")
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json} | {res['status']}")


if __name__ == "__main__":
    main()
