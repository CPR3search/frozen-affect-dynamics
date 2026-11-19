"""Minimum detectable gap of the Aff-Wild2 direction contrasts.

Usage:  python -m affectdyn.inference.affwild_direction_detectable_gaps
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""                    # Run on the CPU.
import argparse, hashlib, json, math, sys, time

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.features.temporal_aggregation import image_readout, video_readout
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof
from affectdyn.inference.bootstrap_detection_power import ClipSet
from affectdyn.inference.calibrated_clustered_auc import ICC, Design, sim_paired, wilson, within_class_corr
from affectdyn.inference.pooled_auc_detection_scale import pooled_auc, pair_counts
from affectdyn.inference.pain_direction_detectable_gaps import sim_paired_alt
from affectdyn.analyses.direction.affwild_dense_direction_contrasts import (subject_bootstrap_vs_half, paired_bootstrap as w34_paired, VIDEO, K,
                                     IMG_SEEDS, VID_SEEDS, SEED as W34_SEED, NBOOT)
from affectdyn.provenance import run_provenance

OUT = "runs/affwild_direction_detectable_gaps"
SEED = 20260923
M = 6
ALPHA = 0.05
TOL = 0.005
CHUNK = 2500
N_NULL_Z = 100_000
N_NULL_BOOT = 20_000
N_STEP = 1000
N_FINAL = 2000
STEPS = 9
POWER = 0.80
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W35-W37 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W35-W37 CRITERIA AMENDMENT"


def mult(draw, G):
    b = draw.shape[0]
    return np.bincount((draw + (np.arange(b) * G)[:, None]).ravel(), minlength=b * G).reshape(b, G).astype(float)


def fast_gaps(cs, C, draws):
    W = mult(draws, cs.G)
    den = (W @ cs.P) * (W @ cs.N)
    ok = den > 0
    return ((W @ C) * W).sum(1)[ok] / den[ok]


def boot_p(cs, C, rng, nboot=NBOOT):
    v = np.concatenate([fast_gaps(cs, C, rng.integers(0, cs.G, size=(min(CHUNK, nboot - j), cs.G)))
                        for j in range(0, nboot, CHUNK)])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return float(min(1.0, max(p, 1.0 / len(v))))


def solve_theta(target, w, tot):
    """The between-subject pair AUC whose pooled AUC on this design equals target."""
    lo, hi = 0.5, 0.995
    for _ in range(60):
        mid = (lo + hi) / 2
        if pooled_auc(mid, w, tot) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def thetas_for_gap(theta0, gap, w, tot):
    """(theta_a, theta_b) = (theta0 + d/2, theta0 - d/2) with the POOLED gap equal to gap."""
    f = lambda d: pooled_auc(theta0 + d / 2, w, tot) - pooled_auc(theta0 - d / 2, w, tot)
    lo, hi = 0.0, 2 * (0.995 - theta0)
    if f(hi) < gap:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) < gap:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return theta0 + d / 2, theta0 - d / 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    stem = "emotion_contrast_mde"
    out_json = f"{OUT}/{stem}.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/{stem}.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(2)

    clips = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    lab = np.load("runs/affwild_clips_features/labels.npz", allow_pickle=True)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    subj_all = np.array([str(s) for s in lab["subject"]])
    d_all = lab["direction"]
    sub = subj_all[idx]
    y = (d_all[idx] == 1).astype(int)
    mask = d_all[idx] != -1
    rel = np.empty(len(clips))
    for s in np.unique(subj_all):
        ix = np.flatnonzero(subj_all == s)
        rel[ix] = np.arange(len(ix)) / max(1, len(ix) - 1)
    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    X = {"clip": image_readout(frames["clip"], K)[idx]}
    for e in VIDEO:
        z = np.load(f"runs/affwild_clips_features/{e}_windows.npz")
        X[e] = video_readout(z["full"], z["K16_up"])
    assign = grouped_folds(sub)
    fold_hash = hashlib.sha256(assign.astype(np.int64).tobytes()).hexdigest()[:16]
    W34 = json.load(open("runs/affwild_dense_direction_contrasts/aw2_contrasts.json"))
    pub_auc = {r["encoder"]: r["auc"] for r in W34["refits"]}
    pub_con = {r["video"]: r for r in W34["contrasts"]}
    log(f"W35 | subset {len(idx)} clips, direction-labelled {int(mask.sum())} | fold hash {fold_hash} | seed {SEED}")

    S = {}
    for e, Xe in X.items():
        seeds = IMG_SEEDS if e == "clip" else VID_SEEDS
        S[e] = np.nanmean([oof(Xe, y, assign, mask, sd, "cpu") for sd in seeds], axis=0)
    keep = mask.copy()
    for s in S.values():
        keep &= ~np.isnan(s)
    yk, gk = y[keep].astype(np.int64), sub[keep]
    aucs = {e: float(_auc(S[e][keep], yk)) for e in S}
    d1 = max(abs(aucs[e] - pub_auc[e]) for e in pub_auc)
    gate1 = {"n_clips": int(keep.sum()), "n_expected": 1206, "max_abs_auc_delta": d1, "tolerance": TOL,
             "aucs": aucs, "passes": bool(keep.sum() == 1206 and d1 <= TOL),
             "input_equality": ("clip membership, labels, groups and folds are rebuilt by W34's own deterministic "
                                "code; no publication-time fold record exists, so their equality to W34's inputs is "
                                "verified indirectly, by the exact AUCs here and the exact bootstrap replay of (ii)")}
    log(f"gate (i) refit: {gate1}")

    rng34 = np.random.default_rng(W34_SEED)
    subject_bootstrap_vs_half(rel[idx][mask], y[mask].astype(np.int64), sub[mask], rng34, NBOOT)
    rep = {}
    for e in VIDEO:
        b = w34_paired(S["clip"][keep], S[e][keep], yk, gk, rng34, NBOOT)
        rep[e] = {"p": b["p_boot"], "ci": b["ci"], "p_pub": pub_con[e]["p_boot"], "ci_pub": pub_con[e]["ci"],
                  "exact": bool(b["p_boot"] == pub_con[e]["p_boot"] and b["ci"] == pub_con[e]["ci"])}
    gate2 = {"contrasts": rep, "passes": bool(all(r["exact"] for r in rep.values()))}
    log(f"gate (ii) exact replay: passes {gate2['passes']} | " +
        ", ".join(f"{e} {r['p']:.4f}/{r['p_pub']:.4f}" for e, r in rep.items()))

    cs = ClipSet("w34", yk, gk, M)
    members = [np.flatnonzero(cs.inv == k) for k in range(cs.G)]
    draws = np.random.default_rng([SEED, 3]).integers(0, cs.G, size=(400, cs.G))
    worst = 0.0
    for e in VIDEO:
        C = cs.counts(S["clip"][keep]) - cs.counts(S[e][keep])
        fv = fast_gaps(cs, C, draws)
        sv = []
        for row in draws:
            i = np.concatenate([members[g] for g in row])
            if yk[i].min() == yk[i].max():
                continue
            sv.append(_auc(S["clip"][keep][i], yk[i]) - _auc(S[e][keep][i], yk[i]))
        worst = max(worst, float(np.max(np.abs(fv - np.asarray(sv)))))
    gate3 = {"max_abs_deviation": worst, "tolerance": 1e-9, "passes": bool(worst <= 1e-9)}
    log(f"gate (iii) fast bootstrap: {gate3}")

    # Estimate generator parameters from refitted scores.
    w, tot = pair_counts(yk, gk)[:2]
    pooled_mean = float(np.mean(list(aucs.values())))
    theta0 = solve_theta(pooled_mean, w, tot)
    rhos = {e: within_class_corr(S["clip"][keep], S[e][keep], yk) for e in VIDEO}
    rho = float(np.mean(list(rhos.values())))
    params = {"pooled_mean_auc": pooled_mean, "theta_between_subject": theta0, "within_subject_pair_share": w / tot,
              "rho_per_pair": rhos, "rho_mean": rho, "rho_min": min(rhos.values()), "rho_max": max(rhos.values()),
              "icc": ICC, "n_groups": int(cs.G), "n_pos": int(cs.n1)}
    log(f"parameters: theta {theta0:.4f} (pooled {pooled_mean:.4f}) rho mean {rho:.4f} "
        f"[{params['rho_min']:.4f}, {params['rho_max']:.4f}]")

    dsg = Design("w34", yk, gk, M)
    rng = np.random.default_rng(SEED)
    # Independently validate the secondary test's critical value.
    zc = np.abs(sim_paired(dsg, theta0, rho, rng, N_NULL_Z))
    crit = float(np.quantile(zc, 1 - ALPHA / M))
    zv = np.abs(sim_paired(dsg, theta0, rho, rng, N_NULL_Z))
    kv = int((zv >= crit).sum())
    lo4, hi4 = wilson(kv, N_NULL_Z)
    gate4 = {"critical_abs_z": crit, "independent_rate": kv / N_NULL_Z, "wilson95": [lo4, hi4],
             "alpha_over_m": ALPHA / M, "passes": bool(lo4 <= ALPHA / M <= hi4)}
    log(f"gate (iv) proxy validation: {gate4}")

    gate4["consequence"] = ("secondary proxy FAILED its validation gate; its values are reported descriptively and "
                            "not interpreted" if not gate4["passes"] else "secondary proxy validated")
    gates = {"i_refit": gate1, "ii_exact_replay": gate2, "iii_fast_bootstrap": gate3, "iv_proxy_null": gate4}
    hard = gate1["passes"] and gate2["passes"] and gate3["passes"]
    res = {"provenance": run_provenance(seeds=[SEED, W34_SEED], device="cpu"),
           "criteria_heading": CRITERIA_HEADING, "amendment_heading": AMENDMENT_HEADING,
           "design": {"m": M, "alpha_over_m": ALPHA / M, "nboot": NBOOT, "fold_hash": fold_hash},
           "parameters": params, "gates": gates}
    if not hard:
        res["status"] = "HOLD: gate failed"
        json.dump(res, open(out_json, "w"), indent=2)
        log(f"wrote {out_json} | {res['status']}; no simulation run")
        logf.close(); return

    def boot_study(theta_a, theta_b, r):
        n = len(yk)
        e1, e2 = rng.standard_normal((1, n)), rng.standard_normal((1, n))
        u1, u2 = rng.standard_normal((1, dsg.K)), rng.standard_normal((1, dsg.K))
        sa = dsg.draw(rng, 1, theta_a, e1, u1)[0]
        sb = dsg.draw(rng, 1, theta_b, r * e1 + math.sqrt(1 - r ** 2) * e2, r * u1 + math.sqrt(1 - r ** 2) * u2)[0]
        p = boot_p(cs, cs.counts(sa) - cs.counts(sb), rng)
        return p, _auc(sa, yk) - _auc(sb, yk)

    def boot_power(gap, r, n):
        th = thetas_for_gap(theta0, gap, w, tot)
        if th is None:
            return None, None
        k = 0
        for _ in range(n):
            p, g = boot_study(th[0], th[1], r)
            k += int(p <= ALPHA / M and g > 0)
        return k / n, wilson(k, n)

    def z_power(gap, r, n):
        th = thetas_for_gap(theta0, gap, w, tot)
        z = sim_paired_alt(dsg, th[0], th[1], r, rng, n)
        k = int((z >= crit).sum())
        return k / n, wilson(k, n)

    gap_hi = pooled_auc(0.995, w, tot) - pooled_auc(2 * theta0 - 0.995, w, tot)

    def mde(powerf, r):
        lo, hi, trace = 0.0, gap_hi, []
        top, tw = powerf(hi, r, N_STEP)
        trace.append({"gap": hi, "power": top, "wilson95": tw, "n": N_STEP})
        if top < POWER:
            return {"exceeds_upper_bound": True, "upper_bound": hi, "power_at_upper": top,
                    "power_at_upper_wilson95": tw, "trace": trace,
                    "note": "a point estimate below 0.80 at the upper bound, not a statistically established bound"}
        for _ in range(STEPS):
            mid = (lo + hi) / 2
            pw, mw = powerf(mid, r, N_STEP)
            trace.append({"gap": mid, "power": pw, "wilson95": mw, "n": N_STEP})
            lo, hi = (mid, hi) if pw < POWER else (lo, mid)
        pw, wi = powerf(hi, r, N_FINAL)
        return {"exceeds_upper_bound": False, "mde_pooled_gap": hi, "search_bracket": [lo, hi],
                "search_bracket_note": "the final bisection bracket on noisy power estimates, not a confidence interval",
                "power_at_mde": pw, "power_wilson95": wi, "n_final": N_FINAL, "trace": trace}

    t1 = time.time()
    k0 = 0; k0dir = 0
    for _ in range(N_NULL_BOOT):
        p, g = boot_study(theta0, theta0, rho)
        k0 += int(p <= ALPHA / M); k0dir += int(p <= ALPHA / M and g > 0)
    null = {"n": N_NULL_BOOT, "rate_two_sided": k0 / N_NULL_BOOT, "wilson95": wilson(k0, N_NULL_BOOT),
            "rate_correct_direction": k0dir / N_NULL_BOOT,
            "rate_correct_direction_wilson95": wilson(k0dir, N_NULL_BOOT), "nominal": ALPHA / M}
    log(f"bootstrap null size at 0.05/6: {null} ({time.time() - t1:.0f}s)")

    W9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))["contrasts"]
    pain = sorted((r["label"], r["gap"]) for r in W9 if r["holm"]["boot_subject"] < ALPHA)
    obs_max = max(abs(r["gap"]) for r in W34["contrasts"])
    bench = [{"benchmark": lab_, "pooled_gap": g} for lab_, g in pain] + \
            [{"benchmark": "largest observed emotion gap in absolute value", "pooled_gap": obs_max}]
    for b in bench:
        pw, wi = boot_power(b["pooled_gap"], rho, N_FINAL)
        zp, zw = z_power(b["pooled_gap"], rho, N_FINAL)
        b.update({"power_bootstrap": pw, "power_bootstrap_wilson95": wi, "power_proxy_z": zp, "power_proxy_wilson95": zw})
        log(f"  {b['benchmark']:40s} gap {b['pooled_gap']:.4f} | bootstrap power {pw:.3f} [{wi[0]:.3f}, {wi[1]:.3f}] "
            f"| proxy {zp:.3f}")
    mdes = {}
    for tag, r in (("rho_mean", rho), ("rho_min", params["rho_min"]), ("rho_max", params["rho_max"])):
        t1 = time.time()
        mdes[tag] = {"rho": r, "bootstrap": mde(boot_power, r), "proxy_z": mde(z_power, r),
                     "proxy_note": ("the proxy uses the critical value set at the mean rho, not recalibrated for "
                                    "this rho, and the proxy failed its validation gate; descriptive only")}
        mb = mdes[tag]["bootstrap"]
        log(f"  MDE at {tag} ({r:.4f}): bootstrap {mb.get('mde_pooled_gap', 'exceeds')} "
            f"| proxy {mdes[tag]['proxy_z'].get('mde_pooled_gap', 'exceeds')} ({time.time() - t1:.0f}s)")
    res.update({"bootstrap_null": null, "benchmarks": bench, "mde": mdes,
                "reading": ("correct-direction first-step power of W34's paired subject bootstrap under the declared "
                            "generator (binormal scores, ICC 0.3, observed labels and group sizes, within-class rho). "
                            "The pain benchmarks are SELECTED significant effects measured at K=2 to 16 on UNBC, at a "
                            "base AUC near 0.87, while W34 tests K=16 alone at a base AUC near 0.57; equal gaps there "
                            "are equal changes in ranking probability, not equal separations. The subset's direction "
                            "labels are position-imbalanced (W34, 0.446). No family verdict is drawn from a power value."),
                "status": "COMPLETE"})
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
