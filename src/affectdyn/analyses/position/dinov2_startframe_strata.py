"""DINOv2 on the Aff-Wild2 peak task within start-frame strata."""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import get_context
from statistics import NormalDist
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.provenance import run_provenance

OUT = "runs/dinov2_startframe_strata"
SEED = 20260926
NBOOT = 10_000
ALPHA, M = 0.05, 4
KS = (2, 4, 8, 16)
ICC = 0.3
S_CHOICES = (10, 20)
BAL = (0.45, 0.55)
N_NULL, N_STEP, N_FINAL, N_BENCH, STEPS = 2000, 1000, 2000, 2000, 9
WORKERS = 16
W28_START_AUC = 0.35897595472770716
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA = "## 2026-09-18 — W38 (D27) PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT = "## 2026-09-18 — W38 CRITERIA AMENDMENT"
ND = NormalDist()


def load():
    clips = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    subj = np.array([str(c["subject"]) for c in clips])
    video = np.array([str(c["video"]) for c in clips])
    traj = np.array([int(c["traj"]) for c in clips])
    start = np.array([int(os.path.splitext(os.path.basename(c["start"]))[0]) for c in clips], dtype=float)
    y = (traj == 2).astype(np.int64)
    z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    w7 = json.load(open("runs/peak_cluster_inference/cluster.json"))
    pub = {c["cell"]: c["auc"] for c in w7["cells"] if c["corpus"] == "aw2_image"}
    return clips, subj, video, start, y, z, pub


def partition(start, keep, S):
    """Strata at the linear-method quantiles of the retained clips' start frames; ties go up."""
    cuts = np.unique(np.quantile(start[keep], np.linspace(0, 1, S + 1)[1:-1], method="linear"))
    return np.searchsorted(cuts, start, side="right"), len(cuts) + 1


def counts(s, y, grp, strata, G, idx):
    """C (wins, ties one half) and P (eligible pairs) between identifier groups, within strata."""
    C = np.zeros((G, G)); P = np.zeros((G, G))
    for k in np.unique(strata[idx]):
        ii = idx[strata[idx] == k]
        pos, neg = ii[y[ii] == 1], ii[y[ii] == 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        d = s[pos][:, None] - s[neg][None, :]
        win = (d > 0).astype(float) + 0.5 * (d == 0)
        Gp = np.zeros((len(pos), G)); Gp[np.arange(len(pos)), grp[pos]] = 1
        Gn = np.zeros((len(neg), G)); Gn[np.arange(len(neg)), grp[neg]] = 1
        C += Gp.T @ win @ Gn
        P += Gp.T @ np.ones((len(pos), len(neg))) @ Gn
    return C, P


def weights(rng, G, B):
    draws = rng.integers(0, G, (B, G))
    W = np.zeros((B, G))
    np.add.at(W, (np.repeat(np.arange(B), G), draws.ravel()), 1)
    return W


def qform(W, A):
    return ((W @ A) * W).sum(axis=1)


def boot(C, P, W):
    """Stratified AUC per resample, NaN where the eligible-pair denominator is zero."""
    num, den = qform(W, C), qform(W, P)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def pvalue(v):
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None, 0
    p = 2.0 * min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return float(min(1.0, max(p, 1.0 / len(v)))), len(v)


def summarize(C, P, W):
    v = boot(C, P, W)
    p, nv = pvalue(v)
    ok = v[~np.isnan(v)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return {"auc": float(C.sum() / P.sum()), "ci": [float(lo), float(hi)], "p_boot": p,
            "attempted": int(len(v)), "valid": int(nv)}, v


def holm(ps):
    """Local Holm adjustment; the shared implementation is
    inference.aggregation_permutation_tests.holm. Family membership and size are specified by the
    caller.
    """
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def mw_direct(s, y):
    """Mann-Whitney AUC with average ranks, independent of the pair-count code."""
    order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); ss = s[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2), n1 * n0


def explicit(s, y, grp, strata, draw, members):
    """Replicate whole groups, then count within-stratum pairs directly (numerator, denominator)."""
    ix = np.concatenate([members[g] for g in draw])
    ss, yy, st = s[ix], y[ix], strata[ix]
    num = den = 0.0
    for k in np.unique(st):
        m = st == k
        if yy[m].min() == yy[m].max():
            continue
        u, d = mw_direct(ss[m], yy[m]); num += u; den += d
    return num, den


def pooled_scale(theta, q):
    return (1 - q) * theta + q * ND.cdf(ND.inv_cdf(theta) / np.sqrt(1 - ICC))


def solve_theta(target, q):
    lo, hi = 1e-6, 1 - 1e-6
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pooled_scale(mid, q) < target else (lo, mid)
    return (lo + hi) / 2


_SIM = {}


def _sim_init(y, grp, strata, idx, G, P):
    _SIM.update(y=y, grp=grp, strata=strata, idx=idx, G=G, P=P)


def _sim_study(args):
    theta, seed = args
    y, grp, strata, idx, G, P = (_SIM[k] for k in ("y", "grp", "strata", "idx", "G", "P"))
    rng = np.random.default_rng(seed)
    mu = np.sqrt(2) * ND.inv_cdf(theta)
    u = rng.standard_normal(G)
    s = mu * y + np.sqrt(1 - ICC) * rng.standard_normal(len(y)) + np.sqrt(ICC) * u[grp]
    C, _ = counts(s, y, grp, strata, G, idx)
    v = boot(C, P, weights(rng, G, NBOOT))
    p, _nv = pvalue(v)
    a = C.sum() / P.sum()
    return (p is not None and p <= ALPHA / M and a > 0.5, p is not None and p <= ALPHA,
            p is not None and p <= ALPHA / M, a)


def run_studies(pool, theta, n, seed_base):
    res = pool.map(_sim_study, [(theta, seed_base + i) for i in range(n)], chunksize=8)
    k = sum(r[0] for r in res)
    return {"theta_between_subject": theta, "n": n, "power_first_step_correct_direction": k / n,
            "wilson95": wilson(k, n), "mean_stratified_auc": float(np.mean([r[3] for r in res])),
            "reject_two_sided_0.05": sum(r[1] for r in res) / n,
            "reject_two_sided_0.05_wilson95": wilson(sum(r[1] for r in res), n),
            "reject_two_sided_alpha_over_m": sum(r[2] for r in res) / n,
            "reject_two_sided_alpha_over_m_wilson95": wilson(sum(r[2] for r in res), n)}


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tag = "startframe_stratified"
    logf = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    clips, subj, video, start, y, z, pub = load()
    groups = sorted(set(subj)); gi = {g: i for i, g in enumerate(groups)}; grp = np.array([gi[g] for g in subj])
    G = len(groups)
    vids = sorted(set(video)); vi = {v: i for i, v in enumerate(vids)}; vgrp = np.array([vi[v] for v in video])
    members = [np.flatnonzero(grp == g) for g in range(G)]
    gates = {}

    align = bool(np.array_equal(z["aw2img__y"].astype(int), y) and
                 all(str(a) == b for a, b in zip(z["aw2img__subject"], subj)))
    w28 = _auc(start, y)
    gates["i_alignment_and_w28"] = {"alignment": align, "start_frame_auc_legacy": w28,
                                    "stored": W28_START_AUC, "abs_delta": abs(w28 - W28_START_AUC),
                                    "passes": bool(align and abs(w28 - W28_START_AUC) <= 1e-15)}
    log(f"gate (i): {gates['i_alignment_and_w28']}")

    keep = z["aw2img__mask"].astype(bool)
    idx = np.flatnonzero(keep)

    one = np.zeros(len(y), dtype=int)
    # Select the partition using only labels and start frames.
    W_bal = weights(np.random.default_rng(SEED + 2), G, NBOOT)
    part = {}
    for S in S_CHOICES:
        st, S_eff = partition(start, keep, S)
        Cb, Pb = counts(start, y, grp, st, G, idx)
        bal, _ = summarize(Cb, Pb, W_bal)
        per = []
        for k in range(S_eff):
            ii = idx[st[idx] == k]; pos, neg = ii[y[ii] == 1], ii[y[ii] == 0]
            per.append({"stratum": k, "clips": int(len(ii)), "positives": int(len(pos)), "pairs": int(len(pos) * len(neg)),
                        "subjects": int(len(np.unique(grp[ii]))), "videos": int(len(np.unique(vgrp[ii]))),
                        "positive_bearing_groups": int(len(np.unique(grp[pos]))),
                        "frame_range": [float(start[ii].min()), float(start[ii].max())] if len(ii) else None,
                        "start_quantiles_pos": [float(q) for q in np.quantile(start[pos], [.25, .5, .75])] if len(pos) else None,
                        "start_quantiles_neg": [float(q) for q in np.quantile(start[neg], [.25, .5, .75])] if len(neg) else None,
                        "within_stratum_start_auc": (float(mw_direct(start[ii], y[ii])[0] / mw_direct(start[ii], y[ii])[1])
                                                     if len(pos) and len(neg) else None)})
        passes = bool(BAL[0] <= bal["ci"][0] and bal["ci"][1] <= BAL[1])
        Pall_same = counts(np.zeros(len(y)), y, grp, one, G, idx)[1]
        Pv_st = counts(np.zeros(len(y)), y, vgrp, st, len(vids), idx)[1]
        Pv_all = counts(np.zeros(len(y)), y, vgrp, one, len(vids), idx)[1]
        part[S] = {"effective_S": S_eff, "balance": bal, "balance_passes": passes, "strata": per,
                   "eligible_pair_coverage": float(Pb.sum() / Pall_same.sum()),
                   "within_subject_share_all": float(np.trace(Pall_same) / Pall_same.sum()),
                   "within_subject_share_stratified": float(np.trace(Pb) / Pb.sum()),
                   "within_video_share_all": float(np.trace(Pv_all) / Pv_all.sum()),
                   "within_video_share_stratified": float(np.trace(Pv_st) / Pv_st.sum()),
                   "single_class_strata": [p_["stratum"] for p_ in per if p_["positives"] in (0, p_["clips"])]}
        log(f"  S={S} (effective {S_eff}): balance AUC {bal['auc']:.4f} CI [{bal['ci'][0]:.4f}, {bal['ci'][1]:.4f}] "
            f"-> {'PASS' if passes else 'fail'} | coverage {part[S]['eligible_pair_coverage']:.3f} | "
            f"within-subject {part[S]['within_subject_share_all']:.4f} -> {part[S]['within_subject_share_stratified']:.4f}")
    selected = next((S for S in S_CHOICES if part[S]["balance_passes"]), None)
    log(f"selected S: {selected}")

    # Access DINOv2 scores only after selecting the partition.
    finite = [np.isfinite(z[f"aw2img__dinov2_K{k}"]) for k in KS]
    same_members = all(np.array_equal(f & keep, keep) for f in finite)
    rows = {}
    for k in KS:
        s = z[f"aw2img__dinov2_K{k}"]
        leg = _auc(s[idx], y[idx])
        C, P = counts(s, y, grp, one, G, idx)
        u, d = mw_direct(s[idx], y[idx])
        rows[f"K{k}"] = {"legacy": leg, "w7": pub[f"dinov2_K{k}"], "legacy_delta": abs(leg - pub[f"dinov2_K{k}"]),
                         "halftie_qform": C.sum() / P.sum(), "halftie_direct": u / d,
                         "halftie_delta": abs(C.sum() / P.sum() - u / d), "cross_class_ties_effect":
                         abs(leg - u / d)}
    g2 = same_members and all(r["legacy_delta"] <= 1e-12 and r["halftie_delta"] <= 1e-9 for r in rows.values())
    gates["ii_s1_reproduction"] = {"identical_finite_membership": bool(same_members), "rows": rows, "passes": bool(g2)}
    log(f"gate (ii): passes {g2} | " + ", ".join(f"{k} legacy d {r['legacy_delta']:.1e} halftie d "
                                                  f"{r['halftie_delta']:.1e}" for k, r in rows.items()))

    # Check the quadratic form against explicit clip replication over 20 resamples.
    strata10, _ = partition(start, keep, 10)
    s2 = z["aw2img__dinov2_K2"]
    C, P = counts(s2, y, grp, strata10, G, idx)
    rng_g = np.random.default_rng(SEED + 1)
    worst = 0.0
    kept_members = [np.intersect1d(m, idx) for m in members]
    for _ in range(20):
        draw = rng_g.integers(0, G, G)
        w = np.bincount(draw, minlength=G).astype(float)
        nq, dq = w @ C @ w, w @ P @ w
        ne, de = explicit(s2, y, grp, strata10, draw, kept_members)
        worst = max(worst, abs(nq - ne) / max(1.0, abs(ne)), abs(dq - de) / max(1.0, abs(de)))
    gates["iii_explicit_resamples"] = {"max_rel_delta": worst, "tolerance": 1e-9, "passes": bool(worst <= 1e-9)}
    log(f"gate (iii): {gates['iii_explicit_resamples']}")

    def case(s, yy, gg, st):
        s, yy, gg, st = map(np.asarray, (s, yy, gg, st)); Gc = int(gg.max()) + 1
        ii = np.arange(len(yy)); Cc, Pc = counts(s.astype(float), yy, gg, st, Gc, ii)
        mem = [np.flatnonzero(gg == g) for g in range(Gc)]
        out = []
        for draw in (np.arange(Gc), np.zeros(Gc, dtype=int), np.array([Gc - 1] * Gc)):
            w = np.bincount(draw, minlength=Gc).astype(float)
            ne, de = explicit(s.astype(float), yy, gg, st, draw, mem) if len(np.unique(yy[np.concatenate(
                [mem[g] for g in draw])])) == 2 else (0.0, 0.0)
            out.append((w @ Cc @ w, w @ Pc @ w, ne, de))
        return out
    cases = {
        "cross_class_ties": case([1, 1, 0, 2, 1], [1, 0, 0, 1, 1], [0, 0, 1, 1, 2], [0, 0, 0, 0, 0]),
        "repeated_subjects": case([3, 1, 2, 0], [1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]),
        "single_class_stratum": case([1, 2, 3, 4], [1, 1, 0, 0], [0, 1, 0, 1], [0, 0, 1, 1]),
        "zero_eligible_pairs": case([1, 2, 3, 4], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 2, 3]),
    }
    unit_ok = all(abs(a - c) < 1e-12 and abs(b - d) < 1e-12 for rows_ in cases.values() for a, b, c, d in rows_)
    gates["iv_unit_cases"] = {"passes": bool(unit_ok),
                              "zero_eligible_denominator": cases["zero_eligible_pairs"][0][1]}
    log(f"gate (iv): {gates['iv_unit_cases']}")

    Sv = selected or S_CHOICES[0]
    st_v, _ = partition(start, keep, Sv)
    _, P_v = counts(np.zeros(len(y)), y, grp, st_v, G, idx)
    q_v = float(np.trace(P_v) / P_v.sum())
    rng_v = np.random.default_rng(SEED + 3); th = 0.60; mu = np.sqrt(2) * ND.inv_cdf(th); vals = []
    for _ in range(200):
        u = rng_v.standard_normal(G)
        s = mu * y + np.sqrt(1 - ICC) * rng_v.standard_normal(len(y)) + np.sqrt(ICC) * u[grp]
        Cs, Ps = counts(s, y, grp, st_v, G, idx); vals.append(Cs.sum() / Ps.sum())
    pred = pooled_scale(th, q_v); se = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    gates["v_transform"] = {"S": Sv, "q_within_subject": q_v, "theta": th, "predicted": pred,
                            "simulated_mean": float(np.mean(vals)), "se": se,
                            "passes": bool(abs(np.mean(vals) - pred) <= 3 * se + 1e-4)}
    log(f"gate (v): {gates['v_transform']}")

    hard = all(g["passes"] for g in gates.values())
    base = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRITERIA,
            "amendment_heading": AMENDMENT, "gates": gates, "partitions": {str(k): v for k, v in part.items()},
            "selected_S": selected}
    if not hard:
        base["status"] = "HOLD: a gate failed"
        json.dump(base, open(f"{OUT}/{tag}.json", "w"), indent=2)
        log(f"wrote {OUT}/{tag}.json | {base['status']}"); return

    results = {}
    if selected is None:
        log("neither partition passes the balance screen: DESCRIPTIVE ONLY, reported at S = 10")
    S_use = selected or S_CHOICES[0]
    st, _ = partition(start, keep, S_use)
    rng = np.random.default_rng(SEED)
    W = weights(rng, G, NBOOT)                         # Share resamples across all tests and both estimands.
    raw = {}
    for name in [f"dinov2_K{k}" for k in KS] + ["clip_K16"]:
        s = z[f"aw2img__{name}"]
        Cs, Ps = counts(s, y, grp, st, G, idx)
        C1, P1 = counts(s, y, grp, one, G, idx)
        r, v = summarize(Cs, Ps, W)
        v1 = boot(C1, P1, W)
        d = v1 - v; d = d[~np.isnan(d)]
        r.update({"published_auc": pub[name], "pooled_halftie": float(C1.sum() / P1.sum()),
                  "pooled_minus_stratified": float(C1.sum() / P1.sum() - r["auc"]),
                  "pooled_minus_stratified_ci": ([float(x) for x in np.percentile(d, [2.5, 97.5])]
                                                 if len(d) else [float("nan"), float("nan")])})
        results[name] = r; raw[name] = r["p_boot"]
        log(f"  {name:11s} stratified {r['auc']:.4f} CI [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] p {r['p_boot']:.4f} "
            f"valid {r['valid']}/{r['attempted']} | published {pub[name]:.4f} | pooled-minus-stratified "
            f"{r['pooled_minus_stratified']:+.4f}")
    fam = [f"dinov2_K{k}" for k in KS]
    if any(results[n]["valid"] == 0 for n in results):
        base.update({"results": results, "status": "HOLD: zero valid resamples"})
        json.dump(base, open(f"{OUT}/{tag}.json", "w"), indent=2)
        log("HOLD: zero valid resamples"); return
    adj = holm(np.array([raw[n] for n in fam]))
    for n, a in zip(fam, adj):
        results[n]["holm_4"] = float(a)
    above = [n for n in fam if results[n]["auc"] > 0.5 and results[n]["holm_4"] <= ALPHA]
    below = [n for n in fam if results[n]["auc"] < 0.5 and results[n]["holm_4"] <= ALPHA]
    if selected is None:
        reading = "DESCRIPTIVE ONLY: no partition passed the balance screen"
    elif above:
        reading = "R2: above-chance discrimination retained within start-frame strata by " + ", ".join(above)
    else:
        reading = "R3: no granularity shows significant above-chance discrimination within start-frame strata"
    log(f"reading: {reading}; reversals: {below or 'none'}")

    # Evaluate sensitivity and null size at the selected partition.
    _, P_s = counts(np.zeros(len(y)), y, grp, st, G, idx)
    q_s = float(np.trace(P_s) / P_s.sum())
    bench = min(pub[n] for n in fam)
    sens = {"q_within_subject": q_s, "benchmark_stratified_scale": bench}
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_sim_init, initargs=(y, grp, st, idx, G, P_s)) as pool:
        sens["null"] = run_studies(pool, 0.5, N_NULL, SEED * 10)
        log(f"  null size: {sens['null']}")
        th_b = solve_theta(bench, q_s)
        sens["at_benchmark"] = run_studies(pool, th_b, N_BENCH, SEED * 10 + 100_000)
        log(f"  power at benchmark {bench:.4f}: {sens['at_benchmark']['power_first_step_correct_direction']:.3f} "
            f"{sens['at_benchmark']['wilson95']}")
        lo, hi, trace = 0.5, 0.75, []
        for step in range(STEPS):
            mid = (lo + hi) / 2
            r_ = run_studies(pool, solve_theta(mid, q_s), N_STEP, SEED * 10 + 200_000 + step * 10_000)
            trace.append({"stratified_auc": mid, **r_})
            lo, hi = (mid, hi) if r_["power_first_step_correct_direction"] < 0.8 else (lo, mid)
        fin = run_studies(pool, solve_theta(hi, q_s), N_FINAL, SEED * 10 + 400_000)
        sens["mde"] = {"mde_stratified_auc": hi, "search_bracket": [lo, hi],
                       "search_bracket_note": "the final bisection bracket on noisy power estimates, not a confidence interval",
                       "power_at_mde": fin["power_first_step_correct_direction"], "power_wilson95": fin["wilson95"],
                       "trace": trace}
        log(f"  MDE {hi:.4f} (bracket {lo:.4f}-{hi:.4f}), power {fin['power_first_step_correct_direction']:.3f}")

    base.update({"S_used": S_use, "results": results, "reading": reading, "reversals": below,
                 "sensitivity": sens, "status": "COMPLETE",
                 "scope": ("Stratifying changes the pair population; neither persistence nor attenuation identifies "
                           "start frame as the explanation. Residual within-stratum association, subject and video "
                           "composition and other confounds remain possible; the scores are those of one fit.")})
    json.dump(base, open(f"{OUT}/{tag}.json", "w"), indent=2)
    log(f"wrote {OUT}/{tag}.json | COMPLETE")


if __name__ == "__main__":
    main()
