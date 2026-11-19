"""Endpoint magnitude as a label-side cue for the pain peak task."""
from __future__ import annotations
import json, os, sys, time

import numpy as np

from affectdyn.provenance import run_provenance

OUT = "runs/endpoint_magnitude_controls"
SEED = 20260912
NBOOT = 10000
NPERM = 20000
MDE_NSIM = 500
MDE_STEPS = 7
CALIPER = 0.5                      # Measure the caliper in PSPI units; the label margin is 1.
KS = (4, 16)                       # Use K=4 as the primary shape granularity.
PRIMARY_K = 4


def midrank_auc(s, y):
    """AUC counting a tie as one half, which is the Mann-Whitney definition."""
    s = np.asarray(s, float)
    order = np.argsort(s, kind="stable")
    uniq, inv, cnt = np.unique(s[order], return_inverse=True, return_counts=True)
    r = np.empty(len(s))
    r[order] = (np.cumsum(cnt) - cnt + (cnt + 1) / 2.0)[inv]
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def subject_bootstrap(s, y, g, rng, nboot=NBOOT):
    subs = np.unique(g)
    idx = {u: np.flatnonzero(g == u) for u in subs}
    vals = []
    for _ in range(nboot):
        draw = rng.choice(len(subs), size=len(subs), replace=True)
        take = np.concatenate([idx[subs[j]] for j in draw])
        yb = y[take]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        vals.append(midrank_auc(s[take], yb))
    v = np.asarray(vals, float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    share = min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"auc": midrank_auc(s, y), "ci": [float(lo), float(hi)],
            "p_boot": max(2.0 * share, 1.0 / max(1, len(v))),
            "excludes_half": bool(lo > 0.5 or hi < 0.5),
            "n_resamples_used": int(len(v))}


def match_on_magnitude(dm, y, caliper=CALIPER):
    """Pair each peaked clip with the monotonic clip nearest in endpoint magnitude."""
    peaked = [int(i) for i in np.flatnonzero(y == 1)]
    mono = [int(i) for i in np.flatnonzero(y == 0)]
    peaked.sort(key=lambda i: (dm[i], i))
    used, pairs, dropped = set(), [], []
    for i in peaked:
        best, bestd = None, None
        for j in mono:
            if j in used:
                continue
            d = abs(dm[i] - dm[j])
            if d <= caliper and (bestd is None or d < bestd):
                best, bestd = j, d
        if best is None:
            dropped.append(i)
            continue
        used.add(best)
        pairs.append((i, best, float(bestd)))
    return pairs, dropped


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / den) if den > 0 else float("nan")


def spearman_perm_p(a, b, rng, nperm=NPERM):
    obs = abs(spearman(a, b))
    hits = sum(abs(spearman(a, rng.permutation(b))) >= obs - 1e-12 for _ in range(nperm))
    return float((1 + hits) / (1 + nperm))


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/endpoint_magnitude.log", "w")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    traj, subject, slope = lab["traj"], lab["subject"], lab["slope"]
    y_all = (traj == 2).astype(np.int64)
    rng = np.random.default_rng(SEED)

    res = {"provenance": run_provenance(seeds=[SEED]), "seed": SEED, "nboot": NBOOT, "nperm": NPERM,
           "caliper_pspi": CALIPER, "primary_K": PRIMARY_K, "gate": {}, "tests": {}}

    same_subject = bool(subject.shape == W7["unbc__subject"].shape and (subject == W7["unbc__subject"]).all())
    same_y = bool((y_all == W7["unbc__y"]).all())
    mask = traj != -1
    mono_ok = bool((np.abs(slope[mask & (y_all == 0)]) >= 1 - 1e-9).all())
    res["gate"] = {"i_subject_and_label_match_w7": bool(same_subject and same_y),
                   "ii_monotonic_magnitude_at_least_1": mono_ok,
                   "passes": bool(same_subject and same_y and mono_ok)}
    p(f"pipeline gate: {res['gate']}")
    if not res["gate"]["passes"]:
        res["status"] = "HOLD: pipeline gate failed"
        json.dump(res, open(f"{OUT}/endpoint_magnitude.json", "w"), indent=2)
        p("HOLD"); return

    keep = W7["unbc__keep"].astype(bool)
    y = y_all[keep]
    g = subject[keep]
    dm = np.abs(slope[keep])
    res["n_clips"] = int(keep.sum())
    res["n_peaked"] = int(y.sum())
    res["n_subjects"] = int(len(set(g.tolist())))
    p(f"{res['n_clips']} shape-labeled clips, {res['n_peaked']} peaked, over {res['n_subjects']} subjects")
    for name, sel in (("peaked", y == 1), ("monotonic", y == 0)):
        q = np.percentile(dm[sel], [25, 50, 75])
        res.setdefault("endpoint_magnitude", {})[name] = {
            "n": int(sel.sum()), "median": float(q[1]), "iqr": [float(q[0]), float(q[2])],
            "min": float(dm[sel].min()), "max": float(dm[sel].max())}
        p(f"  |d| {name:10s} median {q[1]:.2f} IQR {q[0]:.2f} to {q[2]:.2f} "
          f"range {dm[sel].min():.2f} to {dm[sel].max():.2f}")

    res["tests"]["a_magnitude_alone"] = subject_bootstrap(dm, y, g, rng)
    p(f"(a) |d| alone: AUC {res['tests']['a_magnitude_alone']['auc']:.4f} "
      f"CI {[round(x, 4) for x in res['tests']['a_magnitude_alone']['ci']]} "
      f"p {res['tests']['a_magnitude_alone']['p_boot']:.4f}")

    pairs, dropped = match_on_magnitude(dm, y, CALIPER)
    sel = np.array(sorted([i for pr in pairs for i in pr[:2]]), dtype=int)
    res["matching"] = {"n_pairs": len(pairs), "n_peaked_dropped": len(dropped),
                       "max_abs_difference": float(max([pr[2] for pr in pairs], default=float("nan"))),
                       "mean_abs_difference": float(np.mean([pr[2] for pr in pairs])) if pairs else float("nan"),
                       "n_clips_matched": int(len(sel)),
                       "n_subjects_matched": int(len(set(g[sel].tolist())))}
    p(f"(b) matching: {len(pairs)} pairs, {len(dropped)} peaked clips dropped, "
      f"largest |d| difference {res['matching']['max_abs_difference']:.3f}, "
      f"{res['matching']['n_subjects_matched']} subjects")
    res["tests"]["b_magnitude_alone_on_matched"] = subject_bootstrap(dm[sel], y[sel], g[sel], rng)
    p(f"    |d| on the matched set: AUC {res['tests']['b_magnitude_alone_on_matched']['auc']:.4f} "
      f"CI {[round(x, 4) for x in res['tests']['b_magnitude_alone_on_matched']['ci']]}")

    for k in KS:
        s = W7[f"unbc__clip_K{k}"][keep]
        full = subject_bootstrap(s, y, g, rng)
        matched = subject_bootstrap(s[sel], y[sel], g[sel], rng)
        rho_within = {}
        for cname, cy in (("peaked", 1), ("monotonic", 0)):
            m = y == cy
            rho_within[cname] = {"spearman": spearman(s[m], dm[m]),
                                 "p_perm": spearman_perm_p(s[m], dm[m], rng),
                                 "n": int(m.sum())}
        res["tests"][f"clip_K{k}"] = {"full": full, "matched": matched, "c_within_class": rho_within}
        p(f"    CLIP K={k}: full AUC {full['auc']:.4f} CI {[round(x, 4) for x in full['ci']]} "
          f"(excludes 0.5 {full['excludes_half']}) | matched AUC {matched['auc']:.4f} "
          f"CI {[round(x, 4) for x in matched['ci']]} (excludes 0.5 {matched['excludes_half']})")
        for cname, r_ in rho_within.items():
            p(f"      (c) {cname:10s} rho(score, |d|) {r_['spearman']:+.3f} p {r_['p_perm']:.4f} n {r_['n']}")

    # Estimate the matched design's detectable effect.
    from affectdyn.inference.bootstrap_detection_power import ClipSet, mde_boot
    mrng = np.random.default_rng([SEED, 14])

    def _mde(name, yy, gg):
        cs = ClipSet(name, yy, gg, 1)
        out = {"design": cs.describe()}
        for icc in (0.3, 0.0):
            out[f"icc{icc}"] = mde_boot(cs, icc, mrng, MDE_NSIM, NBOOT, MDE_STEPS,
                                        lambda m: p("      " + m.strip()))
        return out

    p("(d) minimum detectable effect of the bootstrap that produced the matched result")
    res["mde"] = {"matched": _mde("matched", y[sel], g[sel]), "full": _mde("full", y, g)}
    for nm in ("matched", "full"):
        p(f"    {nm}: MDE {res['mde'][nm]['icc0.3']['mde_auc_at_80pc_power']:.3f} at ICC 0.3, "
          f"{res['mde'][nm]['icc0.0']['mde_auc_at_80pc_power']:.3f} at ICC 0, on "
          f"{res['mde'][nm]['design']['n_positive']} positives over "
          f"{res['mde'][nm]['design']['n_subjects']} subjects")

    a = res["tests"]["a_magnitude_alone"]
    pk = res["tests"][f"clip_K{PRIMARY_K}"]
    if a["excludes_half"] and pk["full"]["excludes_half"] and not pk["matched"]["excludes_half"]:
        verdict = "SUPPORTED"
    elif pk["matched"]["excludes_half"]:
        verdict = "NOT SUPPORTED"
    else:
        verdict = "INCONCLUSIVE"
    mde_m = res["mde"]["matched"]["icc0.3"]["mde_auc_at_80pc_power"]
    res["verdict"] = {
        "primary_K": PRIMARY_K, "endpoint_magnitude_account": verdict,
        "magnitude_alone_excludes_half": a["excludes_half"],
        "magnitude_alone_direction": ("peaked clips have the smaller endpoint magnitude"
                                      if a["auc"] < 0.5 else
                                      "peaked clips have the larger endpoint magnitude"),
        "clip_full_excludes_half": pk["full"]["excludes_half"],
        "clip_matched_excludes_half": pk["matched"]["excludes_half"],
        "clip_matched_auc": pk["matched"]["auc"], "clip_full_auc": pk["full"]["auc"],
        "matched_mde_icc0.3": mde_m,
        "matched_point_estimate_inside_full_ci": bool(pk["full"]["ci"][0] <= pk["matched"]["auc"]
                                                      <= pk["full"]["ci"][1]),
        "note": (f"the matched arm is a non-detection on {res['matching']['n_pairs']} pairs with an "
                 f"MDE of {mde_m:.3f}, so it is never demonstrated absence, and the verdict rests on "
                 "an interval that width alone could have produced")}
    res["status"] = "COMPLETE"
    json.dump(res, open(f"{OUT}/endpoint_magnitude.json", "w"), indent=2)
    p(f"\nVERDICT at K={PRIMARY_K}: the endpoint-magnitude account is {verdict}")
    p(f"wrote {OUT}/endpoint_magnitude.json")
    logf.close()


if __name__ == "__main__":
    main()
