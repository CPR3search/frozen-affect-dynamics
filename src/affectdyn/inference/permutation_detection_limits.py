"""Simulated minimum detectable effects.

Usage:  python -m affectdyn.inference.permutation_detection_limits            # writes runs/peak_cluster_inference/mde.json
        python -m affectdyn.inference.permutation_detection_limits --print
"""
from __future__ import annotations
import argparse, datetime, json, os, subprocess, sys

import numpy as np

OUT = "runs/peak_cluster_inference/mde.json"
ALPHA = 0.05
POWER = 0.80
N_NULL = 20000
N_ALT = 4000
SEED = 20260910


def _git(*a, default="unavailable"):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True,
                              timeout=10).stdout.strip() or default
    except Exception:                                                  # noqa: BLE001
        return default


def provenance():
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", default="")),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "seed": SEED, "n_null": N_NULL, "n_alt": N_ALT, "alpha": ALPHA, "power": POWER,
        "inputs": ["runs/unbc_clip_pipeline/stage1_feats.npz", "runs/affwild_clips_features/labels.npz"],
        "note": "Labels only; no probe is refitted and no feature matrix is read.",
    }


def auc(scores, y):
    """Rank-based AUC, vectorised over the trailing axis of `scores` if 2-D."""
    s = np.atleast_2d(scores)
    order = np.argsort(s, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    n = s.shape[1]
    ar = np.arange(1, n + 1, dtype=float)
    for i in range(s.shape[0]):   # Break ties stably in continuous simulated scores.
        ranks[i, order[i]] = ar
    npos = y.sum()
    nneg = n - npos
    rsum = ranks[:, y == 1].sum(axis=1)
    return (rsum - npos * (npos + 1) / 2.0) / (npos * nneg)


def critical_auc(y, groups, rng, unit, m_family):
    """AUC threshold a single test must exceed to survive Holm in a family of m_family."""
    n = len(y)
    scores = rng.standard_normal((N_NULL, n))   # Generate scores independently of labels under the null.
    if unit == "clip":
        ys = np.tile(y, (N_NULL, 1))
        for i in range(N_NULL):
            ys[i] = rng.permutation(y)
    else:
        # Permute labels within subjects.
        uniq = np.unique(groups)
        idx = [np.where(groups == g)[0] for g in uniq]
        ys = np.empty((N_NULL, n), dtype=int)
        for i in range(N_NULL):
            row = np.empty(n, dtype=int)
            for ix in idx:
                row[ix] = rng.permutation(y[ix])
            ys[i] = row
    a = np.empty(N_NULL)
    for i in range(N_NULL):
        yi = ys[i]
        if yi.sum() == 0 or yi.sum() == n:
            a[i] = 0.5
            continue
        a[i] = auc(scores[i], yi)[0]
    return float(np.quantile(a, 1.0 - ALPHA / m_family))


def power_at(theta, y, groups, rng, icc):
    """Fraction of simulated studies whose AUC clears the critical value, at true AUC theta."""
    n = len(y)
    mu = np.sqrt(2.0) * _probit(theta)   # Set the separation so P(pos > neg) = theta.
    uniq, inv = np.unique(groups, return_inverse=True)
    out = np.empty(N_ALT)
    for i in range(N_ALT):
        b = rng.standard_normal(len(uniq)) * np.sqrt(icc) if icc > 0 else np.zeros(len(uniq))
        s = mu * y + b[inv] + rng.standard_normal(n) * np.sqrt(max(1e-9, 1.0 - icc))
        out[i] = auc(s, y)[0]
    return out


def _probit(p):
    """Inverse standard normal CDF (Acklam), adequate for the 0.5-0.95 range used here."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mde(y, groups, m_family, unit, icc, rng):
    crit = critical_auc(y, groups, rng, unit, m_family)
    lo, hi = 0.50, 0.99
    for _ in range(18):   # Bisect theta to reach the target power.
        mid = (lo + hi) / 2
        pw = float((power_at(mid, y, groups, rng, icc) >= crit).mean())
        if pw < POWER:
            lo = mid
        else:
            hi = mid
    return {"critical_auc": round(crit, 4), "mde_auc_at_80pc_power": round(hi, 4)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--print", dest="show", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    u = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    u_sub, u_traj = u["subject"], u["traj"]
    keep = u_traj != -1
    uy = (u_traj[keep] == 2).astype(int); ug = u_sub[keep]

    a = np.load("runs/affwild_clips_features/labels.npz")
    ay_all = (a["traj"] == 2).astype(int); ag_all = a["subject"]

    out = {"provenance": provenance(), "published_floor": {"unbc": 0.610, "aff_wild2": 0.535},
           "corpora": {}}

    for name, y, g, m in (("unbc", uy, ug, 36), ("aff_wild2", ay_all, ag_all, 18)):
        npos = int(y.sum())
        nsub_pos = int(len(np.unique(g[y == 1])))
        entry = {"n_clips": int(len(y)), "n_positive": npos,
                 "n_subjects": int(len(np.unique(g))), "n_subjects_with_a_positive": nsub_pos,
                 "holm_family": m, "results": {}}
        for unit, icc in (("clip", 0.0), ("subject", 0.3)):
            entry["results"][unit] = mde(y, g, m, unit, icc, rng)
        out["corpora"][name] = entry
        if args.show:
            print(f"\n{name}: {npos} positives across {nsub_pos} subjects, Holm family {m}")
            for unit, r in entry["results"].items():
                print(f"   {unit:8} null -> critical AUC {r['critical_auc']:.3f}, "
                      f"MDE at 80% power {r['mde_auc_at_80pc_power']:.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
