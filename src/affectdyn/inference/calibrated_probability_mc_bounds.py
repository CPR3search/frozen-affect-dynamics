"""Monte Carlo error of simulated p values near 0.05."""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np

from affectdyn.provenance import run_provenance

OUT = "runs/calibrated_probability_mc_bounds"
N = 100_000
ALPHA = 0.05
LO, HI = 0.01, 0.10


def holm(ps):
    """Local Holm adjustment; the shared implementation is
    inference.aggregation_permutation_tests.holm. Family membership and size are specified by the
    caller.
    """
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


_LG = None


def _lchoose(n):
    global _LG
    if _LG is None or len(_LG) != n + 1:
        j = np.arange(n + 1)
        lg = np.array([math.lgamma(x + 1) for x in range(n + 1)])
        _LG = lg[n] - lg[j] - lg[n - j]
    return _LG


def binom_cdf(k, n, q):
    """P(X <= k) for X ~ Binomial(n, q), summed in log space."""
    if q <= 0:
        return 1.0
    if q >= 1:
        return 1.0 if k >= n else 0.0
    j = np.arange(k + 1)
    lp = _lchoose(n)[: k + 1] + j * math.log(q) + (n - j) * math.log1p(-q)
    m = lp.max()
    return float(min(1.0, math.exp(m) * np.exp(lp - m).sum()))


def clopper_pearson(k, n, level):
    """Exact two-sided interval for k/n at the given confidence level."""
    a = (1 - level) / 2

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2
    lower = 0.0 if k == 0 else solve(lambda q: 1 - binom_cdf(k - 1, n, q) < a)      # Invert the upper binomial tail.
    upper = 1.0 if k == n else solve(lambda q: binom_cdf(k, n, q) > a)              # Invert the lower binomial tail.
    return lower, upper


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/calibrated_mc_error.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    w12 = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    fams = {"direction_contrasts_24": w12["families"]["direction_contrasts_24"]["rows"],
            "aw2_shape_18": w12["families"]["aw2_shape_18"]["rows"],
            "F5_pain_order_controls_5": w12["families"]["w11_exploratory_5"]["rows"]}
    w30 = json.load(open("runs/pain_order_holm_family/pain_control_family.json"))

    gate_rows, ok = [], True
    for fam, rows in fams.items():
        for r in rows:
            if r["p_method"] != "simulation-calibrated":
                continue
            x = (N + 1) * r["p_calibrated"]; k = int(round(x)) - 1
            good = abs(x - round(x)) <= 1e-6 and 0 <= k <= N and abs((k + 1) / (N + 1) - r["p_calibrated"]) <= 1e-15
            ok &= good; gate_rows.append((fam, r["test"], k, good))
    repro = {}
    for fam, rows in fams.items():
        adj = holm([r["p_calibrated"] for r in rows])
        pub = ([r["holm_calibrated"] for r in rows] if "holm_calibrated" in rows[0]
               else None)
        if pub is None:
            w30c = {x["test"]: x for x in w30["rows"]}
            pub = [w30c[r["test"]]["holm_F5_calibrated"] if r["test"] in w30c else None for r in rows]
        d = max(abs(a - b) for a, b in zip(adj, pub) if b is not None) if any(b is not None for b in pub) else None
        repro[fam] = {"max_abs_delta_published_holm": None if d is None else float(d),
                      "passes": bool(d is not None and d <= 1e-12)}
        ok &= repro[fam]["passes"]
    gate = {"count_recovery": all(g[3] for g in gate_rows), "n_recovered": len(gate_rows),
            "holm_reproduction": repro, "passes": bool(ok)}
    log(f"gate: {gate}")
    res = {"provenance": run_provenance(device="cpu"),
           "criteria_heading": "## 2026-09-18 — W39-W44 PRE-DECLARED CRITERIA",
           "amendment_heading": "## 2026-09-18 — W39-W44 CRITERIA AMENDMENT", "N": N, "gate": gate}
    if not ok:
        res["status"] = "HOLD: gate failed"
        json.dump(res, open(f"{OUT}/calibrated_mc_error.json", "w"), indent=2); log("HOLD"); return

    out = {}
    for fam, rows in fams.items():
        p = np.array([r["p_calibrated"] for r in rows])
        sim = [r["p_method"] == "simulation-calibrated" for r in rows]
        m_sim = sum(sim); level = 1 - ALPHA / m_sim
        lo_v, hi_v = p.copy(), p.copy()
        for i, r in enumerate(rows):
            if sim[i]:
                k = int(round((N + 1) * r["p_calibrated"])) - 1
                lo_v[i], hi_v[i] = clopper_pearson(k, N, level)
        adj, adj_lo, adj_hi = holm(p), holm(lo_v), holm(hi_v)
        sel = [i for i in range(len(rows)) if LO <= adj[i] <= HI]
        rep = []
        for i in sel:
            cls = ("MC-stable significant" if adj_hi[i] < ALPHA else
                   "MC-stable non-significant" if adj_lo[i] >= ALPHA else "MC-unresolved")
            rep.append({"test": rows[i]["test"], "raw_p": float(p[i]), "raw_simultaneous95": [float(lo_v[i]), float(hi_v[i])],
                        "holm": float(adj[i]), "holm_from_bounds": [float(adj_lo[i]), float(adj_hi[i])],
                        "classification": cls})
            log(f"  {fam:26s} {rows[i]['test']:28s} raw {p[i]:.5f} [{lo_v[i]:.5f}, {hi_v[i]:.5f}] "
                f"Holm {adj[i]:.4f} [{adj_lo[i]:.4f}, {adj_hi[i]:.4f}] -> {cls}")
        out[fam] = {"m": len(rows), "m_simulated": m_sim, "simultaneous_level_per_test": level, "selected": rep}
    res.update({"families": out,
                "w19_grid": ("not computable: W19 stored neither observed-test exceedance counts nor p values at its "
                             "20,000-study settings, and its quantile-based Wilson fields are not substituted"),
                "status": "COMPLETE"})
    json.dump(res, open(f"{OUT}/calibrated_mc_error.json", "w"), indent=2)
    log(f"wrote {OUT}/calibrated_mc_error.json | COMPLETE")


if __name__ == "__main__":
    main()
