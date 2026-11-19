"""Bootstrap draw variation of the matched against mixed-granularity comparison."""
from __future__ import annotations
import json, os, sys, time, zlib
from multiprocessing import get_context
import numpy as np

from affectdyn.analyses.direction.mixed_granularity_contrasts import paired_bootstrap, holm, rng_for, VIDEO, NBOOT
from affectdyn.provenance import run_provenance

OUT = "runs/granularity_bootstrap_variation"
SEED = 20260927
REPS = 200
BIG = 100_000
ALPHA = 0.05
WORKERS = 16
_D = {}


def wilson(k, n, z=1.959963984540054):
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def classify(lo, hi):
    return "draw-dependent" if lo >= 0.05 else "stable" if hi < 0.05 else "unresolved"


def pairs():
    m = {v: ("clip_K2", f"{v}_K2", f"K= 2: CLIP vs {v}") for v in VIDEO}
    u = {v: ("clip_K16", f"{v}_K2", f"unmatched: clip_K16 vs {v}_K2") for v in VIDEO}
    return m, u


def _init(S, y, g):
    _D.update(S=S, y=y, g=g)


def _coupled(rep):
    S, y, g = _D["S"], _D["y"], _D["g"]
    m, u = pairs(); pm, pu = [], []
    for v in VIDEO:
        seed = [SEED, zlib.crc32(v.encode()), rep]
        pm.append(paired_bootstrap(S[m[v][0]], S[m[v][1]], y, g, "subject", np.random.default_rng(seed), NBOOT)["p_boot"])
        pu.append(paired_bootstrap(S[u[v][0]], S[u[v][1]], y, g, "subject", np.random.default_rng(seed), NBOOT)["p_boot"])
    return list(holm(pm)), list(holm(pu))


def _independent(rep):
    S, y, g = _D["S"], _D["y"], _D["g"]
    m, u = pairs(); out = {}
    for fam, d in (("M6_K2", m), ("U6", u)):
        ps = [paired_bootstrap(S[d[v][0]], S[d[v][1]], y, g, "subject",
                               np.random.default_rng([SEED, zlib.crc32(d[v][2].encode()), rep]), NBOOT)["p_boot"]
              for v in VIDEO]
        out[fam] = list(holm(ps))
    return out


def _big(args):
    fam, v = args
    S, y, g = _D["S"], _D["y"], _D["g"]
    d = pairs()[0] if fam == "M6_K2" else pairs()[1]
    b = paired_bootstrap(S[d[v][0]], S[d[v][1]], y, g, "subject",
                         np.random.default_rng([SEED, zlib.crc32(d[v][2].encode()), 10 ** 6]), BIG)
    return fam, v, b["p_boot"]


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/matching_draws.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    Z = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    keep = Z["keep"].astype(bool)
    y, g = Z["y"][keep].astype(np.int64), Z["subject"][keep]
    S = {k: Z[k][keep] for k in Z.files if k not in ("subject", "y", "keep")}
    W26 = json.load(open("runs/mixed_granularity_contrasts/unmatched_contrasts.json"))
    m, u = pairs()

    ok, rows = True, []
    for fam, d, pub in (("U6", u, W26["U6"]), ("M6_K2", m, W26["M6_K2"])):
        pubv = {r["video"]: r for r in pub}; ps = []
        for v in VIDEO:
            b = paired_bootstrap(S[d[v][0]], S[d[v][1]], y, g, "subject", rng_for(d[v][2] + ":subject"), NBOOT)
            same = b["p_boot"] == pubv[v]["p"] and b["ci"] == pubv[v]["ci"]
            ok &= same; ps.append(b["p_boot"]); rows.append({"family": fam, "video": v, "exact": bool(same)})
        adj = holm(ps)
        hok = all(abs(a - pubv[v]["holm"]) <= 1e-12 for a, v in zip(adj, VIDEO))
        ok &= hok; rows.append({"family": fam, "holm_exact": bool(hok)})
    gate = {"rows": rows, "passes": bool(ok)}
    log(f"gate: passes {ok}")
    base = {"provenance": run_provenance(seeds=[SEED], device="cpu"),
            "criteria_heading": "## 2026-09-18 — W39-W44 PRE-DECLARED CRITERIA",
            "amendment_heading": "## 2026-09-18 — W39-W44 CRITERIA AMENDMENT", "gate": gate}
    if not ok:
        base["status"] = "HOLD: gate failed"
        json.dump(base, open(f"{OUT}/matching_draws.json", "w"), indent=2); log("HOLD"); return

    pub = {"M6_K2": {r["video"]: r["holm"] for r in W26["M6_K2"]}, "U6": {r["video"]: r["holm"] for r in W26["U6"]}}
    ctx = get_context("fork")
    with ctx.Pool(WORKERS, initializer=_init, initargs=(S, y, g)) as pool:
        coupled = pool.map(_coupled, range(REPS), chunksize=2)
        log(f"coupled replicates done ({time.time() - t0:.0f}s)")
        indep = pool.map(_independent, range(REPS), chunksize=2)
        log(f"independent replicates done ({time.time() - t0:.0f}s)")
        big = pool.map(_big, [(f, v) for f in ("M6_K2", "U6") for v in VIDEO])
    big = {(f, v): p for f, v, p in big}

    ti = VIDEO.index("timesformer")
    combo = {"matched_sig_unmatched_not": 0, "unmatched_sig_matched_not": 0, "both_sig": 0, "neither": 0}
    ndiff = []
    for hm, hu in coupled:
        sm, su = hm[ti] < ALPHA, hu[ti] < ALPHA
        combo["matched_sig_unmatched_not" if sm and not su else "unmatched_sig_matched_not" if su and not sm
              else "both_sig" if sm else "neither"] += 1
        ndiff.append(sum((a < ALPHA) != (b < ALPHA) for a, b in zip(hm, hu)))
    primary = {k: {"count": c, "share": c / REPS, "wilson95": wilson(c, REPS)} for k, c in combo.items()}
    primary["n_differing_decisions_distribution"] = {str(k): int(sum(1 for x in ndiff if x == k)) for k in range(7)}
    log(f"PRIMARY (coupled) TimeSformer: {({k: v['count'] for k, v in primary.items() if k in combo})}")

    marginal = {}
    for fam in ("M6_K2", "U6"):
        for j, v in enumerate(VIDEO):
            vals = np.array([r[fam][j] for r in indep]); logged = pub[fam][v] < ALPHA
            flips = int(sum((x < ALPHA) != logged for x in vals)); w = wilson(flips, REPS)
            marginal[f"{fam}:{v}"] = {"published_holm": pub[fam][v], "logged_significant": bool(logged),
                                      "median": float(np.median(vals)), "p2.5_p97.5": [float(np.percentile(vals, 2.5)),
                                                                                       float(np.percentile(vals, 97.5))],
                                      "min_max": [float(vals.min()), float(vals.max())], "flips": flips,
                                      "flip_share_wilson95": w, "classification": classify(*w),
                                      "raw_p_at_B100000": big[(fam, v)]}
    for fam in ("M6_K2", "U6"):
        adjb = holm([big[(fam, v)] for v in VIDEO])
        for v, a in zip(VIDEO, adjb):
            marginal[f"{fam}:{v}"]["holm_at_B100000"] = float(a)
    for k in ("M6_K2:timesformer", "U6:timesformer"):
        r = marginal[k]
        log(f"  {k:22s} published {r['published_holm']:.4f} | median {r['median']:.4f} {r['p2.5_p97.5']} | flips "
            f"{r['flips']}/{REPS} {r['flip_share_wilson95']} -> {r['classification']} | B=100k {r['holm_at_B100000']:.4f}")
    base.update({"reps": REPS, "nboot": NBOOT, "primary_coupled": primary, "marginal": marginal,
                 "reading": ("the published runs differed on one borderline decision; the coupled share is the "
                             "directional disagreement frequency under identical resamples, not the probability that "
                             "matching changes the scientific conclusion"),
                 "status": "COMPLETE"})
    json.dump(base, open(f"{OUT}/matching_draws.json", "w"), indent=2)
    log(f"wrote {OUT}/matching_draws.json | COMPLETE")


if __name__ == "__main__":
    main()
