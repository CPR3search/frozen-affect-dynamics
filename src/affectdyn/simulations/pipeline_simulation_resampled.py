"""Pipeline simulation of the headline decision with the probes refitted.

Usage: python -m affectdyn.simulations.pipeline_simulation_resampled --tune|--verify|--run R [--procs 18]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json, math, sys, time
from multiprocessing import get_context
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                        # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                  # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout                                   # noqa: E402
from affectdyn.inference.calibrated_clustered_auc import z_paired, within_class_corr          # noqa: E402
from affectdyn.provenance import run_provenance                                         # noqa: E402

OUT = "runs/pipeline_simulation_resampled"
SEED = 20261169
KS = (2, 4)
GRID_S = (0.25, 0.5, 1.0, 2.0, 4.0)
GRID_C = (0.0, 0.5, 0.75)
T_AUC, T_AUC_TOL, T_R, T_R_TOL = 0.82, 0.03, 0.54, 0.10
ALPHA = 0.05
Z1 = 1.6448536269514722            # Use the one-sided 95% normal quantile.
LIMIT = 0.075

_W = {}


def _load():
    import affectdyn.data.unbc_clip_pipeline as stage1_probe
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]; direction = lab["direction"].astype(int)
    clips = stage1_probe.build_clips()
    paths = [c["paths"] for c in clips]
    uniq = {p: i for i, p in enumerate(sorted({p for ps in paths for p in ps}))}
    fidx = np.array([[uniq[p] for p in ps] for ps in paths])               # Unique-frame indices have shape [599, 16].
    seq = np.array([ps[0].split("/")[-2] for ps in paths])
    order = {}
    for s_ in np.unique(seq):
        ii = np.flatnonzero(seq == s_)
        order[s_] = ii[np.argsort([paths[i][0] for i in ii])]
    F = np.load("runs/temporal_aggregation/img_frames.npz")["clip"].astype(np.float32)    # Frame features have shape [599, 16, 768].
    U = np.zeros((len(uniq), F.shape[2]), np.float32)
    U[fidx.ravel()] = F.reshape(-1, F.shape[2])
    sd = U.std(0).astype(np.float32)
    elig = np.unique(sub[direction != -1])
    lf = _loso_folds(sub)
    null = np.abs(np.load(f"{OUT}/w12_direction_null.npy"))
    return {"sub": sub, "direction": direction, "fidx": fidx, "F": F, "sd": sd, "nU": len(uniq), "seq": seq,
            "order": order, "elig": elig, "fold": lf, "null": np.sort(null)}


def _init():
    _W.update(_load())


def p_ref(z):
    nul = _W["null"]
    k = len(nul) - np.searchsorted(nul, abs(z), side="left")           # Count absolute null statistics at least as large as |z|.
    return float((1 + k) / (1 + len(nul)))


def fixed_step(X, y, fid, mask):
    import torch, torch.nn as nn
    torch.set_num_threads(1)
    nfold = int(fid.max()) + 1; specs, W0, b0 = [], [], []
    for rep in range(5):
        torch.manual_seed(1000 + rep)
        for k in range(nfold):
            te = (fid == k) & mask; tr = (fid != k) & mask
            if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
                continue
            clf = nn.Linear(X.shape[1], 1)
            specs.append((rep, tr, te)); W0.append(clf.weight.detach().clone()[0]); b0.append(clf.bias.detach().clone())
    n, D = X.shape; Fn = len(specs)
    Xs = np.empty((Fn, n, D), np.float32); M = np.zeros((Fn, n), np.float32)
    for f, (_, tr, _) in enumerate(specs):
        mu, sdv = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs[f] = (X - mu) / sdv; M[f, tr] = 1.0
    Xt = torch.tensor(Xs); Mt = torch.tensor(M); yt = torch.tensor(y.astype(np.float32))
    W = torch.nn.Parameter(torch.stack(W0)); b = torch.nn.Parameter(torch.cat(b0))
    opt = torch.optim.Adam([W, b], 1e-2, weight_decay=1e-3); cnt = Mt.sum(1)
    for _ in range(300):
        z = torch.einsum("fnd,fd->fn", Xt, W) + b[:, None]
        l = nn.functional.binary_cross_entropy_with_logits(z, yt.expand_as(z), reduction="none")
        loss = ((l * Mt).sum(1) / cnt).sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (torch.einsum("fnd,fd->fn", Xt, W) + b[:, None]).numpy()
    acc = np.zeros((5, n)); seen = np.zeros((5, n), bool)
    for f, (rep, _, te) in enumerate(specs):
        acc[rep, te] = z[f, te]; seen[rep, te] = True
    return np.where(seen.all(0), acc.mean(0), np.nan)


def converged(X, y, fid, mask, audit):
    import affectdyn.probes.nested_probe_fits as w51
    s = np.full(len(y), np.nan)
    for k in range(int(fid.max()) + 1):
        te = (fid == k) & mask; tr = (fid != k) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        xt = w51.standardize(X, tr, tr).astype(np.float64); xe = w51.standardize(X, tr, te).astype(np.float64)
        X1 = np.hstack([xt, np.ones((len(xt), 1))]); X1e = np.hstack([xe, np.ones((len(xe), 1))])
        a, g = w51.newton_kernel(X1 @ X1.T, X1, y[tr], 1e-3)
        s[te] = X1e @ (X1.T @ a)
        audit["fits"] += 1; audit["uncertified"] += int(g > w51.TOL); audit["max_grad"] = max(audit["max_grad"], g)
    return s


def study(job):
    """job = (scenario, index, s, c, decide)."""
    scen, idx, s, c, decide = job
    W = _W
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0 if scen == "A" else 1, idx, int(s * 100), int(c * 100)]))
    dirn = W["direction"].copy()
    nshort = 0
    if scen == "A":
        for s_, ii in W["order"].items():
            L = len(ii)
            if L < 4:
                nshort += 1; continue
            off = int(rng.integers(1, L))
            dirn[ii] = np.roll(W["direction"][ii], off)
    draw = rng.integers(0, len(W["elig"]), len(W["elig"]))
    rows, clus, osub = [], [], []
    for copy, g in enumerate(draw):
        m = np.flatnonzero(W["sub"] == W["elig"][g])
        rows.append(m); clus += [copy] * len(m); osub += [W["elig"][g]] * len(m)
    idx_rows = np.concatenate(rows); clus = np.array(clus); osub = np.array(osub)
    yy = (dirn[idx_rows] == 1).astype(int); mm = dirn[idx_rows] != -1
    if mm.sum() == 0 or yy[mm].min() == yy[mm].max():
        return {"scenario": scen, "index": idx, "skipped": "single class"}
    fid = np.array([W["fold"][x] for x in osub])
    N0 = rng.standard_normal((W["nU"], W["F"].shape[2]), dtype=np.float32)
    arms = []
    for j in (1, 2):
        Nj = rng.standard_normal((W["nU"], W["F"].shape[2]), dtype=np.float32)
        noise = (s * (c * N0 + math.sqrt(1 - c * c) * Nj)) * W["sd"]
        arms.append(W["F"][idx_rows] + noise[W["fidx"][idx_rows]])
    out = {"scenario": scen, "index": idx, "s": s, "c": c, "n_short_sequences": nshort,
           "n_clusters": int(len(np.unique(clus[mm]))), "fits": {}}
    audit = {"fits": 0, "uncertified": 0, "max_grad": 0.0}
    for fit in ("fixed", "converged"):
        rec = {}
        for k in KS:
            Xa, Xb = image_readout(arms[0], k), image_readout(arms[1], k)
            if fit == "fixed":
                sa, sb = fixed_step(Xa, yy, fid, mm), fixed_step(Xb, yy, fid, mm)
            else:
                sa, sb = converged(Xa, yy, fid, mm, audit), converged(Xb, yy, fid, mm, audit)
            ok = mm & ~np.isnan(sa) & ~np.isnan(sb)
            r = {"auc_a": _auc(sa[ok], yy[ok]), "auc_b": _auc(sb[ok], yy[ok]),
                 "r": within_class_corr(sa[ok], sb[ok], yy[ok]), "n": int(ok.sum())}
            if decide:
                d, z = z_paired(sa[ok], sb[ok], yy[ok], clus[ok])
                r["z"] = float(z); r["p"] = p_ref(z); r["degenerate"] = bool(not np.isfinite(z) or z == 0.0)
            rec[f"K{k}"] = r
        if decide:
            ps = sorted(rec[f"K{k}"]["p"] for k in KS)
            rec["event"] = bool(2 * ps[0] < ALPHA)        # Holm with two tests rejects at least one when 2*min(p) < alpha.
        out["fits"][fit] = rec
    out["audit"] = audit
    return out


def wilson1(k, n, side):
    p = k / n; den = 1 + Z1 * Z1 / n; ctr = (p + Z1 * Z1 / (2 * n)) / den
    h = Z1 * math.sqrt(p * (1 - p) / n + Z1 * Z1 / (4 * n * n)) / den
    return ctr + h if side == "upper" else ctr - h


def run_jobs(jobs, procs, sink):
    ctx = get_context("spawn")
    with ctx.Pool(procs, initializer=_init) as pool:
        for r in pool.imap_unordered(study, jobs, chunksize=1):
            sink(r)


def summarize_regime(recs):
    agg = {}
    for fit in ("fixed", "converged"):
        for k in KS:
            a = [x["fits"][fit][f"K{k}"] for x in recs if "fits" in x]
            aucs = [v for r in a for v in (r["auc_a"], r["auc_b"])]
            agg[f"{fit}_K{k}"] = {"median_auc": float(np.median(aucs)), "median_r": float(np.median([r["r"] for r in a]))}
    return agg


def distance(agg):
    return max(max(abs(v["median_auc"] - T_AUC) / T_AUC_TOL, abs(v["median_r"] - T_R) / T_R_TOL) for v in agg.values())


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tune", action="store_true"); g.add_argument("--verify", action="store_true")
    g.add_argument("--run", type=int)
    ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mode = "tune" if a.tune else "verify" if a.verify else "run"
    logf = open(f"{OUT}/calibration_{mode}.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    prov = run_provenance(seeds=[SEED])
    if a.tune:
        recs = {}
        jobs = [("B", 100000 + i, s, c, False) for s in GRID_S for c in GRID_C for i in range(20)]
        run_jobs(jobs, a.procs, lambda r: recs.setdefault((r["s"], r["c"]), []).append(r))
        table = []
        for (s, c), rr in sorted(recs.items()):
            agg = summarize_regime(rr); dist = distance(agg)
            table.append({"s": s, "c": c, "agg": agg, "distance": dist, "meets": dist <= 1.0})
            log(f"s {s:<5} c {c:<5} dist {dist:6.3f} " + " ".join(f"{k}:{v['median_auc']:.3f}/{v['median_r']:.3f}"
                                                              for k, v in agg.items()))
        meets = [t for t in table if t["meets"]]
        best = min(meets or table, key=lambda t: t["distance"])
        choice = {"s": best["s"], "c": best["c"], "meets_targets": bool(meets), "distance": best["distance"]}
        json.dump({"provenance": prov, "table": table, "choice": choice}, open(f"{OUT}/calibration_tune.json", "w"),
                  indent=2, default=float)
        log(f"CHOICE {choice}")
        return
    ch = json.load(open(f"{OUT}/calibration_tune.json"))["choice"]
    s, c = ch["s"], ch["c"]
    if a.verify:
        recs = []
        run_jobs([("B", 200000 + i, s, c, False) for i in range(20)], a.procs, recs.append)
        agg = summarize_regime(recs); dist = distance(agg)
        res = {"provenance": prov, "choice": ch, "agg": agg, "distance": dist, "verified": dist <= 1.0}
        json.dump(res, open(f"{OUT}/calibration_verify.json", "w"), indent=2, default=float)
        log(f"VERIFY {res['verified']} distance {dist:.3f} {agg}")
        return
    R = a.run
    path = f"{OUT}/calibration_studies.jsonl"
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done.add((r["scenario"], r["index"]))
    fh = open(path, "a")

    def sink(r):
        fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
    jobs = [(sc, i, s, c, True) for i in range(R) for sc in ("A", "B") if (sc, i) not in done]
    log(f"RUN R={R} per scenario at s={s} c={c}; {len(jobs)} studies to go")
    run_jobs(jobs, a.procs, sink)
    fh.close()
    recs = [json.loads(l) for l in open(path)]
    res = {"provenance": prov, "choice": ch, "R": R, "events": {}}
    for sc in ("A", "B"):
        rr = [r for r in recs if r["scenario"] == sc and "fits" in r and r["index"] < R]
        res[f"skipped_{sc}"] = sum(1 for r in recs if r["scenario"] == sc and "skipped" in r)
        res[f"arm_auc_{sc}"] = summarize_regime(rr)
        for fit in ("fixed", "converged"):
            n = len(rr); k = sum(r["fits"][fit]["event"] for r in rr)
            lo, hi = wilson1(k, n, "lower"), wilson1(k, n, "upper")
            outcome = "MET" if hi <= LIMIT else "EXCEEDED" if lo > LIMIT else "INCONCLUSIVE"
            per_k = {f"K{kk}": sum(r["fits"][fit][f"K{kk}"]["p"] < ALPHA for r in rr) / n for kk in KS}
            degen = sum(r["fits"][fit][f"K{kk}"]["degenerate"] for r in rr for kk in KS)
            res["events"][f"{sc}_{fit}"] = {"n": n, "k": k, "rate": k / n, "wilson1_lower": lo, "wilson1_upper": hi,
                                            "outcome": outcome, "per_K_rate_at_0.05": per_k, "degenerate": degen}
            log(f"{sc} {fit}: {k}/{n} = {k / n:.4f} one-sided Wilson [{lo:.4f}, {hi:.4f}] -> {outcome}")
    aud = [r["audit"] for r in recs if "audit" in r and r["index"] < R]
    nf = sum(x["fits"] for x in aud); nu = sum(x["uncertified"] for x in aud)
    ndeg = sum(v["degenerate"] for v in res["events"].values()); nst = sum(v["n"] for v in res["events"].values())
    res["failures"] = {"converged_fits": nf, "uncertified": nu, "degenerate_z": ndeg,
                       "gate_fail": bool((nf and nu / nf > 0.01) or (nst and ndeg / (2 * nst) > 0.01))}
    outs = [v["outcome"] for v in res["events"].values()]
    if res["failures"]["gate_fail"]:
        outs.append("INCONCLUSIVE")
    res["claim_level"] = ("met the scoped simulation criterion" if all(o == "MET" for o in outs)
                          else "not met" if any(o == "EXCEEDED" for o in outs) else "INCONCLUSIVE")
    json.dump(res, open(f"{OUT}/calibration.json", "w"), indent=2, default=float)
    log(f"CLAIM LEVEL: {res['claim_level']}")


if __name__ == "__main__":
    main()
