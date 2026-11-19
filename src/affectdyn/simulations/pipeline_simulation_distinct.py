"""Pipeline simulation of the headline CLIP-VideoMAEv2 decision with each subject entered once.

Same generator and decision as affectdyn.simulations.pipeline_simulation_resampled (chance-level labels shifted within recorded sequences, both
fits, K=2 and 4, Holm across both), except the subject draw. --arm all enters all 23 direction-eligible subjects
once in their own leave-one-subject-out folds; --arm sub18 draws 18 of the 23 without replacement (descriptive).

Usage: python -m affectdyn.simulations.pipeline_simulation_distinct --arm all|sub18 --run R [--procs 18]
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse, json, math, sys, time
from multiprocessing import get_context
import numpy as np

import affectdyn.simulations.pipeline_simulation_resampled as w69                                           # noqa: E402
from affectdyn.features.unbc_pretraining_comparison import _auc                                        # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout                                   # noqa: E402
from affectdyn.inference.calibrated_clustered_auc import z_paired, within_class_corr          # noqa: E402
from affectdyn.provenance import run_provenance                                         # noqa: E402

OUT = "runs/pipeline_simulation_distinct"
SEED = 20261170
S_NOISE, C_NOISE = 2.0, 0.0
M_SUB = 18


def study(job):
    arm, idx = job
    W = w69._W
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 0 if arm == "all" else 1, idx]))
    dirn = W["direction"].copy(); nshort = 0
    for s_, ii in W["order"].items():
        L = len(ii)
        if L < 4:
            nshort += 1; continue
        dirn[ii] = np.roll(W["direction"][ii], int(rng.integers(1, L)))
    n_el = len(W["elig"])
    draw = rng.permutation(n_el) if arm == "all" else rng.choice(n_el, M_SUB, replace=False)
    rows, clus, osub = [], [], []
    for copy, g in enumerate(draw):
        m = np.flatnonzero(W["sub"] == W["elig"][g])
        rows.append(m); clus += [copy] * len(m); osub += [W["elig"][g]] * len(m)
    idx_rows = np.concatenate(rows); clus = np.array(clus); osub = np.array(osub)
    assert len(np.unique(osub)) == len(draw)                         # Require distinct sampled subjects.
    yy = (dirn[idx_rows] == 1).astype(int); mm = dirn[idx_rows] != -1
    if mm.sum() == 0 or yy[mm].min() == yy[mm].max():
        return {"arm": arm, "index": idx, "skipped": "single class"}
    fid = np.array([W["fold"][x] for x in osub])
    N0 = rng.standard_normal((W["nU"], W["F"].shape[2]), dtype=np.float32)
    arms = []
    for j in (1, 2):
        Nj = rng.standard_normal((W["nU"], W["F"].shape[2]), dtype=np.float32)
        noise = (S_NOISE * (C_NOISE * N0 + math.sqrt(1 - C_NOISE ** 2) * Nj)) * W["sd"]
        arms.append(W["F"][idx_rows] + noise[W["fidx"][idx_rows]])
    out = {"arm": arm, "index": idx, "n_short_sequences": nshort, "n_clusters": int(len(np.unique(clus[mm]))),
           "fits": {}}
    audit = {"fits": 0, "uncertified": 0, "max_grad": 0.0}
    for fit in ("fixed", "converged"):
        rec = {}
        for k in w69.KS:
            Xa, Xb = image_readout(arms[0], k), image_readout(arms[1], k)
            if fit == "fixed":
                sa, sb = w69.fixed_step(Xa, yy, fid, mm), w69.fixed_step(Xb, yy, fid, mm)
            else:
                sa, sb = w69.converged(Xa, yy, fid, mm, audit), w69.converged(Xb, yy, fid, mm, audit)
            ok = mm & ~np.isnan(sa) & ~np.isnan(sb)
            d, z = z_paired(sa[ok], sb[ok], yy[ok], clus[ok])
            rec[f"K{k}"] = {"auc_a": _auc(sa[ok], yy[ok]), "auc_b": _auc(sb[ok], yy[ok]),
                            "r": within_class_corr(sa[ok], sb[ok], yy[ok]), "n": int(ok.sum()), "z": float(z),
                            "p": w69.p_ref(z), "degenerate": bool(not np.isfinite(z) or z == 0.0)}
        rec["event"] = bool(2 * min(rec[f"K{k}"]["p"] for k in w69.KS) < w69.ALPHA)
        out["fits"][fit] = rec
    out["audit"] = audit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("all", "sub18"), required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--procs", type=int, default=18)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ch = json.load(open("runs/pipeline_simulation_resampled/calibration_tune.json"))["choice"]
    assert (ch["s"], ch["c"]) == (S_NOISE, C_NOISE), ch
    logf = open(f"{OUT}/{a.arm}_run.log", "a"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    path = f"{OUT}/{a.arm}_studies.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["index"] for l in open(path)}
    fh = open(path, "a")
    jobs = [(a.arm, i) for i in range(a.run) if i not in done]
    log(f"W70 arm={a.arm} R={a.run}; {len(jobs)} studies to go; generator s={S_NOISE} c={C_NOISE}")
    ctx = get_context("spawn")
    with ctx.Pool(a.procs, initializer=w69._init) as pool:
        for r in pool.imap_unordered(study, jobs, chunksize=1):
            fh.write(json.dumps(r, default=float) + "\n"); fh.flush()
    fh.close()
    recs = [json.loads(l) for l in open(path)]
    recs = [r for r in recs if r["index"] < a.run]
    rr = [r for r in recs if "fits" in r]
    res = {"provenance": run_provenance(seeds=[SEED]), "arm": a.arm, "R": a.run,
           "skipped": sum("skipped" in r for r in recs), "events": {}}
    for fit in ("fixed", "converged"):
        n = len(rr); k = sum(r["fits"][fit]["event"] for r in rr)
        lo, hi = w69.wilson1(k, n, "lower"), w69.wilson1(k, n, "upper")
        outcome = "MET" if hi <= w69.LIMIT else "EXCEEDED" if lo > w69.LIMIT else "INCONCLUSIVE"
        degen = sum(r["fits"][fit][f"K{kk}"]["degenerate"] for r in rr for kk in w69.KS)
        res["events"][fit] = {"n": n, "k": k, "rate": k / n, "wilson1_lower": lo, "wilson1_upper": hi,
                              "outcome": outcome, "degenerate": degen,
                              "median_arm_auc": {f"K{kk}": float(np.median([v for r in rr for v in
                                                 (r["fits"][fit][f"K{kk}"]["auc_a"], r["fits"][fit][f"K{kk}"]["auc_b"])]))
                                                 for kk in w69.KS}}
        log(f"{a.arm} {fit}: {k}/{n} = {k / n:.4f} one-sided Wilson [{lo:.4f}, {hi:.4f}] -> {outcome}")
    aud = [r["audit"] for r in rr]
    nf = sum(x["fits"] for x in aud); nu = sum(x["uncertified"] for x in aud)
    ndeg = sum(v["degenerate"] for v in res["events"].values())
    res["failures"] = {"converged_fits": nf, "uncertified": nu, "degenerate_z": ndeg,
                       "gate_fail": bool((nf and nu / nf > 0.01) or (rr and ndeg / (4 * len(rr)) > 0.01))}
    outs = [v["outcome"] for v in res["events"].values()]
    if res["failures"]["gate_fail"]:
        res["outcome"] = "INCONCLUSIVE"
    else:
        res["outcome"] = ("MET" if all(o == "MET" for o in outs) else "EXCEEDED" if any(o == "EXCEEDED" for o in outs)
                          else "INCONCLUSIVE")
    if a.arm == "sub18":
        res["outcome_note"] = "descriptive only (reference null simulated for the 23-subject layout)"
    json.dump(res, open(f"{OUT}/{a.arm}.json", "w"), indent=2, default=float)
    log(f"OUTCOME {res['outcome']}")


if __name__ == "__main__":
    main()
