"""CLIP against VideoMAEv2 on pain direction under subject-grouped 5-fold partitions.

Usage: python -m affectdyn.analyses.sensitivity.subject_partition_contrasts
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")
import json, sys, time, itertools
import numpy as np

import affectdyn.probes.nested_probe_fits as w51                                          # noqa: E402
from affectdyn.features.unbc_pretraining_comparison import _auc                                        # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                  # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout                    # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES                                        # noqa: E402
from affectdyn.inference.calibrated_clustered_auc import z_paired                             # noqa: E402
from affectdyn.analyses.peak.endpoint_magnitude_controls import midrank_auc                          # noqa: E402
from affectdyn.provenance import run_provenance                                         # noqa: E402

OUT = "runs/subject_partition_contrasts"
PSEEDS = [20261101, 20261102, 20261103, 20261104, 20261105]
N_REP = 5
KS = (2, 4)
PAIRS = [("clip", "videomaev2")]
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA = "## 2026-11-08 — REVISION PROTOCOL: N1-N4 (W66-W69) PRE-DECLARED CRITERIA"


def fixed_step(X, y, fid, mask):
    import torch, torch.nn as nn
    nfold = int(fid.max()) + 1
    specs, W0, b0 = [], [], []
    for rep in range(N_REP):
        torch.manual_seed(1000 + rep)
        for k in range(nfold):
            te = (fid == k) & mask; tr = (fid != k) & mask
            if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
                continue
            clf = nn.Linear(X.shape[1], 1)
            specs.append((rep, k, tr, te)); W0.append(clf.weight.detach().clone()[0]); b0.append(clf.bias.detach().clone())
    F = len(specs); n, D = X.shape
    Xs = np.empty((F, n, D), np.float32); M = np.zeros((F, n), np.float32)
    for f, (_, _, tr, _) in enumerate(specs):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xs[f] = (X - mu) / sd; M[f, tr] = 1.0
    Xt = torch.tensor(Xs); Mt = torch.tensor(M); yt = torch.tensor(y.astype(np.float32))
    W = torch.nn.Parameter(torch.stack(W0)); b = torch.nn.Parameter(torch.cat(b0))
    opt = torch.optim.Adam([W, b], 1e-2, weight_decay=1e-3)
    cnt = Mt.sum(1)
    for _ in range(300):
        z = torch.einsum("fnd,fd->fn", Xt, W) + b[:, None]
        l = nn.functional.binary_cross_entropy_with_logits(z, yt.expand_as(z), reduction="none")
        loss = ((l * Mt).sum(1) / cnt).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (torch.einsum("fnd,fd->fn", Xt, W) + b[:, None]).numpy()
    acc = np.zeros((N_REP, n)); seen = np.zeros((N_REP, n), bool)
    for f, (rep, _, _, te) in enumerate(specs):
        acc[rep, te] = z[f, te]; seen[rep, te] = True
    return np.where(seen.all(0), acc.mean(0), np.nan)


def converged(X, y, sub, fid, mask):
    audit = {"inner_splits": 0, "max_inner_grad": 0.0, "max_outer_grad": 0.0, "outer_fits": 0, "outer_uncertified": 0}
    s, _ = w51.nested_scores(X, y, sub, mask, fid, (1e-3,), audit, lambda m: None, "w67")
    return s, audit


def partition(subs_eligible, seed):
    perm = np.random.default_rng(seed).permutation(subs_eligible)
    return {s: i % 5 for i, s in enumerate(perm)}


def canon(assign, subs):
    groups = {}
    for s in subs:
        groups.setdefault(assign[s], []).append(s)
    return frozenset(frozenset(v) for v in groups.values())


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/partition.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]; y = (lab["direction"] == 1).astype(int); md = lab["direction"] != -1
    IMG = np.load("runs/temporal_aggregation/img_frames.npz"); vpath, tmpl = VIDEO_SOURCES["videomaev2"]; zv = np.load(vpath)
    X = {}
    for k in KS:
        X[f"clip_K{k}"] = image_readout(IMG["clip"], k)
        X[f"videomaev2_K{k}"] = video_readout(zv["full"], zv[tmpl.format(k=k)])
    P = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz"); C = np.load("runs/converged_probe_fits/converged_scores.npz")
    keep = P["keep"].astype(bool)
    res = {"provenance": run_provenance(seeds=PSEEDS), "criteria_heading": CRITERIA, "gates": {}, "partitions": {},
           "loso": {}}

    lf = _loso_folds(sub); fid_loso = np.array([lf[s] for s in sub])
    gate = {}
    for c, Xc in X.items():
        sf = fixed_step(Xc, y, fid_loso, md)
        da = abs(_auc(sf[keep], y[keep]) - _auc(P[c][keep], y[keep]))
        sc, aud = converged(Xc, y, sub, fid_loso, md)
        ds = float(np.nanmax(np.abs(sc[keep] - C[f"f1__{c}"][keep])))
        gate[c] = {"fixed_auc_delta": float(da), "converged_max_score_delta": ds,
                   "converged_uncertified": aud["outer_uncertified"]}
        res["loso"][c] = {"fixed": sf, "converged": sc}
        log(f"gate {c}: fixed AUC delta {da:.2e}, converged score delta {ds:.2e}")
    res["gates"]["loso_reproduction"] = gate
    res["gates"]["loso_pass"] = all(v["fixed_auc_delta"] <= 1e-4 and v["converged_max_score_delta"] <= 1e-9
                                    and v["converged_uncertified"] == 0 for v in gate.values())

    elig = np.unique(sub[md])
    assigns = {s: partition(elig, s) for s in PSEEDS}
    cs = [canon(a, elig) for a in assigns.values()]
    distinct = all(x != y_ for x, y_ in itertools.combinations(cs, 2))
    sizes = {s: sorted(np.bincount(list(a.values()), minlength=5).tolist()) for s, a in assigns.items()}
    res["gates"]["partitions_distinct_modulo_relabel"] = bool(distinct)
    res["gates"]["fold_sizes"] = sizes
    log(f"partitions distinct: {distinct}; fold sizes {sizes}")
    if not (res["gates"]["loso_pass"] and distinct):
        res["status"] = "HOLD: gate failed"
        json.dump(res, open(f"{OUT}/partition.json", "w"), indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
        log("HOLD"); return

    null = np.abs(np.load("runs/pipeline_simulation_resampled/w12_direction_null.npy"))

    def contrast(sa, sb, mask_ok):
        d = midrank_auc(sa[mask_ok], y[mask_ok]) - midrank_auc(sb[mask_ok], y[mask_ok])
        dz, z = z_paired(sa[mask_ok], sb[mask_ok], y[mask_ok], sub[mask_ok])
        p = float((1 + (null >= abs(z)).sum()) / (1 + len(null)))
        return {"gap": float(d), "z": float(z), "p_published_reference_descriptive": p}

    for fit in ("fixed", "converged"):
        res["loso"][f"contrasts_{fit}"] = {}
        for k in KS:
            a, b = res["loso"][f"clip_K{k}"][fit], res["loso"][f"videomaev2_K{k}"][fit]
            ok = keep & ~np.isnan(a) & ~np.isnan(b)
            res["loso"][f"contrasts_{fit}"][f"K{k}"] = contrast(a, b, ok)
    store = {}
    for s, asg in assigns.items():
        fid = np.array([asg.get(x, -1) for x in sub])
        rec = {"fold_sizes": sizes[s], "single_class_test_folds": [], "training_two_classes": True, "fits": {}}
        for f in range(5):
            te = (fid == f) & md; tr = (fid != f) & (fid >= 0) & md
            if len(set(y[te].tolist())) < 2:
                rec["single_class_test_folds"].append(f)
            if len(set(y[tr].tolist())) < 2:
                rec["training_two_classes"] = False
            if te.sum() == 0:
                raise RuntimeError(f"partition {s}: empty test fold {f}")
        for fit in ("fixed", "converged"):
            rec["fits"][fit] = {}
            for k in KS:
                if fit == "fixed":
                    a = fixed_step(X[f"clip_K{k}"], y, fid, md); b = fixed_step(X[f"videomaev2_K{k}"], y, fid, md)
                    unc = 0
                else:
                    a, au = converged(X[f"clip_K{k}"], y, sub, fid, md); b, bu = converged(X[f"videomaev2_K{k}"], y, sub, fid, md)
                    unc = au["outer_uncertified"] + bu["outer_uncertified"]
                ok = md & ~np.isnan(a) & ~np.isnan(b)
                r = contrast(a, b, ok); r["n_scored"] = int(ok.sum()); r["uncertified"] = unc
                if ok.sum() != md.sum() or not rec["training_two_classes"] or unc:
                    raise RuntimeError(f"partition {s} {fit} K={k}: coverage {ok.sum()}/{md.sum()}, "
                                       f"training classes {rec['training_two_classes']}, uncertified {unc}")
                rec["fits"][fit][f"K{k}"] = r
                store[f"p{s}_{fit}_clip_K{k}"] = a; store[f"p{s}_{fit}_videomaev2_K{k}"] = b
                log(f"partition {s} {fit:9s} K={k}: gap {r['gap']:+.4f} z {r['z']:+.3f} (n {r['n_scored']})")
        res["partitions"][str(s)] = rec
    for fit in ("fixed", "converged"):
        gaps = [res["partitions"][str(s)]["fits"][fit][f"K{k}"]["gap"] for s in PSEEDS for k in KS]
        npos = sum(g > 0 for g in gaps)
        res[f"summary_{fit}"] = {"n_positive_of_10": npos,
                                 "sentence": ("the ordering persisted across 5 subject-grouped partitions" if npos == 10
                                              else f"the ordering reversed in {10 - npos} of 10 partition-granularity combinations"),
                                 "gap_range": [min(gaps), max(gaps)]}
        log(f"{fit}: {res[f'summary_{fit}']}")
    for c in X:
        for fit in ("fixed", "converged"):
            store[f"loso_{fit}_{c}"] = res["loso"][c][fit]
        del res["loso"][c]
    np.savez_compressed(f"{OUT}/partition_scores.npz", **store)
    res["status"] = "complete"
    json.dump(res, open(f"{OUT}/partition.json", "w"), indent=2, default=float)
    log(f"wrote {OUT}/partition.json")


if __name__ == "__main__":
    main()
