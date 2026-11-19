"""CLIP against VideoMAEv2 on Aff-Wild2 along a shared penalty path.

Usage:  python -m affectdyn.analyses.sensitivity.affwild_regularization_path
        python -m affectdyn.analyses.sensitivity.affwild_regularization_path [--procs 16] # the full stage
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Run on the CPU.
import argparse, copy, json, re, sys, time                    # noqa: E402

import numpy as np   # noqa: E402

from affectdyn.features.unbc_pretraining_comparison import _auc                                          # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout                      # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                                  # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import grouped_folds                                        # noqa: E402
from affectdyn.analyses.direction.affwild_dense_direction_contrasts import (IMG_SEEDS, VID_SEEDS, paired_bootstrap,    # noqa: E402
                                     subject_bootstrap_vs_half)
from affectdyn.inference.calibrated_probability_mc_bounds import clopper_pearson                       # noqa: E402
import affectdyn.probes.converged_probe_fits as stage_w46_probe_convergence   # noqa: E402,F401
import affectdyn.probes.nested_probe_fits as w51                                            # noqa: E402
import affectdyn.analyses.direction.affwild_transfer_test as w63                                                # noqa: E402
from affectdyn.provenance import run_provenance                                           # noqa: E402

OUT = "runs/affwild_regularization_path"
W63 = "runs/affwild_transfer_test"
SEED = 20260964
W63_SEED = w63.SEED
NBOOT = w63.NBOOT   # Use 10,000 bootstrap draws.
N_NULL = w63.N_NULL_Z   # Use 100,000 null draws.
ALPHA = 0.05
KS = w63.KS   # Evaluate K=2 and K=4.
ENC = w63.ENC   # Evaluate VideoMAEv2.
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
KEY_LAMBDAS = (1e-3, 1.0, 10.0)
SELECTED = (1.0, 10.0)   # Evaluate the selected penalties.
HEAVIER = (100.0, 1000.0)   # These penalties approach the same mean-difference limit.
WDS = (1e-4, 1e-3, 1e-2, 1e-1)
STEPS = (300, 1000)
PUBLISHED = (1e-3, 300)
LR = 1e-2
B_REFIT = 2_000
TOL = w51.TOL   # Require convergence within 1e-8.
G1_TOL, G2_TOL, GC_TOL = 1e-8, 1e-6, 1e-6
PUB_HOLM2_CONV = {2: 0.7272, 4: 0.0072}
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-10-02 — W64 AND W65 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMEND = "## 2026-10-02 — W64 AND W65 CRITERIA AMENDED"


def lam_key(lam):
    return f"{lam:g}"


def load_data():
    lab = np.load("runs/affwild_clips_features/labels.npz", allow_pickle=True)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    subj_all = np.array([str(s) for s in lab["subject"]]); d_all = lab["direction"]
    sub = subj_all[idx]
    y = (d_all[idx] == 1).astype(int)
    mask = d_all[idx] != -1
    assign = grouped_folds(sub)
    ex = json.load(open(f"{W63}/extract.json"))
    assert ex["status"] == "COMPLETE" and not ex["smoke"], ex["status"]
    assert ex["gate_g1_extraction_replay"]["passes"], "W63 extraction gate did not pass"
    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    pubz = np.load(f"runs/affwild_clips_features/{ENC}_windows.npz")
    newz = np.load(w63.NEW)
    X = {}
    for k in KS:
        X[("clip", k)] = image_readout(frames["clip"], k)[idx]
        X[(ENC, k)] = video_readout(pubz["full"], newz[f"K{k}_up"])
    S = np.load(f"{W63}/oof_scores.npz")
    stored = {f: S[f] for f in S.files}
    assert np.array_equal(stored["y"], y) and np.array_equal(stored["mask"], mask)
    assert np.array_equal(stored["fold"], assign) and np.array_equal(stored["subject"], sub)
    T = json.load(open(f"{W63}/transfer.json"))
    cells = {(c["K"], c["fit"]): c for c in T["cells"]}
    nested_lams = {}
    for k in KS:
        fd = cells[(k, "nested")]["fit_detail"]
        for e in ("clip", ENC):
            assert fd[e]["n_folds"] == int(assign.max()) + 1 and len(fd[e]["lambdas"]) == fd[e]["n_folds"]
            nested_lams[(e, k)] = [float(v) for v in fd[e]["lambdas"]]   # Preserve fold order from 0 through 4.
    return {"lab": lab, "idx": idx, "subj_all": subj_all, "d_all": d_all, "sub": sub, "y": y, "mask": mask,
            "assign": assign, "X": X, "stored": stored, "T": T, "cells": cells, "nested_lams": nested_lams,
            "extraction_sha256": ex["output"]["sha256"]}


def primal_scores(X, y, assign, mask, lam_of_fold):
    s = np.full(len(y), np.nan); recs = []
    for k in range(int(assign.max()) + 1):
        te, tr = (assign == k) & mask, (assign != k) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        lam = lam_of_fold(k)
        xt = w51.standardize(X, tr, tr); xe = w51.standardize(X, tr, te)
        th, g, stalled = w51.newton_primal(xt, y[tr], lam)
        s[te] = np.hstack([xe.astype(np.float64), np.ones((te.sum(), 1))]) @ th
        recs.append({"fold": k, "lambda": lam, "grad_inf": g, "stalled": bool(stalled),
                     "certified": bool(g <= TOL and not stalled)})
    return s, recs


def oof_wd(X, y, assign, mask, seed, wd, steps, lr=LR):
    """At (1e-3, 300) it is stage_w3_probe.oof line for line (G2)."""
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    scores = np.full(len(y), np.nan, np.float32)
    for k in range(assign.max() + 1):
        te, tr = (assign == k) & mask, (assign != k) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32, device="cpu")
        yt = torch.tensor(y[tr], dtype=torch.float32, device="cpu")
        xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32, device="cpu")
        clf = nn.Linear(X.shape[1], 1).to("cpu")
        opt = torch.optim.Adam(clf.parameters(), lr, weight_decay=wd)
        for _ in range(steps):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores[te] = clf(xe).squeeze(-1).cpu().numpy()
    return scores


def fixed_step_wd(X, y, assign, mask, seeds, wd, steps):
    return np.nanmean([oof_wd(X, y, assign, mask, sd, wd, steps) for sd in seeds], axis=0)


def boot_detail(sa, sb, y, g, rng):
    """The 2 must agree exactly and leave the generator in the same state."""
    clone = np.random.default_rng(); clone.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    rec = paired_bootstrap(sa, sb, y, g, clone, NBOOT)
    codes = np.unique(g, return_inverse=True)[1]
    mem = [np.flatnonzero(codes == j) for j in range(codes.max() + 1)]; G = len(mem); vals = []; disc = 0
    for _ in range(NBOOT):
        i = np.concatenate([mem[j] for j in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            disc += 1; continue
        vals.append(_auc(sa[i], y[i]) - _auc(sb[i], y[i]))
    v = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = float(min(1.0, max(2.0 * min(float((v <= 0).mean()), float((v >= 0).mean())), 1.0 / len(v))))
    assert p == rec["p_boot"] and [float(lo), float(hi)] == rec["ci"], (p, rec)
    assert rng.bit_generator.state == clone.bit_generator.state
    return {**rec, "n_valid": int(len(v)), "discarded": disc, "k_le0": int((v <= 0).sum()),
            "k_ge0": int((v >= 0).sum()), "boot_mean": float(v.mean())}


def mc_classify(raw, bounds):
    adj, lo, hi = holm(raw), holm([b[0] for b in bounds]), holm([b[1] for b in bounds])
    out = []
    for i in range(len(raw)):
        cls = ("MC-stable significant" if hi[i] < ALPHA else
               "MC-stable non-significant" if lo[i] >= ALPHA else "MC-unresolved")
        out.append({"holm2": float(adj[i]), "holm2_from_bounds": [float(lo[i]), float(hi[i])],
                    "raw_bounds": [float(bounds[i][0]), float(bounds[i][1])], "classification": cls})
    return out


def boot_bounds(b):
    n = b["n_valid"]; k = min(b["k_le0"], b["k_ge0"])
    lo, hi = clopper_pearson(k, n, 1 - ALPHA / len(KS))
    f = lambda q: min(1.0, max(2.0 * q, 1.0 / n))                                   # noqa: E731
    return f(lo), f(hi)


def cal_bounds(c):
    x = (N_NULL + 1) * c["p_simulated"]; k = int(round(x)) - 1
    assert abs(x - round(x)) <= 1e-6 and 0 <= k <= N_NULL, x
    return clopper_pearson(k, N_NULL, 1 - ALPHA / len(KS))


def decide(holm_p, gap):
    if holm_p < ALPHA:
        return "neg" if gap < 0 else "pos" if gap > 0 else "ns"
    return "ns"


def test_point(sc, sv, d, states, label, log):
    y, mask, sub, assign = d["y"], d["mask"], d["sub"], d["assign"]
    per_k = {}
    for k in KS:
        keep = mask & ~np.isnan(sc[k]) & ~np.isnan(sv[k])
        yk, gk = y[keep].astype(np.int64), sub[keep]
        rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(states[k])
        bp = boot_detail(sc[k][keep], sv[k][keep], yk, gk, rng)
        cal = w63.calibrated(sc[k][keep], sv[k][keep], yk, gk, rng, f"{label}:K{k}", log)
        ac, av = float(_auc(sc[k][keep], yk)), float(_auc(sv[k][keep], yk))
        pf_c, sd_c, nf = w63.fold_auc(sc[k], y, assign, keep)
        pf_v, sd_v, _ = w63.fold_auc(sv[k], y, assign, keep)
        per_k[k] = {"K": k, "n": int(keep.sum()),
                    "keep_matches_w63": bool(np.array_equal(keep, d["stored"][f"keep_K{k}_converged"])),
                    "auc_clip": ac, f"auc_{ENC}": av, "gap_pooled": ac - av,
                    "per_fold": {"clip": pf_c, "clip_sd": sd_c, ENC: pf_v, f"{ENC}_sd": sd_v, "n_folds": nf,
                                 "note": "descriptive (O1); the family is on the pooled scale"},
                    "bootstrap": bp, "calibrated": cal}
    for proc, rawf, bnd in (("bootstrap", lambda c: c["bootstrap"]["p_boot"], lambda c: boot_bounds(c["bootstrap"])),
                            ("calibrated", lambda c: c["calibrated"]["p_simulated"], lambda c: cal_bounds(c["calibrated"]))):
        rows = [per_k[k] for k in KS]
        mc = mc_classify([rawf(c) for c in rows], [bnd(c) for c in rows])
        for c, m in zip(rows, mc):
            c[f"holm2_{proc}"] = m["holm2"]
            c[f"decision_{proc}"] = decide(m["holm2"], c["gap_pooled"])
            c[f"mc_{proc}"] = m
    for k in KS:
        c = per_k[k]
        log(f"  {label} K={k} CLIP {c['auc_clip']:.4f} {ENC} {c[f'auc_{ENC}']:.4f} gap {c['gap_pooled']:+.4f} | "
            f"boot Holm2 {c['holm2_bootstrap']:.4f} ({c['decision_bootstrap']}, {c['mc_bootstrap']['classification']}) "
            f"| cal Holm2 {c['holm2_calibrated']:.5f} ({c['decision_calibrated']}, "
            f"{c['mc_calibrated']['classification']})")
    return {f"K{k}": per_k[k] for k in KS}


def category(dec, lams):
    """dec: {lambda: 'neg'|'pos'|'ns'} at K=4 over the CERTIFIED lambdas `lams`."""
    neg = [l for l in lams if dec[l] == "neg"]; pos = [l for l in lams if dec[l] == "pos"]
    if neg and pos:
        return "SIGN FLIPS WITH PENALTY"
    if len(neg) == len(lams):
        return "PENALTY-ROBUST REVERSAL"
    if dec[1e-3] == "neg" and all(dec[l] != "neg" for l in SELECTED):
        return "PENALTY-DEPENDENT REVERSAL"
    return "MIXED"


def read_rule(armA, proc, certified):
    """Classify one procedure, accounting for uncertified fits and Monte Carlo uncertainty."""
    lams = [l for l in LAMBDAS if certified[l]]
    excluded = [lam_key(l) for l in LAMBDAS if not certified[l]]
    if any(not certified[l] for l in KEY_LAMBDAS):
        return {"category": "HELD", "reason": f"uncertified at a key lambda: {excluded}", "excluded": excluded}
    c4 = {l: armA[lam_key(l)]["K4"] for l in lams}
    dec = {l: c4[l][f"decision_{proc}"] for l in lams}
    unres = [l for l in lams if c4[l][f"mc_{proc}"]["classification"] == "MC-unresolved"]

    def flip(l):
        if dec[l] != "ns":
            return "ns"
        return "neg" if c4[l]["gap_pooled"] < 0 else "pos"

    def cats(over):
        out = set()
        for mask_ in range(2 ** len(over)):
            d2 = dict(dec)
            for j, l in enumerate(over):
                if mask_ >> j & 1:
                    d2[l] = flip(l)
            out.add(category(d2, lams))
        return sorted(out)

    point = category(dec, lams)
    key_unres = [l for l in unres if l in KEY_LAMBDAS]
    cat_key = cats(key_unres); cat_all = cats(unres)
    final = "MC-UNRESOLVED" if len(cat_key) > 1 else point
    ranges = {s: [lam_key(l) for l in lams if dec[l] == s] for s in ("neg", "pos", "ns")}
    heavier = {"not_detected_at_100_and_1000": bool(all(dec.get(l, "ns") != "neg" for l in HEAVIER)
                                                     and all(certified[l] for l in HEAVIER)),
               "note": "Lambda 100 and 1000 approach the same mean-difference limit; extend the range to heavier "
                       "penalties only when both satisfy the decision rule."}
    return {"category": final, "point_category": point, "decisions_K4": {lam_key(l): dec[l] for l in lams},
            "ranges_K4": {"reversal_significant_gap_lt_0": ranges["neg"],
                          "clip_higher_significant_gap_gt_0": ranges["pos"], "not_significant": ranges["ns"]},
            "mc_unresolved_lambdas_K4": [lam_key(l) for l in unres],
            "categories_reachable_by_key_unresolved": cat_key,
            "categories_reachable_by_any_unresolved": cat_all, "heavier_clause": heavier,
            "excluded_uncertified": excluded,
            "robust_scope_note": ("ROBUST is evaluated over certified lambdas only; uncertified lambdas are "
                                  "excluded and listed separately.")}


OUTCOME_TEXT = {
    "PENALTY-ROBUST REVERSAL": (
        "The reversal persists across the certified common-penalty range. "
        "The nested fit has a non-significant CLIP-higher gap with penalties "
        "selected per encoder and fold. Encoder ordering is not consistent "
        "across all fits."),
    "PENALTY-DEPENDENT REVERSAL": (
        "The reversal is detected at the untuned penalty (fixed-step and converged at 1e-3) and is not detected at "
        "the selected penalties (1 and 10). Extend this range to heavier penalties only if non-detection also holds "
        "at 100 and 1000."),
    "SIGN FLIPS WITH PENALTY": (
        "The reversal is detected at the untuned penalty and not detected at the selected penalties (1 and 10). A "
        "significant CLIP-higher gap at some lambda is selected from an uncorrected 8-point path, is descriptive, "
        "and does not establish a general cross-corpus ordering."),
    "MIXED": (
        "Each decision applies to its reported lambda range. The reversal is limited to "
        "the range where it is detected and is not penalty-robust."),
    "HELD": "A key lambda (1e-3, 1 or 10) has an uncertified or stalled outer fit; classification is withheld.",
    "MC-UNRESOLVED": "Classification depends on a Monte Carlo-unresolved decision at 1e-3, 1 or 10.",
}
FIXED_CONSEQUENCE = ("The fixed-step and converged fits share one penalty, "
                     "so the three fits span two regularization regimes. Arms B and C provide context and do not "
                     "override arm A's category.")


_W = {}


def _init(payload):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _W.update(payload)


def armc_payload(d):
    m = d["mask"]
    rows = np.flatnonzero(m)
    subm = d["sub"][rows]
    groups = np.unique(subm)
    members = [np.flatnonzero(subm == g_) for g_ in groups]
    return {"X": {f"{e}|{k}": d["X"][(e, k)][rows].astype(d["X"][(e, k)].dtype) for e in ("clip", ENC) for k in KS},
            "y": d["y"][rows].astype(int), "fold": d["assign"][rows], "members": members,
            "nested": {f"{e}|{k}": d["nested_lams"][(e, k)] for e in ("clip", ENC) for k in KS},
            "rows": rows}


def kernel_scores(X, y, fold, settings):
    """Returns {name: scores}, a certificate tally, and a discard flag."""
    out = {s: np.full(len(y), np.nan) for s in settings}; cert = {s: [0, 0, 0.0] for s in settings}
    for k in range(int(fold.max()) + 1):
        te, tr = fold == k, fold != k
        if te.sum() == 0:
            continue
        if len(set(y[tr].tolist())) < 2:
            return None, None
        xt = w51.standardize(X, tr, tr).astype(np.float64); xe = w51.standardize(X, tr, te).astype(np.float64)
        X1 = np.hstack([xt, np.ones((len(xt), 1))]); X1e = np.hstack([xe, np.ones((len(xe), 1))])
        G = X1 @ X1.T
        for s, lf in settings.items():
            a, g = w51.newton_kernel(G, X1, y[tr], lf(k))
            out[s][te] = X1e @ (X1.T @ a)
            cert[s][0] += 1; cert[s][1] += int(g > TOL); cert[s][2] = max(cert[s][2], g)
    return out, cert


def _settings(nested):
    return {"lam1e-3": (lambda k: 1e-3), "nested_fixed": (lambda k, L=nested: L[k])}


def _armc_rep(draw):
    W = _W
    idx = np.sort(np.concatenate([W["members"][g_] for g_ in draw]))
    yy, ff = W["y"][idx], W["fold"][idx]
    if yy.min() == yy.max():
        return None
    res, certs = {}, {}
    for k in KS:
        sc = {}
        for e in ("clip", ENC):
            s, c = kernel_scores(W["X"][f"{e}|{k}"][idx], yy, ff, _settings(W["nested"][f"{e}|{k}"]))
            if s is None:
                return None   # Discard resamples with a single-class training fold.
            sc[e] = s; certs[f"{e}|{k}"] = c
        for st in ("lam1e-3", "nested_fixed"):
            ok = ~np.isnan(sc["clip"][st]) & ~np.isnan(sc[ENC][st])
            res[f"K{k}|{st}"] = float(_auc(sc["clip"][st][ok], yy[ok]) - _auc(sc[ENC][st][ok], yy[ok]))
    unc = {st: sum(certs[c][st][1] for c in certs) for st in ("lam1e-3", "nested_fixed")}
    fits = {st: sum(certs[c][st][0] for c in certs) for st in ("lam1e-3", "nested_fixed")}
    mg = {st: max(certs[c][st][2] for c in certs) for st in ("lam1e-3", "nested_fixed")}
    return {"gaps": res, "uncertified": unc, "fits": fits, "max_grad": mg}


def gate_g1(d, log):
    out, scores = {}, {}
    worst = 0.0; uncert = 0
    for k in KS:
        for e in ("clip", ENC):
            s, recs = primal_scores(d["X"][(e, k)], d["y"], d["assign"], d["mask"], lambda f: 1e-3)
            ref = d["stored"][f"{e}_K{k}_converged"]; keep = d["stored"][f"keep_K{k}_converged"]
            assert np.array_equal(np.isnan(s) & d["mask"], np.isnan(ref) & d["mask"])
            dl = float(np.nanmax(np.abs(s[keep] - ref[keep])))
            out[f"{e}_K{k}"] = {"max_abs_score_delta": dl, "uncertified": sum(not r["certified"] for r in recs),
                                "max_grad_inf": max(r["grad_inf"] for r in recs)}
            worst = max(worst, dl); uncert += out[f"{e}_K{k}"]["uncertified"]; scores[(e, k)] = s
    g = {"per_cell": out, "max_abs_score_delta": worst, "tolerance": G1_TOL, "uncertified": uncert,
         "passes": bool(worst <= G1_TOL and uncert == 0)}
    log(f"G1 converged at 1e-3 vs runs/affwild_transfer_test: max |score delta| {worst:.3e} (tol {G1_TOL:g}), uncertified {uncert} "
        f"-> {'PASS' if g['passes'] else 'FAIL'}")
    return g, scores


def gate_g2(d, log):
    out, worst, gaps = {}, 0.0, {}
    sc = {}
    for k in KS:
        for e, seeds in (("clip", IMG_SEEDS), (ENC, VID_SEEDS)):
            s = fixed_step_wd(d["X"][(e, k)], d["y"], d["assign"], d["mask"], seeds, *PUBLISHED)
            ref = d["stored"][f"{e}_K{k}_fixed"]
            assert np.array_equal(np.isnan(s) & d["mask"], np.isnan(ref) & d["mask"])
            dl = float(np.nanmax(np.abs(s[d["mask"]] - ref[d["mask"]])))
            out[f"{e}_K{k}"] = dl; worst = max(worst, dl); sc[(e, k)] = s
        keep = d["mask"] & ~np.isnan(sc[("clip", k)]) & ~np.isnan(sc[(ENC, k)])
        yk = d["y"][keep].astype(np.int64)
        gap = float(_auc(sc[("clip", k)][keep], yk) - _auc(sc[(ENC, k)][keep], yk))
        pub = d["cells"][(k, "fixed")]["gap_pooled"]
        gaps[f"K{k}"] = {"gap": gap, "gap_w63": pub, "abs_delta": abs(gap - pub)}
    gmax = max(v["abs_delta"] for v in gaps.values())
    g = {"max_abs_score_delta": worst, "per_cell": out, "tolerance_scores": G2_TOL, "gaps": gaps,
         "max_abs_gap_delta": gmax, "tolerance_gap": 1e-12,
         "passes": bool(worst <= G2_TOL and gmax <= 1e-12)}
    log(f"G2 fixed-step at (1e-3, 300) vs runs/affwild_transfer_test: max |score delta| {worst:.3e} (tol {G2_TOL:g}); gap deltas "
        f"{ {k: v['abs_delta'] for k, v in gaps.items()} } -> {'PASS' if g['passes'] else 'FAIL'}")
    return g, sc


def gate_g3(d, log):
    rng = np.random.default_rng(W63_SEED)
    T, cells, st = d["T"], d["cells"], d["stored"]
    y, mask, sub, idx, subj_all, d_all = d["y"], d["mask"], d["sub"], d["idx"], d["subj_all"], d["d_all"]
    checks = []

    def chk(name, got, want):
        checks.append({"call": name, "got": got, "w63": want, "equal": bool(got == want)})

    man = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    assert len(man) == len(subj_all)
    start = np.array([float(re.findall(r"(\d+)", os.path.basename(c["start"]))[0]) for c in man])
    rel = np.empty(len(subj_all))
    for s in np.unique(subj_all):
        ix = np.flatnonzero(subj_all == s)
        order = np.argsort(start[ix], kind="mergesort")
        r = np.empty(len(ix)); r[order] = np.arange(len(ix))
        rel[ix] = r / max(1, len(ix) - 1)
    full_m = d_all != -1
    yf = (d_all[full_m] == 1).astype(np.int64)
    pa = subject_bootstrap_vs_half(rel[idx][mask], y[mask].astype(np.int64), sub[mask], rng, NBOOT)
    pb = subject_bootstrap_vs_half(rel[full_m], yf, subj_all[full_m], rng, NBOOT)
    pd_ = T["position_diagnostics"]
    chk("position subset_relative_position", [pa["ci"], pa["p_boot"]],
        [pd_["subset_relative_position"]["ci"], pd_["subset_relative_position"]["p_boot"]])
    chk("position full_set_relative_position", [pb["ci"], pb["p_boot"]],
        [pd_["full_set_relative_position"]["ci"], pd_["full_set_relative_position"]["p_boot"]])
    states, conv_p = {}, {}
    for k in KS:
        for fit in ("fixed", "converged", "nested"):
            keep = st[f"keep_K{k}_{fit}"]
            sc, sv = st[f"clip_K{k}_{fit}"], st[f"{ENC}_K{k}_{fit}"]
            assert np.array_equal(keep, mask & ~np.isnan(sc) & ~np.isnan(sv))
            yk, gk = y[keep].astype(np.int64), sub[keep]
            if fit == "converged":
                states[k] = copy.deepcopy(rng.bit_generator.state)
            bp = paired_bootstrap(sc[keep], sv[keep], yk, gk, rng, NBOOT)
            cal = w63.calibrated(sc[keep], sv[keep], yk, gk, rng, f"replay K{k}:{fit}", log)
            c = cells[(k, fit)]
            chk(f"K{k} {fit} paired_bootstrap", [bp["p_boot"], bp["ci"]], [c["bootstrap"]["p_boot"], c["bootstrap"]["ci"]])
            chk(f"K{k} {fit} calibrated p_simulated", cal["p_simulated"], c["calibrated"]["p_simulated"])
            if fit == "converged":
                conv_p[k] = bp["p_boot"]
    h = holm([conv_p[k] for k in KS])
    holm2 = {k: float(h[i]) for i, k in enumerate(KS)}
    for k in KS:
        chk(f"K{k} converged Holm-2 bootstrap vs declared", holm2[k], PUB_HOLM2_CONV[k])
        chk(f"K{k} converged Holm-2 bootstrap vs transfer.json", holm2[k], cells[(k, "converged")]["holm2_bootstrap"])
    exact = all(c["equal"] for c in checks)
    within = all(abs(holm2[k] - PUB_HOLM2_CONV[k]) <= 0.005 for k in KS) and \
        all((holm2[k] < ALPHA) == (PUB_HOLM2_CONV[k] < ALPHA) for k in KS)
    g = {"holm2_converged_bootstrap": {f"K{k}": holm2[k] for k in KS},
         "declared": {f"K{k}": PUB_HOLM2_CONV[k] for k in KS},
         "replay": "EXACT" if exact else ("WITHIN 0.005, SAME DECISIONS" if within else "FAILED"),
         "n_checks": len(checks), "n_equal": sum(c["equal"] for c in checks), "checks": checks,
         "states_captured": {f"K{k}": states[k]["state"]["state"] for k in KS},
         "passes": bool(exact),
         "note": ("the amended R4 makes G3 an EXACT replay; the original declaration's 0.005 fallback is reported "
                  "but does not pass the gate, since every lambda's bootstrap is drawn from the captured states")}
    for c in checks:
        if not c["equal"]:
            log(f"   G3 MISMATCH {c['call']}: got {c['got']} w63 {c['w63']}")
    log(f"G3 replay: {g['n_equal']}/{g['n_checks']} calls equal; converged Holm-2 K=2 {holm2[2]} (declared 0.7272), "
        f"K=4 {holm2[4]} (declared 0.0072) | {g['replay']} -> {'PASS' if g['passes'] else 'FAIL'}")
    return g, states


def gate_gc(d, primal13, log):
    P = armc_payload(d)
    ident = np.arange(len(P["members"]))
    idx = np.sort(np.concatenate([P["members"][g_] for g_ in ident])); rows = P["rows"][idx]   # Match the refit worker's row ordering.
    yy, ff = P["y"][idx], P["fold"][idx]
    out, worst, worst_w63 = {}, 0.0, 0.0
    for k in KS:
        for e in ("clip", ENC):
            sk, cert = kernel_scores(P["X"][f"{e}|{k}"][idx], yy, ff, _settings(P["nested"][f"{e}|{k}"]))
            pn, recs = primal_scores(d["X"][(e, k)], d["y"], d["assign"], d["mask"],
                                     lambda f, L=d["nested_lams"][(e, k)]: L[f])
            d13 = float(np.max(np.abs(sk["lam1e-3"] - primal13[(e, k)][rows])))
            dn = float(np.max(np.abs(sk["nested_fixed"] - pn[rows])))
            dw = float(np.max(np.abs(pn[rows] - d["stored"][f"{e}_K{k}_nested"][rows])))
            out[f"{e}_K{k}"] = {"kernel_vs_primal_1e-3": d13, "kernel_vs_primal_nested_lambdas": dn,
                                "primal_nested_vs_w63_nested_scores": dw,
                                "kernel_uncertified": {s: cert[s][1] for s in cert},
                                "primal_nested_uncertified": sum(not r["certified"] for r in recs)}
            worst = max(worst, d13, dn); worst_w63 = max(worst_w63, dw)
    unc = sum(sum(v["kernel_uncertified"].values()) for v in out.values())
    g = {"per_cell": out, "max_abs_score_delta": worst, "tolerance": GC_TOL, "kernel_uncertified": unc,
         "primal_nested_vs_w63_nested_max_abs": worst_w63,
         "passes": bool(worst <= GC_TOL and unc == 0)}
    log(f"GC arm C identity replicate: kernel vs primal max |score delta| {worst:.3e} (tol {GC_TOL:g}), kernel "
        f"uncertified {unc}; descriptive: primal at nested lambdas vs W63 nested scores {worst_w63:.3e} "
        f"-> {'PASS' if g['passes'] else 'FAIL'}")
    return g


def run_gates(d, log):
    gates = {}
    g1, primal13 = gate_g1(d, log); gates["G1"] = g1
    if not g1["passes"]:
        return gates, None, None, False
    g2, _ = gate_g2(d, log); gates["G2"] = g2
    if not g2["passes"]:
        return gates, primal13, None, False
    g3, states = gate_g3(d, log); gates["G3"] = g3
    if not g3["passes"]:
        return gates, primal13, None, False
    gc = gate_gc(d, primal13, log); gates["GC_identity_replicate"] = gc
    return gates, primal13, states, bool(gc["passes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--b-refit", type=int, default=B_REFIT)
    a = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    out_json = f"{OUT}/penalty.json"
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite")
    if os.path.exists("RESEARCH_LOG-D3-W1.md"):   # Validate criteria when the optional source file exists.
        log_text = open("RESEARCH_LOG-D3-W1.md").read()
        assert CRIT in log_text and AMEND in log_text, "criteria headings not found in the research log"
    os.makedirs(OUT, exist_ok=True)
    prov = run_provenance(seeds=[W63_SEED, SEED], device="cpu")
    t0 = time.time(); logf = open(f"{OUT}/penalty.log", "w")

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(4)   # Limit CPU execution to four threads.
    d = load_data()
    log(f"W64 | torch {torch.__version__} | numpy {np.__version__} | clips {int(d['mask'].sum())} | groups "
        f"{len(np.unique(d['sub'][d['mask']]))} | folds {int(d['assign'].max()) + 1} | nested lambdas "
        f"{ {f'{e}_K{k}': v for (e, k), v in d['nested_lams'].items()} }")


    gates, primal13, states, ok = run_gates(d, log)
    res = {"provenance": prov, "criteria_heading": "Penalty-sensitivity evaluation criteria",
           "amendment_heading": "Additional fit-certification and classification criteria",
           "data": {"n_direction_labelled": int(d["mask"].sum()), "n_onset": int(d["y"][d["mask"]].sum()),
                    "n_groups": int(len(np.unique(d["sub"][d["mask"]]))), "folds": int(d["assign"].max()) + 1,
                    "features_sha256_w63_extraction": d["extraction_sha256"],
                    "seed_asymmetry": "CLIP fixed-step averages 3 seeds, VideoMAEv2 5 (W63's convention, disclosed)"},
           "lambdas": [lam_key(l) for l in LAMBDAS], "grid": {"weight_decay": list(WDS), "steps": list(STEPS),
                                                               "lr": LR, "published": list(PUBLISHED)},
           "nested_lambdas_w63": {f"{e}_K{k}": v for (e, k), v in d["nested_lams"].items()},
           "gates": gates, "fixed_consequence": FIXED_CONSEQUENCE}
    if not ok:
        res["status"] = "HELD: gate failed"
        json.dump(res, open(out_json, "w"), indent=1); log(res["status"]); return

    armA, saveA, certified = {}, {}, {}
    for lam in LAMBDAS:
        sc, sv, recs = {}, {}, {}
        for k in KS:
            if lam == 1e-3:
                sc[k], sv[k] = primal13[("clip", k)], primal13[(ENC, k)]
                _, rc = primal_scores(d["X"][("clip", k)], d["y"], d["assign"], d["mask"], lambda f: lam)
                _, rv = primal_scores(d["X"][(ENC, k)], d["y"], d["assign"], d["mask"], lambda f: lam)
            else:
                sc[k], rc = primal_scores(d["X"][("clip", k)], d["y"], d["assign"], d["mask"], lambda f: lam)
                sv[k], rv = primal_scores(d["X"][(ENC, k)], d["y"], d["assign"], d["mask"], lambda f: lam)
            recs[f"clip_K{k}"], recs[f"{ENC}_K{k}"] = rc, rv
            saveA[f"lam{lam_key(lam)}__clip_K{k}"] = sc[k]; saveA[f"lam{lam_key(lam)}__{ENC}_K{k}"] = sv[k]
        cert = all(r["certified"] for rs in recs.values() for r in rs)
        certified[lam] = cert
        tp = test_point(sc, sv, d, states, f"A lam {lam_key(lam)}", log)
        tp["certificates"] = {c: {"uncertified": sum(not r["certified"] for r in rs),
                                  "stalled": sum(r["stalled"] for r in rs),
                                  "max_grad_inf": max(r["grad_inf"] for r in rs)} for c, rs in recs.items()}
        tp["certified"] = cert
        tp["k2_descriptive"] = ("K=2 is reported at every lambda; a significant K=2 result of either sign is "
                                "descriptive and does not determine the category.")
        armA[lam_key(lam)] = tp
        log(f"  A lam {lam_key(lam)} certified {cert}")
    np.savez_compressed(f"{OUT}/armA_scores.npz", **saveA)
    best = {}
    for k in KS:
        for e in ("clip", ENC):
            vals = {lam_key(l): armA[lam_key(l)][f"K{k}"][f"auc_{e}"] for l in LAMBDAS if certified[l]}
            bl = max(vals, key=vals.get)
            best[f"{e}_K{k}"] = {"lambda": bl, "auc": vals[bl]}
    a13 = armA[lam_key(1e-3)]
    consistency = {f"K{k}": {"boot_p": a13[f"K{k}"]["bootstrap"]["p_boot"],
                             "boot_p_w63": d["cells"][(k, "converged")]["bootstrap"]["p_boot"],
                             "cal_p": a13[f"K{k}"]["calibrated"]["p_simulated"],
                             "cal_p_w63": d["cells"][(k, "converged")]["calibrated"]["p_simulated"]} for k in KS}
    for v in consistency.values():
        v["equal"] = bool(v["boot_p"] == v["boot_p_w63"] and v["cal_p"] == v["cal_p_w63"])
    rule = {proc: read_rule(armA, proc, certified) for proc in ("calibrated", "bootstrap")}
    cat = rule["calibrated"]["category"]
    res["arm_A"] = {"per_lambda": armA, "certified": {lam_key(l): certified[l] for l in LAMBDAS},
                    "lambda_maximizing_pooled_auc": best,
                    "lambda_maximizing_note": "DESCRIPTIVE ONLY, chosen on held-out scores",
                    "lambda_1e-3_reproduces_w63_converged_tests": consistency,
                    "no_correction_across_lambda": True}
    res["reading_rule"] = {"deciding_procedure": "calibrated", "category": cat,
                           "calibrated": rule["calibrated"], "bootstrap_beside": rule["bootstrap"],
                           "outcome_text": OUTCOME_TEXT.get(cat, OUTCOME_TEXT["MIXED"]),
                           "fixed_consequence": FIXED_CONSEQUENCE}
    log(f"ARM A CATEGORY (calibrated, deciding): {cat} | bootstrap beside: {rule['bootstrap']['category']}")
    json.dump(res, open(out_json + ".partial", "w"), indent=1)

    armB, saveB = {}, {}
    for wd in WDS:
        for steps in STEPS:
            sc, sv = {}, {}
            for k in KS:
                sc[k] = fixed_step_wd(d["X"][("clip", k)], d["y"], d["assign"], d["mask"], IMG_SEEDS, wd, steps)
                sv[k] = fixed_step_wd(d["X"][(ENC, k)], d["y"], d["assign"], d["mask"], VID_SEEDS, wd, steps)
                saveB[f"wd{wd:g}_s{steps}__clip_K{k}"] = sc[k]; saveB[f"wd{wd:g}_s{steps}__{ENC}_K{k}"] = sv[k]
            tp = test_point(sc, sv, d, states, f"B wd {wd:g} steps {steps}", log)
            tp["published_setting"] = bool((wd, steps) == PUBLISHED)
            armB[f"wd{wd:g}_steps{steps}"] = tp
    np.savez_compressed(f"{OUT}/armB_scores.npz", **saveB)
    summB = {proc: {s: [g for g, tp in armB.items() if tp["K4"][f"decision_{proc}"] == s] for s in ("neg", "pos", "ns")}
             for proc in ("calibrated", "bootstrap")}
    res["arm_B"] = {"per_grid_point": armB, "K4_decisions_summary": summB,
                    "note": ("qualifies the wording and never overrides arm A's category; every grid point draws "
                             "W63's converged-bootstrap state, so the published point's bootstrap p differs from "
                             "W63's fixed-step value, which was drawn from the preceding state")}
    json.dump(res, open(out_json + ".partial", "w"), indent=1)

    P = armc_payload(d)
    Gn = len(P["members"])
    draws = np.random.default_rng([SEED, 3]).integers(0, Gn, (a.b_refit, Gn))
    from multiprocessing import get_context
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"   # Spawned workers inherit these thread limits.
    t1 = time.time(); reps = []
    with get_context("spawn").Pool(a.procs, initializer=_init, initargs=(P,)) as pool:
        for i, r in enumerate(pool.imap(_armc_rep, list(draws), chunksize=2)):
            reps.append(r)
            if (i + 1) % 100 == 0:
                log(f"  C {i + 1}/{len(draws)} ({time.time() - t1:.0f}s)")
    keys = [f"K{k}|{st}" for k in KS for st in ("lam1e-3", "nested_fixed")]
    disc = sum(r is None for r in reps)
    G = {key: np.array([r["gaps"][key] if r is not None else np.nan for r in reps]) for key in keys}
    U = {st: np.array([r["uncertified"][st] if r is not None else 0 for r in reps]) for st in ("lam1e-3", "nested_fixed")}
    np.savez_compressed(f"{OUT}/armC_gaps.npz", draws=draws, **{k.replace("|", "__"): v for k, v in G.items()},
                        **{f"uncertified__{s}": v for s, v in U.items()})
    obs = {"lam1e-3": {k: armA[lam_key(1e-3)][f"K{k}"]["gap_pooled"] for k in KS},
           "nested_fixed": {k: d["cells"][(k, "nested")]["gap_pooled"] for k in KS}}

    def summ(x, o):
        x = x[~np.isnan(x)]
        lo, hi = np.percentile(x, [2.5, 97.5])
        return {"n": int(len(x)), "observed_gap": float(o), "boot_mean": float(x.mean()),
                "ci95_percentile": [float(lo), float(hi)], "share_at_or_below_0": float((x <= 0).mean()),
                "share_at_or_above_0": float((x >= 0).mean())}
    armC = {}
    for key in keys:
        k, st = key.split("|"); kk = int(k[1:])
        armC[key] = summ(G[key], obs[st][kk])
        bad = U[st] > 0
        if bad.any():
            armC[key]["excluding_replicates_with_uncertified_fits"] = summ(G[key][~bad], obs[st][kk])
        log(f"  C {key}: observed {obs[st][kk]:+.4f} mean {armC[key]['boot_mean']:+.4f} CI "
            f"[{armC[key]['ci95_percentile'][0]:+.4f}, {armC[key]['ci95_percentile'][1]:+.4f}] "
            f"share<=0 {armC[key]['share_at_or_below_0']:.3f} share>=0 {armC[key]['share_at_or_above_0']:.3f}")
    res["arm_C"] = {"B": int(a.b_refit), "draw_seed": [SEED, 3], "n_groups": Gn, "discarded_lost_class": int(disc),
                    "summaries": armC,
                    "uncertified": {st: {"replicates_with_any": int((U[st] > 0).sum()), "fits": int(U[st].sum()),
                                         "total_fits": int(sum(r["fits"][st] for r in reps if r is not None)),
                                         "max_grad": float(max(r["max_grad"][st] for r in reps if r is not None))}
                                    for st in ("lam1e-3", "nested_fixed")},
                    "note": ("qualifies, never decides. Setting (ii) holds W63's nested per-fold lambdas fixed; it is "
                             "conditional on W63's selection and is NOT a nested refit. Identical draws across "
                             "encoders, K and settings; duplicates stay in their group's original fold"),
                    "seconds": round(time.time() - t1, 1)}
    res["status"] = "COMPLETE"
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(out_json, "w"), indent=1)
    if os.path.exists(out_json + ".partial"):
        os.remove(out_json + ".partial")
    log(f"OUTCOME {cat}: {res['reading_rule']['outcome_text']}")
    log(f"COMPLETE in {res['seconds']}s"); logf.close()


if __name__ == "__main__":
    main()
