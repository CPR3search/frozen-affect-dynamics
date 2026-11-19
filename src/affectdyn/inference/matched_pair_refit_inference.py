"""Refit nulls and within-pair shares on the position-matched set.

Usage:  python -m affectdyn.inference.matched_pair_refit_inference
        python -m affectdyn.inference.matched_pair_refit_inference [--procs 18] [--force]
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

_T = os.environ.get("W65_THREADS", "1")
_ENV_KEEP = {v: _T for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
os.environ.update(_ENV_KEEP)
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")
import argparse, json, math, sys, time                                               # noqa: E402
from statistics import NormalDist                                                    # noqa: E402
import numpy as np                                                                   # noqa: E402

import affectdyn.analyses.peak.position_matched_peak_analysis as w59                                                     # noqa: E402
import affectdyn.probes.nested_probe_fits as w51                                                 # noqa: E402
import affectdyn.inference.peak_refit_null_centering as w52                                                   # noqa: E402
import affectdyn.inference.calibrated_clustered_auc as w12                                           # noqa: E402
from affectdyn.features.unbc_pretraining_comparison import _auc                                               # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout                           # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import grouped_folds                                             # noqa: E402
from affectdyn.provenance import run_provenance                                                # noqa: E402
if _CVD is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CVD
os.environ.update(_ENV_KEEP)

OUT = "runs/matched_pair_refit_inference"
SEED = 20260965
ALPHA = 0.05
M_FAMILY = 18
R_CENTRING = 1_000
R_DETECTION = 2_000
N_FLIP = 10_000
N_BOOT = 10_000
N_VALIDATION = 100_000
N_STEP = 2_000
STEPS = 10   # Bisect delta over [0, DELTA_HI] at resolution 1.5 / 2**10.
DELTA_HI = 1.5
SIM_BATCH = 250   # Share 10,000 fresh flip vectors across each batch of studies.
ICCS = (0.0, 0.3, 0.6)
POWER = 0.80
TOLERANCE = 1.5
CALIPER = "32"
FITS = ("fixed_step", "converged", "nested")
NULLS = ("pair_swap", "within_subject", "free")
DETECTION = (("siglip_K16", "fixed_step"), ("clip_K8", "nested"), ("clip_K16", "nested"))
G1_TOL, G3_TOL, NESTED_TOL = 1e-12, 1e-8, 1e-8
FIXED_TOL = 1e-5
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-10-02 — W64 AND W65 PRE-DECLARED CRITERIA (written before any code or run)"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMEND = "## 2026-10-02 — W64 AND W65 CRITERIA AMENDED after a design review, before any code or run"
ND = NormalDist()


def load_design():
    design = json.load(open("runs/position_matched_peak_analysis/matched_subset.json"))
    assert design["status"] == "FROZEN" and str(design["chosen_caliper"]) == CALIPER
    midx = np.load("runs/position_matched_peak_analysis/matched_subset_idx.npy").astype(int)
    lab = np.load("runs/affwild_clips_features/labels.npz")
    y = (lab["traj"][midx] == 2).astype(int); g = lab["subject"][midx]
    assert (lab["traj"][midx] != -1).all()
    pairs_full = np.array(design["ladder"][CALIPER]["pairs"], dtype=int)   # Each row pairs a peak with a negative clip.
    return midx, y, g, pairs_full


def build_features(midx, status):
    X = {}
    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    for e in w59.IMAGE:
        fr = frames[e][midx]
        for k in w59.KS:
            X[f"{e}_K{k}"] = image_readout(fr, k)
    pub = np.load("runs/affwild_clips_features/video_subset_idx.npy").astype(int); pos_pub = {int(i): k for k, i in enumerate(pub)}
    for e in w59.VIDEO:
        if status[e] != "tested":
            continue
        # Materialize arrays once to avoid repeated decompression on key access.
        cache = dict(np.load(f"runs/affwild_clips_features/{e}_windows.npz")); new = dict(np.load(f"runs/position_matched_peak_analysis/{e}_ordered_new.npz"))
        pos_new = {int(i): k for k, i in enumerate(new["idx"])}
        full = np.stack([cache["full"][pos_pub[i]] if i in pos_pub else new["full"][pos_new[i]] for i in midx])
        W = np.stack([cache["K16_up"][pos_pub[i]] if i in pos_pub else new["K16_up"][pos_new[i]] for i in midx])
        X[f"{e}_K16"] = video_readout(full, W)
    return X


def all_cells():
    return [f"{e}_K{k}" for e in w59.IMAGE for k in w59.KS] + [f"{e}_K16" for e in w59.VIDEO]


def is_image(cell):
    return cell.split("_K")[0] in w59.IMAGE


def new_audit():
    return {"inner_splits": 0, "max_inner_grad": 0.0, "max_outer_grad": 0.0, "outer_fits": 0, "outer_uncertified": 0}


def merge_audit(a, b):
    for k in ("inner_splits", "outer_fits", "outer_uncertified"):
        a[k] += b[k]
    for k in ("max_inner_grad", "max_outer_grad"):
        a[k] = max(a[k], b[k])
    return a


def refit(X, y, g, fit, image, audit):
    mask = np.ones(len(y), bool)
    if fit == "fixed_step":
        return w59.fit_scores(X, y, g, mask, fit, image)
    grid = (1e-3,) if fit == "converged" else w59.W56_GRID
    s, _ = w51.nested_scores(X, y, g, mask, grouped_folds(g), grid, audit, lambda m: None, "w65")
    return s


def null_auc(s, y):
    k = ~np.isnan(s)
    return float(_auc(s[k], y[k]))


def pair_swap(y, pp, pn, rng):
    """Each pair's two labels exchanged with probability one half."""
    yp = y.copy(); sw = rng.random(len(pp)) < 0.5
    yp[pp[sw]] = y[pn[sw]]; yp[pn[sw]] = y[pp[sw]]
    return yp


def draw_nulls(y, g, pp, pn, R):
    """One list of R label vectors per null, shared by every cell and fit (common random numbers)."""
    mask = np.ones(len(y), bool); out = {}
    for j, kind in enumerate(NULLS):
        rng = np.random.default_rng([SEED, 1, j])
        gen = {"pair_swap": lambda: pair_swap(y, pp, pn, rng),
               "within_subject": lambda: w52.within_subject_perm(y, mask, g, rng),
               "free": lambda: w52.global_perm(y, mask, rng)}[kind]
        out[kind] = np.stack([gen() for _ in range(R)]).astype(np.int8)
    return out


_W = {}


def _init(state):
    _W.update(state)
    import torch
    torch.set_num_threads(int(_T))


def _null_job(job):
    family, kind, cell, fit, r0, r1 = job
    X, g = _W["X"][cell], _W["g"]; L = _W["perms"][kind]
    audit = new_audit(); out = np.empty(r1 - r0); t0 = time.time()
    for j, r in enumerate(range(r0, r1)):
        yp = L[r].astype(int)
        out[j] = null_auc(refit(X, yp, g, fit, is_image(cell), audit), yp)
    return job, out, audit, time.time() - t0


def pair_indicators(s, pp, pn):
    return (s[pp] > s[pn]).astype(float) + 0.5 * (s[pp] == s[pn])


def subject_sums(I, pair_subj_code, n_subj):
    return np.bincount(pair_subj_code, weights=I - 0.5, minlength=n_subj)


def flip_p(c, E):
    """Two-sided sign-flip p for T = sum_s c_s against T_null = E c, p = (1 + #|T_null| >= |T|) / (1 + F). c: [S] or [B, S]; E: [F, S] of +-1."""
    c2 = np.atleast_2d(c); T = c2.sum(1); Tn = c2 @ E.T   # Tn has shape [B, F].
    cnt = (np.abs(Tn) >= np.abs(T)[:, None] - 1e-9).sum(1)
    p = (1 + cnt) / (1 + E.shape[0])
    return (p if np.ndim(c) == 2 else float(p[0])), T


def arm_b_cell(I, pair_subj_code, n_subj, n_per_subj, E, Wb):
    share = float(I.mean())
    c = subject_sums(I, pair_subj_code, n_subj)
    p_flip, _ = flip_p(c, E)
    sumI = np.bincount(pair_subj_code, weights=I, minlength=n_subj)
    v = (Wb @ sumI) / (Wb @ n_per_subj)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p_boot = float(min(1.0, max(2 * min((v <= 0.5).mean(), (v >= 0.5).mean()), 1.0 / len(v))))
    return {"share": share, "n_pairs": int(len(I)), "ties": int((I == 0.5).sum()), "p_flip": float(p_flip),
            "ci_subject_boot": [float(lo), float(hi)], "p_boot": p_boot}


def sim_rejections(delta, icc, sizes, n_studies, rng, alpha_m):
    """Simulate the sign-flip test on d = delta + u_subject + e."""
    S = len(sizes); code = np.repeat(np.arange(S), sizes); n = int(sizes.sum())
    onehot = np.zeros((n, S)); onehot[np.arange(n), code] = 1.0
    rej = rej_pos = done = 0
    while done < n_studies:
        b = min(SIM_BATCH, n_studies - done)
        d = delta + math.sqrt(icc) * rng.standard_normal((b, S))[:, code] + math.sqrt(1 - icc) * rng.standard_normal((b, n))
        I = (d > 0).astype(float) - 0.5
        c = I @ onehot   # Subject sums have shape [b, S].
        E = rng.integers(0, 2, (N_FLIP, S)).astype(float) * 2 - 1
        p, T = flip_p(c, E)
        hit = p <= alpha_m
        rej += int(hit.sum()); rej_pos += int((hit & (T > 0)).sum()); done += b
    return rej, rej_pos


def _bench_job(args):
    icc, sizes, seed_key, part = args
    alpha_m = ALPHA / M_FAMILY; t0 = time.time()
    if part == "validation":
        rng = np.random.default_rng([SEED, 3, *seed_key])
        n = seed_key[-1]   # Read the number of studies in this chunk.
        r, rp = sim_rejections(0.0, icc, sizes, n, rng, alpha_m)
        return {"icc": icc, "part": part, "n": n, "rej": r, "rej_pos": rp, "sec": time.time() - t0}
    rng = np.random.default_rng([SEED, 4, *seed_key])
    lo, hi, trace = 0.0, DELTA_HI, []
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        _, rp = sim_rejections(mid, icc, sizes, N_STEP, rng, alpha_m)
        pw = rp / N_STEP
        trace.append({"delta": round(mid, 6), "share": ND.cdf(mid), "power": pw, "wilson": list(w12.wilson(rp, N_STEP))})
        lo, hi = (mid, hi) if pw < POWER else (lo, mid)
    _, rp = sim_rejections(hi, icc, sizes, N_STEP, rng, alpha_m)
    return {"icc": icc, "part": part, "delta": hi, "bracket": [lo, hi], "share": ND.cdf(hi),
            "share_bracket": [ND.cdf(lo), ND.cdf(hi)], "power_at_mde": rp / N_STEP,
            "power_at_mde_wilson": list(w12.wilson(rp, N_STEP)), "trace": trace, "sec": time.time() - t0}


def run_gates(log, X, y, g, midx, pairs_full, w59j, stored, cells, tested):
    """No null comparison and no within-pair share is computed here."""
    res = {}

    def verdict(name, ok, info):
        res[name] = {**info, "passes": bool(ok)}
        log(f"{name}: {info} -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise GateFailure(name)

    d1, d1r, n1 = 0.0, 0.0, 0
    for fit in FITS:
        rows = {r["cell"]: r for r in w59j["shape_family"][fit]}
        for c in tested:
            s = stored[f"{fit}__{c}"]; auc_z, _ = w12.z_single(s, y, g)
            d1 = max(d1, abs(auc_z - rows[c]["auc"])); d1r = max(d1r, abs(_auc(s, y) - auc_z)); n1 += 1
    verdict("G1", d1 <= G1_TOL and d1r <= G1_TOL,
            {"aucs_checked": n1, "max_abs_auc_delta_vs_w59": d1, "max_abs_rank_auc_minus_z_single_auc": d1r,
             "tolerance": G1_TOL})

    pos = {int(i): k for k, i in enumerate(midx)}
    inside = all(int(a) in pos and int(b) in pos for a, b in pairs_full)
    flat = pairs_full.ravel(); disjoint = len(set(flat.tolist())) == len(flat)
    covers = set(flat.tolist()) == set(midx.tolist())
    pp = np.array([pos[int(a)] for a, _ in pairs_full]) if inside else None
    pn = np.array([pos[int(b)] for _, b in pairs_full]) if inside else None
    labels_ok = inside and bool((y[pp] == 1).all() and (y[pn] == 0).all())
    fold = grouped_folds(g)
    same_subj = inside and bool((g[pp] == g[pn]).all())
    same_fold = inside and bool((fold[pp] == fold[pn]).all())
    sizes = np.unique(g[pp], return_counts=True)[1] if inside else np.array([])
    verdict("G2", inside and disjoint and covers and labels_ok and same_subj and same_fold and len(pairs_full) == 225,
            {"n_pairs": int(len(pairs_full)), "inside_450": inside, "disjoint": disjoint, "cover_450": covers,
             "one_peak_one_negative": labels_ok, "one_subject_per_pair": same_subj, "one_fold_per_pair": same_fold,
             "subjects_with_pairs": int(len(sizes)), "subjects_holding_1_pair": int((sizes == 1).sum()),
             "max_pairs_per_subject": int(sizes.max()) if len(sizes) else 0})

    d3, aud3 = 0.0, new_audit()
    for c in tested:
        s = refit(X[c], y, g, "converged", is_image(c), aud3)
        d3 = max(d3, float(np.nanmax(np.abs(s - stored[f"converged__{c}"]))))
    verdict("G3", d3 <= G3_TOL and aud3["outer_uncertified"] == 0,
            {"cells": len(tested), "max_abs_score_delta_vs_w59_converged": d3, "tolerance": G3_TOL,
             "outer_fits": aud3["outer_fits"], "outer_uncertified": aud3["outer_uncertified"],
             "max_outer_grad": aud3["max_outer_grad"]})

    # Evaluate detection cells using their own fits.
    for c, fit in DETECTION:
        aud = new_audit(); t0 = time.time()
        s = refit(X[c], y, g, fit, is_image(c), aud); ref = stored[f"{fit}__{c}"]
        d = float(np.nanmax(np.abs(s.astype(np.float64) - ref.astype(np.float64))))
        da = abs(null_auc(s, y) - null_auc(ref, y))
        tol = FIXED_TOL if fit == "fixed_step" else NESTED_TOL
        verdict(f"G4_{fit}_{c}", d <= tol and da == 0.0 and aud["outer_uncertified"] == 0,
                {"max_abs_score_delta_vs_w59": d, "abs_auc_delta": da, "tolerance": tol,
                 "outer_fits": aud["outer_fits"], "outer_uncertified": aud["outer_uncertified"],
                 "max_inner_grad": aud["max_inner_grad"], "refit_seconds": round(time.time() - t0, 2)})
    return res, pp, pn, fold


class GateFailure(Exception):
    pass


def holm18(p_by_cell, cells):
    ps = [p_by_cell.get(c, 1.0) for c in cells]   # Assign TimeSformer a p-value of 1.
    return dict(zip(cells, (float(a) for a in w59.holm(ps))))


def read_detection(rows, w59j):
    """R9/R10 keep rule and the 'above own centre' rule, per detection cell."""
    out = {}
    for c, fit in DETECTION:
        r = rows[c]; obs = r["observed_auc"]
        sig = {k: r[k]["holm"] < ALPHA for k in NULLS}
        below = {k: obs < r[k]["centring"]["median"] for k in NULLS}
        keep = sig["pair_swap"] and sig["within_subject"] and below["pair_swap"] and below["within_subject"]
        above = [k for k in NULLS if sig[k] and not below[k]]
        w = {x["cell"]: x for x in w59j["shape_family"][fit]}[c]
        fixed_agree = bool(obs > 0.5 and w["holm"]["calibrated"] < ALPHA and w["holm"]["boot_subject"] < ALPHA)
        out[c] = {"fit": fit, "observed_auc": obs,
                  "significant_against": sig, "observed_below_null_median": below,
                  "null_medians": {k: r[k]["centring"]["median"] for k in NULLS},
                  "null_median_shift_from_half": {k: r[k]["centring"]["median"] - 0.5 for k in NULLS},
                  "centring_classes": {k: r[k]["centring"]["class"] for k in NULLS},
                  "supports_centring_at_half": all(r[k]["centring"]["class"] == "negligible" for k in NULLS),
                  "keep_sentence": bool(keep),
                  "wording": ("below chance is what the design demonstrably resolves" if keep else
                              "below chance under fixed-score tests (state the measured null centres)"),
                  "significant_above_own_null_centre": above,
                  "may_be_called_above_chance": bool(above and fixed_agree)}
    return out


def read_arm_b(armb, cells_tested):
    per_cell = {}
    for c in cells_tested:
        up = [f for f in FITS if armb[f][c]["holm_flip"] < ALPHA and armb[f][c]["share"] > 0.5]
        dn = [f for f in FITS if armb[f][c]["holm_flip"] < ALPHA and armb[f][c]["share"] < 0.5]
        if len(up) == len(FITS):
            cat = "ROBUST PAIRED-DESIGN DETECTION (discrimination between peaked windows and matched adjacent negatives)"
        elif up:
            cat = "FIT-DEPENDENT SENSITIVITY FINDING above one half"
        elif dn:
            cat = "SIGNIFICANTLY BELOW ONE HALF (never a position artifact)"
        else:
            cat = "NOT DETECTED on the paired estimand"
        per_cell[c] = {"above_half_fits": up, "below_half_fits": dn, "category": cat}
    any_up = any(v["above_half_fits"] for v in per_cell.values())
    return per_cell, any_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=18); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/matched_null.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    prov = run_provenance(seeds=[SEED], device="cpu", extra={"threads_per_process": int(_T), "procs": args.procs})
    tag = "matched_null"
    logf = open(f"{OUT}/{tag}.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(int(_T))
    midx, y, g, pairs_full = load_design()
    w59j = json.load(open("runs/position_matched_peak_analysis/analysis.json")); assert w59j["status"] == "COMPLETE"
    stored = np.load("runs/position_matched_peak_analysis/analysis_scores.npz"); assert np.array_equal(stored["idx"], midx)
    status = w59j["encoder_status"]
    cells = all_cells()
    X = build_features(midx, status); tested = [c for c in cells if c in X]
    assert len(tested) == 17 and "timesformer_K16" not in tested
    log(f"W65 | {len(midx)} clips, {int(y.sum())} peaked, {len(set(g.tolist()))} subjects, {len(tested)} tested cells, "
        f"threads/process {_T}, torch threads {torch.get_num_threads()}")


    try:
        gates, pp, pn, fold = run_gates(log, X, y, g, midx, pairs_full, w59j, stored, cells, tested)
    except GateFailure as e:
        log(f"STOP: gate {e} failed")
        json.dump({"provenance": prov, "criteria_heading": CRIT, "amendment_heading": AMEND,
                   "status": f"HOLD: gate {e} failed"}, open(out_json, "w"), indent=1)
        logf.close(); return

    res = {"provenance": prov, "criteria_heading": CRIT, "amendment_heading": AMEND, "gates": gates,
           "n_clips": int(len(midx)), "n_pairs": int(len(pp)), "n_subjects": int(len(set(g.tolist()))),
           "cells_tested": tested, "family_size": M_FAMILY, "timesformer": "enters every 18-test family at p = 1"}
    subj_codes, pair_subj_code = np.unique(g[pp], return_inverse=True)
    n_subj = len(subj_codes); n_per_subj = np.bincount(pair_subj_code, minlength=n_subj).astype(float)

    # Compute the benchmark before accessing scores.
    from multiprocessing import get_context
    VCH = 20
    bjobs = [(icc, n_per_subj.astype(int), (i, j, N_VALIDATION // VCH), "validation")
             for i, icc in enumerate(ICCS) for j in range(VCH)]
    bjobs += [(icc, n_per_subj.astype(int), (i,), "bisection") for i, icc in enumerate(ICCS)]
    perms = draw_nulls(y, g, pp, pn, R_DETECTION)
    state = {"X": X, "g": g, "perms": perms}
    jobs = []
    for c, fit in DETECTION:
        step = 50 if fit == "nested" else 250
        jobs += [("detection", k, c, fit, r, min(R_DETECTION, r + step)) for k in NULLS for r in range(0, R_DETECTION, step)]
    jobs += [("centring", k, c, "converged", r, min(R_CENTRING, r + 250)) for c in tested for k in NULLS
             for r in range(0, R_CENTRING, 250)]
    log(f"benchmark jobs {len(bjobs)}, null jobs {len(jobs)} over {args.procs} processes")
    with get_context("spawn").Pool(args.procs, initializer=_init, initargs=(state,)) as pool:
        bench_out = list(pool.imap_unordered(_bench_job, bjobs))
        bench = {}
        for i, icc in enumerate(ICCS):
            v = [b for b in bench_out if b["icc"] == icc and b["part"] == "validation"]
            n = sum(b["n"] for b in v); r = sum(b["rej"] for b in v); rp = sum(b["rej_pos"] for b in v)
            wl = w12.wilson(r, n); nominal = ALPHA / M_FAMILY
            valid = bool(wl[1] <= TOLERANCE * nominal)
            bis = next(b for b in bench_out if b["icc"] == icc and b["part"] == "bisection")
            bench[str(icc)] = {"validation": {"n_null_studies": n, "first_step_rejections": r, "rate": r / n,
                                              "wilson": list(wl), "nominal": nominal, "bound": TOLERANCE * nominal,
                                              "rejections_share_above_half": rp, "passes": valid},
                               ("detectable_share" if valid else "not_quotable_bisection"): bis}
            log(f"benchmark ICC {icc}: validation rate {r / n:.5f} {wl} vs bound {TOLERANCE * nominal:.5f} "
                f"{'PASS' if valid else 'FAIL'}; bisection share {bis['share']:.4f} (delta {bis['delta']:.4f}) "
                f"power {bis['power_at_mde']:.3f} {bis['power_at_mde_wilson']}")
        res["arm_b_benchmark"] = {"model": "d = delta + u_subject + e, var(u) = ICC, var(e) = 1 - ICC, share = Phi(delta)",
                                  "omits": ["within-pair feature similarity", "a fold-level probe effect"],
                                  "declared_icc": 0.3, "n_flip": N_FLIP, "n_per_step": N_STEP, "steps": STEPS,
                                  "flip_vectors": f"fresh {N_FLIP} per batch of {SIM_BATCH} simulated studies",
                                  "power_counts": "first-step rejections (p <= 0.05/18) with share above one half",
                                  "by_icc": bench}

        jobs.sort(key=lambda j: 0 if j[3] == "nested" else 1)
        nulls = {}; audits = {}; t1 = time.time(); done = 0
        for job, vals, aud, sec in pool.imap_unordered(_null_job, jobs):
            fam, kind, c, fit, r0, r1 = job
            nulls.setdefault((fam, kind, c, fit), {})[r0] = vals
            merge_audit(audits.setdefault((fam, fit), new_audit()), aud); done += 1
            if done % 25 == 0:
                log(f"  null jobs {done}/{len(jobs)} ({time.time() - t1:.0f}s)")
    null = {k: np.concatenate([v[r] for r in sorted(v)]) for k, v in nulls.items()}

    armA = {"nulls": list(NULLS), "R": {"centring": R_CENTRING, "detection": R_DETECTION},
            "certificates": {f"{f}:{fit}": a for (f, fit), a in audits.items()}}
    # Check centring with converged fits.
    cen = {k: {} for k in NULLS}
    for kind in NULLS:
        for i, c in enumerate(tested):
            obs = null_auc(stored[f"converged__{c}"], y); nv = null[("centring", kind, c, "converged")]
            cen[kind][c] = {"observed_auc": obs, "p": w52.tail_p(nv, obs),
                            "centring": w52.centring(nv, np.random.default_rng([SEED, 2, 0, NULLS.index(kind), i]))}
        for c, a in holm18({c: cen[kind][c]["p"] for c in tested}, cells).items():
            if c in cen[kind]:
                cen[kind][c]["holm"] = a
                cen[kind][c]["separates"] = (None if a >= ALPHA else
                                             "below own null centre" if cen[kind][c]["observed_auc"] < cen[kind][c]["centring"]["median"]
                                             else "above own null centre")
    armA["centring_family"] = {
        "fit": "converged (lambda 1e-3)", "role": "characterizes centring; never decides the sentence",
        "by_null": cen,
        "supports_centring_at_half": {c: all(cen[k][c]["centring"]["class"] == "negligible" for k in NULLS) for c in tested},
        "class_counts": {k: {cl: sum(cen[k][c]["centring"]["class"] == cl for c in tested)
                             for cl in ("negligible", "biased low", "biased high", "unresolved")} for k in NULLS}}
    # Apply Holm across three detection tests within each null draw.
    det = {c: {"fit": fit, "observed_auc": {x["cell"]: x for x in w59j["shape_family"][fit]}[c]["auc"]}
           for c, fit in DETECTION}
    for kind in NULLS:
        for j, (c, fit) in enumerate(DETECTION):
            nv = null[("detection", kind, c, fit)]; obs = det[c]["observed_auc"]
            det[c][kind] = {"p": w52.tail_p(nv, obs),
                            "centring": w52.centring(nv, np.random.default_rng([SEED, 2, 1, NULLS.index(kind), j]))}
        for (c, _), a in zip(DETECTION, w59.holm([det[c][kind]["p"] for c, _ in DETECTION])):
            det[c][kind]["holm"] = float(a)
    armA["detection_family"] = {"cells": [list(x) for x in DETECTION], "selected": "fixed from the published detections",
                                "holm": "across the 3 within each null", "by_cell": det,
                                "reading": read_detection(det, w59j)}
    res["arm_a"] = armA
    for c, r in armA["detection_family"]["reading"].items():
        log(f"A detection {c} ({r['fit']}): keep {r['keep_sentence']} | " +
            "; ".join(f"{k} holm {det[c][k]['holm']:.4f} median {r['null_medians'][k]:.4f} {r['centring_classes'][k]}"
                      for k in NULLS))

    rng_b = np.random.default_rng([SEED, 5])
    E = rng_b.integers(0, 2, (N_FLIP, n_subj)).astype(float) * 2 - 1
    draws = rng_b.integers(0, n_subj, (N_BOOT, n_subj))
    Wb = np.zeros((N_BOOT, n_subj)); np.add.at(Wb, (np.repeat(np.arange(N_BOOT), n_subj), draws.ravel()), 1)
    armb = {}
    for fit in FITS:
        rows = {c: arm_b_cell(pair_indicators(stored[f"{fit}__{c}"], pp, pn), pair_subj_code, n_subj, n_per_subj, E, Wb)
                for c in tested}
        hf = holm18({c: r["p_flip"] for c, r in rows.items()}, cells)
        hb = holm18({c: r["p_boot"] for c, r in rows.items()}, cells)
        for c in rows:
            rows[c]["holm_flip"] = hf[c]; rows[c]["holm_boot"] = hb[c]
        armb[fit] = rows
        log(f"B {fit}: " + "; ".join(f"{c} {r['share']:.3f} flip {r['holm_flip']:.4f} boot {r['holm_boot']:.4f}"
                                     for c, r in rows.items()))
    per_cell, any_up = read_arm_b(armb, tested)
    b03 = bench["0.3"]
    sens = (f"detectable share {b03['detectable_share']['share']:.3f} at ICC 0.3" if "detectable_share" in b03
            else "of unstated sensitivity (benchmark validation failed at ICC 0.3)")
    res["arm_b"] = {"deciding_test": "subject-level sign flip, 10,000 flips, two-sided against one half, Holm across 18 per fit",
                    "sensitivity": "subject bootstrap, B = 10,000, pairs nested in subjects",
                    "by_fit": armb, "reading": per_cell,
                    "summary": ("at least one cell significantly above one half; see reading (never trajectory shape or order)"
                                if any_up else f"no cell significantly above one half under any fit: the non-detection "
                                               f"also holds on the paired estimand, {sens}; never 'absent'")}
    uncert = sum(a["outer_uncertified"] for a in audits.values())
    res["uncertified_outer_fits"] = int(uncert)
    res["status"] = "COMPLETE" if uncert == 0 else "COMPLETE: uncertified outer fits in a null, see certificates"
    np.savez_compressed(f"{OUT}/null_draws.npz", **{f"{f}__{k}__{c}__{fit}": v for (f, k, c, fit), v in null.items()},
                        **{f"labels__{k}": v for k, v in perms.items()})
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json} | {res['status']}"); logf.close()


if __name__ == "__main__":
    main()
