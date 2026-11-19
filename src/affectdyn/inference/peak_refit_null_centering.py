"""Refit null centring and a subject-stratified AUC.

Usage:  python -m affectdyn.inference.peak_refit_null_centering
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, json, sys, time
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                              # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds, _oof_scores                           # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP     # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS                                          # noqa: E402
from affectdyn.analyses.position.dinov2_startframe_strata import mw_direct                               # noqa: E402
from affectdyn.provenance import run_provenance, audit_loso                                   # noqa: E402

OUT = "runs/peak_refit_null_centering"
SEED = 20260952
LAM = 1e-3
TOL = 1e-8
MAXIT = 200
R = 200
R_SPOT = 50
B_MED = 2_000
B_STRAT = 10_000
SPOT = ("clip_K4", "xclip_K8", "timesformer_K2")
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W51-W53 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W51-W53 CRITERIA AMENDMENT"


def _sig(z):
    return np.exp(-np.logaddexp(0.0, -z))


class FoldKernel:

    def __init__(self, X, tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6                 # Match the float32 normalization used for significance tests.
        xt = ((X[tr] - mu) / sd).astype(np.float32).astype(np.float64)
        xe = ((X[te] - mu) / sd).astype(np.float32).astype(np.float64)
        self.X1 = np.hstack([xt, np.ones((len(xt), 1))])
        X1e = np.hstack([xe, np.ones((len(xe), 1))])
        self.G = self.X1 @ self.X1.T
        self.Ge = X1e @ self.X1.T
        self.n = len(xt)

    def fit(self, y):
        """Newton in the representer coordinates; returns test scores and the primal certificate."""
        G, n = self.G, self.n
        y = y.astype(np.float64)
        a = np.zeros(n)

        def J(a):
            z = G @ a
            return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * LAM * a @ z)

        Jc = J(a)
        for it in range(MAXIT + 1):
            z = G @ a; s = _sig(z)
            u = (s - y) / n + LAM * a
            g = self.X1.T @ u
            if np.abs(g).max() <= TOL or it == MAXIT:
                break
            D = s * (1.0 - s) / n
            step = -np.linalg.solve(D[:, None] * G + LAM * np.eye(n), u)
            slope = float(u @ (G @ step)); t = 1.0                  # Compute J's directional derivative in alpha.
            for _ in range(50):
                Jn = J(a + t * step)
                if Jn <= Jc + 1e-4 * t * slope:
                    break
                t /= 2.0
            else:
                break
            a = a + t * step; Jc = Jn
        return self.Ge @ a, float(np.abs(g).max())


def within_subject_perm(y, mask, sub, rng):
    yp = y.copy()
    for s in np.unique(sub[mask]):
        ii = np.flatnonzero(mask & (sub == s))
        yp[ii] = y[ii][rng.permutation(len(ii))]
    return yp


def global_perm(y, mask, rng):
    yp = y.copy(); ii = np.flatnonzero(mask)
    yp[ii] = y[ii][rng.permutation(len(ii))]
    return yp


def tail_p(null, obs):
    R_ = len(null)
    return float(min(1.0, 2 * min((1 + (null >= obs).sum()) / (R_ + 1), (1 + (null <= obs).sum()) / (R_ + 1))))


def centring(null, rng):
    meds = np.array([np.median(null[rng.integers(0, len(null), len(null))]) for _ in range(B_MED)])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    cls = ("negligible" if 0.49 <= lo and hi <= 0.51 else "biased low" if hi < 0.49 else
           "biased high" if lo > 0.51 else "unresolved")
    return {"mean": float(null.mean()), "median": float(np.median(null)),
            "p2.5_p97.5": [float(x) for x in np.percentile(null, [2.5, 97.5])],
            "share_below_half": float((null < 0.5).mean()), "median_ci": [float(lo), float(hi)], "class": cls}


def stratified(s, y, sub, keep, rng):
    """Within-subject pairs only, ties one half; bootstrap weighting each subject LINEARLY by its multiplicity."""
    subs = np.unique(sub[keep]); C = np.zeros(len(subs)); P = np.zeros(len(subs))
    for j, g in enumerate(subs):
        ii = np.flatnonzero(keep & (sub == g)); pos = ii[y[ii] == 1]; neg = ii[y[ii] == 0]
        if len(pos) and len(neg):
            d = s[pos][:, None] - s[neg][None, :]
            C[j] = (d > 0).sum() + 0.5 * (d == 0).sum(); P[j] = len(pos) * len(neg)
    mixed = int((P > 0).sum())
    if mixed == 0:
        return {"status": "NOT ESTIMABLE", "mixed_subjects": 0}
    draws = rng.integers(0, len(subs), (B_STRAT, len(subs)))
    W = np.zeros((B_STRAT, len(subs))); np.add.at(W, (np.repeat(np.arange(B_STRAT), len(subs)), draws.ravel()), 1)
    num, den = W @ C, W @ P
    ok = den > 0; v = num[ok] / den[ok]
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = float(min(1.0, max(2 * min((v <= 0.5).mean(), (v >= 0.5).mean()), 1.0 / len(v))))
    yk = y[keep]; n1 = int(yk.sum()); n0 = int(len(yk) - n1)
    return {"auc_within_subject": float(C.sum() / P.sum()), "ci": [float(lo), float(hi)], "p_boot": p,
            "mixed_subjects": mixed, "within_pairs": int(P.sum()),
            "share_of_all_pairs": float(P.sum() / (n1 * n0)), "invalid_resamples": int((~ok).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/refit_null.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/refit_null.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(1)
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]; folds = _loso_folds(sub); audit_loso(sub, folds)
    ys = (lab["traj"] == 2).astype(int); ms = lab["traj"] != -1
    IMG = np.load("runs/temporal_aggregation/img_frames.npz"); X = {}
    for e in IMG_ENCODERS:
        for k in KS:
            X[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for vname, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            if tmpl.format(k=k) in z.files:
                X[f"{vname}_K{k}"] = video_readout(z["full"], z[tmpl.format(k=k)])
    cells = [c for c in X]
    assert len(cells) == 36
    S7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz"); ukeep = S7["unbc__keep"].astype(bool)
    C46 = np.load("runs/converged_probe_fits/converged_scores.npz")

    nf = max(folds.values()) + 1
    fold_sets = []
    for k in range(nf):
        te = np.array([folds[s] == k for s in sub]) & ms
        tr = np.array([folds[s] != k for s in sub]) & ms
        if te.sum() == 0 or len(set(ys[tr].tolist())) < 2:
            continue
        fold_sets.append((tr, te))
    log(f"W52 | 36 UNBC shape cells, {len(fold_sets)} evaluable LOSO folds, lambda {LAM}")

    kern = {}
    for c in cells:
        kern[c] = [FoldKernel(X[c], tr, te) for tr, te in fold_sets]
    log("  Gram matrices built")

    def pipeline(c, y):
        sc = np.full(len(y), np.nan); worst = 0.0
        for (tr, te), K in zip(fold_sets, kern[c]):
            s_te, g = K.fit(y[tr]); sc[te] = s_te; worst = max(worst, g)
        return sc, worst

    obs, gate_d, gmax = {}, {}, 0.0
    for c in cells:
        sc, g = pipeline(c, ys); gmax = max(gmax, g)
        obs[c] = float(_auc(sc[ukeep], ys[ukeep]))
        ref = float(_auc(C46[f"unbc__{c}"][ukeep], ys[ukeep]))
        gate_d[c] = abs(obs[c] - ref)
    g_a = {"max_abs_auc_delta_vs_w46": max(gate_d.values()), "max_grad_inf": gmax,
           "passes": bool(max(gate_d.values()) <= 1e-9 and gmax <= TOL)}
    log(f"gate (a): {g_a}")
    s0 = S7["unbc__clip_K4"]; one = np.zeros(len(sub), dtype=object); one[:] = "all"
    st = stratified(s0, ys, one, ukeep, np.random.default_rng(0))
    u, d = mw_direct(s0[ukeep], ys[ukeep])
    g_c = {"one_block_auc": st["auc_within_subject"], "mann_whitney": u / d,
           "passes": bool(abs(st["auc_within_subject"] - u / d) <= 1e-12)}
    log(f"gate (c): {g_c}")
    t1 = time.time(); pipeline("clip_K4", within_subject_perm(ys, ms, sub, np.random.default_rng(1)))
    log(f"  timing pilot: one permutation of one cell {time.time() - t1:.2f}s")
    if not (g_a["passes"] and g_c["passes"]):
        json.dump({"provenance": run_provenance(device="cpu"), "gates": {"a": g_a, "c": g_c},
                   "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        log(f"{'HOLD'}"); logf.close(); return

    rng_w, rng_g = np.random.default_rng(SEED), np.random.default_rng([SEED, 1])
    perms_w = [within_subject_perm(ys, ms, sub, rng_w) for _ in range(R)]
    perms_g = [global_perm(ys, ms, rng_g) for _ in range(R)]
    null_w = np.zeros((len(cells), R)); null_g = np.zeros((len(cells), R))
    for r in range(R):
        for i, c in enumerate(cells):
            sc, _ = pipeline(c, perms_w[r]); null_w[i, r] = _auc(sc[ukeep], perms_w[r][ukeep])
            sc, _ = pipeline(c, perms_g[r]); null_g[i, r] = _auc(sc[ukeep], perms_g[r][ukeep])
        if r % 20 == 0:
            log(f"  null permutation {r + 1}/{R}")
    rng_b = np.random.default_rng([SEED, 2])
    res_a = {}
    for i, c in enumerate(cells):
        res_a[c] = {"observed_converged_auc": obs[c],
                    "within_subject": {**centring(null_w[i], rng_b), "p_two_sided": tail_p(null_w[i], obs[c])},
                    "global": {**centring(null_g[i], rng_b), "p_two_sided": tail_p(null_g[i], obs[c])}}
    cls = {k: sum(res_a[c]["within_subject"]["class"] == k for c in cells)
           for k in ("negligible", "biased low", "biased high", "unresolved")}
    log(f"(a) within-subject null centring classes: {cls}")

    # Check Adam fits on the same permutations.
    spot = {}
    for c in SPOT:
        i = cells.index(c); adam = np.zeros(R_SPOT)
        for r in range(R_SPOT):
            acc = None
            for rep in range(N_REP):
                torch.manual_seed(2000 + rep)
                s_ = _oof_scores(X[c], perms_w[r], sub, ms, folds)
                acc = s_ if acc is None else acc + s_
            sc = acc / N_REP
            adam[r] = _auc(sc[ukeep], perms_w[r][ukeep])
        newt = null_w[i, :R_SPOT]
        rb = np.random.default_rng([SEED, 4]); diffs = []
        for _ in range(B_MED):
            j = rb.integers(0, R_SPOT, R_SPOT); diffs.append(np.median(adam[j]) - np.median(newt[j]))
        spot[c] = {"adam_null_median": float(np.median(adam)), "newton_null_median": float(np.median(newt)),
                   "difference_median": float(np.median(adam) - np.median(newt)),
                   "difference_ci": [float(x) for x in np.percentile(diffs, [2.5, 97.5])],
                   "adam_null_p2.5_p97.5": [float(x) for x in np.percentile(adam, [2.5, 97.5])]}
        log(f"  spot {c}: {spot[c]}")

    rng_s = np.random.default_rng([SEED, 3]); res_c = {"unbc_shape": {}, "unbc_direction": {}, "emotion": {}}
    for c in cells:
        res_c["unbc_shape"][c] = stratified(S7[f"unbc__{c}"], ys, sub, ukeep, rng_s)
    S1 = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz"); k1 = S1["keep"].astype(bool)
    yd = (lab["direction"] == 1).astype(int)
    for c in [n for n in S1.files if n not in ("keep", "y", "subject")]:
        res_c["unbc_direction"][c] = stratified(S1[c], yd, sub, k1, rng_s)
    a_sub, a_y, a_m = S7["aw2img__subject"], S7["aw2img__y"].astype(int), S7["aw2img__mask"].astype(bool)
    res_c["emotion"]["dinov2_K2"] = stratified(S7["aw2img__dinov2_K2"], a_y, a_sub,
                                               a_m & ~np.isnan(S7["aw2img__dinov2_K2"]), rng_s)
    v_sub, v_y, v_m = S7["aw2vid__subject"], S7["aw2vid__y"].astype(int), S7["aw2vid__mask"].astype(bool)
    res_c["emotion"]["vjepa_K16"] = stratified(S7["aw2vid__vjepa_K16"], v_y, v_sub,
                                               v_m & ~np.isnan(S7["aw2vid__vjepa_K16"]), rng_s)
    log(f"(c) emotion: {res_c['emotion']}")

    np.savez_compressed(f"{OUT}/refit_null_draws.npz", cells=np.array(cells), null_within=null_w, null_global=null_g)
    res = {"provenance": run_provenance(seeds=[SEED, 2000], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "amendment_heading": AMENDMENT_HEADING, "lambda": LAM, "R": R, "R_spot": R_SPOT,
           "gates": {"a": g_a, "c": g_c}, "a_refit_null": res_a, "a_centring_classes": cls,
           "a_spot_check_adam": spot, "c_within_subject_auc": res_c,
           "note": "descriptive: the minimum attainable p (1/201) exceeds 0.05/36, so no Holm verdict is drawn",
           "status": "COMPLETE"}
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json}"); logf.close()


if __name__ == "__main__":
    main()
