"""Initial nested-fit analysis using five penalties from 1e-4 to 1.0. The article's nine-penalty nested
fit is implemented in affectdyn.probes.extended_nested_probe_fits.

Usage:  python -m affectdyn.probes.nested_probe_fits [--work-dir DIR]
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "6")
import argparse, json, shutil, sys, tempfile, time
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                              # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                        # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS            # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS                                          # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import grouped_folds                                            # noqa: E402
from affectdyn.probes.converged_probe_fits import f1_pass, w7_pass                            # noqa: E402
from affectdyn.provenance import run_provenance, audit_loso                                   # noqa: E402

OUT = "runs/nested_probe_fits"
SEED = 20260951
GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TOL = 1e-8
MAXIT = 200
N_INNER = 5
ALPHA = 0.05
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W51-W53 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W51-W53 CRITERIA AMENDMENT"


def _sig(z):
    return np.exp(-np.logaddexp(0.0, -z))


def newton_primal(xtr, ytr, lam, th0=None):
    X1 = np.hstack([xtr.astype(np.float64), np.ones((len(xtr), 1))])
    y = ytr.astype(np.float64); n, p = X1.shape
    th = np.zeros(p) if th0 is None else th0.copy()

    def obj(t):
        z = X1 @ t
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * t @ t)
    J = obj(th); stalled = False
    for it in range(MAXIT + 1):
        s = _sig(X1 @ th)
        g = X1.T @ (s - y) / n + lam * th
        if np.abs(g).max() <= TOL or it == MAXIT:
            break
        w = s * (1.0 - s) / n
        if n < p:
            A = X1 * np.sqrt(w)[:, None]
            K = A @ A.T; K[np.diag_indices_from(K)] += lam
            step = -(g - A.T @ np.linalg.solve(K, A @ g)) / lam
        else:
            H = (X1 * w[:, None]).T @ X1; H[np.diag_indices_from(H)] += lam
            step = -np.linalg.solve(H, g)
        slope = float(g @ step); t = 1.0
        for _ in range(50):
            Jn = obj(th + t * step)
            if Jn <= J + 1e-4 * t * slope:
                break
            t /= 2.0
        else:
            stalled = True; break
        th = th + t * step; J = Jn
    s = _sig(X1 @ th); g = X1.T @ (s - y) / n + lam * th
    return th, float(np.abs(g).max()), stalled


def newton_kernel(G, X1, y, lam, a0=None):
    """The same optimum in the representer coordinates (penalty on weight and bias): theta = X1' alpha."""
    n = len(y); y = y.astype(np.float64); a = np.zeros(n) if a0 is None else a0.copy()

    def J(a):
        z = G @ a
        return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * lam * a @ z)
    Jc = J(a)
    for it in range(MAXIT + 1):
        z = G @ a; s = _sig(z); u = (s - y) / n + lam * a
        g = X1.T @ u
        if np.abs(g).max() <= TOL or it == MAXIT:
            break
        D = s * (1.0 - s) / n
        step = -np.linalg.solve(D[:, None] * G + lam * np.eye(n), u)
        slope = float(u @ (G @ step)); t = 1.0
        for _ in range(50):
            Jn = J(a + t * step)
            if Jn <= Jc + 1e-4 * t * slope:
                break
            t /= 2.0
        else:
            break
        a = a + t * step; Jc = Jn
    return a, float(np.abs(X1.T @ ((_sig(G @ a) - y) / n + lam * a)).max())


def standardize(X, tr, rows):
    """Standardize rows with training-row means and standard deviations plus 1e-6; return float32
    features matching the fixed-step preprocessing.
    """
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return ((X[rows] - mu) / sd).astype(np.float32)


def bce(z, y):
    return np.logaddexp(0.0, z) - y * z


def select_lambda(X, y, sub, tr, outer_k, grid, audit):
    subs = np.array(sorted(set(sub[tr].tolist())))
    perm = np.random.default_rng([SEED, outer_k]).permutation(len(subs))
    inner_of = {s: i % N_INNER for i, s in enumerate(subs[perm])}
    inner = np.array([inner_of.get(s, -1) for s in sub])
    dev = {lam: [] for lam in grid}
    for j in range(N_INNER):
        ite = tr & (inner == j); itr = tr & (inner != j) & (inner >= 0)
        assert not set(sub[ite].tolist()) & set(sub[itr].tolist())
        if ite.sum() == 0 or len(set(y[itr].tolist())) < 2:
            continue
        xt = standardize(X, itr, itr).astype(np.float64); xe = standardize(X, itr, ite).astype(np.float64)
        audit["inner_splits"] += 1
        n, p1 = xt.shape[0], xt.shape[1] + 1
        X1 = np.hstack([xt, np.ones((n, 1))]); X1e = np.hstack([xe, np.ones((len(xe), 1))])
        use_kernel = n < p1
        if use_kernel:
            G = X1 @ X1.T; Ge = X1e @ X1.T; a = None
        th = None
        for lam in sorted(grid, reverse=True):                     # Warm-start fits along a descending penalty path.
            if use_kernel:
                a, g = newton_kernel(G, X1, y[itr], lam, a); z = Ge @ a
            else:
                th, g, _ = newton_primal(xt, y[itr], lam, th); z = X1e @ th
            audit["max_inner_grad"] = max(audit["max_inner_grad"], g)
            dev[lam].append(bce(z, y[ite].astype(np.float64)))
    mean_dev = {lam: float(np.concatenate(v).mean()) for lam, v in dev.items()}
    best = min(mean_dev.values())
    chosen = max(lam for lam, v in mean_dev.items() if v == best)
    return chosen, mean_dev


def nested_scores(X, y, sub, mask, fold_of, grid, audit, log, tag):
    scores = np.full(len(y), np.nan); picks = []
    for k in range(int(fold_of.max()) + 1):
        te = (fold_of == k) & mask; tr = (fold_of != k) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        lam, md = (grid[0], {grid[0]: None}) if len(grid) == 1 else select_lambda(X, y, sub, tr, k, grid, audit)
        xt = standardize(X, tr, tr); xe = standardize(X, tr, te)
        th, g, stalled = newton_primal(xt, y[tr], lam)
        audit["max_outer_grad"] = max(audit["max_outer_grad"], g); audit["outer_fits"] += 1
        audit["outer_uncertified"] += int(g > TOL or stalled)
        scores[te] = np.hstack([xe.astype(np.float64), np.ones((te.sum(), 1))]) @ th
        picks.append({"fold": k, "lambda": lam, "boundary": lam in (min(grid), max(grid)) and len(grid) > 1,
                      "deviance": md})
    return scores, picks


def synthetic_selection_test():
    rng = np.random.default_rng([SEED, 99]); out = {}
    n, p = 120, 400
    X = rng.standard_normal((n, p)).astype(np.float32); y = (rng.random(n) < 0.3).astype(int)
    sub = np.repeat(np.arange(24), 5); tr = np.ones(n, bool)
    audit = {"inner_splits": 0, "max_inner_grad": 0.0}
    lam, _ = select_lambda(X, y, sub, tr, 0, GRID, audit); out["noise_n_lt_p"] = {"chosen": lam, "expected": max(GRID)}
    n, p = 2000, 5
    X = rng.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] * 4 + rng.standard_normal(n) * 0.5 > 0).astype(int); sub = np.repeat(np.arange(100), 20)
    tr = np.ones(n, bool)
    lam, _ = select_lambda(X, y, sub, tr, 0, GRID, audit); out["signal_low_dim"] = {"chosen": lam, "expected": min(GRID)}
    out["passes"] = all(v["chosen"] == v["expected"] for k, v in out.items() if isinstance(v, dict))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true"); ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    out_json = f"{OUT}/nested_probe.json"
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    prov = run_provenance(seeds=[SEED], device="cpu")                # Capture provenance before writing outputs.
    logf = open(f"{OUT}/nested_probe.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    u_sub = lab["subject"]; folds = _loso_folds(u_sub); audit_loso(u_sub, folds)
    u_fold = np.array([folds[s] for s in u_sub])
    yd = (lab["direction"] == 1).astype(int); md = lab["direction"] != -1
    ys = (lab["traj"] == 2).astype(int); ms = lab["traj"] != -1
    IMG = np.load("runs/temporal_aggregation/img_frames.npz"); UX = {}
    for e in IMG_ENCODERS:
        for k in KS:
            UX[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for vname, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            if tmpl.format(k=k) in z.files:
                UX[f"{vname}_K{k}"] = video_readout(z["full"], z[tmpl.format(k=k)])
    F1_CELLS = [n for n in UX if n.startswith("clip_") or not n.split("_K")[0] in IMG_ENCODERS]
    M = json.load(open("runs/pain_encoder_maps/map.json")); f1_rows = [r for r in M["primary"] if r["a"] in UX and r["b"] in UX]
    alab = np.load("runs/affwild_clips_features/labels.npz")
    a_sub = alab["subject"]; a_y = (alab["traj"] == 2).astype(int); a_mask = alab["traj"] != -1
    a_fold = grouped_folds(a_sub)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    v_sub = a_sub[idx]; v_y = (alab["traj"][idx] == 2).astype(int); v_mask = alab["traj"][idx] != -1
    v_fold = grouped_folds(v_sub)
    W7J = json.load(open("runs/peak_cluster_inference/cluster.json")); order = [(c["corpus"], c["cell"]) for c in W7J["cells"]]
    S7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    pref = {"unbc": "unbc__", "aw2_image": "aw2img__", "aw2_video": "aw2vid__"}
    u_keep = S7["unbc__keep"]
    corpus_data = {"unbc": (ys, u_keep, u_sub), "aw2_image": (a_y, a_mask, a_sub), "aw2_video": (v_y, v_mask, v_sub)}
    saved7 = {c: {n: S7[pref[c] + n] for cc, n in order if cc == c} for c in pref}
    S1 = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz"); saved1 = {n: S1[n] for n in F1_CELLS}; keep1 = S1["keep"]
    C46 = np.load("runs/converged_probe_fits/converged_scores.npz")
    frames_aw2 = np.load("runs/affwild_clips_features/img_frames.npz")

    def X_of(corpus, name):
        if corpus in ("unbc", "dir"):
            return UX[name]
        if corpus == "aw2_image":
            e, k = name.rsplit("_K", 1); return image_readout(frames_aw2[e], int(k))
        z = np.load(f"runs/affwild_clips_features/{name.rsplit('_K', 1)[0]}_windows.npz"); return video_readout(z["full"], z["K16_up"])

    spec = [("dir", n, yd, md, u_sub, u_fold) for n in F1_CELLS] + \
           [(c, n, *{"unbc": (ys, ms, u_sub, u_fold), "aw2_image": (a_y, a_mask, a_sub, a_fold),
                     "aw2_video": (v_y, v_mask, v_sub, v_fold)}[c]) for c, n in order]

    def ref_key(corpus, name):
        return f"f1__{name}" if corpus == "dir" else f"{pref[corpus]}{name}"

    def run_all(grid, tag):
        audit = {"inner_splits": 0, "max_inner_grad": 0.0, "max_outer_grad": 0.0, "outer_fits": 0,
                 "outer_uncertified": 0}
        scores, picks = {}, {}
        for corpus, name, y, mask, sub, fold_of in spec:
            t1 = time.time()
            sc, pk = nested_scores(X_of(corpus, name), y, sub, mask, fold_of, grid, audit, log, tag)
            scores[(corpus, name)] = sc; picks[(corpus, name)] = pk
            log(f"  {tag} {corpus:9s} {name:18s} lambdas {[p['lambda'] for p in pk][:8]}{'...' if len(pk) > 8 else ''} "
                f"({time.time() - t1:.0f}s)")
        return scores, picks, audit

    g1b = synthetic_selection_test(); log(f"G1b synthetic selection: {g1b}")
    W9 = {c["label"]: c for c in json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))["contrasts"]}
    r1 = f1_pass(saved1, keep1, yd, u_sub, f1_rows)
    w9_surv = sorted(l for l, c in W9.items() if c["holm"]["boot_subject"] < ALPHA)
    g3_f1 = all(r["p"] == W9[r["label"]]["subject_boot"]["p_boot"] for r in r1) and \
        sorted(r["label"] for r in r1 if r["survives"]) == w9_surv
    c7, p7, states7 = w7_pass(saved7, order, corpus_data)
    L = {(c["corpus"], c["cell"]): c for c in W7J["cells"]}
    g3_f7 = all(c["p_boot_subject"] == L[(c["corpus"], c["cell"])]["subject"]["p_boot"] for c in c7)
    log(f"G3 replays: F1 {g3_f1} F2/F3 {g3_f7}")
    s1, _, a1 = run_all((1e-3,), "G1")
    g1d = max(float(np.nanmax(np.abs(s1[k] - C46[ref_key(*k)]))) for k in s1)
    g1 = {"max_abs_score_delta_vs_w46": g1d, "audit": a1, "passes": bool(g1d <= 1e-9 and a1["outer_uncertified"] == 0)}
    log(f"G1 one-point grid vs W46: {g1}")
    gates = {"G1": g1, "G1b": g1b, "G3": {"f1": bool(g3_f1), "f2f3": bool(g3_f7)}}
    if not (g1["passes"] and g1b["passes"] and g3_f1 and g3_f7):
        json.dump({"provenance": prov, "gates": gates, "status": "HOLD: gate failed"}, open(out_json, "w"), indent=2)
        log("HOLD"); logf.close(); return

    sN, picksN, aN = run_all(GRID, "nested")
    g2 = {"audit": aN, "passes": aN["outer_uncertified"] == 0}
    log(f"G2 outer certificates: {g2}")
    nest1 = {n: sN[("dir", n)] for n in F1_CELLS}
    nest7 = {c: {n: sN[(c, n)] for cc, n in order if cc == c} for c in pref}
    rn1 = f1_pass(nest1, keep1, yd, u_sub, f1_rows)
    c7n, p7n, _ = w7_pass(nest7, order, corpus_data, states7)
    W46 = json.load(open("runs/converged_probe_fits/convergence.json"))["M3"]
    conv1 = {r["label"]: r for r in W46["F1_converged"]}
    conv7 = {(c["corpus"], c["cell"]): c for c in W46["F2F3_converged"]}
    ch1 = [{"label": a["label"], "published": a["survives"], "nested": b["survives"],
            "converged_w46": conv1[a["label"]]["survives"], "gap_nested": b["gap"], "holm_nested": b["holm"]}
           for a, b in zip(r1, rn1) if a["survives"] != b["survives"] or conv1[a["label"]]["survives"] != b["survives"]]
    key = lambda c: (c["significant"], c["direction"] if c["significant"] else None)          # noqa: E731
    ch7 = [{"test": f"{a['corpus']}:{a['cell']}", "published": key(a), "nested": key(b),
            "converged_w46": key(conv7[(a["corpus"], a["cell"])]), "auc_published": a["auc"], "auc_nested": b["auc"],
            "holm_nested": b["holm_boot_subject"]}
           for a, b in zip(c7, c7n) if key(a) != key(b) or key(conv7[(a["corpus"], a["cell"])]) != key(b)]
    log(f"F1 survivors published {sum(r['survives'] for r in r1)} nested {sum(r['survives'] for r in rn1)}; changes {ch1}")
    log(f"F2/F3 changes {ch7}")

    npz = f"{OUT}/nested_scores.npz"
    np.savez_compressed(npz, keep_f1=keep1, **{f"f1__{n}": v for n, v in nest1.items()},
                        **{f"{pref[c]}{n}": v for c in nest7 for n, v in nest7[c].items()})
    import affectdyn.inference.converged_clustered_calibration as w49
    work = os.path.abspath(args.work_dir or tempfile.mkdtemp(prefix="w51_"))
    assert not work.startswith(os.path.abspath(".") + os.sep)
    pub = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))
    jcal, consumed = w49.run_pass(work, "nested_primary", True, False, np.load(npz), pub["calibration"]["parameters"])
    g2cal = w49.compare(jcal, pub, ["w8_order_controls_4", "w11_exploratory_5"])
    cal = {}
    for f in ("direction_contrasts_24", "unbc_shape_36", "aw2_shape_18", "cross_corpus_18"):
        P = {r["test"]: r for r in pub["families"][f]["rows"]}
        cal[f] = {"n_significant_published": pub["families"][f]["n_significant_calibrated"],
                  "n_significant_nested": jcal["families"][f]["n_significant_calibrated"],
                  "changes_vs_w12": [{"test": r["test"], "published": P[r["test"]]["decision_calibrated"],
                                      "nested": r["decision_calibrated"], "holm_nested": r["holm_calibrated"]}
                                     for r in jcal["families"][f]["rows"]
                                     if r["decision_calibrated"] != P[r["test"]]["decision_calibrated"]],
                  "mc": w49.mc(jcal["families"][f])}
        log(f"  calibrated {f}: {cal[f]['n_significant_published']} -> {cal[f]['n_significant_nested']} "
            f"{[c['test'] for c in cal[f]['changes_vs_w12']]}")
    shutil.copy(os.path.join(work, "nested_primary", "calibrated_recheck.json"), f"{OUT}/nested_calibrated_recheck.json")

    aucs = {}
    for corpus, name, y, mask, sub, fo in spec:
        sc = sN[(corpus, name)]
        m = (keep1 if corpus == "dir" else corpus_data[corpus][1]) & ~np.isnan(sc)
        aucs[f"{corpus}:{name}"] = float(_auc(sc[m], y[m]))
    lam_counts = {}
    for k, pk in picksN.items():
        for p_ in pk:
            lam_counts[p_["lambda"]] = lam_counts.get(p_["lambda"], 0) + 1
    res = {"provenance": prov, "criteria_heading": CRITERIA_HEADING, "amendment_heading": AMENDMENT_HEADING,
           "grid": list(GRID), "n_inner": N_INNER, "gates": {**gates, "G2": g2, "G2_calibrated_controls_mismatches": g2cal},
           "lambda_selection_counts": {str(k): v for k, v in sorted(lam_counts.items())},
           "boundary_selections": sum(p_["boundary"] for pk in picksN.values() for p_ in pk),
           "picks": {f"{c}:{n}": pk for (c, n), pk in picksN.items()},
           "aucs": aucs,
           "F1_nested": rn1, "F1_survivors": {"published": sum(r["survives"] for r in r1),
                                              "nested": sum(r["survives"] for r in rn1)},
           "F1_changes": ch1, "F2F3_nested": c7n, "F2F3_changes": ch7, "cross_corpus_nested": p7n,
           "calibrated": cal, "n_arrays_substituted": len(consumed),
           "status": "COMPLETE" if (g2["passes"] and not g2cal) else "HOLD: G2 failed"}
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json} | {res['status']}"); logf.close()


if __name__ == "__main__":
    main()
