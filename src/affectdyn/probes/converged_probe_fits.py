"""Converged probe fit and its effect on the decisions.

Usage:  python -m affectdyn.probes.converged_probe_fits
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

N_THREADS = 6
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_THREADS))

import argparse, json, sys, time, zlib
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                              # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                        # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP     # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                                      # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS                                          # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, N_SEED                                    # noqa: E402
from affectdyn.inference.peak_cluster_inference import (perm_test, bootstrap, diff_bootstrap, torch_cpu_only,  # noqa: E402
                              NPERM, NBOOT, SEED as W7_SEED)
from affectdyn.analyses.sensitivity.fixed_step_penalty_sweep import paired_bootstrap                        # noqa: E402
from affectdyn.provenance import run_provenance, audit_loso                                   # noqa: E402

OUT = "runs/converged_probe_fits"
W9_SEED = 20260911
SEED = 20260946
ALPHA = 0.05
LAM = 1e-3
TOL_GRAD = 1e-8
MAXIT = 200
ARMIJO_C = 1e-4
MAX_HALVINGS = 50
AUC_TOL = 0.005
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W45-W46 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W45-W46 CRITERIA AMENDMENT"


def adam_unbc(X, y, sub, mask, folds, cap):
    """stage1d_significance._oof_scores verbatim (== stage_w32 oof_scores at 1e-3, 300), plus capture."""
    import torch, torch.nn as nn
    scores = np.full(len(y), np.nan, np.float32)
    nfold = max(folds.values()) + 1
    for k in range(nfold):
        te = np.array([folds[s] == k for s in sub]) & mask
        tr = np.array([folds[s] != k for s in sub]) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        xt = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        clf = nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        xe = torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        with torch.no_grad():
            scores[te] = clf(xe).squeeze(-1).numpy()
        cap(k, te, xt, y[tr], xe, clf)
    return scores


def adam_aw2(X, y, assign, mask, seed, cap):
    """stage_w3_probe.oof verbatim on the CPU, plus capture."""
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
        opt = torch.optim.Adam(clf.parameters(), 1e-2, weight_decay=1e-3)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(clf(xt).squeeze(-1), yt)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            scores[te] = clf(xe).squeeze(-1).cpu().numpy()
        cap(k, te, xt, y[tr], xe, clf)
    return scores


def _sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def objective(X1, y, th):
    z = X1 @ th
    return float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * LAM * th @ th)


def newton(xtr32, ytr):
    """Minimize mean logistic loss + LAM/2*(||w||^2+b^2) in float64 from zero. Input features must
    already be standardized using training-fold statistics. Return fitted parameters and convergence
    diagnostics.
    """
    X1 = np.hstack([xtr32.astype(np.float64), np.ones((len(xtr32), 1))])
    y = ytr.astype(np.float64); n, p = X1.shape
    th = np.zeros(p); J = objective(X1, y, th)
    rec = {"n": n, "p": p, "iterations": 0, "stalled": False}
    for it in range(MAXIT + 1):
        s = _sigmoid(X1 @ th)
        g = X1.T @ (s - y) / n + LAM * th
        if np.abs(g).max() <= TOL_GRAD:
            break
        if it == MAXIT:
            break
        w = s * (1.0 - s) / n
        if n < p:                                            # Use the Woodbury identity when samples are fewer than features.
            A = X1 * np.sqrt(w)[:, None]
            K = A @ A.T; K[np.diag_indices_from(K)] += LAM
            step = -(g - A.T @ np.linalg.solve(K, A @ g)) / LAM
        else:
            H = (X1 * w[:, None]).T @ X1; H[np.diag_indices_from(H)] += LAM
            step = -np.linalg.solve(H, g)
        slope = float(g @ step); t = 1.0
        for _ in range(MAX_HALVINGS):
            Jn = objective(X1, y, th + t * step)
            if Jn <= J + ARMIJO_C * t * slope:
                break
            t /= 2.0
        else:
            rec["stalled"] = True
            break
        th = th + t * step; J = Jn; rec["iterations"] = it + 1
    s = _sigmoid(X1 @ th); g = X1.T @ (s - y) / n + LAM * th
    rec.update({"grad_inf": float(np.abs(g).max()), "J_star": objective(X1, y, th),
                "objective_error_bound": float(g @ g / (2 * LAM)), "theta_norm": float(np.linalg.norm(th)),
                "certified": bool(np.abs(g).max() <= TOL_GRAD)})
    return th, rec, X1, y


class Cell:

    def __init__(self, n):
        self.folds = {}; self.n = n

    def capture(self, seed_idx):
        def cap(k, te, xt, ytr, xe, clf):
            f = self.folds.setdefault(k, {"te": te, "xt": xt.numpy().copy(), "ytr": np.asarray(ytr),
                                          "xe": xe.numpy().copy(), "adam": []})
            assert np.array_equal(f["te"], te)
            th = np.concatenate([clf.weight.detach().numpy().ravel(), clf.bias.detach().numpy().ravel()])
            f["adam"].append(th.astype(np.float64))
        return cap

    def solve(self):
        conv = np.full(self.n, np.nan, np.float64); recs = []
        for k in sorted(self.folds):
            f = self.folds[k]
            th, rec, X1, y = newton(f["xt"], f["ytr"])
            Js = rec["J_star"]; tn = max(rec["theta_norm"], 1e-300)
            rec["fold"] = int(k)
            rec["adam_rel_excess"] = [float((objective(X1, y, a) - Js) / Js) for a in f["adam"]]
            rec["adam_rel_param_dist"] = [float(np.linalg.norm(a - th) / tn) for a in f["adam"]]
            rec["adam_grad_inf"] = [float(np.abs(X1.T @ (_sigmoid(X1 @ a) - y) / len(y) + LAM * a).max())
                                    for a in f["adam"]]
            conv[f["te"]] = np.hstack([f["xe"].astype(np.float64), np.ones((len(f["xe"]), 1))]) @ th
            recs.append(rec)
            del f["xt"], f["xe"]
        return conv, recs


def f1_pass(scores, keep, y, sub, rows):
    yk, gk = y[keep].astype(np.int64), sub[keep]
    out = []
    for r in rows:
        sa, sb = scores[r["a"]][keep], scores[r["b"]][keep]
        rng = np.random.default_rng([W9_SEED, zlib.crc32((r["label"] + ":subject").encode())])
        b = paired_bootstrap(sa, sb, yk, gk, rng, NBOOT)
        out.append({"label": r["label"], "a": r["a"], "b": r["b"], "gap": float(_auc(sa, yk) - _auc(sb, yk)),
                    "p": b["p_boot"], "ci": b["ci"]})
    for r, a in zip(out, holm([x["p"] for x in out])):
        r["holm"] = float(a); r["significant"] = bool(a < ALPHA); r["survives"] = bool(a < ALPHA and r["gap"] > 0)
    return out


def w7_pass(scores, order, corpus_data, states=None):
    """Replay peak_cluster_inference procedures in the supplied cell order using a fresh generator
    seeded with W7_SEED; optionally verify saved RNG boundaries.
    """
    rng = np.random.default_rng(W7_SEED); seen = []
    cells, kept = [], {}

    def mark():
        st = rng.bit_generator.state["state"]["state"]
        if states is not None:
            assert states[len(seen)] == st, f"rng desynchronized at boundary {len(seen)}"
        seen.append(st)

    for corpus, name in order:
        y, mask, groups = corpus_data[corpus]
        s = scores[corpus][name]
        keep = mask & ~np.isnan(s)
        sk, yk, gk = s[keep], y[keep].astype(np.int64), groups[keep]
        kept[(corpus, name)] = (sk, yk, gk)
        _, pc = perm_test(sk, yk, gk, "clip", rng, NPERM); mark()
        _, ps = perm_test(sk, yk, gk, "subject", rng, NPERM); mark()
        bc = bootstrap(sk, yk, gk, "clip", rng, NBOOT); mark()
        bs = bootstrap(sk, yk, gk, "subject", rng, NBOOT); mark()
        cells.append({"corpus": corpus, "cell": name, "family": "unbc" if corpus == "unbc" else "aw2",
                      "auc": float(_auc(sk, yk)), "p_perm_clip": pc, "p_perm_subject": ps,
                      "p_boot_clip": bc["p_boot"], "p_boot_subject": bs["p_boot"], "ci_subject": bs["ci"]})
    for fam in ("unbc", "aw2"):
        fc = [c for c in cells if c["family"] == fam]
        for c, a in zip(fc, holm([c["p_boot_subject"] for c in fc])):
            c["holm_boot_subject"] = float(a)
            c["significant"] = bool(a < ALPHA)
            c["direction"] = "above_chance" if c["auc"] > 0.5 else "below_chance"
    u = {n for c, n in order if c == "unbc"}
    ai = {n for c, n in order if c == "aw2_image"}
    av = {n for c, n in order if c == "aw2_video"}
    pairs = []
    for name in sorted(u & (ai | av)):
        other = "aw2_image" if name in ai else "aw2_video"
        d = diff_bootstrap(kept[("unbc", name)], kept[(other, name)], rng, NBOOT); mark()
        pairs.append({"cell": name, **{k: d[k] for k in ("diff", "p_boot", "ci")}})
    for p, a in zip(pairs, holm([p["p_boot"] for p in pairs])):
        p["p_holm"] = float(a); p["excludes_zero_after_holm"] = bool(a < ALPHA)
    return cells, pairs, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    out_json = f"{OUT}/convergence.json"
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/convergence.log", "w")
    t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True)
        if logf:
            logf.write(line + "\n"); logf.flush()

    torch = torch_cpu_only()
    log(f"Converged probe fits | torch {torch.__version__} | numpy {np.__version__}")

    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    u_sub = lab["subject"]; folds = _loso_folds(u_sub); audit_loso(u_sub, folds)
    yd = (lab["direction"] == 1).astype(int); md = lab["direction"] != -1
    ys = (lab["traj"] == 2).astype(int); ms = lab["traj"] != -1
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    UX = {}
    for e in IMG_ENCODERS:
        for k in KS:
            UX[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for vname, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            if tmpl.format(k=k) in z.files:
                UX[f"{vname}_K{k}"] = video_readout(z["full"], z[tmpl.format(k=k)])
    M = json.load(open("runs/pain_encoder_maps/map.json"))
    F1_CELLS = [n for n in UX if n.startswith("clip_") or not n.split("_K")[0] in IMG_ENCODERS]
    f1_rows = [r for r in M["primary"] if r["a"] in UX and r["b"] in UX]
    alab = np.load("runs/affwild_clips_features/labels.npz")
    a_sub = alab["subject"]; a_y = (alab["traj"] == 2).astype(int); a_mask = alab["traj"] != -1
    a_assign = grouped_folds(a_sub)
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    v_sub = a_sub[idx]; v_y = (alab["traj"][idx] == 2).astype(int); v_mask = alab["traj"][idx] != -1
    v_assign = grouped_folds(v_sub)

    W7J = json.load(open("runs/peak_cluster_inference/cluster.json"))
    order = [(c["corpus"], c["cell"]) for c in W7J["cells"]]
    S7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    pref = {"unbc": "unbc__", "aw2_image": "aw2img__", "aw2_video": "aw2vid__"}
    u_keep = S7["unbc__keep"]
    corpus_data = {"unbc": (ys, u_keep, u_sub), "aw2_image": (a_y, a_mask, a_sub), "aw2_video": (v_y, v_mask, v_sub)}
    assert np.array_equal(S7["aw2vid__idx"], idx) and np.array_equal(S7["aw2img__mask"], a_mask)
    saved7 = {c: {n: S7[pref[c] + n] for cc, n in order if cc == c} for c in pref}
    S1 = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz")
    saved1 = {n: S1[n] for n in F1_CELLS}; keep1 = S1["keep"]

    W9 = {c["label"]: c for c in json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))["contrasts"]}
    r1 = f1_pass(saved1, keep1, yd, u_sub, f1_rows)
    w9_surv = sorted(l for l, c in W9.items() if c["holm"]["boot_subject"] < ALPHA)
    f1_exact = all(r["p"] == W9[r["label"]]["subject_boot"]["p_boot"] for r in r1) and \
        sorted(r["label"] for r in r1 if r["survives"]) == w9_surv
    log(f"G1a F1 replay of saved scores: exact {f1_exact} | survivors {sum(r['survives'] for r in r1)}")
    c7, p7, states7 = w7_pass(saved7, order, corpus_data)
    L = {(c["corpus"], c["cell"]): c for c in W7J["cells"]}
    f7_exact = all(c["p_perm_clip"] == L[(c["corpus"], c["cell"])]["clip"]["p_perm"]
                   and c["p_perm_subject"] == L[(c["corpus"], c["cell"])]["subject"]["p_perm"]
                   and c["p_boot_clip"] == L[(c["corpus"], c["cell"])]["clip"]["p_boot"]
                   and c["p_boot_subject"] == L[(c["corpus"], c["cell"])]["subject"]["p_boot"]
                   and c["holm_boot_subject"] == L[(c["corpus"], c["cell"])]["holm_pooled"]["boot_subject"]
                   for c in c7)
    LP = {p["cell"]: p for p in W7J["cross_corpus"]}
    pairs_exact = all(p["p_boot"] == LP[p["cell"]]["p_boot"] for p in p7)
    log(f"G1a F2/F3 replay of saved scores: cells exact {f7_exact} | cross-corpus exact {pairs_exact}")
    g1a = {"f1_exact": bool(f1_exact), "f2f3_exact": bool(f7_exact), "cross_corpus_exact": bool(pairs_exact)}


    refit1, conv1, refit7, conv7, fold_recs, disp = {}, {}, {p: {} for p in pref}, {p: {} for p in pref}, {}, {}

    def run_cell(tag, n, fits):
        c = Cell(n); reps = []
        for i, f in enumerate(fits):
            reps.append(f(c.capture(i)))
        conv, recs = c.solve()
        fold_recs[tag] = recs
        R = np.array(reps, dtype=np.float64)
        disp[tag] = {"mean_clip_score_sd": float(np.nanmean(np.nanstd(R, axis=0))) if len(reps) > 1 else 0.0}
        return reps, conv

    torch.set_num_threads(1)
    for name in F1_CELLS:
        def mk(rep, X=UX[name]):
            def f(cap):
                torch.manual_seed(1000 + rep); return adam_unbc(X, yd, u_sub, md, folds, cap)
            return f
        reps, conv = run_cell(f"dir:{name}", len(yd), [mk(r) for r in range(N_REP)])
        refit1[name] = np.mean(reps, axis=0); conv1[name] = conv
        log(f"  F1 {name}: {sum(r['certified'] for r in fold_recs['dir:' + name])}/{len(fold_recs['dir:' + name])} certified")
    for corpus, name in order:
        if corpus == "unbc":
            torch.set_num_threads(1)

            def mk(rep, X=UX[name]):
                def f(cap):
                    torch.manual_seed(2000 + rep); return adam_unbc(X, ys, u_sub, ms, folds, cap)
                return f
            reps, conv = run_cell(f"unbc:{name}", len(ys), [mk(r) for r in range(N_REP)])
            acc = None
            for s in reps:
                acc = s if acc is None else acc + s
            refit7["unbc"][name] = acc / N_REP
        elif corpus == "aw2_image":
            torch.set_num_threads(N_THREADS)
            e, k = name.rsplit("_K", 1)
            X = image_readout(np.load("runs/affwild_clips_features/img_frames.npz")[e], int(k))
            reps, conv = run_cell(f"aw2img:{name}", len(a_y),
                                  [(lambda cap, r=r: adam_aw2(X, a_y, a_assign, a_mask, 100 + r, cap))
                                   for r in range(N_SEED)])
            refit7["aw2_image"][name] = np.nanmean(reps, axis=0); del X
        else:
            torch.set_num_threads(2)
            z = np.load(f"runs/affwild_clips_features/{name.rsplit('_K', 1)[0]}_windows.npz")
            X = video_readout(z["full"], z["K16_up"])
            reps, conv = run_cell(f"aw2vid:{name}", len(v_y),
                                  [(lambda cap, r=r: adam_aw2(X, v_y, v_assign, v_mask, 300 + r, cap))
                                   for r in range(N_REP)])
            refit7["aw2_video"][name] = np.nanmean(reps, axis=0)
        conv7[corpus][name] = conv
        tag = [t for t in fold_recs if t.endswith(":" + name) and t.split(":")[0] == {"unbc": "unbc",
               "aw2_image": "aw2img", "aw2_video": "aw2vid"}[corpus]][0]
        log(f"  {corpus} {name}: {sum(r['certified'] for r in fold_recs[tag])}/{len(fold_recs[tag])} certified")

    uncert = [(t, r["fold"], r["grad_inf"], r["stalled"]) for t, rs in fold_recs.items() for r in rs if not r["certified"]]
    g2 = {"n_folds": sum(len(v) for v in fold_recs.values()), "uncertified": uncert, "passes": not uncert}
    log(f"G2 certificate: {g2['n_folds'] - len(uncert)}/{g2['n_folds']} folds certified")

    for name in F1_CELLS:
        assert np.array_equal(np.isnan(conv1[name]) & md, np.isnan(saved1[name]) & md), name
    for corpus, name in order:
        m = corpus_data[corpus][1]
        assert np.array_equal(np.isnan(conv7[corpus][name]) & m, np.isnan(saved7[corpus][name]) & m), name

    # Compare local Adam refits with saved scores.
    yk1 = yd[keep1].astype(np.int64)
    d1 = {n: abs(float(_auc(refit1[n][keep1], yk1)) - float(_auc(saved1[n][keep1], yk1))) for n in F1_CELLS}
    s1 = {n: float(np.nanmax(np.abs(refit1[n] - saved1[n]))) for n in F1_CELLS}
    rf1 = f1_pass(refit1, keep1, yd, u_sub, f1_rows)
    c7r, p7r, _ = w7_pass(refit7, order, corpus_data, states7)
    d7, s7 = {}, {}
    for corpus, name in order:
        y, m, _ = corpus_data[corpus]
        kk = m & ~np.isnan(saved7[corpus][name])
        d7[f"{corpus}:{name}"] = abs(float(_auc(refit7[corpus][name][kk], y[kk])) -
                                     float(_auc(saved7[corpus][name][kk], y[kk])))
        s7[f"{corpus}:{name}"] = float(np.nanmax(np.abs(refit7[corpus][name] - saved7[corpus][name])))
    dec = lambda cs: [(c["corpus"], c["cell"], c["significant"], c["direction"]) for c in cs]  # noqa: E731
    g1b = {"max_abs_auc_delta_f1": max(d1.values()), "max_abs_score_delta_f1": max(s1.values()),
           "max_abs_auc_delta_f2f3": max(d7.values()), "max_abs_score_delta_f2f3": max(s7.values()),
           "f1_decisions_identical": [r["survives"] for r in rf1] == [r["survives"] for r in r1],
           "f2f3_decisions_identical": dec(c7r) == dec(c7)}
    g1b["passes"] = bool(g1b["max_abs_auc_delta_f1"] <= AUC_TOL and g1b["max_abs_auc_delta_f2f3"] <= AUC_TOL
                         and g1b["f1_decisions_identical"] and g1b["f2f3_decisions_identical"])
    log(f"G1b refit: {g1b}")

    # Apply both family procedures to converged scores.
    rc1 = f1_pass(conv1, keep1, yd, u_sub, f1_rows)
    c7c, p7c, _ = w7_pass(conv7, order, corpus_data, states7)
    m2 = {f"dir:{n}": float(_auc(conv1[n][keep1], yk1) - _auc(saved1[n][keep1], yk1)) for n in F1_CELLS}
    for corpus, name in order:
        y, m, _ = corpus_data[corpus]; kk = m & ~np.isnan(saved7[corpus][name])
        m2[f"{corpus}:{name}"] = float(_auc(conv7[corpus][name][kk], y[kk]) - _auc(saved7[corpus][name][kk], y[kk]))
    ch1 = [{"label": a["label"], "published": a["survives"], "converged": b["survives"], "gap_published": a["gap"],
            "gap_converged": b["gap"], "holm_published": a["holm"], "holm_converged": b["holm"]}
           for a, b in zip(r1, rc1) if a["survives"] != b["survives"]]
    ch7 = [{"test": f"{a['corpus']}:{a['cell']}", "published": [a["significant"], a["direction"]],
            "converged": [b["significant"], b["direction"]], "auc_published": a["auc"], "auc_converged": b["auc"],
            "holm_published": a["holm_boot_subject"], "holm_converged": b["holm_boot_subject"]}
           for a, b in zip(c7, c7c) if (a["significant"], a["direction"] if a["significant"] else None)
           != (b["significant"], b["direction"] if b["significant"] else None)]
    chp = [{"cell": a["cell"], "published": a["excludes_zero_after_holm"], "converged": b["excludes_zero_after_holm"]}
           for a, b in zip(p7, p7c) if a["excludes_zero_after_holm"] != b["excludes_zero_after_holm"]]
    verdict = "DECISION-STABLE" if not (ch1 or ch7) else "OPTIMIZATION-DEPENDENT"

    def summ(prefixes):
        ex = [x for t, rs in fold_recs.items() if t.split(":")[0] in prefixes for r in rs for x in r["adam_rel_excess"]]
        pd_ = [x for t, rs in fold_recs.items() if t.split(":")[0] in prefixes for r in rs for x in r["adam_rel_param_dist"]]
        return {"n": len(ex), "rel_excess_median": float(np.median(ex)), "rel_excess_max": float(np.max(ex)),
                "rel_excess_min": float(np.min(ex)), "rel_param_dist_median": float(np.median(pd_)),
                "rel_param_dist_max": float(np.max(pd_))}
    m1 = {"F1_direction": summ({"dir"}), "F2_unbc_shape": summ({"unbc"}), "F3_aw2_shape": summ({"aw2img", "aw2vid"})}
    log(f"M1 {json.dumps(m1)}")
    log(f"M2 max |AUC delta| {max(abs(v) for v in m2.values()):.4f}")
    log(f"F1 survivors published {sum(r['survives'] for r in r1)} converged {sum(r['survives'] for r in rc1)} changes {ch1}")
    log(f"F2/F3 changes {ch7}")
    log(f"cross-corpus changes {chp}")
    log(f"VERDICT {verdict}")

    np.savez_compressed(f"{OUT}/converged_scores.npz", keep_f1=keep1,
                        **{f"f1__{n}": v for n, v in conv1.items()}, **{f"f1refit__{n}": v for n, v in refit1.items()},
                        **{f"{pref[c]}{n}": v for c in conv7 for n, v in conv7[c].items()},
                        **{f"refit_{pref[c]}{n}": v for c in refit7 for n, v in refit7[c].items()})
    status = "COMPLETE" if (all(g1a.values()) and g1b["passes"] and g2["passes"]) else "HOLD: gate failed"
    res = {"provenance": run_provenance(seeds=[W9_SEED, W7_SEED, SEED], device="cpu"),
           "criteria_heading": CRITERIA_HEADING, "amendment_heading": AMENDMENT_HEADING,
           "objective": "mean BCE + (0.001/2)(||w||^2 + b^2) on the published float32 standardized features, in float64",
           "solver": {"method": "Newton, Woodbury when n < p, Armijo", "tol_grad_inf": TOL_GRAD, "maxit": MAXIT,
                      "armijo_c": ARMIJO_C, "max_halvings": MAX_HALVINGS},
           "gates": {"G1a_replay": g1a, "G1b_refit": g1b, "G2_certificate": g2},
           "M1_shortfall": m1, "M2_auc_delta": m2,
           "M3": {"F1_published": r1, "F1_converged": rc1, "F1_changes": ch1,
                  "F1_survivors": {"published": sum(r["survives"] for r in r1), "converged": sum(r["survives"] for r in rc1)},
                  "F2F3_published": c7, "F2F3_converged": c7c, "F2F3_changes": ch7,
                  "cross_corpus_published": p7, "cross_corpus_converged": p7c, "cross_corpus_changes": chp},
           "adam_seed_dispersion": disp, "fold_records": fold_recs,
           "verdict": verdict if status == "COMPLETE" else None, "status": status}
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json} | {status}")
    logf.close()


if __name__ == "__main__":
    main()
