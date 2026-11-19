"""Family-size Monte Carlo resolution and refit nulls for the valence and cross-dataset tests.

Usage:  python -m affectdyn.inference.family_resolution_refit_nulls family
        python -m affectdyn.inference.family_resolution_refit_nulls refitnull [--procs 10]
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse, json, sys, time                                                     # noqa: E402
import numpy as np                                                                   # noqa: E402


OUT = "runs/family_resolution_refit_nulls"
SEED = 20260958
N_PERM = 2_000_000
BATCH = 20_000
R_NULL = 1_000
ALPHA = 0.05
CELLS = [("aw2_image", "dinov2_K2"), ("aw2_image", "dinov2_K4"), ("aw2_image", "dinov2_K8"),
         ("aw2_image", "dinov2_K16"), ("aw2_image", "clip_K16"), ("aw2_video", "vjepa_K16"),
         ("aw2_video", "videomae_K16"), ("aw2_video", "videomaev2_K16"), ("unbc", "vjepa_K16")]
PREF = {"unbc": "unbc__", "aw2_image": "aw2img__", "aw2_video": "aw2vid__"}
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-09-22 — W56-W59 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMEND = "## 2026-09-22 — W56-W59 CRITERIA AMENDMENT"


def family():
    import affectdyn.reporting.make_tables as mt
    from affectdyn.inference.calibrated_probability_mc_bounds import clopper_pearson
    from affectdyn.features.unbc_pretraining_comparison import _auc
    from affectdyn.provenance import run_provenance
    Z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    keep = Z["unbc__keep"].astype(bool); y = Z["unbc__y"][keep].astype(int)
    shape = json.load(open("runs/pain_encoder_maps/shape.json")); rows = {r["cell"]: r for r in shape["rows"]}
    cells = sorted(rows)
    S = np.stack([Z[f"unbc__{c}"][keep] for c in cells], 1)                     # Score arrays have shape [n, 36].
    n, n1 = len(y), int(y.sum()); n0 = n - n1
    ranks = np.empty_like(S)
    for j in range(S.shape[1]):
        o = np.argsort(S[:, j], kind="stable"); ranks[o, j] = np.arange(1, n + 1)
    obs = np.array([_auc(S[:, j], y) for j in range(S.shape[1])])
    gate = float(np.max(np.abs(obs - np.array([rows[c]["auc"] for c in cells]))))
    dist = np.abs(obs - 0.5) - 1e-12
    k = np.zeros(len(cells), np.int64); rng = np.random.default_rng(SEED); t0 = time.time()
    for b in range(N_PERM // BATCH):
        idx = np.argsort(rng.random((BATCH, n)), axis=1)[:, :n1]                  # Sample a uniform positive set for each draw.
        a = (ranks[idx].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)                   # AUC arrays have shape [BATCH, 36].
        k += (np.abs(a - 0.5) >= dist).sum(0)
        if (b + 1) % 10 == 0:
            print(f"[{time.time() - t0:6.0f}s] {(b + 1) * BATCH:,} permutations", flush=True)
    p = (k + 1) / (N_PERM + 1)
    M_UNION = len(cells); LEVEL = 1 - ALPHA / M_UNION
    bnd = {c: clopper_pearson(int(k[j]), N_PERM, LEVEL) for j, c in enumerate(cells)}
    pc = dict(zip(cells, p))
    fam, pts = list(mt.IMAGE), []
    for extra in [None] + list(mt.ENTRY_ORDER):
        if extra:
            fam.append(extra)
        cs = [f"{e}_K{kk}" for e in fam for kk in mt.KS if f"{e}_K{kk}" in rows]
        i = cs.index("clip_K4")
        ap = mt._holm_adj([pc[c] for c in cs])[i]
        lo = mt._holm_adj([bnd[c][0] for c in cs])[i]; hi = mt._holm_adj([bnd[c][1] for c in cs])[i]
        cls = "MC-stable significant" if hi < ALPHA else "MC-stable non-significant" if lo >= ALPHA else "MC-unresolved"
        pts.append({"family_size": len(cs), "encoders": list(fam), "holm": ap, "bounds": [lo, hi], "class": cls})
        print(f"m={len(cs):2d} holm {ap:.4f} [{lo:.4f}, {hi:.4f}] {cls}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    res = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRIT, "amendment_heading": AMEND,
           "n_perm": N_PERM, "gate_max_abs_auc_delta_vs_w2": gate, "gate_passes": bool(gate <= 1e-9),
           "clip_K4": {"k": int(k[cells.index("clip_K4")]), "p": float(pc["clip_K4"]),
                       "p_published_20000": rows["clip_K4"]["p"]},
           "p": {c: float(pc[c]) for c in cells}, "points": pts,
           "status": "COMPLETE" if gate <= 1e-9 else "HOLD: gate failed"}
    json.dump(res, open(f"{OUT}/family_mc.json", "w"), indent=1)
    print(f"wrote {OUT}/family_mc.json | {res['status']}")


def _data(corpus, cell):
    from affectdyn.features.temporal_aggregation import image_readout, video_readout
    from affectdyn.probes.affwild_image_fixed_step import grouped_folds
    from affectdyn.inference.loso_permutation_tests import _loso_folds
    if corpus == "unbc":
        lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
        from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES
        e, kk = cell.rsplit("_K", 1)
        path, tmpl = VIDEO_SOURCES[e]; z = np.load(path)
        X = video_readout(z["full"], z[tmpl.format(k=int(kk))])
        sub = lab["subject"]; y = (lab["traj"] == 2).astype(int); mask = lab["traj"] != -1
        f = _loso_folds(sub); fold = np.array([f[s] for s in sub])
        return X, y, mask, sub, fold
    lab = np.load("runs/affwild_clips_features/labels.npz")
    if corpus == "aw2_image":
        e, kk = cell.rsplit("_K", 1)
        X = image_readout(np.load("runs/affwild_clips_features/img_frames.npz")[e], int(kk))
        sub = lab["subject"]; y = (lab["traj"] == 2).astype(int); mask = lab["traj"] != -1
    else:
        idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
        z = np.load(f"runs/affwild_clips_features/{cell.rsplit('_K', 1)[0]}_windows.npz")
        X = video_readout(z["full"], z["K16_up"])
        sub = lab["subject"][idx]; y = (lab["traj"][idx] == 2).astype(int); mask = lab["traj"][idx] != -1
    return X, y, mask, sub, grouped_folds(sub)


def _job(args):
    import affectdyn.probes.nested_probe_fits as w51
    import affectdyn.inference.peak_refit_null_centering as w52
    from affectdyn.features.unbc_pretraining_comparison import _auc
    corpus, cell, kind, jid = args
    X, y, mask, sub, fold = _data(corpus, cell)
    audit = {"inner_splits": 0, "max_inner_grad": 0.0, "max_outer_grad": 0.0, "outer_fits": 0, "outer_uncertified": 0}
    fit = lambda yy: w51.nested_scores(X, yy, sub, mask, fold, (1e-3,), audit, lambda m: None, "w58")[0]  # noqa: E731
    s_obs = fit(y)
    ref = np.load("runs/converged_probe_fits/converged_scores.npz")[PREF[corpus] + cell]
    gate = float(np.nanmax(np.abs(s_obs - ref)))
    k = mask & ~np.isnan(s_obs); obs = float(_auc(s_obs[k], y[k]))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, jid]))
    null = []
    for r in range(R_NULL):
        yp = w52.within_subject_perm(y, mask, sub, rng) if kind == "within" else w52.global_perm(y, mask, rng)
        s = fit(yp); kk = mask & ~np.isnan(s); null.append(float(_auc(s[kk], yp[kk])))
    null = np.array(null)
    return {"corpus": corpus, "cell": cell, "kind": kind, "gate_max_abs_score_delta": gate, "observed_auc": obs,
            "p": w52.tail_p(null, obs), "centring": w52.centring(null, np.random.default_rng([SEED, jid, 7])),
            "certificates": dict(audit), "null": null.tolist()}


def refitnull(procs):
    from multiprocessing import get_context
    from affectdyn.provenance import run_provenance
    import affectdyn.reporting.make_tables as mt
    os.makedirs(OUT, exist_ok=True)
    jobs = [(c, n, kind, j) for j, (kind, (c, n)) in enumerate([(k_, cn) for k_ in ("within", "global") for cn in CELLS])]
    prov = run_provenance(seeds=[SEED], device="cpu"); t0 = time.time()
    out = []
    with get_context("spawn").Pool(procs) as pool:
        for r in pool.imap_unordered(_job, jobs):
            out.append(r)
            print(f"[{time.time() - t0:7.0f}s] {r['kind']:6s} {r['corpus']}:{r['cell']} obs {r['observed_auc']:.4f} "
                  f"p {r['p']:.4f} gate {r['gate_max_abs_score_delta']:.1e} {r['centring']['class']}", flush=True)
    fam = {}
    for kind in ("within", "global"):
        rs = sorted([r for r in out if r["kind"] == kind], key=lambda r: CELLS.index((r["corpus"], r["cell"])))
        adj = mt._holm_adj([r["p"] for r in rs])
        fam[kind] = [{**{x: r[x] for x in ("corpus", "cell", "observed_auc", "p", "centring", "gate_max_abs_score_delta",
                                           "certificates")}, "holm": float(a), "confirmed": bool(a < ALPHA)}
                     for r, a in zip(rs, adj)]
    gate_ok = all(r["gate_max_abs_score_delta"] <= 1e-9 for r in out)
    uncert = sum(r["certificates"]["outer_uncertified"] for r in out)
    res = {"provenance": prov, "criteria_heading": CRIT, "amendment_heading": AMEND, "R": R_NULL,
           "gate_passes": bool(gate_ok), "uncertified_fits": int(uncert), "families": fam,
           "status": "COMPLETE" if gate_ok and uncert == 0 else "HOLD: gate or certificate failed"}
    json.dump(res, open(f"{OUT}/refit_null.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/refit_null_draws.npz", **{f"{r['kind']}__{r['corpus']}__{r['cell']}": np.array(r["null"])
                                                         for r in out})
    print(f"wrote {OUT}/refit_null.json | {res['status']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("part", choices=["family", "refitnull"])
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    family() if a.part == "family" else refitnull(a.procs)
