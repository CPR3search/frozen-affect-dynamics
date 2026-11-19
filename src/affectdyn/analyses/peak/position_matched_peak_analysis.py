"""Peak task on the position-matched Aff-Wild2 set."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")
import argparse, json, math, sys, time                                               # noqa: E402
import numpy as np                                                                   # noqa: E402

import affectdyn.probes.nested_probe_fits as w51                                                 # noqa: E402
if _CVD is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CVD
import affectdyn.inference.peak_cluster_inference as w7                                                        # noqa: E402
import affectdyn.inference.calibrated_clustered_auc as w12                                           # noqa: E402
import affectdyn.inference.calibrated_detection_limits as w18                                             # noqa: E402
from affectdyn.features.unbc_pretraining_comparison import _auc                                               # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout                           # noqa: E402
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof                                        # noqa: E402
from affectdyn.provenance import run_provenance                                                # noqa: E402

OUT = "runs/position_matched_peak_analysis"
SEED = 20260959
IMAGE = ("clip", "dinov2", "siglip")
VIDEO = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")
KS = (2, 4, 8, 16)
NPERM, NBOOT, N_NULL = 20_000, 10_000, 100_000
ALPHA = 0.05
W56_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-09-22 — W56-W59 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMEND = "## 2026-09-22 — W56-W59 CRITERIA AMENDMENT"


def holm(ps):
    """Local Holm adjustment; the shared implementation is
    inference.aggregation_permutation_tests.holm. Family membership and size are specified by the
    caller.
    """
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def fit_scores(X, y, sub, mask, fit, image):
    assign = grouped_folds(sub)
    if fit == "fixed_step":
        seeds = [100 + r for r in range(3)] if image else [300 + r for r in range(5)]
        return np.nanmean([oof(X, y, assign, mask, s, "cpu") for s in seeds], axis=0)
    grid = (1e-3,) if fit == "converged" else W56_GRID
    audit = {"inner_splits": 0, "max_inner_grad": 0.0, "max_outer_grad": 0.0, "outer_fits": 0, "outer_uncertified": 0}
    s, _ = w51.nested_scores(X, y, sub, mask, assign, grid, audit, lambda m: None, "w59")
    return s


def paired_boot(sa, sb, y, g, rng, nboot):
    """Local fixed-score paired cluster bootstrap, returning AUC(a)-AUC(b), a percentile interval and a
    two-sided bootstrap p value. Related shared implementation:
    inference.pain_direction_cluster_inference.paired_bootstrap.
    """
    mem = w7._members(g); G = len(mem); d = []
    for _ in range(nboot):
        i = np.concatenate([mem[k] for k in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            continue
        d.append(_auc(sa[i], y[i]) - _auc(sb[i], y[i]))
    d = np.array(d); obs = _auc(sa, y) - _auc(sb, y)
    p = min(1.0, max(1.0 / len(d), 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {"gap": float(obs), "ci": [float(x) for x in np.percentile(d, [2.5, 97.5])], "p_boot": float(p)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    out_json = f"{OUT}/analysis.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    prov = run_provenance(seeds=[SEED], device="cpu")
    logf = open(f"{OUT}/analysis.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    rng = np.random.default_rng(SEED)
    design = json.load(open(f"{OUT}/matched_subset.json")); assert design["status"] == "FROZEN"
    midx = np.load(f"{OUT}/matched_subset_idx.npy").astype(int)
    lab = np.load("runs/affwild_clips_features/labels.npz")
    y = (lab["traj"][midx] == 2).astype(int); g = lab["subject"][midx]; mask = lab["traj"][midx] != -1
    assert mask.all()
    res = {"provenance": prov, "criteria_heading": CRIT, "amendment_heading": AMEND,
           "n_clips": int(len(midx)), "n_peaked": int(y.sum()), "n_subjects": int(len(set(g.tolist())))}

    # Estimate the MDE before accessing scores.
    d = w12.Design("aw2_matched", y, g, 18)
    mde = w18.mde(d, rng, log)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    within_share = float((g[pos][:, None] == g[neg][None, :]).mean())
    th = mde["mde_auc_at_80pc_power"]; icc = w12.ICC
    from statistics import NormalDist
    nd = NormalDist()
    pooled = (1 - within_share) * th + within_share * nd.cdf(nd.inv_cdf(th) / math.sqrt(1 - icc))
    res["mde"] = {**mde, "within_subject_pair_share": within_share, "mde_pooled_scale": float(pooled)}
    log(f"1. MDE: theta {th:.4f}, pooled scale {pooled:.4f} (within-subject pair share {within_share:.4f})")
    folds_counts = {}
    assign = grouped_folds(g)
    for k in range(assign.max() + 1):
        folds_counts[str(k)] = {"peaked": int(y[assign == k].sum()), "negative": int((y[assign == k] == 0).sum())}
    res["per_fold_class_counts"] = folds_counts

    pub = np.load("runs/affwild_clips_features/video_subset_idx.npy").astype(int); pos_pub = {int(i): k for k, i in enumerate(pub)}
    perms = np.load("runs/affwild_peak_order_controls/perms.npz")["perms"][:, midx]                                  # Permutations have shape [3, n, 16].
    rows = np.arange(len(midx))[:, None]
    status, X, CTRL = {}, {}, {}
    frames = np.load("runs/affwild_clips_features/img_frames.npz")
    for e in IMAGE:
        fr = frames[e][midx]
        for k in KS:
            X[f"{e}_K{k}"] = image_readout(fr, k)
        CTRL[e] = {"static": fr.mean(1), "shuffled": [image_readout(fr[rows, perms[r]], 16) for r in range(3)]}
        status[e] = "tested"
    gate_auc = {}
    for e in VIDEO:
        ej = json.load(open(f"{OUT}/{e}_extract.json"))
        if ej["status"] != "COMPLETE":
            status[e] = f"not tested ({ej['status']})"; continue
        cache = np.load(f"runs/affwild_clips_features/{e}_windows.npz"); new = np.load(f"{OUT}/{e}_ordered_new.npz")
        pos_new = {int(i): k for k, i in enumerate(new["idx"])}
        full = np.stack([cache["full"][pos_pub[i]] if i in pos_pub else new["full"][pos_new[i]] for i in midx])
        W = np.stack([cache["K16_up"][pos_pub[i]] if i in pos_pub else new["K16_up"][pos_new[i]] for i in midx])
        X[f"{e}_K16"] = video_readout(full, W)
        sh = np.load(f"{OUT}/{e}_shuffled_full.npz"); assert np.array_equal(sh["idx"], midx)
        CTRL[e] = {"static": W.mean(1),
                   "shuffled": [video_readout(sh["full_shuf"][r], W[rows, perms[r]]) for r in range(3)]}
        gz = np.load(f"{OUT}/{e}_gate.npz"); gi = gz["idx"]; gr = [pos_pub[int(i)] for i in gi]
        Xc = video_readout(cache["full"][gr], cache["K16_up"][gr]); Xr = video_readout(gz["full"], gz["K16_up"])
        yg = (lab["traj"][gi] == 2).astype(int); sg = lab["subject"][gi]; mg = np.ones(len(gi), bool)
        a_c = _auc(fit_scores(Xc, yg, sg, mg, "fixed_step", False), yg)
        a_r = _auc(fit_scores(Xr, yg, sg, mg, "fixed_step", False), yg)
        gate_auc[e] = {"cached": float(a_c), "reextracted": float(a_r), "abs_delta": float(abs(a_c - a_r)),
                       "passes": bool(abs(a_c - a_r) <= 0.002)}
        if not gate_auc[e]["passes"]:
            status[e] = "not tested (AUC gate failed)"; X.pop(f"{e}_K16"); CTRL.pop(e)
        else:
            status[e] = "tested"
    res["encoder_status"] = status; res["gate_auc"] = gate_auc
    log(f"2. encoders: {status}; AUC gate {gate_auc}")
    cells = [f"{e}_K{k}" for e in IMAGE for k in KS] + [f"{e}_K16" for e in VIDEO]

    z_null = np.abs(w12.sim_single(d, 0.5, rng, N_NULL)[0])
    fam, scores = {}, {}
    for fit in ("fixed_step", "converged", "nested"):
        rows_ = []
        for c in cells:
            if c not in X:
                rows_.append({"cell": c, "tested": False, "p": {k: 1.0 for k in ("perm_clip", "perm_within", "boot_clip",
                                                                                   "boot_subject", "calibrated")}})
                continue
            s = fit_scores(X[c], y, g, mask, fit, c.split("_K")[0] in IMAGE); scores[(fit, c)] = s
            k_ = ~np.isnan(s); sk, yk, gk = s[k_], y[k_], g[k_]
            _, pc = w7.perm_test(sk, yk, gk, "clip", rng, NPERM)
            _, pw = w7.perm_test(sk, yk, gk, "subject", rng, NPERM)
            bc = w7.bootstrap(sk, yk, gk, "clip", rng, NBOOT); bs = w7.bootstrap(sk, yk, gk, "subject", rng, NBOOT)
            auc, z = w12.z_single(sk, yk, gk)
            pcal = float((1 + (z_null >= abs(z)).sum()) / (1 + len(z_null)))
            rows_.append({"cell": c, "tested": True, "auc": float(auc), "ci_subject": bs["ci"], "z": float(z),
                          "p": {"perm_clip": pc, "perm_within": pw, "boot_clip": bc["p_boot"],
                                "boot_subject": bs["p_boot"], "calibrated": pcal}})
        for proc in ("perm_clip", "perm_within", "boot_clip", "boot_subject", "calibrated"):
            for r_, a in zip(rows_, holm([r_["p"][proc] for r_ in rows_])):
                r_.setdefault("holm", {})[proc] = float(a)
        fam[fit] = rows_
        log(f"3. {fit}: " + "; ".join(f"{r_['cell']} {r_.get('auc', float('nan')):.3f} "
                                      f"boot {r_['holm']['boot_subject']:.4f} cal {r_['holm']['calibrated']:.4f}"
                                      for r_ in rows_ if r_["tested"]))
    res["shape_family"] = fam

    W7Z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz"); ukeep = W7Z["unbc__keep"].astype(bool)
    uy = W7Z["unbc__y"].astype(int); ug = W7Z["unbc__subject"]
    usrc = {"fixed_step": lambda c: W7Z[f"unbc__{c}"], "converged": lambda c: np.load("runs/converged_probe_fits/converged_scores.npz")[f"unbc__{c}"]}
    if os.path.exists("runs/extended_nested_probe_fits/nested_scores.npz"):
        N56 = np.load("runs/extended_nested_probe_fits/nested_scores.npz"); usrc["nested"] = lambda c: N56[f"unbc__{c}"]
    du = w12.Design("unbc_shape", uy[ukeep], ug[ukeep], 36)
    au, vu = w12.sim_single(du, 0.6, rng, N_NULL)[1:]; aw, vw = w12.sim_single(d, 0.6, rng, N_NULL)[1:]
    zb_null = np.abs((aw - au) / np.sqrt(vu + vw))
    cross = {}
    for fit in usrc:
        rows_ = []
        for c in cells:
            if (fit, c) not in scores:
                rows_.append({"cell": c, "tested": False, "p_boot": 1.0, "p_cal": 1.0}); continue
            su = usrc[fit](c); ku = ukeep & ~np.isnan(su); sw = scores[(fit, c)]; kw = ~np.isnan(sw)
            dd = w7.diff_bootstrap((su[ku], uy[ku], ug[ku]), (sw[kw], y[kw], g[kw]), rng, NBOOT)
            zb = w12.z_between(su[ku], uy[ku], ug[ku], sw[kw], y[kw], g[kw])
            zval = zb[-1] if isinstance(zb, tuple) else zb
            rows_.append({"cell": c, "tested": True, "diff": float(dd["diff"]), "ci": dd["ci"], "p_boot": dd["p_boot"],
                          "z": float(zval), "p_cal": float((1 + (zb_null >= abs(zval)).sum()) / (1 + len(zb_null)))})
        for key_, hk in (("p_boot", "holm_boot"), ("p_cal", "holm_cal")):
            for r_, a in zip(rows_, holm([r_[key_] for r_ in rows_])):
                r_[hk] = float(a)
        cross[fit] = rows_
        log(f"4. cross-corpus {fit}: " + "; ".join(f"{r_['cell']} {r_['diff']:+.3f} boot {r_['holm_boot']:.3f} "
                                                   f"cal {r_['holm_cal']:.3f}" for r_ in rows_ if r_["tested"]))
    res["cross_corpus"] = cross

    # Evaluate order controls for all nine encoders at K=16.
    ctrl = {}
    for fit in ("fixed_step", "converged", "nested"):
        rows_ = []
        for e in IMAGE + VIDEO:
            c = f"{e}_K16"
            for kind in ("shuffled", "static"):
                if (fit, c) not in scores:
                    rows_.append({"test": f"{c}:{kind}", "tested": False, "p_boot": 1.0}); continue
                img = e in IMAGE
                if kind == "static":
                    sc = fit_scores(CTRL[e]["static"], y, g, mask, fit, img)
                else:
                    sc = np.mean([fit_scores(Xs, y, g, mask, fit, img) for Xs in CTRL[e]["shuffled"]], axis=0)
                so = scores[(fit, c)]; k_ = ~np.isnan(so) & ~np.isnan(sc)
                pb = paired_boot(so[k_], sc[k_], y[k_], g[k_], rng, NBOOT)
                rows_.append({"test": f"{c}:{kind}", "tested": True, "control_auc": float(_auc(sc[k_], y[k_])), **pb})
        for r_, a in zip(rows_, holm([r_["p_boot"] for r_ in rows_])):
            r_["holm"] = float(a)
        ctrl[fit] = rows_
        log(f"5. controls {fit}: " + "; ".join(f"{r_['test']} gap {r_['gap']:+.3f} holm {r_['holm']:.3f}"
                                               for r_ in rows_ if r_["tested"]))
    res["order_controls"] = ctrl
    res["status"] = "COMPLETE"
    json.dump(res, open(out_json, "w"), indent=1)
    np.savez_compressed(f"{OUT}/analysis_scores.npz", idx=midx, **{f"{f}__{c}": s for (f, c), s in scores.items()})
    log(f"wrote {out_json} | COMPLETE"); logf.close()


if __name__ == "__main__":
    main()
