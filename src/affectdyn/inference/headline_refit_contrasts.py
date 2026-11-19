"""Headline contrasts under refitting and on the within-cluster estimand."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os
_CVD = os.environ.get("CUDA_VISIBLE_DEVICES")
import argparse, json, sys, time                                                     # noqa: E402
import numpy as np                                                                   # noqa: E402

import affectdyn.probes.nested_probe_fits as w51                                                 # noqa: E402  (sets CVD="")
if _CVD is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CVD
from affectdyn.features.unbc_pretraining_comparison import _auc                                               # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                         # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS             # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS                                           # noqa: E402
from affectdyn.provenance import run_provenance                                                # noqa: E402

OUT = "runs/headline_refit_contrasts"
SEED = 20260957
B_WITHIN = 10_000
B_REFIT = 2_000
R_NULL = 200
N_REP = 5
ALPHA = 0.05
HEADLINE = [("clip_K2", "videomaev2_K2"), ("clip_K4", "videomaev2_K4"), ("clip_K8", "videomaev2_K8"),
            ("clip_K16", "videomaev2_K16"), ("clip_K2", "videomae_K2")]
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-22 — W56-W59 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-22 — W56-W59 CRITERIA AMENDMENT"


def auc_half(s, y):
    """Mann-Whitney AUC with half credit for ties."""
    p, n = s[y == 1], s[y == 0]
    if len(p) == 0 or len(n) == 0:
        return np.nan
    d = p[:, None] - n[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def within_aucs(s, y, sub, keep):
    """{subject: within-subject AUC} over subjects holding both classes among kept clips."""
    out = {}
    for g in np.unique(sub[keep]):
        m = keep & (sub == g)
        if y[m].min() != y[m].max():
            out[g] = auc_half(s[m], y[m])
    return out


def holm(ps):
    """Local Holm adjustment; the shared implementation is
    inference.aggregation_permutation_tests.holm. Family membership and size are specified by the
    caller.
    """
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


def fixed_step_scores(X, y, osub, mask, device):
    """stage_w2_map.oof_reps averaged over 5 seeds, with every fold's probe trained in one batched Adam run."""
    import torch, torch.nn as nn
    folds = _loso_folds(osub); nfold = max(folds.values()) + 1
    fid = np.array([folds[s] for s in osub])
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
    Xt = torch.tensor(Xs, device=device); Mt = torch.tensor(M, device=device)
    yt = torch.tensor(y.astype(np.float32), device=device)
    W = torch.nn.Parameter(torch.stack(W0).to(device)); b = torch.nn.Parameter(torch.cat(b0).to(device))
    opt = torch.optim.Adam([W, b], 1e-2, weight_decay=1e-3)
    cnt = Mt.sum(1)
    for _ in range(300):
        z = torch.einsum("fnd,fd->fn", Xt, W) + b[:, None]
        l = nn.functional.binary_cross_entropy_with_logits(z, yt.expand_as(z), reduction="none")
        loss = ((l * Mt).sum(1) / cnt).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = (torch.einsum("fnd,fd->fn", Xt, W) + b[:, None]).cpu().numpy()
    acc = np.zeros((N_REP, n), np.float64); seen = np.zeros((N_REP, n), bool)
    for f, (rep, _, _, te) in enumerate(specs):
        acc[rep, te] = z[f, te]; seen[rep, te] = True
    out = np.where(seen.all(0), acc.mean(0), np.nan)
    return out


def converged_scores(X, y, osub, mask, primal=False):
    folds = _loso_folds(osub); fid = np.array([folds[s] for s in osub])
    audit = {"inner_splits": 0, "max_inner_grad": 0.0, "max_outer_grad": 0.0, "outer_fits": 0, "outer_uncertified": 0}
    if primal:
        s, _ = w51.nested_scores(X, y, osub, mask, fid, (1e-3,), audit, lambda m: None, "w57")
        return s, audit
    s = np.full(len(y), np.nan)
    for k in range(int(fid.max()) + 1):
        te = (fid == k) & mask; tr = (fid != k) & mask
        if te.sum() == 0 or len(set(y[tr].tolist())) < 2:
            continue
        xt = w51.standardize(X, tr, tr).astype(np.float64); xe = w51.standardize(X, tr, te).astype(np.float64)
        X1 = np.hstack([xt, np.ones((len(xt), 1))]); X1e = np.hstack([xe, np.ones((len(xe), 1))])
        a, g = w51.newton_kernel(X1 @ X1.T, X1, y[tr], 1e-3)
        s[te] = X1e @ (X1.T @ a)
        audit["outer_fits"] += 1; audit["max_outer_grad"] = max(audit["max_outer_grad"], g)
        audit["outer_uncertified"] += int(g > w51.TOL)
    return s, audit


_W = {}


def _load():
    """Everything a worker needs, rebuilt from the logged caches in each spawned process."""
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]; y = (lab["direction"] == 1).astype(int); md = lab["direction"] != -1
    IMG = np.load("runs/temporal_aggregation/img_frames.npz"); UX = {}
    for e in IMG_ENCODERS:
        for k in KS:
            UX[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for vname, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            if tmpl.format(k=k) in z.files:
                UX[f"{vname}_K{k}"] = video_readout(z["full"], z[tmpl.format(k=k)])
    keep = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz")["keep"].astype(bool)
    usubs = np.unique(sub[md])
    return {"sub": sub, "y": y, "md": md, "UX": UX, "keep": keep, "usubs": usubs,
            "members": {g: np.flatnonzero(sub == g) for g in usubs}}


def _init():
    _W.update(_load())


def gaps(sc, yy, osub, mm, draw, usubs):
    """Pooled gap and Delta_subject (each copy of a drawn subject counted once) for the headline contrasts."""
    out = {}
    for a, b in HEADLINE:
        ka = mm & ~np.isnan(sc[a]) & ~np.isnan(sc[b])
        pooled = _auc(sc[a][ka], yy[ka]) - _auc(sc[b][ka], yy[ka])
        wa = []
        for g in np.unique(osub[ka]):
            mg = ka & (osub == g)
            if yy[mg].min() != yy[mg].max():
                mult = int((draw == np.flatnonzero(usubs == g)[0]).sum())
                h = mg.sum() // mult
                wa += [auc_half(sc[a][mg][:h], yy[mg][:h]) - auc_half(sc[b][mg][:h], yy[mg][:h])] * mult
        out[f"{a}|{b}"] = {"pooled": pooled, "delta_subject": float(np.mean(wa)) if wa else np.nan}
    return out


def _conv_rep(draw):
    W = _W; idx = np.concatenate([W["members"][W["usubs"][g]] for g in draw])
    osub = W["sub"][idx]; yy = W["y"][idx]; mm = (W["md"] & W["keep"])[idx]
    if yy[mm].min() == yy[mm].max():
        return None, None
    cells = sorted({c for pr in HEADLINE for c in pr}); sc, cert = {}, [0, 0, 0.0]
    for c in cells:
        sc[c], au = converged_scores(W["UX"][c][idx], yy, osub, mm)
        cert[0] += au["outer_fits"]; cert[1] += au["outer_uncertified"]; cert[2] = max(cert[2], au["max_outer_grad"])
    return gaps(sc, yy, osub, mm, draw, W["usubs"]), cert


def _null_job(job):
    r_, kind = job
    W = _W; mm0 = W["md"] & W["keep"]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 57, r_, 0 if kind == "within" else 1]))
    yp = w52_perm(kind, W["y"], mm0, W["sub"], rng)
    out = {}
    for c in sorted(W["UX"]):
        if c.split("_K")[0] in ("dinov2", "siglip"):
            continue
        s, _ = converged_scores(W["UX"][c], yp, W["sub"], mm0)
        k2 = mm0 & ~np.isnan(s); out[c] = _auc(s[k2], yp[k2])
    return kind, out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--b-refit", type=int, default=B_REFIT); ap.add_argument("--r-null", type=int, default=R_NULL)
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/headline_refit.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prov = run_provenance(seeds=[SEED], device=device)
    logf = open(f"{OUT}/headline_refit.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:8.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W57 | device {device} | B_refit {args.b_refit} | R_null {args.r_null}")
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]; y = (lab["direction"] == 1).astype(int); md = lab["direction"] != -1
    IMG = np.load("runs/temporal_aggregation/img_frames.npz"); UX = {}
    for e in IMG_ENCODERS:
        for k in KS:
            UX[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for vname, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            if tmpl.format(k=k) in z.files:
                UX[f"{vname}_K{k}"] = video_readout(z["full"], z[tmpl.format(k=k)])
    M = json.load(open("runs/pain_encoder_maps/map.json"))
    rows = [r for r in M["primary"] if r["a"] in UX and r["b"] in UX]
    assert len(rows) == 24, len(rows)
    S32 = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz"); keep = S32["keep"].astype(bool)
    C46 = np.load("runs/converged_probe_fits/converged_scores.npz")
    stored = {"fixed_step": {n: S32[n] for n in UX if n in S32.files},
              "converged": {n: C46[f"f1__{n}"] for n in UX if f"f1__{n}" in C46.files}}
    for tag, path in (("nested_w51", "runs/nested_probe_fits/nested_scores.npz"), ("nested_w56", "runs/extended_nested_probe_fits/nested_scores.npz")):
        if os.path.exists(path):
            Z = np.load(path); stored[tag] = {n: Z[f"f1__{n}"] for n in UX if f"f1__{n}" in Z.files}
    rng = np.random.default_rng(SEED)

    # Estimate within-subject AUC from stored scores.
    within = {}
    for fit, S in stored.items():
        W = {n: within_aucs(S[n], y, sub, keep) for n in S}
        subs = sorted(set.intersection(*[set(W[r["a"]]) & set(W[r["b"]]) for r in rows]))
        G = len(subs); draws = rng.integers(0, G, (B_WITHIN, G))
        res = []
        for r in rows:
            d = np.array([W[r["a"]][g] - W[r["b"]][g] for g in subs])
            boot = d[draws].mean(1)
            p = min(1.0, max(1.0 / B_WITHIN, 2 * min((boot <= 0).mean(), (boot >= 0).mean())))
            res.append({"label": r["label"], "a": r["a"], "b": r["b"], "delta_subject": float(d.mean()),
                        "ci": [float(x) for x in np.percentile(boot, [2.5, 97.5])], "p_boot": float(p)})
        adj = holm([x["p_boot"] for x in res])
        for x, h in zip(res, adj):
            x["holm"] = float(h); x["significant_clip_higher"] = bool(h < ALPHA and x["delta_subject"] > 0)
        cls = {g: (int((y[keep & (sub == g)] == 1).sum()), int((y[keep & (sub == g)] == 0).sum()))
               for g in np.unique(sub[keep])}
        within[fit] = {"subjects_mixed": subs, "n_mixed": G,
                       "excluded_single_class": {str(g): cls[g] for g in cls if g not in subs}, "contrasts": res,
                       "n_significant": int(sum(x["holm"] < ALPHA for x in res))}
        log(f"(ii) {fit}: {G} mixed subjects; significant {within[fit]['n_significant']}: "
            f"{[x['label'] for x in res if x['holm'] < ALPHA]}")
    np.savez_compressed(f"{OUT}/within_subject_aucs.npz",
                        **{f"{fit}__{n}": np.array([within_aucs(S[n], y, sub, keep).get(g, np.nan)
                                                     for g in within[fit]["subjects_mixed"]])
                           for fit, S in stored.items() for n in S})

    cells = sorted({c for pr in HEADLINE for c in pr})
    usubs = np.unique(sub[md])
    members = {g: np.flatnonzero(sub == g) for g in usubs}

    cert = {"outer_fits": 0, "outer_uncertified": 0, "max_outer_grad": 0.0}

    def replicate(draw, fit, primal=False):
        idx = np.concatenate([members[usubs[g]] for g in draw])
        osub = sub[idx]; yy = y[idx]; mm = (md & keep)[idx]
        out = {}
        for c in cells:
            if fit == "fixed_step":
                out[c] = fixed_step_scores(UX[c][idx], yy, osub, mm, device)
            else:
                out[c], au = converged_scores(UX[c][idx], yy, osub, mm, primal)
                cert["outer_fits"] += au["outer_fits"]; cert["outer_uncertified"] += au["outer_uncertified"]
                cert["max_outer_grad"] = max(cert["max_outer_grad"], au["max_outer_grad"])
        return out, yy, osub, mm

    ident = np.arange(len(usubs))
    gate = {}
    for fit, ref, primal in (("fixed_step", stored["fixed_step"], False), ("converged", stored["converged"], True),
                             ("converged_kernel", stored["converged"], False)):
        sc, yy, osub, mm = replicate(ident, "fixed_step" if fit == "fixed_step" else "converged", primal)
        order_idx = np.concatenate([members[usubs[g]] for g in ident])
        d_auc = {c: abs(_auc(sc[c][mm], yy[mm]) - _auc(ref[c][order_idx][mm], yy[mm])) for c in cells}
        d_score = max(float(np.nanmax(np.abs(sc[c][mm] - ref[c][order_idx][mm]))) for c in cells)
        tol_ok = {"fixed_step": max(d_auc.values()) <= 1e-4, "converged": d_score <= 1e-9,
                  "converged_kernel": d_score <= 1e-6}[fit]
        gate[fit] = {"max_abs_auc_delta": max(d_auc.values()), "max_abs_score_delta": d_score, "passes": bool(tol_ok)}
        log(f"gate {fit}: {gate[fit]}")
    res = {"provenance": prov, "criteria_heading": CRITERIA_HEADING, "amendment_heading": AMENDMENT_HEADING,
           "gates": gate, "within_subject": within}
    if not all(g["passes"] for g in gate.values()):
        res["status"] = "HOLD: identity gate failed"
        json.dump(res, open(out_json, "w"), indent=1); log(res["status"]); return

    G = len(usubs); draws = rng.integers(0, G, (args.b_refit, G))
    FITS = ("fixed_step", "converged")
    refit = {fit: {f"{a}|{b}": {"pooled": [], "delta_subject": []} for a, b in HEADLINE} for fit in FITS}
    discarded = {fit: 0 for fit in FITS}
    from multiprocessing import get_context
    pool = get_context("spawn").Pool(args.procs, initializer=_init)
    conv_async = pool.map_async(_conv_rep, list(draws), chunksize=4)
    null_jobs = [(r_, kind) for r_ in range(args.r_null) for kind in ("within", "global")]
    for bi, draw in enumerate(draws):
        sc, yy, osub, mm = replicate(draw, "fixed_step")
        if yy[mm].min() == yy[mm].max():
            discarded["fixed_step"] += 1; continue
        for k_, v in gaps(sc, yy, osub, mm, draw, usubs).items():
            for est in v:
                refit["fixed_step"][k_][est].append(v[est])
        if (bi + 1) % 100 == 0:
            log(f"(i) fixed-step refit {bi + 1}/{len(draws)}")
    for res_c, cc in conv_async.get():
        if res_c is None:
            discarded["converged"] += 1; continue
        cert["outer_fits"] += cc[0]; cert["outer_uncertified"] += cc[1]; cert["max_outer_grad"] = max(cert["max_outer_grad"], cc[2])
        for k_, v in res_c.items():
            for est in v:
                refit["converged"][k_][est].append(v[est])
    log("(i) converged refit complete")
    summ = {}
    for fit in refit:
        summ[fit] = {}
        for key_, v in refit[fit].items():
            summ[fit][key_] = {}
            for est, xs in v.items():
                x = np.array([t for t in xs if t == t])
                share = float((x <= 0).mean()); nb = len(x)
                se = np.sqrt(share * (1 - share) / max(nb, 1))
                summ[fit][key_][est] = {"n": int(nb), "mean": float(x.mean()), "ci": [float(q) for q in np.percentile(x, [2.5, 97.5])],
                                        "share_at_or_below_0": share, "share_mc_se": float(se)}
        log(f"(i) {fit}: " + "; ".join(f"{k} pooled {v['pooled']['ci']} subj {v['delta_subject']['ci']}"
                                       for k, v in summ[fit].items()))
    res["refit_bootstrap"] = {"B": int(args.b_refit), "discarded": discarded, "summary": summ,
                              "converged_certificates": dict(cert)}

    # Compute paired refit nulls with converged fits.
    all_cells = sorted({r["a"] for r in rows} | {r["b"] for r in rows})
    null = {kind: {c: [] for c in all_cells} for kind in ("within", "global")}
    for kind, out in pool.imap(_null_job, null_jobs, chunksize=2):
        for c in all_cells:
            null[kind][c].append(out[c])
    pool.close(); pool.join()
    log(f"(iii) null complete, R {args.r_null}")
    import affectdyn.inference.peak_refit_null_centering as w52
    cent = {kind: {c: w52.centring(np.array(v), rng) for c, v in null[kind].items()} for kind in null}
    paired = {kind: {f"{a}|{b}": {"mean_gap": float(np.mean(np.array(null[kind][a]) - np.array(null[kind][b]))),
                                  "p2.5_p97.5": [float(q) for q in np.percentile(np.array(null[kind][a]) - np.array(null[kind][b]), [2.5, 97.5])]}
                     for a, b in HEADLINE} for kind in null}
    res["refit_null"] = {"R": int(args.r_null), "centring": cent, "paired_gap_under_null": paired}
    res["status"] = "COMPLETE"
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json} | COMPLETE"); logf.close()


def w52_perm(kind, y, mask, sub, rng):
    import affectdyn.inference.peak_refit_null_centering as w52
    return w52.within_subject_perm(y, mask, sub, rng) if kind == "within" else w52.global_perm(y, mask, rng)


if __name__ == "__main__":
    main()
