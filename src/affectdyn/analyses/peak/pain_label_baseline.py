"""Label-only baseline against the pain peak-task encoder scores, with an alternative eligibility rule.

Usage: python -m affectdyn.analyses.peak.pain_label_baseline
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, json, sys, time
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                       # noqa: E402
from affectdyn.inference.loso_permutation_tests import _oof_scores, _loso_folds                    # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS, N_REP  # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                               # noqa: E402
from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES, KS                                   # noqa: E402
from affectdyn.inference.calibrated_clustered_auc import z_paired, p_from_z                  # noqa: E402
from affectdyn.analyses.peak.endpoint_magnitude_controls import midrank_auc                         # noqa: E402
from affectdyn.provenance import run_provenance                                        # noqa: E402

OUT = "runs/pain_label_baseline"
SEED = 20261108
NBOOT = 10_000
ALPHA = 0.05
ENCODERS = list(IMG_ENCODERS) + list(VIDEO_SOURCES)
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA = "## 2026-11-08 — REVISION PROTOCOL: N1-N4 (W66-W69) PRE-DECLARED CRITERIA"


def members(g):
    u = np.unique(g)
    return [np.flatnonzero(g == x) for x in u]


def paired_boot(sa, sb, y, g, rng, nboot):
    mem = members(g); G = len(mem); vals = []; skipped = 0
    for _ in range(nboot):
        i = np.concatenate([mem[j] for j in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            skipped += 1; continue
        vals.append(midrank_auc(sa[i], y[i]) - midrank_auc(sb[i], y[i]))
    v = np.asarray(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    p = 2.0 * min(float((v <= 0).mean()), float((v >= 0).mean()))
    return {"ci": [float(lo), float(hi)], "p_boot": float(min(1.0, max(p, 1.0 / len(v)))), "skipped": skipped}


def single_boot(s, y, g, rng, nboot):
    mem = members(g); G = len(mem); vals = []; skipped = 0
    for _ in range(nboot):
        i = np.concatenate([mem[j] for j in rng.integers(0, G, G)])
        if y[i].min() == y[i].max():
            skipped += 1; continue
        vals.append(midrank_auc(s[i], y[i]))
    v = np.asarray(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    share = min(float((v <= 0.5).mean()), float((v >= 0.5).mean()))
    return {"auc": midrank_auc(s, y), "ci": [float(lo), float(hi)],
            "p_boot": float(max(2 * share, 1.0 / len(v))), "skipped": skipped}


def cells():
    out = {}
    IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    for e in IMG_ENCODERS:
        for k in KS:
            out[f"{e}_K{k}"] = image_readout(IMG[e], k)
    for name, (path, tmpl) in VIDEO_SOURCES.items():
        z = np.load(path)
        for k in KS:
            key = tmpl.format(k=k)
            if key in z.files:
                out[f"{name}_K{k}"] = video_readout(z["full"], z[key])
    return out


def refit(X, y, sub, mask, folds):
    import torch
    acc = None
    for rep in range(N_REP):
        torch.manual_seed(2000 + rep)
        s = _oof_scores(X, y, sub, mask, folds)
        acc = s if acc is None else acc + s
    return acc / N_REP


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    nboot = NBOOT
    out = OUT
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/label_baseline.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    rng = np.random.default_rng(SEED)
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    traj, sub, slope = lab["traj"], lab["subject"], lab["slope"]
    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    keep = W7["unbc__keep"].astype(bool)
    y_all = (traj == 2).astype(np.int64)
    res = {"provenance": run_provenance(seeds=[SEED]), "criteria_heading": CRITERIA, "nboot": nboot,
           "gates": {}, "primary": {}, "converged_descriptive": {}, "A1": {}}

    y, g, dm = y_all[keep], sub[keep], np.abs(slope[keep])
    base = -dm
    g1 = midrank_auc(dm, y)
    res["gates"]["G1_w14_magnitude_auc"] = {"value": g1, "target": 0.1801188, "pass": abs(g1 - 0.1801188) < 1e-4}
    pub = {r["cell"]: r["auc"] for r in json.load(open("runs/pain_encoder_maps/shape.json"))["rows"]}
    g2 = {c: abs(_auc(W7[f"unbc__{c}"][keep], y) - pub[c]) for c in pub}
    res["gates"]["G2_published_shape_aucs"] = {"estimator": "stage-level _auc (the published estimator)",
                                              "max_abs_delta": max(g2.values()), "n": len(g2),
                                              "pass": max(g2.values()) < 1e-4}
    g2m = {c: abs(midrank_auc(W7[f"unbc__{c}"][keep], y) - pub[c]) for c in pub}
    res["gates_failed_as_written"] = {"G2_midrank_as_declared": {
        "max_abs_delta": max(g2m.values()), "n_over_1e-4": int(sum(v > 1e-4 for v in g2m.values())),
        "pass": max(g2m.values()) <= 1e-4}}
    log(f"345-clip set: {int(keep.sum())} clips, {int(y.sum())} peaked, {len(np.unique(g))} subjects")
    log(f"G1 {res['gates']['G1_w14_magnitude_auc']}")
    log(f"G2 max |delta| {max(g2.values()):.2e} over {len(g2)}")

    X = cells()
    folds = _loso_folds(sub)
    mask_pub = traj != -1
    s_chk = refit(X["clip_K4"], y_all, sub, mask_pub, folds)
    d3 = float(np.nanmax(np.abs(s_chk[keep] - W7["unbc__clip_K4"][keep])))
    res["gates"]["G3_refit_reproduces_clip_K4"] = {"max_abs_score_delta": d3, "pass": d3 < 1e-5}
    log(f"G3 refit clip_K4 max |score delta| {d3:.2e}")
    if not all(v["pass"] for v in res["gates"].values()):
        res["status"] = "HOLD: gate failed"
        json.dump(res, open(f"{out}/label_baseline.json", "w"), indent=2); log("HOLD"); return

    rows = []
    for e in ENCODERS:
        for k in KS:
            c = f"{e}_K{k}"
            s = W7[f"unbc__{c}"][keep].astype(np.float64)
            d = midrank_auc(base, y) - midrank_auc(s, y)
            b = paired_boot(base, s, y, g, rng, nboot)
            dz, z = z_paired(base, s, y, g)
            rows.append({"cell": c, "auc_baseline": midrank_auc(base, y), "auc_encoder": midrank_auc(s, y),
                         "diff": d, "diff_clustered": dz, "boot": b, "z": z, "p_z": p_from_z(z)})
    for r, h in zip(rows, holm([r["boot"]["p_boot"] for r in rows])):
        r["holm_boot"] = h
    for r, h in zip(rows, holm([r["p_z"] for r in rows])):
        r["holm_z"] = h
    counts = {"baseline_higher": 0, "encoder_higher": 0, "disagree": 0, "neither": 0}
    for r in rows:
        rb, rz = r["holm_boot"] < ALPHA, r["holm_z"] < ALPHA
        if rb and rz:
            if np.sign(r["diff"]) != np.sign(r["diff_clustered"]):
                raise RuntimeError(f"{r['cell']}: procedures reject with opposite signs, outside the declared map")
            r["outcome"] = "baseline_higher" if r["diff"] > 0 else "encoder_higher"
        elif rb != rz:                          # Count disagreement when exactly one decision rejects.
            r["outcome"] = "disagree"
        else:
            r["outcome"] = "neither"
        r["degenerate_z"] = bool(not np.isfinite(r["z"]) or r["z"] == 0.0)
        counts[r["outcome"]] += 1
        log(f"  {r['cell']:16s} base {r['auc_baseline']:.3f} enc {r['auc_encoder']:.3f} D {r['diff']:+.3f} "
            f"boot Holm {r['holm_boot']:.4f} z Holm {r['holm_z']:.4f} -> {r['outcome']}")
    res["primary"] = {"rows": rows, "counts": counts}
    log(f"(i) outcomes: {counts}")

    C = np.load("runs/converged_probe_fits/converged_scores.npz"); kc = C["keep_f1"].astype(bool) if "keep_f1" in C.files else None
    for e in ENCODERS:
        for k in KS:
            key = f"unbc__{e}_K{k}"
            if key in C.files:
                s = C[key]
                res["converged_descriptive"][f"{e}_K{k}"] = {"auc": midrank_auc(s[keep], y)}
    res["converged_descriptive"]["note"] = ("UNBC shape converged scores (runs/converged_probe_fits unbc__ keys); descriptive only"
                                            if res["converged_descriptive"] else "no UNBC shape converged keys found")

    y1 = y_all.copy()
    mask1 = np.ones(len(y1), bool)                  # Include every cached clip under the same peak rule.
    npos = int(y1.sum()); nneg = int(len(y1) - npos)
    pos_sub = np.unique(sub[y1 == 1])
    share = max(np.mean(sub[y1 == 1] == s_) for s_ in pos_sub)
    train_ok = all(len(set(y1[(np.array([folds[s_] != f for s_ in sub]))].tolist())) == 2
                   for f in range(max(folds.values()) + 1))
    sup = {"n_clips": int(len(y1)), "n_peaked": npos, "n_nonpeaked": nneg,
           "n_subjects": int(len(np.unique(sub))), "n_subjects_with_peaked": int(len(pos_sub)),
           "n_subjects_with_nonpeaked": int(len(np.unique(sub[y1 == 0]))),
           "max_share_of_peaked_from_one_subject": float(share), "every_training_fold_two_classes": bool(train_ok),
           "added_vs_published": int(len(y1) - int(mask_pub.sum()))}
    sup["pass"] = bool(npos >= 15 and len(pos_sub) >= 8 and nneg >= 50 and train_ok)
    res["A1"]["support"] = sup
    log(f"A1 support: {sup}")
    if sup["pass"]:
        dm1 = np.abs(slope)
        res["A1"]["baseline"] = single_boot(-dm1, y1, sub, rng, nboot)
        lo, hi = res["A1"]["baseline"]["ci"]
        res["A1"]["baseline"]["ci_excludes_0.820"] = bool(lo > 0.820 or hi < 0.820)
        log(f"A1 -|d| AUC {res['A1']['baseline']['auc']:.4f} CI [{lo:.4f}, {hi:.4f}]")
        arow = []
        for c, Xc in X.items():
            s1 = refit(Xc, y1, sub, mask1, folds)
            ok = ~np.isnan(s1)
            r = single_boot(s1[ok], y1[ok], sub[ok], rng, nboot); r["cell"] = c; r["n_scored"] = int(ok.sum())
            arow.append(r); log(f"  A1 {c:16s} AUC {r['auc']:.3f} CI [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
        for r, h in zip(arow, holm([r["p_boot"] for r in arow])):
            r["holm"] = h
        res["A1"]["encoders"] = arow
        res["A1"]["n_significant"] = int(sum(r["holm"] < ALPHA for r in arow))
        log(f"A1 encoders significant after Holm-36: {res['A1']['n_significant']}")
    skipped = [r["boot"]["skipped"] for r in rows] + [r.get("skipped", 0) for r in res["A1"].get("encoders", [])]
    ndeg = sum(r["degenerate_z"] for r in rows)
    res["failure_gate"] = {"max_discarded_resamples": int(max(skipped)), "degenerate_z": int(ndeg),
                           "pass": bool(max(skipped) <= 0.01 * nboot and ndeg <= 0.01 * len(rows))}
    log(f"failure gate {res['failure_gate']}; G2 as declared (midrank) {res['gates_failed_as_written']}")
    res["status"] = "complete"
    json.dump(res, open(f"{out}/label_baseline.json", "w"), indent=2, default=float)
    log("wrote " + f"{out}/label_baseline.json")


if __name__ == "__main__":
    main()
