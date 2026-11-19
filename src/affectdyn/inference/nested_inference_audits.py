"""Deterministic supplement to the nested-fit analyses.

Usage:  python -m affectdyn.inference.nested_inference_audits --part tables      (b-e)
        python -m affectdyn.inference.nested_inference_audits --part replay
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"
import argparse, io, json, platform, subprocess, sys, time
from contextlib import redirect_stdout
import numpy as np

from affectdyn.inference.aggregation_permutation_tests import holm                                         # noqa: E402
from affectdyn.provenance import run_provenance                                  # noqa: E402

OUT = "runs/nested_inference_audits"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W54 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W54 CRITERIA AMENDMENT"
KS = (2, 4, 8, 16)


def _numerics():
    buf = io.StringIO()
    with redirect_stdout(buf):
        np.show_config()
    return {"numpy": np.__version__, "threads": {v: os.environ[v] for v in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
            "numpy_config": buf.getvalue()}


def part_replay():
    import affectdyn.inference.peak_refit_null_centering as w52
    from affectdyn.features.unbc_pretraining_comparison import _auc
    from affectdyn.inference.loso_permutation_tests import _loso_folds
    from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS
    from affectdyn.analyses.direction.pain_direction_encoder_map import VIDEO_SOURCES
    from affectdyn.provenance import audit_loso

    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/certificate_replay.json"
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite")
    prov_start = run_provenance(device="cpu")
    logf = open(f"{OUT}/certificate_replay.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    # Match the refit-null procedure's setup.
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
    D = np.load("runs/peak_refit_null_centering/refit_null_draws.npz")
    assert cells == [str(c) for c in D["cells"]], "cell order differs from W52's saved draws"
    S7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz"); ukeep = S7["unbc__keep"].astype(bool)
    nf = max(folds.values()) + 1
    fold_sets = []
    for k in range(nf):
        te = np.array([folds[s] == k for s in sub]) & ms
        tr = np.array([folds[s] != k for s in sub]) & ms
        if te.sum() == 0 or len(set(ys[tr].tolist())) < 2:
            continue
        fold_sets.append((tr, te))
    kern = {c: [w52.FoldKernel(X[c], tr, te) for tr, te in fold_sets] for c in cells}
    log(f"W54(a) | {len(cells)} cells, {len(fold_sets)} folds, lambda {w52.LAM}, TOL {w52.TOL}, MAXIT {w52.MAXIT}, "
        f"threads {os.environ['OPENBLAS_NUM_THREADS']}")

    def pipeline(c, y):
        sc = np.full(len(y), np.nan); g = np.zeros(len(fold_sets))
        for j, ((tr, te), K) in enumerate(zip(fold_sets, kern[c])):
            s_te, g[j] = K.fit(y[tr]); sc[te] = s_te
        return float(_auc(sc[ukeep], y[ukeep])), g

    rng_w, rng_g = np.random.default_rng(w52.SEED), np.random.default_rng([w52.SEED, 1])
    perms_w = [w52.within_subject_perm(ys, ms, sub, rng_w) for _ in range(w52.R)]
    perms_g = [w52.global_perm(ys, ms, rng_g) for _ in range(w52.R)]
    n_r = w52.R
    null = {"within": np.full((len(cells), w52.R), np.nan), "global": np.full((len(cells), w52.R), np.nan)}
    cert = {"within": np.zeros((len(cells), w52.R, len(fold_sets))), "global": np.zeros((len(cells), w52.R, len(fold_sets)))}
    for r in range(n_r):
        for i, c in enumerate(cells):
            null["within"][i, r], cert["within"][i, r] = pipeline(c, perms_w[r])
            null["global"][i, r], cert["global"][i, r] = pipeline(c, perms_g[r])
        if r % 20 == 0:
            log(f"  replay permutation {r + 1}/{n_r}")
    delta, fails = {}, {}
    for k in null:
        dd = np.abs(null[k][:, :n_r] - D[f"null_{k}"][:, :n_r])
        bad = ~np.isfinite(dd) | (dd > 1e-12) | ~np.isfinite(cert[k][:, :n_r]).all(axis=2)
        delta[k] = float(np.nanmax(dd)) if np.isfinite(dd).any() else float("nan")
        fails[k] = [{"cell": cells[i], "draw": int(r)} for i, r in zip(*np.nonzero(bad))]
    gate = {"max_abs_null_auc_delta": delta, "tolerance_rank_identical": 1e-12, "permutations_replayed": n_r,
            "failures": fails, "passes": not any(fails.values())}
    log(f"gate: {gate}")
    res = {}
    for k in ("within", "global"):
        cfk = cert[k]
        res[k] = {"pipelines": int(len(cells) * w52.R), "fold_fits": int(cfk.size),
                  "max_certificate": float(cfk.max()),
                  "fold_fits_uncertified": int((cfk > w52.TOL).sum()),
                  "pipelines_with_an_uncertified_fold": int((cfk > w52.TOL).any(axis=2).sum()),
                  "uncertified_by_cell": {c: int((cfk[i] > w52.TOL).sum()) for i, c in enumerate(cells)
                                          if (cfk[i] > w52.TOL).any()}}
    out = {"provenance_start": prov_start, "provenance_end": run_provenance(device="cpu"), "numerics": _numerics(),
           "criteria_heading": CRITERIA_HEADING, "amendment_heading": AMENDMENT_HEADING, "gate": gate,
           "certificates": res, "note": "fold sets frozen on the observed labels, as in W52",
           "status": "COMPLETE" if gate["passes"] else "HOLD: replay does not reproduce W52's draws"}
    json.dump(out, open(out_json, "w"), indent=2)
    log(f"certificates: {json.dumps(res)}")
    log(f"wrote {out_json}"); logf.close()


def part_perfold():
    m = json.load(open("runs/pain_encoder_maps/map.json"))["table"]
    rows = []
    for k in KS:
        a, b = m[f"clip_K{k}"], m[f"videomaev2_K{k}"]
        rows.append({"K": k, "clip_per_fold_mean": a["per_fold_mean"], "videomaev2_per_fold_mean": b["per_fold_mean"],
                     "gap_per_fold_mean": a["per_fold_mean"] - b["per_fold_mean"],
                     "gap_pooled": a["pooled_oof"] - b["pooled_oof"]})
    return {"source": "runs/pain_encoder_maps/map.json table[].per_fold_mean and pooled_oof", "rows": rows,
            "aggregation": {"per_fold_mean": "mean over folds of within-fold AUCs, each fold AUC averaged over 5 fits",
                            "pooled": "AUC of out-of-fold scores averaged over the 5 fits"},
            "note": "descriptive only; per-fold arrays are not stored, so no paired fold-level difference or interval"}


def part_holm():
    d = json.load(open("runs/nested_probe_fits/nested_probe.json"))
    out, ok = {}, True
    for fam in ("unbc", "aw2"):
        fc = [c for c in d["F2F3_nested"] if c["family"] == fam]
        keys = [f"{c['corpus']}:{c['cell']}" for c in fc]
        assert len(set(keys)) == len(keys) == {"unbc": 36, "aw2": 18}[fam], fam
        assert all(np.isfinite(c[p]) for c in fc for p in ("p_perm_clip", "p_perm_subject", "p_boot_clip",
                                                            "p_boot_subject"))
        res = {f"{c['corpus']}:{c['cell']}": {"auc": c["auc"], "direction": "above" if c["auc"] > 0.5 else "below"}
               for c in fc}
        for proc in ("p_perm_clip", "p_perm_subject", "p_boot_clip", "p_boot_subject"):
            for c, a in zip(fc, holm([c[proc] for c in fc])):
                key = f"{c['corpus']}:{c['cell']}"
                res[key][proc] = c[proc]; res[key]["holm_" + proc] = float(a)
        for c in fc:
            ok &= res[f"{c['corpus']}:{c['cell']}"]["holm_p_boot_subject"] == c["holm_boot_subject"]
        sig = {proc: sorted(k for k, v in res.items() if v["holm_" + proc] < 0.05)
               for proc in ("p_perm_clip", "p_perm_subject", "p_boot_clip", "p_boot_subject")}
        out[fam] = {"n": len(fc), "tests": res, "significant": sig}
    return out, bool(ok)


def part_cal_vs_w49():
    a = json.load(open("runs/converged_clustered_calibration/primary_calibrated_recheck.json"))["families"]
    b = json.load(open("runs/nested_probe_fits/nested_calibrated_recheck.json"))["families"]
    out = {}
    for fam in a:
        ra = {r["test"]: r for r in a[fam]["rows"]}; rb = {r["test"]: r for r in b[fam]["rows"]}
        assert set(ra) == set(rb), fam
        primary = fam != "w11_exploratory_5"
        ch = [{"test": t, "w49_converged": ra[t].get("decision_calibrated"), "w51_nested": rb[t].get("decision_calibrated"),
               "holm_w49": ra[t].get("holm_calibrated"), "holm_w51": rb[t].get("holm_calibrated")}
              for t in ra if ra[t].get("decision_calibrated") != rb[t].get("decision_calibrated")]
        out[fam] = {"n": len(ra), "primary": primary, "n_significant_w49": sum(bool(r.get("decision_calibrated")) for r in ra.values()),
                    "n_significant_w51": sum(bool(r.get("decision_calibrated")) for r in rb.values()), "changes": ch}
    out["n_primary_tests"] = sum(v["n"] for v in out.values() if isinstance(v, dict) and v["primary"])
    out["w8_order_controls_unchanged"] = not out["w8_order_controls_4"]["changes"]
    return out


def part_retro():
    d51 = json.load(open("runs/nested_probe_fits/nested_probe.json")); d46 = json.load(open("runs/converged_probe_fits/convergence.json"))
    au = d51["gates"]["G2"]["audit"]
    c46, c51 = d46["provenance"]["git_commit"], d51["provenance"]["git_commit"]
    r = subprocess.run(["git", "diff", "--quiet", c46, c51, "--", "src/stage_w46_probe_convergence.py"],
                       capture_output=True)
    same = None if r.returncode not in (0, 1) else r.returncode == 0   # None: commit history unavailable
    g46 = d46["gates"]["G1a_replay"]
    return {"inner_certificates": {"inner_splits": au["inner_splits"], "max_inner_grad": au["max_inner_grad"],
                                   "tolerance": 1e-8, "passes": bool(au["max_inner_grad"] <= 1e-8)},
            "replay_via_w46": {"w46_commit": c46, "w46_dirty": d46["provenance"]["git_dirty"], "w51_commit": c51,
                               "w46_functions_unchanged": same, "w46_G1a_replay": g46,
                               "passes": None if same is None else bool(same and all(g46.values()))},
            "open": "eligibility-mask and held-out-row audit records were not written by W51 (code-verified only)"}


AXES = ("Reference", "Inference", "Comparison set", "Probe fit", "Representation", "Sample", "Corpus")
W50_AXIS = {
    "published subject bootstrap": ("Reference", False),
    "clip bootstrap": ("Inference", False),
    "subject swap permutation": ("Inference", False),
    "calibrated (published fit)": ("Inference", False),
    "27 calibrated null models (min over configurations)": ("Inference", False),
    "6-test slice U6": ("Comparison set", False),
    "6-test slice M6_K2": ("Comparison set", False),
    "6-test slice M6_K16": ("Comparison set", False),
    "probe setting wd0.001_steps300": ("DUPLICATE", False),
    "converged probe, subject bootstrap": ("Probe fit", True),
    "converged probe, calibrated (primary nuisances)": ("Probe fit", True),
    "converged probe, calibrated (secondary nuisances)": ("Probe fit", True),
    "appearance term matched": ("Representation", True),
    "native V-JEPA-2 windows (alternative family)": ("Representation", False),
    "uncapped sample, subject bootstrap": ("Sample", True),
}


def _inference(name):
    if name.startswith("27 calibrated"):
        return "calibrated, range over 27 null models including the published one"
    if "calibrated" in name:
        return "calibrated" + (" (secondary nuisances)" if "secondary" in name else "")
    if name == "clip bootstrap":
        return "clip bootstrap"
    if name == "subject swap permutation":
        return "subject swap permutation"
    return "subject bootstrap"


def _axis(name):
    if name in W50_AXIS:
        return W50_AXIS[name]
    if name.startswith("probe setting "):
        return ("Probe fit", True)
    raise KeyError(name)


def _parse(label):
    k, pair = label.split(":", 1)
    return int(k.replace("K=", "").strip()), pair.split(" vs ")[1].strip()


def part_axes():
    w50 = json.load(open("runs/videomaev2_contrast_stability/videomaev2_stability.json"))
    rows = []
    for name, e in w50["variants"].items():
        ax, changed = _axis(name)
        if ax == "DUPLICATE":
            continue
        rows.append({"variant": name, "axis": ax, "target_scores_changed": changed, "inference": _inference(name),
                     "tested_K": e["tested_K"],
                     "videomaev2_significant_K": e["videomaev2_significant_K"],
                     "other_encoders_surviving": e["other_encoders_surviving"], "mc": e["mc"], "source": e["note"]})
    d = json.load(open("runs/nested_probe_fits/nested_probe.json"))
    surv = [_parse(r["label"]) for r in d["F1_nested"] if r["survives"]]
    assert len(surv) == d["F1_survivors"]["nested"]
    rows.append({"variant": "nested probe, subject bootstrap", "axis": "Probe fit", "target_scores_changed": True,
                 "inference": "subject bootstrap",
                 "tested_K": list(KS), "videomaev2_significant_K": sorted(k for k, v in surv if v == "videomaev2"),
                 "other_encoders_surviving": sorted({v for k, v in surv if v != "videomaev2"}), "mc": {},
                 "source": "runs/nested_probe_fits/nested_probe.json F1_nested[].survives"})
    cal = json.load(open("runs/nested_probe_fits/nested_calibrated_recheck.json"))["families"]["direction_contrasts_24"]["rows"]
    cs = [_parse(r["test"]) for r in cal if r["decision_calibrated"]]
    assert len(cs) == d["calibrated"]["direction_contrasts_24"]["n_significant_nested"]
    mc = {r["test"]: r["classification"] for r in d["calibrated"]["direction_contrasts_24"]["mc"]["rows"]
          if "videomaev2" in r["test"]}
    rows.append({"variant": "nested probe, calibrated", "axis": "Probe fit", "target_scores_changed": True,
                 "inference": "calibrated (re-estimated nuisances)",
                 "tested_K": list(KS), "videomaev2_significant_K": sorted(k for k, v in cs if v == "videomaev2"),
                 "other_encoders_surviving": sorted({v for k, v in cs if v != "videomaev2"}), "mc": mc,
                 "source": "runs/nested_probe_fits/nested_calibrated_recheck.json direction_contrasts_24; MC runs/nested_probe_fits calibrated.mc"})
    w34 = json.load(open("runs/affwild_dense_direction_contrasts/aw2_contrasts.json"))["contrasts"]
    v2 = [c for c in w34 if c["video"] == "videomaev2"][0]
    rows.append({"variant": "Aff-Wild2 emotion, subject bootstrap", "axis": "Corpus", "target_scores_changed": True,
                 "inference": "subject bootstrap", "note": "direction labels position-imbalanced (runs/affwild_dense_direction_contrasts)",
                 "tested_K": [16], "videomaev2_significant_K": [], "other_encoders_surviving":
                 sorted(c["video"] for c in w34 if c["survives"]), "mc": {},
                 "videomaev2_gap": v2["gap"], "videomaev2_ci": v2["ci"], "source": "runs/affwild_dense_direction_contrasts/aw2_contrasts.json"})
    assert not any(c["survives"] for c in w34)
    pain = [r for r in rows if r["axis"] not in ("Reference", "Corpus")]
    all4 = [r for r in pain if r["tested_K"] == list(KS)]
    ch4 = [r for r in all4 if r["target_scores_changed"]]
    counts = {}
    for r in pain:
        for o in r["other_encoders_surviving"]:
            counts[o] = counts.get(o, 0) + 1
    summary = {
        "n_pain_variants_excluding_reference": len(pain),
        "by_axis": {a: sum(r["axis"] == a for r in rows) for a in AXES},
        "n_pain_variants_changing_clip_or_videomaev2_scores": sum(r["target_scores_changed"] for r in pain),
        "videomaev2_K_count_all4": [min(len(r["videomaev2_significant_K"]) for r in all4),
                                    max(len(r["videomaev2_significant_K"]) for r in all4)],
        "videomaev2_K_count_all4_score_changing": [min(len(r["videomaev2_significant_K"]) for r in ch4),
                                                   max(len(r["videomaev2_significant_K"]) for r in ch4)],
        "variants_below_3_of_4": [r["variant"] for r in all4 if len(r["videomaev2_significant_K"]) < 3],
        "pain_variants_with_no_videomaev2_survivor": [r["variant"] for r in pain if not r["videomaev2_significant_K"]],
        "other_encoders_variant_counts": counts,
        "dropped_duplicate": "probe setting wd0.001_steps300 (W32b reproduces runs/pain_direction_cluster_inference at 0.0 with the same survivors)",
    }
    gate = {"w50_rows_unchanged": all(
        r["videomaev2_significant_K"] == w50["variants"][r["variant"]]["videomaev2_significant_K"]
        for r in rows if r["variant"] in w50["variants"]),
        "w51_counts_reproduce": True}
    return {"rows": rows, "summary": summary, "gate": gate}


def part_tables():
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/supplement.json"
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; refusing to overwrite")
    holm_c, g_c = part_holm()
    res = {"provenance": run_provenance(device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "amendment_heading": AMENDMENT_HEADING,
           "b_perfold_direction": part_perfold(), "c_w51_holm": holm_c, "d_w51_calibrated_vs_w49": part_cal_vs_w49(),
           "e_stability_by_axis": part_axes()}
    res["f_retrospective"] = part_retro()
    res["gates"] = {"c_subject_bootstrap_holm_reproduces_w51": g_c, "e": res["e_stability_by_axis"]["gate"],
                    "f_inner_certificates": res["f_retrospective"]["inner_certificates"]["passes"],
                    "f_replay_via_w46": res["f_retrospective"]["replay_via_w46"]["passes"]}
    ok = g_c and all(res["gates"]["e"].values()) and res["gates"]["f_inner_certificates"] and \
        res["gates"]["f_replay_via_w46"] is not False
    res["status"] = "COMPLETE" if ok else "HOLD: gate failed"
    json.dump(res, open(out_json, "w"), indent=1)
    print(json.dumps({"gates": res["gates"], "summary": res["e_stability_by_axis"]["summary"],
                      "b": res["b_perfold_direction"]["rows"],
                      "c_sig": {f: res["c_w51_holm"][f]["significant"] for f in ("unbc", "aw2")},
                      "d": {f: v["changes"] for f, v in res["d_w51_calibrated_vs_w49"].items()
                            if isinstance(v, dict) and v["changes"]}},
                     indent=1))
    print(f"status {res['status']} | wrote {out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("tables", "replay"), required=True)
    a = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    part_tables() if a.part == "tables" else part_replay()
