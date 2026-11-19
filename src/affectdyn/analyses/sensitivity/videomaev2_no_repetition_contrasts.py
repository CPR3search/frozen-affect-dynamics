"""CLIP against VideoMAEv2 on pain direction without frame repetition.

Usage: python -m affectdyn.analyses.sensitivity.videomaev2_no_repetition_contrasts
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, math, sys, time
import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                        # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                  # noqa: E402
from affectdyn.features.temporal_aggregation import image_readout, video_readout                    # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                                # noqa: E402
from affectdyn.inference.calibrated_clustered_auc import Clustered, cov, z_paired             # noqa: E402
from affectdyn.analyses.peak.endpoint_magnitude_controls import midrank_auc                          # noqa: E402
from affectdyn.analyses.sensitivity.subject_partition_contrasts import fixed_step, converged             # noqa: E402
from affectdyn.provenance import run_provenance                                         # noqa: E402

OUT = "runs/videomaev2_no_repetition_contrasts"
MARGIN = 0.035
Z90 = 1.6448536269514722
KS = (2, 4)


def p_ref(z, null):
    return float((1 + (np.abs(null) >= abs(z)).sum()) / (1 + len(null)))


def classify(g, sig, lo, hi):
    if sig and g > 0 and lo >= -MARGIN and hi <= MARGIN:
        return "a", ("the gap persisted under the unrepeated input, and the change from the repeated input lay within "
                     "+-0.035 (a descriptive interval, not an equivalence test)")
    if sig and g > 0:
        return "b", "the gap persisted, but a material input-protocol effect could not be excluded"
    if sig and g < 0:
        return "c", "under the unrepeated input the ordering reversed"
    if not sig and hi < -MARGIN:
        return "d", "the pain advantage depended materially on the repeated-window input"
    return "e", "the gap was not significant under the unrepeated input and the change could not be resolved beyond the margin"


def main():
    logf = open(f"{OUT}/analysis.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    ex = json.load(open(f"{OUT}/extract.json")); rp = json.load(open("runs/pipeline_simulation_resampled/replay.json"))
    null = np.load("runs/pipeline_simulation_resampled/w12_direction_null.npy")
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub = lab["subject"]; y = (lab["direction"] == 1).astype(int); md = lab["direction"] != -1
    lf = _loso_folds(sub); fid = np.array([lf[s] for s in sub])
    P = np.load("runs/fixed_step_penalty_sweep/scores_wd0.001_steps300.npz"); keep = P["keep"].astype(bool)
    Z = np.load(f"{OUT}/videomaev2_native.npz"); IMG = np.load("runs/temporal_aggregation/img_frames.npz")
    res = {"provenance": run_provenance(seeds=[]), "extraction_gates_pass": ex["all_gates_pass"],
           "replay_gate_pass": rp["gate_pass"], "G1_auc": {}, "fits": {}}
    if not (ex["all_gates_pass"] and rp["gate_pass"]):
        res["status"] = "could not be validated (an upstream gate failed)"
        json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=2); log(res["status"]); return

    X = {}
    for k in KS:
        X[f"clip_K{k}"] = image_readout(IMG["clip"], k)
        X[f"rep_K{k}"] = video_readout(Z["full"], Z[f"K{k}_up"])
        X[f"nat_K{k}"] = video_readout(Z["full"], Z[f"K{k}_native"])
    S = {"fixed": {}, "converged": {}}
    for c, Xc in X.items():
        S["fixed"][c] = fixed_step(Xc, y, fid, md)
        S["converged"][c], au = converged(Xc, y, sub, fid, md)
        res.setdefault("converged_audit", {})[c] = au
        if au["outer_uncertified"]:
            raise RuntimeError(f"{c}: {au['outer_uncertified']} uncertified converged fits")
        log(f"scored {c}")
    g1 = {}
    for k in KS:
        da = abs(_auc(S["fixed"][f"rep_K{k}"][keep], y[keep]) - _auc(P[f"videomaev2_K{k}"][keep], y[keep]))
        gp = _auc(P[f"clip_K{k}"][keep], y[keep]) - _auc(P[f"videomaev2_K{k}"][keep], y[keep])
        gn = _auc(S["fixed"][f"clip_K{k}"][keep], y[keep]) - _auc(S["fixed"][f"rep_K{k}"][keep], y[keep])
        g1[f"K{k}"] = {"vmv2_auc_delta": float(da), "gap_delta": float(abs(gp - gn))}
    res["G1_auc"] = {"rows": g1, "pass": all(v["vmv2_auc_delta"] <= 1e-4 and v["gap_delta"] <= 1e-4 for v in g1.values())}
    log(f"G1 (AUC part) {res['G1_auc']}")
    if not res["G1_auc"]["pass"]:
        res["status"] = "could not be validated (G1 AUC part failed)"
        json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=2, default=float); log(res["status"]); return

    store = {}
    for fit in ("fixed", "converged"):
        rows = {}
        for k in KS:
            a, r, n = S[fit][f"clip_K{k}"], S[fit][f"rep_K{k}"], S[fit][f"nat_K{k}"]
            ok = keep & ~np.isnan(a) & ~np.isnan(r) & ~np.isnan(n)
            if ok.sum() != keep.sum():
                raise RuntimeError(f"{fit} K={k}: coverage {ok.sum()}/{keep.sum()}")
            yy, gg = y[ok], sub[ok]
            g, z = z_paired(a[ok], n[ok], yy, gg)
            cr, cn = Clustered(r[ok], yy, gg), Clustered(n[ok], yy, gg)
            dlt = cr.auc - cn.auc; se = math.sqrt(max(cov(cr, cr) + cov(cn, cn) - 2 * cov(cr, cn), 0.0))
            rows[f"K{k}"] = {"auc_clip": midrank_auc(a[ok], yy), "auc_rep": cr.auc, "auc_native": cn.auc,
                             "gap_native": g, "gap_repeated": midrank_auc(a[ok], yy) - cr.auc, "z": z, "p_ref": p_ref(z, null),
                             "delta": dlt, "delta_ci90": [dlt - Z90 * se, dlt + Z90 * se], "n": int(ok.sum())}
            for nm, v in (("clip", a), ("rep", r), ("nat", n)):
                store[f"{fit}_{nm}_K{k}"] = v
        for kk, h in zip(rows, holm([rows[kk]["p_ref"] for kk in rows])):
            rows[kk]["holm2"] = h
            lo, hi = rows[kk]["delta_ci90"]
            rows[kk]["outcome"], rows[kk]["sentence"] = classify(rows[kk]["gap_native"], h < 0.05, lo, hi)
            log(f"{fit} {kk}: CLIP {rows[kk]['auc_clip']:.4f} rep {rows[kk]['auc_rep']:.4f} native "
                f"{rows[kk]['auc_native']:.4f} | gap native {rows[kk]['gap_native']:+.4f} Holm-2 {h:.5f} | Delta "
                f"{rows[kk]['delta']:+.4f} CI90 [{lo:+.4f}, {hi:+.4f}] -> ({rows[kk]['outcome']})")
        res["fits"][fit] = rows
    np.savez_compressed(f"{OUT}/analysis_scores.npz", **store)
    res["status"] = "complete"
    json.dump(res, open(f"{OUT}/analysis.json", "w"), indent=2, default=float)
    log(f"wrote {OUT}/analysis.json")


if __name__ == "__main__":
    main()
