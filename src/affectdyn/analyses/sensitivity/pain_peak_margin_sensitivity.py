"""Margin sensitivity of the pain peak task.

Variants of the label rule (default: peak at frames 4-13 at least 1 PSPI above both endpoints; non-peaked needs
|net change| >= 1): net2, non-peaked needs |net change| >= 2; uniform2, also a peak at least 2 above both endpoints.
For each variant the label-only baseline -|d|, the 36 fixed-step probes refitted and tested against chance, and the
baseline-minus-encoder comparisons under the paired cluster bootstrap and the clustered paired z test, Holm across 36.

Usage: python -m affectdyn.analyses.sensitivity.pain_peak_margin_sensitivity
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse, json, sys, time
import numpy as np

import affectdyn.data.unbc_clip_pipeline as stage1_probe                                                          # noqa: E402
from affectdyn.inference.loso_permutation_tests import _loso_folds                                 # noqa: E402
from affectdyn.inference.aggregation_permutation_tests import holm                                               # noqa: E402
from affectdyn.inference.calibrated_clustered_auc import z_paired, p_from_z                  # noqa: E402
from affectdyn.analyses.peak.endpoint_magnitude_controls import midrank_auc                         # noqa: E402
from affectdyn.analyses.peak.pain_label_baseline import cells, refit, paired_boot, single_boot  # noqa: E402
from affectdyn.provenance import run_provenance                                        # noqa: E402

OUT = "runs/pain_peak_margin_sensitivity"
SEED = 20261171
ALPHA = 0.05


def labels(clips, peak_margin, net_margin):
    y, mask = [], []
    for c in clips:
        p = c["pspi"]; d = p[-1] - p[0]; a = int(np.argmax(p))
        peaked = 3 <= a <= 12 and p[a] >= p[0] + peak_margin and p[a] >= p[-1] + peak_margin
        y.append(int(peaked)); mask.append(bool(peaked or abs(d) >= net_margin))
    return np.array(y), np.array(mask)


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    nboot = 10_000
    out = OUT
    os.makedirs(out, exist_ok=True)
    logf = open(f"{out}/margin.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    rng = np.random.default_rng(SEED)
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    traj, sub, slope = lab["traj"], lab["subject"], lab["slope"]
    clips = stage1_probe.build_clips()
    res = {"provenance": run_provenance(seeds=[SEED]), "nboot": nboot, "gates": {}, "variants": {}}
    g1 = (np.array_equal(np.array([c["traj"] for c in clips]), traj)
          and np.allclose(np.array([c["slope"] for c in clips]), slope))
    y0, m0 = labels(clips, 1, 1)
    g2 = bool(np.array_equal(m0, traj != -1) and np.array_equal(y0[m0], (traj[m0] == 2).astype(int)))
    res["gates"] = {"G1_rebuild_matches_cache": bool(g1), "G2_published_rule_mask": g2, "n_published": int(m0.sum())}
    log(f"gates {res['gates']}")
    if not (g1 and g2):
        res["status"] = "HOLD: gate failed"; json.dump(res, open(f"{out}/margin.json", "w"), indent=2); return
    X = cells(); folds = _loso_folds(sub); dm = np.abs(slope)
    for name, pm, nm in (("V1_net2", 1, 2), ("V2_uniform2", 2, 2)):
        y, mask = labels(clips, pm, nm)
        npos = int(y[mask].sum()); nneg = int(mask.sum() - npos)
        pos_sub = np.unique(sub[mask & (y == 1)])
        train_ok = all(len(set(y[mask & np.array([folds[s_] != f for s_ in sub])].tolist())) == 2
                       for f in range(max(folds.values()) + 1))
        sup = {"n_clips": int(mask.sum()), "n_peaked": npos, "n_nonpeaked": nneg,
               "n_subjects": int(len(np.unique(sub[mask]))), "n_subjects_with_peaked": int(len(pos_sub)),
               "every_training_fold_two_classes": bool(train_ok)}
        sup["pass"] = bool(npos >= 15 and len(pos_sub) >= 8 and nneg >= 50 and train_ok)
        V = {"peak_margin": pm, "net_margin": nm, "support": sup}
        log(f"{name} support {sup}")
        if sup["pass"]:
            ym, gm = y[mask], sub[mask]
            V["baseline"] = single_boot(-dm[mask], ym, gm, rng, nboot)
            enc, comp = [], []
            for c, Xc in X.items():
                s = refit(Xc, y, sub, mask, folds)
                ok = mask & ~np.isnan(s)
                r = single_boot(s[ok], y[ok], sub[ok], rng, nboot); r["cell"] = c; enc.append(r)
                pb = paired_boot(-dm[ok], s[ok], y[ok], sub[ok], rng, nboot)
                d, z = z_paired(-dm[ok], s[ok], y[ok], sub[ok])
                comp.append({"cell": c, "diff": float(midrank_auc(-dm[ok], y[ok]) - midrank_auc(s[ok], y[ok])),
                             "boot": pb, "z": float(z), "p_z": float(p_from_z(z)),
                             "degenerate_z": bool(not np.isfinite(z) or z == 0.0)})
                log(f"  {name} {c:16s} AUC {r['auc']:.3f} diff {comp[-1]['diff']:+.3f}")
            for r, h in zip(enc, holm([r["p_boot"] for r in enc])):
                r["holm"] = h
            hb = holm([r["boot"]["p_boot"] for r in comp]); hz = holm([r["p_z"] for r in comp])
            for r, b, z_ in zip(comp, hb, hz):
                r["holm_boot"], r["holm_z"] = b, z_
                rb, rz = b < ALPHA, z_ < ALPHA
                sgn = np.sign(r["diff"])
                if rb and rz:
                    lo, hi = r["boot"]["ci"]
                    assert (lo > 0) == (sgn > 0) and np.sign(r["z"]) == sgn, f"sign mismatch {r['cell']}"
                r["outcome"] = ("disagree" if rb != rz else "neither" if not rb else
                                "baseline_higher" if sgn > 0 else "encoder_higher")
            V["encoders"] = enc; V["n_encoders_significant"] = int(sum(r["holm"] < ALPHA for r in enc))
            V["comparisons"] = comp
            V["counts"] = {k: int(sum(r["outcome"] == k for r in comp))
                           for k in ("baseline_higher", "encoder_higher", "disagree", "neither")}
            log(f"{name} baseline {V['baseline']['auc']:.3f} {V['baseline']['ci']}; encoders significant "
                f"{V['n_encoders_significant']}; comparison counts {V['counts']}")
        if sup["pass"]:
            sk = [r["boot"]["skipped"] for r in V["comparisons"]] + [r["skipped"] for r in V["encoders"]] \
                 + [V["baseline"]["skipped"]]
            nd = sum(r["degenerate_z"] for r in V["comparisons"])
            V["failure_gate"] = {"max_discarded_resamples": int(max(sk)), "degenerate_z": int(nd),
                                 "pass": bool(max(sk) <= 0.01 * nboot and nd <= 0.01 * len(V["comparisons"]))}
            log(f"{name} failure gate {V['failure_gate']}")
        res["variants"][name] = V
    res["status"] = "complete"
    json.dump(res, open(f"{out}/margin.json", "w"), indent=2, default=float)
    log("wrote " + f"{out}/margin.json")


if __name__ == "__main__":
    main()
