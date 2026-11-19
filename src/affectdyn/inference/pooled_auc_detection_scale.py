"""Minimum detectable effects on the pooled-AUC scale."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import collections, json, math, os, re, sys, time

import numpy as np

from affectdyn.inference.calibrated_clustered_auc import Design, _probit
from affectdyn.provenance import run_provenance

OUT = "runs/pooled_auc_detection_scale"
ICC = 0.3            # Use this subject ICC for bootstrap, calibrated, and within-subject simulations.
T = 16               # Each window contains 16 frames.
UNBC_CAP = 30


def ncdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def pooled_auc(theta, n_within, n_total, icc=ICC):
    """Exact pooled AUC of the simulated alternative, weighting the 2 kinds of pair."""
    if n_total == 0 or icc == 0:
        return float(theta)      # With zero ICC, every pair has ordering probability theta.
    p_within = ncdf(_probit(theta) / math.sqrt(1.0 - icc))
    return float((n_within * p_within + (n_total - n_within) * theta) / n_total)


def pair_counts(y, groups):
    d = Design("x", np.asarray(y).astype(np.int64), groups, 1)
    cp = collections.Counter(d.codes[d.pos]); cn = collections.Counter(d.codes[d.neg])
    within = sum(cp[s] * cn[s] for s in cp)
    return int(within), int(d.M * d.N), int(d.M), int(d.N), int(d.K)


def unbc_cap_effect():
    """Rebuild the UNBC clip set with and without the 30-window cap, counting what the cap drops."""
    recs = json.load(open("data/unbc/manifest.json"))
    by = {}
    for r in recs:
        m = re.search(r"(\d+)$", r["frame"])
        by.setdefault((r["subject"], r["seq"]), []).append(
            (int(m.group(1)) if m else 0, float(r["pspi"])))

    def build(maxper):
        clips, per = [], {}
        for (subj, seq), fr in sorted(by.items()):
            fr.sort()
            for i in range(0, len(fr) - T, T // 2):
                pspi = np.array([w[1] for w in fr[i:i + T]], np.float32)
                if pspi.max() < 1 and pspi.std() < 0.5:
                    continue
                if per.get(subj, 0) >= maxper:
                    continue
                a = int(np.argmax(pspi))
                clips.append({"subject": subj, "seq": seq, "start": i,
                              "peaked": bool(3 <= a <= 12 and pspi[a] >= pspi[0] + 1
                                             and pspi[a] >= pspi[-1] + 1)})
                per[subj] = per.get(subj, 0) + 1
        return clips

    def describe(clips):
        pk = [c for c in clips if c["peaked"]]
        ev = 0
        for key in {(c["subject"], c["seq"]) for c in pk}:
            s = sorted(c["start"] for c in pk if (c["subject"], c["seq"]) == key)
            ev += 1 + sum(1 for a, b in zip(s, s[1:]) if b - a >= T)
        return {"n_windows": len(clips), "n_peaked": len(pk), "n_peak_events": ev,
                "n_peak_bearing_subjects": len({c["subject"] for c in pk}),
                "n_sequences_with_a_peak": len({(c["subject"], c["seq"]) for c in pk})}

    capped, uncapped = describe(build(UNBC_CAP)), describe(build(10 ** 9))
    removed = {k: uncapped[k] - capped[k] for k in capped}
    return {"cap": UNBC_CAP, "with_cap": capped, "without_cap": uncapped, "removed_by_cap": removed}


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/pooled_scale.log", "w")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:6.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    sets = {}
    for name, pre, mask in (("unbc", "unbc", "keep"), ("aw2_full", "aw2img", "mask"),
                            ("aw2_subset", "aw2vid", "mask")):
        k = W7[f"{pre}__{mask}"].astype(bool)
        w, tot, M, N, K = pair_counts(W7[f"{pre}__y"][k], W7[f"{pre}__subject"][k])
        sets[name] = {"n_within_pairs": w, "n_pairs": tot, "M": M, "N": N, "K": K,
                      "within_share": w / tot}
        p(f"{name:11s} M {M:5d} N {N:6d} K {K:4d} | within-subject pairs {w}/{tot} "
          f"= {100*w/tot:.3f} percent")

    # Reconstruct the clip set and verify its count against the run.
    from affectdyn.analyses.peak.endpoint_magnitude_controls import match_on_magnitude                     # noqa: PLC0415
    _lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    _keep = W7["unbc__keep"].astype(bool)
    _y = (_lab["traj"] == 2).astype(np.int64)[_keep]
    _g = _lab["subject"][_keep]
    _pairs, _ = match_on_magnitude(np.abs(_lab["slope"][_keep]), _y)
    _idx = np.array(sorted([i for pr in _pairs for i in pr[:2]]))
    w, tot, M, N, K = pair_counts(_y[_idx], _g[_idx])
    sets["unbc_matched"] = {"n_within_pairs": w, "n_pairs": tot, "M": M, "N": N, "K": K,
                            "within_share": w / tot}
    p(f"{'unbc_matched':11s} M {M:5d} N {N:6d} K {K:4d} | within-subject pairs {w}/{tot} "
      f"= {100*w/tot:.3f} percent")
    _w14 = json.load(open("runs/endpoint_magnitude_controls/endpoint_magnitude.json"))
    assert len(_idx) == _w14["matching"]["n_clips_matched"], (len(_idx), _w14["matching"])

    mde = json.load(open("runs/peak_cluster_inference/mde.json"))
    bp = json.load(open("runs/bootstrap_detection_power/bootstrap_power.json"))
    cp = json.load(open("runs/calibrated_detection_limits/calibrated_power.json"))
    # Store each benchmark's name, clip set, theta, and alternative ICC.
    w7u, w7a = mde["corpora"]["unbc"]["results"], mde["corpora"]["aff_wild2"]["results"]
    items = [
        ("unbc:label_permutation", "unbc", w7u["clip"]["mde_auc_at_80pc_power"], 0.0),
        ("unbc:within_subject_permutation", "unbc", w7u["subject"]["mde_auc_at_80pc_power"], ICC),
        ("unbc:subject_bootstrap", "unbc", bp["clip_sets"]["unbc"]["mde_boot_icc0.3"]["mde_auc_at_80pc_power"], ICC),
        ("unbc:calibrated", "unbc", cp["clip_sets"]["unbc_shape"]["mde_auc_at_80pc_power"], ICC),
        ("aw2_full:label_permutation", "aw2_full", w7a["clip"]["mde_auc_at_80pc_power"], 0.0),
        ("aw2_full:within_subject_permutation", "aw2_full", w7a["subject"]["mde_auc_at_80pc_power"], ICC),
        ("aw2_full:subject_bootstrap", "aw2_full",
         bp["clip_sets"]["aff_wild2_full"]["mde_boot_icc0.3"]["mde_auc_at_80pc_power"], ICC),
        ("aw2_full:calibrated", "aw2_full", cp["clip_sets"]["aw2_full"]["mde_auc_at_80pc_power"], ICC),
        ("aw2_subset:subject_bootstrap", "aw2_subset",
         bp["clip_sets"]["aff_wild2_subset"]["mde_boot_icc0.3"]["mde_auc_at_80pc_power"], ICC),
        ("aw2_subset:calibrated", "aw2_subset", cp["clip_sets"]["aw2_subset"]["mde_auc_at_80pc_power"], ICC),
        ("unbc_matched:label_permutation", "unbc_matched",
         _w14["mde"]["matched"]["icc0.3"]["mde_auc_at_80pc_power"], ICC),
    ]
    p("")
    scale = {}
    for key, cs, th, icc in items:
        s = sets[cs]
        po = pooled_auc(th, s["n_within_pairs"], s["n_pairs"], icc)
        scale[key] = {"clip_set": cs, "theta": float(th), "pooled_auc": po,
                      "shift": po - float(th), "icc": icc}
        p(f"  {key:36s} icc {icc:.1f} theta {th:.4f} -> pooled {po:.4f}  (shift {po - th:+.4f})")

    cap = unbc_cap_effect()
    p("")
    p(f"UNBC {UNBC_CAP}-window cap: with {cap['with_cap']}")
    p(f"                          without {cap['without_cap']}")
    p(f"                          removed {cap['removed_by_cap']}")

    res = {"provenance": run_provenance(), "icc": "per benchmark, see mde[*].icc", "note": (
        "Pooled AUC of the simulated alternative, weighting within-subject pairs at "
        "Phi(Phi^-1(theta)/sqrt(1-ICC)) and between-subject pairs at theta, with each benchmark's own "
        "ICC. The calculation is exact; the label-permutation benchmark has ICC 0."),
        "clip_sets": sets, "mde": scale, "unbc_cap": cap}
    json.dump(res, open(f"{OUT}/pooled_scale.json", "w"), indent=2)
    p(f"wrote {OUT}/pooled_scale.json")
    logf.close()


if __name__ == "__main__":
    main()
