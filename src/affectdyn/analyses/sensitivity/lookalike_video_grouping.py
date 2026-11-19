"""Aff-Wild2 peak task with look-alike video-ID groups merged.

Usage:  python -m affectdyn.analyses.sensitivity.lookalike_video_grouping [--montage DIR]
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Run on the CPU.
import argparse, collections, hashlib, json, sys, time, zlib

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.features.temporal_aggregation import image_readout, video_readout, IMG_ENCODERS
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.probes.affwild_image_fixed_step import grouped_folds, oof
from affectdyn.inference.peak_cluster_inference import bootstrap as w7_bootstrap
from affectdyn.provenance import run_provenance

OUT = "runs/lookalike_video_grouping"
SEED = 20260925
NBOOT = 10_000
CHUNK = 2500
ALPHA = 0.05
TOL = 0.005
NFOLD = 5
KS = (2, 4, 8, 16)
VIDEO = ("vjepa", "videomae", "videomaev2", "timesformer", "xclip", "internvideo2")
IMG_SEEDS, VID_SEEDS = range(100, 103), range(300, 305)
QUANTILES = (5, 25, 50)
N_AUDIT = 8
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-18 — W35-W37 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMENDMENT_HEADING = "## 2026-09-18 — W35-W37 CRITERIA AMENDMENT"


def mult(draw, G):
    b = draw.shape[0]
    return np.bincount((draw + (np.arange(b) * G)[:, None]).ravel(), minlength=b * G).reshape(b, G).astype(float)


def boot(s, y, groups, key):
    b = w7_bootstrap(s, y, groups, "subject", np.random.default_rng([SEED, zlib.crc32(key.encode())]), NBOOT)
    return {"p_boot": b["p_boot"], "ci": b["ci"], "skipped": b["resamples_skipped"]}


class UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def repair(assign, sub, comp_of):
    new = assign.copy()
    moved_ids, moved_clips = set(), 0
    by_comp = collections.defaultdict(list)
    for i, s in enumerate(sub):
        by_comp[comp_of[s]].append(i)
    for comp, rows in by_comp.items():
        ids = {sub[i] for i in rows}
        if len(ids) < 2:
            continue
        counts = np.bincount(assign[rows], minlength=NFOLD)
        target = int(np.argmax(counts))   # Resolve ties with the lowest index.
        for i in rows:
            if new[i] != target:
                new[i] = target; moved_clips += 1; moved_ids.add(sub[i])
    return new, sorted(moved_ids), moved_clips


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--montage", default=None, help="scratch directory for the audit montage, outside runs/")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    if args.montage and os.path.abspath(args.montage).startswith(os.path.abspath("runs")):
        raise SystemExit("the audit montage holds face images and must not be written under runs/")
    os.makedirs(OUT, exist_ok=True)
    stem = "lookalike_grouping"
    out_json = f"{OUT}/{stem}.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    logf = open(f"{OUT}/{stem}.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    torch.set_num_threads(2)
    clips = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    lab = np.load("runs/affwild_clips_features/labels.npz", allow_pickle=True)
    sub_pub = np.array([str(s) for s in lab["subject"]])
    video = np.array([c["video"] for c in clips])
    start = np.array([c["start"] for c in clips])
    traj = lab["traj"]
    y_full = (traj == 2).astype(int)
    mask_full = traj != -1
    idx = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    frames = np.load("runs/affwild_clips_features/img_frames.npz")

    F = frames["dinov2"].astype(np.float32)
    F /= np.linalg.norm(F, axis=2, keepdims=True)
    clip_mean = F.mean(1)
    vids = sorted(set(video))
    vmean = {}
    ref = []
    for v in vids:
        rows = np.flatnonzero(video == v)
        m = clip_mean[rows].mean(0); vmean[v] = m / np.linalg.norm(m)
        if len(rows) >= 4:
            rows = rows[np.argsort(start[rows], kind="stable")]
            h = len(rows) // 2
            a, b = clip_mean[rows[:h]].mean(0), clip_mean[rows[h:]].mean(0)
            ref.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    ref = np.asarray(ref)
    thresholds = {q: float(np.percentile(ref, q)) for q in QUANTILES}
    group_of_video = {v: sub_pub[np.flatnonzero(video == v)[0]] for v in vids}
    V = np.stack([vmean[v] for v in vids])
    sim = V @ V.T
    pairs = [(sim[i, j], vids[i], vids[j]) for i in range(len(vids)) for j in range(i + 1, len(vids))
             if group_of_video[vids[i]] != group_of_video[vids[j]]]
    pairs.sort(reverse=True)
    cross = np.array([p[0] for p in pairs])
    log(f"W37 | {len(vids)} videos, {len(set(sub_pub))} identifier groups | same-session reference over {len(ref)} "
        f"videos, percentiles {thresholds} | cross-group pairs {len(pairs)}, max {cross.max():.4f}, "
        f"median {np.median(cross):.4f} | seed {SEED}")

    cellsX = {}
    for e in IMG_ENCODERS:
        for k in KS:
            cellsX[(f"{e}_K{k}", "image")] = image_readout(frames[e], k)
    for e in VIDEO:
        z = np.load(f"runs/affwild_clips_features/{e}_windows.npz")
        cellsX[(f"{e}_K16", "video")] = video_readout(z["full"], z["K16_up"])
    arms = {"image": (y_full, mask_full, sub_pub, np.arange(len(sub_pub))),
            "video": (y_full[idx], mask_full[idx], sub_pub[idx], idx)}
    pub_assign = {a: grouped_folds(arms[a][2]) for a in arms}
    pub_img = {f"{r['encoder']}_K{r['K']}": r["auc"] for r in json.load(open("runs/affwild_clips_features/probe.json"))["shape"]["rows"]}
    pub_vid = {f"{r['encoder']}_K16": r["auc"] for r in json.load(open("runs/affwild_clips_features/video_probe.json"))["shape"]}
    published = {**pub_img, **pub_vid}
    W7 = {c["cell"]: c for c in json.load(open("runs/peak_cluster_inference/cluster.json"))["cells"] if c["corpus"] != "unbc"}

    def fit_family(assigns, tag):
        scores = {}
        for (cell, arm), X in cellsX.items():
            yy, mm, _, _ = arms[arm]
            seeds = IMG_SEEDS if arm == "image" else VID_SEEDS
            scores[cell] = np.nanmean([oof(X, yy, assigns[arm], mm, sd, "cpu") for sd in seeds], axis=0)
        log(f"  refit {tag}: {len(scores)} tests")
        return scores

    def retest(scores, groups_by_arm, tag):
        rows = []
        for (cell, arm), _ in cellsX.items():
            yy, mm, _, _ = arms[arm]
            s = scores[cell]
            keep = mm & ~np.isnan(s)
            b = boot(s[keep], yy[keep].astype(np.int64), groups_by_arm[arm][keep], cell)
            rows.append({"cell": cell, "arm": arm, "n": int(keep.sum()),
                         "auc": float(_auc(s[keep], yy[keep].astype(np.int64))), **b})
        for r, a in zip(rows, holm([r["p_boot"] for r in rows])):
            r["holm"] = a; r["significant"] = bool(a < ALPHA)
        return rows

    S_a = fit_family(pub_assign, "(a) published folds")
    Z7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    dmax, smax, finite_equal = 0.0, 0.0, True
    for (cell, arm), _ in cellsX.items():
        yy, mm, _, _ = arms[arm]
        stored = Z7[("aw2img__" if arm == "image" else "aw2vid__") + cell]
        keep = mm & ~np.isnan(S_a[cell])
        finite_equal &= bool(np.array_equal(keep, mm & ~np.isnan(stored)))
        dmax = max(dmax, abs(_auc(S_a[cell][keep], yy[keep].astype(np.int64)) - W7[cell]["auc"]))
        smax = max(smax, float(np.nanmax(np.abs(S_a[cell][keep] - stored[keep]))))
    w36 = None
    for fn in ("runs/bootstrap_decision_variation/mc_error.json",):
        if os.path.exists(fn):
            w36 = json.load(open(fn))["gates"].get("aw2_shape18", {}).get("passes"); break
    fold_hash = {a: hashlib.sha256(pub_assign[a].astype(np.int64).tobytes()).hexdigest()[:16] for a in arms}
    gate = {"max_abs_auc_delta_vs_w7": dmax, "max_abs_score_delta_vs_w7_stored": smax,
            "evaluable_clips_equal_w7": finite_equal, "tolerance": TOL, "fold_hash": fold_hash,
            "w36_exact_replay_of_w7_passes": w36,
            "note": ("arm (a) against the published run differs by the resampling draw AND by any refit "
                     "difference, which max_abs_score_delta_vs_w7_stored reports"),
            "passes": bool(dmax <= TOL and finite_equal and w36 is True)}
    log(f"gate: {gate}")
    res = {"provenance": run_provenance(seeds=[SEED], device="cpu"), "criteria_heading": CRITERIA_HEADING,
           "amendment_heading": AMENDMENT_HEADING, "status_frozen": "sensitivity analysis; never the analysis of record",
           "similarity": {"reference_n_videos": int(len(ref)), "reference_percentiles": thresholds,
                          "cross_group_pairs": len(pairs), "cross_max": float(cross.max()),
                          "cross_median": float(np.median(cross)),
                          "cross_pairs_above": {q: int((cross >= t).sum()) for q, t in thresholds.items()},
                          "top_pairs": [{"sim": float(s), "video_a": a, "video_b": b,
                                         "group_a": group_of_video[a], "group_b": group_of_video[b]}
                                        for s, a, b in pairs[:N_AUDIT]]},
           "gate": gate}
    if args.montage:
        from PIL import Image
        os.makedirs(args.montage, exist_ok=True)
        tiles = []
        for s, a, b in pairs[:N_AUDIT]:
            row = []
            for v in (a, b):
                fs = sorted(os.listdir(f"data/cropped_aligned/{v}"))
                row.append(Image.open(f"data/cropped_aligned/{v}/{fs[len(fs) // 2]}").convert("RGB").resize((160, 160)))
            tiles.append(row)
        M = Image.new("RGB", (330, 165 * len(tiles)), "white")
        for i, (l, r) in enumerate(tiles):
            M.paste(l, (0, 165 * i)); M.paste(r, (170, 165 * i))
        M.save(os.path.join(args.montage, "w37_audit_montage.png"))
        log(f"audit montage written to {args.montage} (outside runs/, not committed)")
    if not gate["passes"]:
        res["status"] = "HOLD: gate failed"
        json.dump(res, open(out_json, "w"), indent=2)
        log(f"wrote {out_json} | {res['status']}")
        logf.close(); return

    pub_groups = {a: arms[a][2] for a in arms}
    rows_a = retest(S_a, pub_groups, "(a)")
    pub_sig = {c for c, x in W7.items() if x["holm_pooled"]["boot_subject"] < ALPHA}
    res["arm_a"] = {"rows": rows_a, "significant": sorted(r["cell"] for r in rows_a if r["significant"]),
                    "differs_from_published": sorted({r["cell"] for r in rows_a if r["significant"]} ^ pub_sig)}
    log(f"(a) significant {res['arm_a']['significant']} | differs from published {res['arm_a']['differs_from_published']}")

    scen = {}
    for q, thr in thresholds.items():
        uf = UF(sorted(set(sub_pub)))
        for s, a, b in pairs:
            if s < thr:
                break
            uf.union(group_of_video[a], group_of_video[b])
        comp_of = {g: uf.find(g) for g in set(sub_pub)}
        sizes = collections.Counter(comp_of.values())
        nonsingle = {c: n for c, n in sizes.items() if n > 1}
        merged_sub = np.array([comp_of[s] for s in sub_pub])
        members = collections.defaultdict(list)
        for g_ in sorted(set(sub_pub)):
            members[comp_of[g_]].append(g_)
        info = {"threshold": thr, "components_merged": len(nonsingle),
                "identifiers_in_merged": int(sum(nonsingle.values())),
                "largest_component": int(max(sizes.values())), "clips_in_merged":
                int(sum(1 for s in sub_pub if sizes[comp_of[s]] > 1)),
                "component_members": {c: members[c] for c in nonsingle}}
        rep_assign, moved = {}, {}
        for a in arms:
            _, _, sub_a, rows = arms[a]
            rep_assign[a], ids, nclips = repair(pub_assign[a], sub_a, comp_of)
            span = sum(1 for c in nonsingle if len({int(x) for x, s in zip(pub_assign[a], sub_a) if comp_of[s] == c}) > 1)
            yy, mm = arms[a][0], arms[a][1]
            both = all(len(set(yy[(rep_assign[a] != k) & mm].tolist())) == 2 for k in range(NFOLD))
            moved[a] = {"identifiers_moved": len(ids), "moved_identifiers": ids, "clips_moved": nclips,
                        "components_spanning_folds": span, "every_training_fold_has_both_classes": bool(both)}
        info["moved"] = moved
        if not all(m["every_training_fold_has_both_classes"] for m in moved.values()):
            info["status"] = "NON-EVALUABLE: a repaired training fold lacks a class"
            scen[q] = info; log(f"threshold p{q}: {info['status']}"); continue
        folds_moved = any(m["clips_moved"] > 0 for m in moved.values())
        if folds_moved:
            S_b = fit_family(rep_assign, f"(b) p{q}")
            evaluable = all(np.array_equal(arms[a][1] & ~np.isnan(S_b[c]), arms[a][1] & ~np.isnan(S_a[c]))
                            for (c, a) in cellsX)
            if not evaluable:
                info["status"] = "NON-EVALUABLE: a repaired fold lost evaluable clips"
                scen[q] = info; log(f"threshold p{q}: {info['status']}"); continue
            rows_b = retest(S_b, pub_groups, f"(b) p{q}")
        else:
            S_b, rows_b = S_a, rows_a   # Reuse results when the folds are identical.
        info["fold_check"] = ("informative" if folds_moved else
                              "UNINFORMATIVE: no merged component spans a fold, so the folds are unchanged")
        merged_groups = {a: merged_sub[arms[a][3]] for a in arms}
        rows_c = retest(S_b, merged_groups, f"(c) p{q}")
        sig = lambda rows: {r["cell"] for r in rows if r["significant"]}
        info.update({"status": "EVALUATED", "arm_b": rows_b if folds_moved else "identical to arm (a)", "arm_c": rows_c,
                     "changed_a_to_b": sorted(sig(rows_a) ^ sig(rows_b)),
                     "changed_b_to_c": sorted(sig(rows_b) ^ sig(rows_c)),
                     "changed_a_to_c": sorted(sig(rows_a) ^ sig(rows_c))})
        scen[q] = info
        log(f"threshold p{q} ({thr:.4f}): merged {info['components_merged']} components, moved "
            f"{ {a: m['clips_moved'] for a, m in moved.items()} } | a->b {info['changed_a_to_b']} "
            f"| b->c {info['changed_b_to_c']}")
    res["scenarios"] = scen
    res["visual_audit"] = {"status": "PENDING: judgements of the top pairs are recorded by hand in the research log",
                           "pairs": res["similarity"]["top_pairs"]}
    res["status"] = "COMPLETE"
    json.dump(res, open(out_json, "w"), indent=2)
    log(f"wrote {out_json} | COMPLETE")
    logf.close()


if __name__ == "__main__":
    main()
