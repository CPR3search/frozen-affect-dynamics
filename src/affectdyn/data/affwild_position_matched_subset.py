"""Construction of the position-matched Aff-Wild2 set.

Usage:  python -m affectdyn.data.affwild_position_matched_subset
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import json, os, sys, time
from collections import defaultdict

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc                                                  # noqa: E402
from affectdyn.provenance import run_provenance                                                   # noqa: E402

OUT = "runs/position_matched_peak_analysis"
SEED = 20260959
WINDOW = 16
LADDER = (32, 64, 96)
MARGIN = (0.45, 0.55)
MIN_PEAKS = 200
N_GATE = 32


def match_video(peaks, negs, start, cal):
    """Exact 1:1 matching inside one video: maximum cardinality, then minimum total |delta start|."""
    pk_starts = [start[p] for p in peaks]
    ok_neg = [n for n in negs if all(abs(start[n] - s) >= WINDOW for s in pk_starts)]
    cand = {p: sorted((n for n in ok_neg if abs(start[n] - start[p]) <= cal),
                      key=lambda n: (abs(start[n] - start[p]), start[n])) for p in peaks}
    order = sorted(peaks, key=lambda p: (start[p], p))
    best = {"k": -1, "cost": float("inf"), "pairs": []}

    def dfs(i, used, pairs, cost):
        remaining = len(order) - i
        if len(pairs) + remaining < best["k"]:
            return
        if i == len(order):
            if len(pairs) > best["k"] or (len(pairs) == best["k"] and cost < best["cost"]):
                best.update(k=len(pairs), cost=cost, pairs=list(pairs))
            return
        p = order[i]
        for n in cand[p]:
            if all(abs(start[n] - start[u]) >= WINDOW for u in used):
                used.append(n); pairs.append((p, n))
                dfs(i + 1, used, pairs, cost + abs(start[n] - start[p]))
                used.pop(); pairs.pop()
        dfs(i + 1, used, pairs, cost)                      # Leave this peak unmatched.

    dfs(0, [], [], 0.0)
    return best["pairs"]


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_json = f"{OUT}/matched_subset.json"
    if os.path.exists(out_json):
        raise SystemExit(f"{out_json} exists; the design is frozen once written")
    prov = run_provenance(seeds=[SEED], device="cpu")
    t0 = time.time()
    clips = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    subj = np.array([str(c["subject"]) for c in clips]); vid = np.array([str(c["video"]) for c in clips])
    traj = np.array([int(c["traj"]) for c in clips])
    start = np.array([int(os.path.splitext(os.path.basename(c["start"]))[0]) for c in clips])
    rel = np.empty(len(clips))
    for s in np.unique(subj):
        ix = np.flatnonzero(subj == s); rel[ix] = np.arange(len(ix)) / max(1, len(ix) - 1)
    byv = defaultdict(list)
    for i in range(len(clips)):
        byv[vid[i]].append(i)
    n_peaks = int((traj == 2).sum())
    pp_overlap = sum(1 for v, ix in byv.items() for a in ix for b in ix
                     if a < b and traj[a] == 2 and traj[b] == 2 and abs(start[a] - start[b]) < WINDOW)

    ladder = {}
    for cal in LADDER:
        pairs = []
        for v, ix in byv.items():
            P = [i for i in ix if traj[i] == 2]
            if P:
                pairs += match_video(P, [i for i in ix if traj[i] != 2], start, cal)
        keep = np.array(sorted({i for pr in pairs for i in pr}), dtype=int)
        y = (traj[keep] == 2).astype(int)
        a_rel = float(_auc(rel[keep], y)); a_start = float(_auc(start[keep].astype(float), y))
        d = np.array([abs(start[p] - start[n]) for p, n in pairs])
        passes = bool(MARGIN[0] <= a_rel <= MARGIN[1] and MARGIN[0] <= a_start <= MARGIN[1]
                      and len(pairs) >= MIN_PEAKS)
        ladder[str(cal)] = {"caliper_frames": cal, "matched_peaks": len(pairs), "discarded_peaks": n_peaks - len(pairs),
                            "clips": int(len(keep)), "videos": int(len(set(vid[keep]))),
                            "subjects": int(len(set(subj[keep]))),
                            "subjects_with_peaks": int(len(set(subj[[p for p, _ in pairs]]))),
                            "rel_position_auc": a_rel, "start_frame_auc": a_start,
                            "delta_start_frames": {"median": float(np.median(d)), "max": int(d.max()),
                                                   "hist": {str(k): int((d == k).sum()) for k in sorted(set(d.tolist()))}},
                            "rel_quartiles_pos": np.percentile(rel[keep][y == 1], [25, 50, 75]).tolist(),
                            "rel_quartiles_neg": np.percentile(rel[keep][y == 0], [25, 50, 75]).tolist(),
                            "passes": passes, "pairs": [[int(p), int(n)] for p, n in pairs]}
        print(f"[{time.time() - t0:6.1f}s] caliper {cal}: {len(pairs)} peaks matched, rel AUC {a_rel:.3f}, "
              f"start AUC {a_start:.3f}, passes {passes}", flush=True)

    chosen = next((c for c in LADDER if ladder[str(c)]["passes"]), None)
    rng = np.random.default_rng(SEED)
    pub = np.load("runs/affwild_clips_features/video_subset_idx.npy")
    gate_pos = rng.choice(pub[traj[pub] == 2], N_GATE, replace=False)
    gate_neg = rng.choice(pub[traj[pub] != 2], N_GATE, replace=False)
    res = {"provenance": prov, "criteria_heading": "## 2026-09-22 — W56-W59 PRE-DECLARED CRITERIA",
           "amendment_heading": "## 2026-09-22 — W56-W59 CRITERIA AMENDMENT",
           "n_clips_manifest": len(clips), "n_peaks": n_peaks, "positive_positive_overlapping_pairs": pp_overlap,
           "ladder": ladder, "chosen_caliper": chosen,
           "status": "FROZEN" if chosen else "FAILED: no caliper met the balance margins and the peak minimum",
           "extraction_gate_idx": sorted(int(i) for i in np.concatenate([gate_pos, gate_neg])),
           "note": "40-window cap unchanged; this is the full manifest, not uncapped video."}
    if chosen:
        idx = np.array(sorted({i for pr in ladder[str(chosen)]["pairs"] for i in pr}), dtype=int)
        np.save(f"{OUT}/matched_subset_idx.npy", idx)
        res["n_subset"] = int(len(idx))
        res["already_cached_in_published_subset"] = int(np.isin(idx, pub).sum())
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"wrote {out_json} | {res['status']}", flush=True)


if __name__ == "__main__":
    main()
