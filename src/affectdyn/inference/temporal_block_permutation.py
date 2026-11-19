"""Within-cluster permutation with contiguous blocks and non-overlapping clips."""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import json, os, re, sys, time

import numpy as np

from affectdyn.inference.peak_cluster_inference import auc_ranks, perm_test
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.provenance import run_provenance

OUT = "runs/temporal_block_permutation"
SEED = 20260913
NPERM = 20_000
WINDOW = 16          # Windows overlap when their starts differ by fewer than 16 frames.
BLOCKS = (2, 4, 8)


def clip_order(manifest_clips):
    """Video id and window start frame per clip, in the stored array's own order."""
    vid = np.array([c["video"] for c in manifest_clips])
    start = np.array([int(re.search(r"(\d+)\.jpg$", c["start"]).group(1)) for c in manifest_clips])
    return vid, start


def subject_runs(sub, vid, start, keep_idx):
    """Each subject's clip positions, ordered in time and split where the video changes."""
    out = {}
    for s in np.unique(sub):
        pos = np.flatnonzero(sub == s)
        order = sorted(pos.tolist(), key=lambda i: (str(vid[keep_idx[i]]), int(start[keep_idx[i]])))
        runs, cur = [], [order[0]]
        for a, b in zip(order, order[1:]):
            if vid[keep_idx[a]] == vid[keep_idx[b]]:
                cur.append(b)
            else:
                runs.append(cur)
                cur = [b]
        runs.append(cur)
        out[s] = runs
    return out


def build_blocks(runs, L):
    """Full blocks of L consecutive clips, grouped by subject."""
    blockpos, blk_sub, n_tail = [], [], 0
    for code, (s, rs) in enumerate(sorted(runs.items())):
        for r in rs:
            nfull = len(r) // L
            for i in range(nfull):
                blockpos.append(r[i * L:(i + 1) * L]); blk_sub.append(code)
            n_tail += len(r) - nfull * L
    if not blockpos:
        return None
    return np.asarray(blockpos, np.int64), np.asarray(blk_sub, np.float64), n_tail


def block_perm_p(y, r, blockpos, blk_sub, rng, nperm):
    """Two-sided p on |AUC - 1/2| when whole blocks are permuted within each subject."""
    y = y.astype(np.int64); n = len(y)
    n1 = int(y.sum()); n0 = n - n1
    obs = (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    thr = abs(obs - 0.5) - 1e-12
    nb, L = blockpos.shape
    flat = blockpos.ravel()
    batch = max(1, min(nperm, 2_000_000 // max(1, nb * L)))
    cnt = done = 0
    while done < nperm:
        b = min(batch, nperm - done)
        u = rng.random((b, nb))                       # Offsets in [0, 1) preserve code boundaries.
        order = np.argsort(blk_sub[None, :] + u, axis=1)
        yp = np.repeat(y[None, :], b, axis=0)
        yp[:, flat] = y[blockpos[order]].reshape(b, -1)
        a = (yp @ r - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        cnt += int((np.abs(a - 0.5) >= thr).sum())
        done += b
    return float(obs), float((cnt + 1) / (nperm + 1))


def non_overlapping(vid, start, keep_idx):
    """Greedy maximal set of windows pairwise at least WINDOW frames apart within each video."""
    take = np.zeros(len(keep_idx), bool)
    order = sorted(range(len(keep_idx)), key=lambda i: (str(vid[keep_idx[i]]), int(start[keep_idx[i]])))
    last_v, last_s = None, None
    for i in order:
        v, st = str(vid[keep_idx[i]]), int(start[keep_idx[i]])
        if v != last_v or st - last_s >= WINDOW:
            take[i] = True; last_v, last_s = v, st
    return take


def main():
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/block_permutation.log", "w")
    t0 = time.time()

    def p(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    man = json.load(open("runs/affwild_clips_features/manifest.json"))["clips"]
    vid_all, start_all = clip_order(man)
    Z = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz")
    logged = {(c["corpus"], c["cell"]): c for c in json.load(open("runs/peak_cluster_inference/cluster.json"))["cells"]}
    rng = np.random.default_rng(SEED)

    arms = {
        "aw2_image": (Z["aw2img__y"], Z["aw2img__subject"], Z["aw2img__mask"].astype(bool),
                      np.arange(len(man))),
        "aw2_video": (Z["aw2vid__y"], Z["aw2vid__subject"], Z["aw2vid__mask"].astype(bool),
                      Z["aw2vid__idx"]),
    }
    skip = ("y", "subject", "mask", "idx")
    cells = [(c, k) for c in ("aw2_image", "aw2_video") for k in sorted(Z.files)
             if k.startswith("aw2img__" if c == "aw2_image" else "aw2vid__")
             and k.split("__")[1] not in skip]
    variants = ["free"] + [f"block{L}" for L in BLOCKS] + ["nonoverlap"]
    p(f"W20 | {len(cells)} tests | {NPERM} permutations per variant | seed {SEED}")
    res = {"provenance": run_provenance(seeds=[SEED]), "seed": SEED, "nperm": NPERM,
           "window_frames": WINDOW, "block_lengths": list(BLOCKS), "variants": variants, "tests": {}}

    for corpus, key in cells:
        name = key.split("__")[1]
        y_all, sub_all, mask, idxmap = arms[corpus]
        s = Z[key]
        keep = mask & ~np.isnan(s)
        sk, yk, gk = s[keep], y_all[keep].astype(np.int64), sub_all[keep]
        keep_idx = idxmap[keep]
        r = auc_ranks(sk)
        runs = subject_runs(gk, vid_all, start_all, keep_idx)
        row = {"corpus": corpus, "cell": name, "n": int(len(yk)), "n_pos": int(yk.sum()),
               "auc": float(logged[(corpus, name)]["auc"]), "p": {}, "tail_clips": {}}
        _, row["p"]["free"] = perm_test(sk, yk, gk, "subject", rng, NPERM)
        for L in BLOCKS:
            built = build_blocks(runs, L)
            if built is None:
                row["p"][f"block{L}"] = None; continue
            bp, bs, ntail = built
            row["tail_clips"][f"block{L}"] = int(ntail)
            _, row["p"][f"block{L}"] = block_perm_p(yk, r, bp, bs, rng, NPERM)
        take = non_overlapping(vid_all, start_all, keep_idx)
        s2, y2, g2 = sk[take], yk[take], gk[take]
        if 0 < int(y2.sum()) < len(y2):
            a2, p2 = perm_test(s2, y2, g2, "subject", rng, NPERM)
            row["p"]["nonoverlap"] = p2
            row["nonoverlap"] = {"n": int(len(y2)), "n_pos": int(y2.sum()), "auc": float(a2)}
        else:
            row["p"]["nonoverlap"] = None
        res["tests"][f"{corpus}:{name}"] = row
        p(f"  {corpus:9s} {name:16s} " + "  ".join(
            (f"{v}={row['p'][v]:.4f}" if row["p"].get(v) is not None else f"{v}=--") for v in variants))

    keys = list(res["tests"])
    res["holm"] = {}
    for v in variants:
        ps = [res["tests"][k]["p"].get(v) for k in keys]
        if any(x is None for x in ps):
            res["holm"][v] = None; continue
        adj = holm(ps)
        res["holm"][v] = {k: float(a) for k, a in zip(keys, adj)}
        for k, a in zip(keys, adj):
            res["tests"][k].setdefault("holm", {})[v] = float(a)

    gate_rows = []
    for k in keys:
        t = res["tests"][k]
        lo = logged[(t["corpus"], t["cell"])]["holm_pooled"]["perm_subject"]
        got = t["holm"]["free"]
        gate_rows.append({"test": k, "logged": float(lo), "recomputed": float(got),
                          "delta": float(abs(got - lo)),
                          "decision_matches": bool((lo < 0.05) == (got < 0.05))})
    res["gate"] = {"max_abs_delta": max(g["delta"] for g in gate_rows), "tolerance": 0.02,
                   "all_decisions_match": all(g["decision_matches"] for g in gate_rows),
                   "rows": gate_rows}
    res["gate"]["passes"] = bool(res["gate"]["all_decisions_match"]
                                 and res["gate"]["max_abs_delta"] <= 0.02)

    dep = [f"block{L}" for L in BLOCKS]
    vj = res["tests"]["aw2_video:vjepa_K16"].get("holm", {})
    dn = [res["tests"][k].get("holm", {}) for k in keys if "dinov2" in k]
    res["summary"] = {
        "vjepa_K16_adjusted": {v: vj.get(v) for v in variants if vj.get(v) is not None},
        "vjepa_significant_under_all_block_variants": all(vj.get(v, 1) < 0.05 for v in dep),
        "dinov2_any_becomes_significant": any(h.get(v, 1) < 0.05 for h in dn for v in dep),
        "n_significant_per_variant": {v: (sum(res["tests"][k]["holm"][v] < 0.05 for k in keys)
                                          if res["holm"].get(v) else None) for v in variants},
    }
    res["status"] = "COMPLETE" if res["gate"]["passes"] else "COMPLETE WITH GATE MISMATCH"
    json.dump(res, open(f"{OUT}/block_permutation.json", "w"), indent=2)
    p("")
    p(f"gate: max |delta| {res['gate']['max_abs_delta']:.4f} (tol 0.02), decisions match "
      f"{res['gate']['all_decisions_match']}, passes {res['gate']['passes']}")
    p(f"V-JEPA-2 K=16 adjusted: {res['summary']['vjepa_K16_adjusted']}")
    p(f"significant under every block variant: "
      f"{res['summary']['vjepa_significant_under_all_block_variants']}")
    p(f"any DINOv2 cell becomes significant: {res['summary']['dinov2_any_becomes_significant']}")
    p(f"count per variant: {res['summary']['n_significant_per_variant']}")
    p(f"wrote {OUT}/block_permutation.json | {res['status']}")
    logf.close()


if __name__ == "__main__":
    main()
