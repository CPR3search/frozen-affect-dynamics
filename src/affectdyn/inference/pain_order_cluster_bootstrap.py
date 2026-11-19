"""CLIP's pain order controls with subjects resampled.

Usage:  CUDA_VISIBLE_DEVICES="" python -m affectdyn.inference.pain_order_cluster_bootstrap
"""
from __future__ import annotations
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""                     # Disable CUDA before importing torch.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse, datetime, json, subprocess, sys, time, zlib

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import _auc
from affectdyn.inference.loso_permutation_tests import _loso_folds
from affectdyn.features.temporal_aggregation import image_readout, N_REP
from affectdyn.inference.aggregation_permutation_tests import holm
from affectdyn.analyses.direction.pain_direction_encoder_map import KS
from affectdyn.analyses.peak.pain_peak_order_controls import ENCODER, N_SHUFFLE, _scores
from affectdyn.inference.peak_cluster_inference import torch_cpu_only, bootstrap
from affectdyn.inference.pain_direction_cluster_inference import paired_bootstrap
from affectdyn.provenance import audit_loso

from affectdyn.paths import REPO_STR as REPO
NBOOT = 10000
SEED = 20260911
ALPHA = 0.05
TOL = 0.005
PRIMARY_K = 4
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRITERIA_HEADING = "## 2026-09-11 — W11 PRE-DECLARED CRITERIA"


def criteria_verbatim(heading):
    log = os.path.join(REPO, "RESEARCH_LOG-D3-W1.md")
    if not os.path.exists(log):   # Allow the optional criteria file to be absent.
        return None
    lines = open(log, encoding="utf-8").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading))
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("## "))
    return "\n".join(lines[start:end]).strip()


def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, timeout=10, cwd=REPO).stdout.strip()
    except Exception:                                                          # noqa: BLE001
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="runs/pain_order_cluster_bootstrap")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit(f"run from the repository root {REPO}")
    od = os.path.abspath(args.out_dir)
    for forbidden in ("runs/pain_encoder_maps", "runs/affwild_clips_features"):
        f = os.path.join(REPO, forbidden)
        if od == f or od.startswith(f + os.sep):
            raise SystemExit(f"refusing to write under {forbidden}: the manuscript is validated against it")
    os.makedirs(od, exist_ok=True)
    out_json, out_npz, out_log = (os.path.join(od, f) for f in
                                  ("pain_controls_cluster.json", "pain_controls_cluster_oof.npz",
                                   "pain_controls_cluster.log"))
    if not args.force and (os.path.exists(out_json) or os.path.exists(out_npz)):
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")

    torch = torch_cpu_only()
    torch.set_num_threads(1)                                  # Single-threaded execution reduces overhead for 345 clips.
    nboot = NBOOT
    ks = KS
    n_shuffle = N_SHUFFLE
    criteria = criteria_verbatim(CRITERIA_HEADING)
    logf = open(out_log, "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    log(f"W11 CLIP pain order controls, subject-level | torch {torch.__version__} "
        f"cuda_build={torch.version.cuda} niceness={os.nice(0)} smoke=False")

    # Refit using the shape-control procedure.
    lab = np.load("runs/unbc_clip_pipeline/stage1_feats.npz")
    sub, traj = lab["subject"], lab["traj"]
    y = (traj == 2).astype(int)
    mask = traj != -1
    folds = _loso_folds(sub); audit_loso(sub, folds)
    frames = np.load("runs/temporal_aggregation/img_frames.npz")[ENCODER]                   # Frame features have shape [N, T, D].
    S = {"static": _scores(frames.mean(1), y, sub, mask, folds)}
    log("  refit static")
    for k in ks:
        S[f"ord_K{k}"] = _scores(image_readout(frames, k), y, sub, mask, folds)
        shufs = []
        for sh in range(n_shuffle):
            rng = np.random.default_rng(500 + sh)
            perm = np.stack([rng.permutation(frames.shape[1]) for _ in range(frames.shape[0])])
            shuffled = np.take_along_axis(frames, perm[:, :, None], axis=1)
            shufs.append(_scores(image_readout(shuffled, k), y, sub, mask, folds, seed0=4000 + 10 * sh))
        S[f"shuf_K{k}"] = np.mean(shufs, axis=0)
        log(f"  refit K={k} ({n_shuffle} shuffles)")
    keep = mask.copy()
    for s in S.values():
        keep &= ~np.isnan(s)
    yk, gk = y[keep].astype(np.int64), sub[keep]
    log(f"clips kept {int(keep.sum())}, peaked {int(yk.sum())}, subjects {len(np.unique(gk))}")

    L = json.load(open("runs/pain_encoder_maps/shape_control.json"))
    LS = json.load(open("runs/pain_encoder_maps/shape_ord_vs_static.json"))
    a_static = _auc(S["static"][keep], yk)
    repro = {"static": {"auc": float(a_static), "logged": float(L["static"]["auc"])}, "by_K": {}}
    deltas = [abs(a_static - L["static"]["auc"])]
    for k in ks:
        lk = L["by_K"][str(k)]
        ao, ash = _auc(S[f"ord_K{k}"][keep], yk), _auc(S[f"shuf_K{k}"][keep], yk)
        repro["by_K"][str(k)] = {"auc_ordered": float(ao), "logged_ordered": float(lk["auc_ordered"]),
                                 "auc_shuffled": float(ash), "logged_shuffled": float(lk["auc_shuffled"]),
                                 "gap": float(ao - ash), "logged_gap": float(lk["gap"])}
        deltas += [abs(ao - lk["auc_ordered"]), abs(ash - lk["auc_shuffled"]), abs((ao - ash) - lk["gap"])]
    if PRIMARY_K in ks:
        g_os = _auc(S[f"ord_K{PRIMARY_K}"][keep], yk) - a_static
        repro["ordered_minus_static"] = {"gap": float(g_os), "logged": float(LS["gap"])}
        deltas.append(abs(g_os - LS["gap"]))
    gate = {"tolerance": TOL, "n_clips": int(keep.sum()), "n_clips_logged": int(L["n_clips"]),
            "max_abs_delta": float(max(deltas))}
    gate["passes"] = bool(gate["n_clips"] == gate["n_clips_logged"] and gate["max_abs_delta"] <= TOL)
    log(f"pipeline gate: {gate}")

    def rng_for(name):
        return np.random.default_rng([SEED, zlib.crc32(name.encode())])

    # Resample subjects for paired tests.
    rows = []
    for k in ks:
        so, ss = S[f"ord_K{k}"][keep], S[f"shuf_K{k}"][keep]
        name, lk = f"K{k}:ordered_minus_shuffled", L["by_K"][str(k)]
        row = {"test": name, "K": k, "role": "primary" if k == PRIMARY_K else "exploratory",
               "auc_ordered": float(_auc(so, yk)), "auc_control": float(_auc(ss, yk)),
               "gap": float(_auc(so, yk) - _auc(ss, yk)),
               "subject_boot": paired_bootstrap(so, ss, yk, gk, "subject", rng_for(name + ":subject"), nboot),
               "clip_boot": paired_bootstrap(so, ss, yk, gk, "clip", rng_for(name + ":clip"), nboot),
               "control_vs_chance_subject": bootstrap(ss, yk, gk, "subject", rng_for(name + ":chance"), nboot),
               "logged": {"p_gap_clip_permutation": float(lk["p_gap"]), "ci_gap_clip_bootstrap": lk["ci_gap"]}}
        rows.append(row)
        b = row["subject_boot"]
        log(f"  {name:28s} ordered {row['auc_ordered']:.4f} shuffled {row['auc_control']:.4f} gap {row['gap']:+.4f} "
            f"| subject CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f} p {b['p_boot']:.4f} "
            f"| clip p {row['clip_boot']['p_boot']:.4f} | logged p {lk['p_gap']:.4f}")
    if PRIMARY_K in ks:
        so, st = S[f"ord_K{PRIMARY_K}"][keep], S["static"][keep]
        name = f"K{PRIMARY_K}:ordered_minus_static"
        row = {"test": name, "K": PRIMARY_K, "role": "exploratory",
               "auc_ordered": float(_auc(so, yk)), "auc_control": float(_auc(st, yk)),
               "gap": float(_auc(so, yk) - _auc(st, yk)),
               "subject_boot": paired_bootstrap(so, st, yk, gk, "subject", rng_for(name + ":subject"), nboot),
               "clip_boot": paired_bootstrap(so, st, yk, gk, "clip", rng_for(name + ":clip"), nboot),
               "control_vs_chance_subject": bootstrap(st, yk, gk, "subject", rng_for(name + ":chance"), nboot),
               "logged": {"p_clip_permutation": float(LS["p"]), "ci_clip_bootstrap": LS["ci"]}}
        rows.append(row)
        b = row["subject_boot"]
        log(f"  {name:28s} ordered {row['auc_ordered']:.4f} static {row['auc_control']:.4f} gap {row['gap']:+.4f} "
            f"| subject CI {b['ci'][0]:+.3f} to {b['ci'][1]:+.3f} p {b['p_boot']:.4f} | logged p {LS['p']:.4f}")
    shuffle_rows = [r for r in rows if r["test"].endswith("shuffled")]
    for r, a in zip(shuffle_rows, holm([r["subject_boot"]["p_boot"] for r in shuffle_rows])):
        r["holm_subject_across_shuffle_gaps"] = float(a)

    primary = next((r for r in rows if r["role"] == "primary"), None)
    verdict = None
    if gate["passes"] and primary is not None:
        indicated = primary["gap"] > 0 and primary["subject_boot"]["p_boot"] < ALPHA
        verdict = ("ORDER-DEPENDENT CONTRIBUTION INDICATED (exploratory, uncorrected)" if indicated
                   else "ORDER DEPENDENCE NOT SHOWN WITH SUBJECTS RESAMPLED")
        status = "COMPLETE"
    else:
        status = "HOLD: pipeline gate failed, no verdict is read"

    out = {"provenance": {"timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                          "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
                          "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                          "torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                          "niceness": os.nice(0), "seed": SEED, "nboot": nboot, "smoke": False,
                          "encoder": ENCODER, "n_shuffle": n_shuffle, "n_rep": N_REP,
                          "probe_seeds": {"ordered_static": [3000 + r for r in range(N_REP)],
                                          "shuffled": [[4000 + 10 * sh + r for r in range(N_REP)]
                                                       for sh in range(n_shuffle)]},
                          "shuffle_permutation_seeds": [500 + sh for sh in range(n_shuffle)]},
           "criteria_verbatim": criteria, "status": status, "pipeline_gate": gate, "reproduction": repro,
           "verdict": verdict, "primary_test": primary["test"] if primary else None, "tests": rows}
    json.dump(out, open(out_json, "w"), indent=2)
    np.savez_compressed(out_npz, subject=sub, y=y, keep=keep, **S)
    log(f"status {status} | verdict {verdict}")
    log(f"wrote {out_json} and {out_npz}")
    logf.close()


if __name__ == "__main__":
    main()
