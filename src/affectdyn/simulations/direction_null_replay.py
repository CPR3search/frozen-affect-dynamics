"""Regenerate the direction-contrast null reference used by inference.calibrated_clustered_auc.

Usage: python -m affectdyn.simulations.direction_null_replay
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, sys, time
import numpy as np

import affectdyn.inference.calibrated_clustered_auc as w12                                    # noqa: E402
from affectdyn.features.unbc_pretraining_comparison import _auc                                        # noqa: E402
from affectdyn.provenance import run_provenance                                         # noqa: E402

OUT = "runs/pipeline_simulation_resampled"
PAIRS = {"K2": ("clip_K2", "videomaev2_K2", "K= 2: CLIP vs videomaev2"),
         "K4": ("clip_K4", "videomaev2_K4", "K= 4: CLIP vs videomaev2")}


def advance_single(rng, d, n):
    nn = len(d.y)
    for i in range(0, n, w12.BATCH):
        b = min(w12.BATCH, n - i)
        rng.standard_normal((b, nn)); rng.standard_normal((b, d.K))


def advance_paired(rng, d, n):
    nn = len(d.y)
    for i in range(0, n, w12.BATCH):
        b = min(w12.BATCH, n - i)
        rng.standard_normal((b, nn)); rng.standard_normal((b, nn))
        rng.standard_normal((b, d.K)); rng.standard_normal((b, d.K))


def p_ref(z, null):
    return float((1 + (np.abs(null) >= abs(z)).sum()) / (1 + len(null)))


def main():
    os.makedirs(OUT, exist_ok=True)
    logf = open(f"{OUT}/replay.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    W7 = np.load("runs/peak_cluster_inference/cluster_oof_scores.npz"); W9 = np.load("runs/pain_direction_cluster_inference/direction_cluster_oof.npz")
    C9 = json.load(open("runs/pain_direction_cluster_inference/direction_cluster.json"))
    stored = {r["test"]: r for r in json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))["families"]
              ["direction_contrasts_24"]["rows"]}

    def arr(prefix, mk):
        return W7[f"{prefix}__y"].astype(np.int64), W7[f"{prefix}__subject"], W7[f"{prefix}__{mk}"].astype(bool)
    uy, ug, uk = arr("unbc", "keep"); iy, ig, im = arr("aw2img", "mask"); vy, vg, vm = arr("aw2vid", "mask")
    d_unbc = w12.Design("unbc_shape", uy[uk], ug[uk], 36)
    d_full = w12.Design("aw2_full", iy[im], ig[im], 18)
    d_sub = w12.Design("aw2_subset", vy[vm], vg[vm], 18)
    dk = W9["keep"].astype(bool)
    d_dir = w12.Design("unbc_direction", W9["y"][dk], W9["subject"][dk], 24)
    cells9 = [n for n in W9.files if n not in ("subject", "y", "keep")]
    theta = float(np.median([_auc(W9[n][dk], W9["y"][dk].astype(int)) for n in cells9]))
    rho = float(np.median([w12.within_class_corr(W9[c["a"]][dk], W9[c["b"]][dk], W9["y"][dk])
                           for c in C9["contrasts"]]))
    stored_par = json.load(open("runs/calibrated_clustered_auc/calibrated_recheck.json"))["calibration"]["parameters"]
    log(f"theta {theta:.10f} (stored {stored_par['theta_direction']:.10f}) rho {rho:.10f} "
        f"(stored {stored_par['rho_direction']:.10f})")

    rng = np.random.default_rng(w12.SEED); n = w12.N_NULL
    for d in (d_unbc, d_full, d_sub):
        advance_single(rng, d, n)        # Advance null draws at theta = 0.5.
        advance_single(rng, d, n)        # Advance alternative draws at theta = 0.6.
        log(f"advanced single {d.name}")
    for d in (d_unbc, d_full, d_sub):
        advance_paired(rng, d, n)
        log(f"advanced paired {d.name}")
    z0 = w12.sim_paired(d_dir, theta, rho, rng, n)
    log(f"direction null: {len(z0)} draws, rejection at 0.05 {np.mean(np.abs(z0) >= 1.959964):.4f}")

    gate = {}
    for k, (a, b, name) in PAIRS.items():
        sa, sb, y, g = W9[a][dk], W9[b][dk], W9["y"][dk].astype(int), W9["subject"][dk]
        _, z = w12.z_paired(sa, sb, y, g)
        p = p_ref(z, z0)
        st = stored[name]
        gate[k] = {"z": z, "z_stored": st["z"], "z_delta": abs(z - st["z"]), "p": p, "p_stored": st["p_calibrated"],
                   "p_equal": bool(p == st["p_calibrated"])}
        log(f"gate {k}: z {z:.12f} vs {st['z']:.12f}; p {p:.12e} vs {st['p_calibrated']:.12e}")
    ok = (all(v["z_delta"] <= 1e-9 and v["p_equal"] for v in gate.values())
          and theta == stored_par["theta_direction"] and rho == stored_par["rho_direction"])
    res = {"provenance": run_provenance(seeds=[w12.SEED]), "theta": theta, "rho": rho, "n_null": n,
           "gate": gate, "gate_pass": bool(ok)}
    np.save(f"{OUT}/w12_direction_null.npy", z0)
    json.dump(res, open(f"{OUT}/replay.json", "w"), indent=2, default=float)
    log(f"REPLAY GATE {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
