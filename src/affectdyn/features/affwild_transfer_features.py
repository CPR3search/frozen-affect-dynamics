"""VideoMAEv2 Aff-Wild2 window features at K=2 and 4.

Usage:  python -m affectdyn.features.affwild_transfer_features
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, hashlib, json, os, subprocess, sys, time

import numpy as np

from affectdyn.data.unbc_clip_pipeline import T                                                    # noqa: E402
from affectdyn.features.temporal_aggregation import _device, _window_orders                         # noqa: E402
from affectdyn.data.affwild_valence_clips import build_clips                                   # noqa: E402
from affectdyn.features.affwild_video_features import _loader                                            # noqa: E402

OUT = "runs/affwild_transfer_test"
PUBLISHED = "runs/affwild_clips_features/videomaev2_windows.npz"
SUBSET_IDX = "runs/affwild_clips_features/video_subset_idx.npy"
ENCODER = "videomaev2"
KS = (2, 4)                       # Extract the two evaluated granularities.
K_REPLAY = 16
REL_TOL = 1e-3
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
CRIT = "## 2026-09-26 — W63 PRE-DECLARED CRITERIA"
# Historical lookup key in the optional, unreleased analysis log; not a runnable protocol or a public preregistration identifier.
AMEND = "## 2026-09-26 — W63 CRITERIA AMENDED"


def _sha(path, n=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(n)):
            h.update(b)
    return h.hexdigest()


def environment(dev):
    import torch
    from importlib.metadata import PackageNotFoundError, version as dist_version

    def ver(mod):
        try:
            return dist_version(mod)
        except PackageNotFoundError:
            pass
        except Exception:                                                     # noqa: BLE001
            pass
        try:
            return getattr(__import__(mod), "__version__", "installed, version not exposed")
        except Exception:                                                     # noqa: BLE001
            return "absent"
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "cuda_build": getattr(torch.version, "cuda", None),
            "transformers": ver("transformers"), "timm": ver("timm"),
            "einops": ver("einops"), "easydict": ver("easydict"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_used": str(dev), "venv": sys.prefix,
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                         text=True).stdout.strip(),
            "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                             text=True).stdout.strip())}


def orders_for(k, nframes):
    return [list(map(int, w)) for w in _window_orders(k, upsample_to=nframes)]


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    os.makedirs(OUT, exist_ok=True)
    out_npz = f"{OUT}/videomaev2_windows_k2k4.npz"
    out_json = f"{OUT}/extract.json"
    if os.path.exists(out_npz):
        raise SystemExit(f"{out_npz} exists; refusing to overwrite")

    t0 = time.time()
    logf = open(f"{OUT}/extract.log", "w")

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    import torch
    dev = _device()
    env = environment(dev)
    log(f"device {dev} | torch {env['torch']} cuda {env['cuda_build']} | timm {env['timm']} "
        f"easydict {env['easydict']} | venv {env['venv']}")
    if str(dev) == "cpu":
        raise SystemExit("refusing to extract on the CPU; check the import trap and CUDA_VISIBLE_DEVICES")

    clips = build_clips()
    idx = np.load(SUBSET_IDX).tolist()
    sub = [clips[i] for i in idx]
    log(f"{len(sub)} subset clips (of {len(idx)} published), T={T}")

    fwd, prep, nf = _loader(ENCODER)
    log(f"{ENCODER} loaded, frame budget nf={nf}")

    # Verify order lists against the pain arm.
    orders = {k: orders_for(k, nf) for k in (*KS, K_REPLAY)}
    g3 = {"n_windows_per_k": {k: len(v) for k, v in orders.items()},
          "window_len_per_k": {k: len(v[0]) for k, v in orders.items()},
          "k2_window0": orders[2][0], "k2_window1": orders[2][1]}
    for k, o in orders.items():
        assert len(o) == k, (k, len(o))
        assert all(len(w) == nf for w in o), (k, [len(w) for w in o])
    # Require the two K=2 windows to contain the first and second clip halves.
    first, second = sorted(set(orders[2][0])), sorted(set(orders[2][1]))
    assert first == list(range(T // 2)) and second == list(range(T // 2, T)), (first, second)
    names0 = [os.path.basename(sub[0]["paths"][i]) for i in first]
    names1 = [os.path.basename(sub[0]["paths"][i]) for i in second]
    g3["manifest_check"] = {"clip": sub[0].get("stem", sub[0].get("video", "?")),
                            "k2_window0_frames": names0, "k2_window1_frames": names1,
                            "disjoint": not (set(names0) & set(names1))}
    assert g3["manifest_check"]["disjoint"]
    log(f"G3a orders match _window_orders; K=2 window 0 {names0[0]}..{names0[-1]}, "
        f"window 1 {names1[0]}..{names1[-1]}")

    def emb(clip, order):
        x = torch.from_numpy(np.stack([prep(clip["paths"][i]) for i in order])).to(dev)
        return fwd(x).float().cpu().numpy()

    # Verify that changing frame order changes the representation.
    fwdv = emb(sub[0], orders[2][0])
    revv = emb(sub[0], list(reversed(orders[2][0])))
    g3["reversal_control"] = {"max_abs_diff": float(np.abs(fwdv - revv).max()),
                             "rel_l2": float(np.linalg.norm(fwdv - revv) / (np.linalg.norm(fwdv) + 1e-12)),
                             "embedding_changes": bool(not np.allclose(fwdv, revv))}
    assert g3["reversal_control"]["embedding_changes"], (
        "reversing a K=2 window left the embedding unchanged; the encoder or the frame order is being "
        "collapsed and the K=2 and K=4 features would be order-blind")
    log(f"G3b reversal control: rel L2 {g3['reversal_control']['rel_l2']:.4e}, order matters")

    store = {}
    for k in (*KS, K_REPLAY):
        arrs, t1 = [], time.time()
        for j, c in enumerate(sub):
            arrs.append(np.stack([emb(c, o) for o in orders[k]]))
            if j % 250 == 0:
                el = time.time() - t1
                log(f"  K={k:2d} {j + 1}/{len(sub)}  {el:.0f}s, ~{el / max(j + 1, 1) * (len(sub) - j - 1) / 60:.0f} min left")
        store[f"K{k}_up"] = np.stack(arrs)
        assert store[f"K{k}_up"].shape[1] == k, (k, store[f"K{k}_up"].shape)
        assert np.isfinite(store[f"K{k}_up"]).all()
        log(f"  K={k:2d} done {store[f'K{k}_up'].shape} in {time.time() - t1:.0f}s")

    pub = np.load(PUBLISHED)["K16_up"]
    new = store[f"K{K_REPLAY}_up"]
    n = new.shape[0]
    num = np.linalg.norm((new - pub[:n]).reshape(n, -1), axis=1)
    den = np.linalg.norm(pub[:n].reshape(n, -1), axis=1) + 1e-12
    rel = num / den
    g1 = {"n_clips": int(n), "max_rel_l2_windows": float(rel.max()),
          "median_rel_l2_windows": float(np.median(rel)), "threshold": REL_TOL,
          "passes": bool(rel.max() <= REL_TOL),
          "w59_reference_max_rel_l2_windows": 3.176259951942484e-06}
    log(f"G1 {g1}")

    res = {"criteria_heading": CRIT, "amendment_heading": AMEND, "encoder": ENCODER,
           "environment": env, "weights_sha256": {}, "n_clips": len(sub), "T": T, "frame_budget": nf,
           "ks_extracted": list(KS), "k_replayed": K_REPLAY,
           "gate_g1_extraction_replay": g1, "gate_g3_partition": g3,
           "published_cache": {"path": PUBLISHED, "sha256": _sha(PUBLISHED)},
           "smoke": False, "seconds": round(time.time() - t0, 1)}
    try:                                   # Identify the checkpoint by its content hash.
        rc = json.load(open("runs/meta/run_contract.json"))
        res["weights_sha256"] = {k: v for k, v in (rc.get("weights") or {}).items()
                                 if ENCODER in str(k).lower()}
    except Exception as e:                                                    # noqa: BLE001
        res["weights_sha256"] = {"unavailable": str(e)}

    if not g1["passes"]:
        res["status"] = "HELD at G1: the published VideoMAEv2 Aff-Wild2 cache is not reproduced"
        json.dump(res, open(out_json, "w"), indent=1)
        log(res["status"]); return

    np.savez_compressed(out_npz, **store)
    res["status"] = "COMPLETE"
    res["output"] = {"path": out_npz, "keys": sorted(store), "sha256": _sha(out_npz)}
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_npz} | {res['status']}")


if __name__ == "__main__":
    main()
