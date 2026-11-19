"""Video-encoder features for the position-matched Aff-Wild2 set.

Usage:  <env>/bin/python -m affectdyn.features.position_matched_video_features <encoder>
"""
from __future__ import annotations
from affectdyn.paths import REPO_STR as REPO
import argparse, hashlib, json, os, platform, sys, time

import numpy as np


OUT = "runs/position_matched_peak_analysis"
T = 16
REL_L2_MAX = 1e-3
N_CLIPS = 13276


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def rel_l2(R, C):
    return np.linalg.norm(R - C, axis=-1) / np.maximum(np.linalg.norm(C, axis=-1), 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("encoder"); ap.add_argument("--force", action="store_true")
    args = ap.parse_args(); name = args.encoder
    if os.path.abspath(os.getcwd()) != REPO:
        raise SystemExit("run from the repository root")
    out_json = f"{OUT}/{name}_extract.json"
    if os.path.exists(out_json) and not args.force:
        raise SystemExit(f"{out_json} exists; refusing to overwrite without --force")
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")
    from affectdyn.data.affwild_valence_clips import build_clips
    from affectdyn.features.affwild_video_features import _loader, _orders
    from affectdyn.provenance import run_provenance
    t0 = time.time()
    log_f = open(f"{OUT}/{name}_extract.log", "w")

    def log(m):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    design = json.load(open(f"{OUT}/matched_subset.json"))
    if design["status"] != "FROZEN":
        raise SystemExit("design not frozen")
    midx = np.load(f"{OUT}/matched_subset_idx.npy").astype(int)
    gate_idx = np.array(design["extraction_gate_idx"], dtype=int)
    pub = np.load("runs/affwild_clips_features/video_subset_idx.npy").astype(int)
    pos_pub = {int(i): k for k, i in enumerate(pub)}
    lab = np.load("runs/affwild_clips_features/labels.npz")
    clips = build_clips()
    if len(clips) != N_CLIPS:
        raise SystemExit(f"build_clips gave {len(clips)} clips, expected {N_CLIPS}")
    need = np.concatenate([midx, gate_idx])
    bad = [int(i) for i in need if str(clips[i]["subject"]) != str(lab["subject"][i])
           or int(clips[i]["traj"]) != int(lab["traj"][i]) or len(clips[i]["paths"]) != T]
    if bad:
        raise SystemExit(f"{len(bad)} clips disagree with runs/affwild_clips_features/labels.npz, e.g. {bad[:5]}")
    cache = np.load(f"runs/affwild_clips_features/{name}_windows.npz")
    fwd, prep, nf = _loader(name)
    full_order = list(np.linspace(0, T - 1, nf).round().astype(int))
    win_orders = _orders(nf)

    def frames(c):
        return np.stack([prep(p) for p in c["paths"]])

    def emb(x, order):
        return fwd(torch.from_numpy(np.ascontiguousarray(x[order])).to("cuda")).float().cpu().numpy()

    def ordered(c):
        x = frames(c)
        return emb(x, full_order), np.stack([emb(x, o) for o in win_orders]), x

    gF, gW = [], []
    for i in gate_idx:
        f, w, _ = ordered(clips[i]); gF.append(f); gW.append(w)
    gF, gW = np.stack(gF), np.stack(gW)
    rows = [pos_pub[int(i)] for i in gate_idx]
    eF, eW = rel_l2(gF, cache["full"][rows]), rel_l2(gW, cache["K16_up"][rows])
    gate = {"n_clips": int(len(gate_idx)), "max_rel_l2_full": float(eF.max()), "max_rel_l2_windows": float(eW.max()),
            "median_rel_l2_full": float(np.median(eF)), "threshold": REL_L2_MAX,
            "passes": bool(eF.max() <= REL_L2_MAX and eW.max() <= REL_L2_MAX)}
    np.savez_compressed(f"{OUT}/{name}_gate.npz", idx=gate_idx, full=gF, K16_up=gW)
    log(f"{name} gate: {gate}")
    try:
        import transformers; tv = transformers.__version__
    except Exception:                                                  # noqa: BLE001
        tv = None
    try:
        import easydict; ev = getattr(easydict, "__version__", "unrecorded")
    except Exception:                                                  # noqa: BLE001
        ev = None
    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
           "transformers": tv, "easydict": ev, "gpu": torch.cuda.get_device_name(0),
           "venv": os.path.dirname(os.path.dirname(sys.executable))}
    res = {"provenance": run_provenance(), "environment": env, "encoder": name, "gate": gate,
           "criteria_heading": design["criteria_heading"], "amendment_heading": design["amendment_heading"]}
    if not gate["passes"]:
        res["status"] = "EXCLUDED: extraction gate failed"
        json.dump(res, open(out_json, "w"), indent=1); log(res["status"]); return

    new = np.array([i for i in midx if int(i) not in pos_pub], dtype=int)
    oF, oW = [], []
    for j, i in enumerate(new):
        f, w, _ = ordered(clips[i]); oF.append(f); oW.append(w)
        if j % 50 == 0:
            log(f"ordered {j + 1}/{len(new)}")
    oF, oW = np.stack(oF), np.stack(oW)
    np.savez_compressed(f"{OUT}/{name}_ordered_new.npz", idx=new, full=oF, K16_up=oW)

    perms = np.load("runs/affwild_peak_order_controls/perms.npz")["perms"]   # Permutations have shape [3, 13276, 16].
    sF = np.full((perms.shape[0], len(midx), oF.shape[1]), np.nan, np.float32)
    for j, i in enumerate(midx):
        x = frames(clips[i])
        sel = np.array(full_order)
        for r in range(perms.shape[0]):
            # Order selected frames by their positions in the uniformly shuffled sequence.
            pinv = np.argsort(perms[r, i])
            sF[r, j] = emb(x, list(sel[np.argsort(pinv[sel], kind="stable")]))
        if j % 50 == 0:
            log(f"shuffled {j + 1}/{len(midx)}")
    if np.isnan(sF).any():
        raise SystemExit("NaN left in shuffled embeddings")
    np.savez_compressed(f"{OUT}/{name}_shuffled_full.npz", idx=midx, full_shuf=sF, perms=perms[:, midx])
    res.update({"n_new_ordered": int(len(new)), "n_shuffled_clips": int(len(midx)), "replicates": int(perms.shape[0]),
                "sha": {"ordered_full": sha(oF), "shuffled_full": sha(sF)}, "seconds": round(time.time() - t0, 1),
                "status": "COMPLETE"})
    json.dump(res, open(out_json, "w"), indent=1)
    log(f"wrote {out_json} | COMPLETE")


if __name__ == "__main__":
    main()
