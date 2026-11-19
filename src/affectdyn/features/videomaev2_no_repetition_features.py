"""VideoMAEv2 window features without frame repetition.

Usage (repository root):
  python -m affectdyn.features.videomaev2_no_repetition_features
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips, T, _prep224                              # noqa: E402
from affectdyn.features.temporal_aggregation import _window_orders, _device                          # noqa: E402
from affectdyn.features.videomae_loader import load_videomae_v2                                   # noqa: E402

OUT = "runs/videomaev2_no_repetition_contrasts"
PUBLISHED = "runs/temporal_aggregation/videomaev2_windows.npz"
TOL = 1e-3
KS = (2, 4)
TOKENS = {16: 1568, 8: 784, 4: 392}


def main():
    ap = argparse.ArgumentParser(); a = ap.parse_args()
    import torch
    os.makedirs(OUT, exist_ok=True)
    tag = ""
    logf = open(f"{OUT}/extract{tag}.log", "w"); t0 = time.time()

    def log(m=""):
        line = f"[{time.time() - t0:7.1f}s] {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()

    dev = _device()
    log(f"device {dev} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible; refusing to run a 599-clip extraction on the CPU")
    import easydict
    res = {"env": {"python": sys.executable, "torch": torch.__version__,
                   "easydict": getattr(easydict, "__version__", "1.13 (pinned install)")}}
    models = {nf: load_videomae_v2(device=dev, num_frames=(None if nf == 16 else nf)) for nf in (16, 8, 4)}

    sd = {nf: {k: v for k, v in m.state_dict().items() if "pos_embed" not in k} for nf, m in models.items()}
    same = all(set(sd[nf]) == set(sd[16]) and all(torch.equal(sd[nf][k], sd[16][k]) for k in sd[16]) for nf in (8, 4))
    toks = {nf: int(m.model.pos_embed.shape[1]) for nf, m in models.items()}
    prefix = all(torch.allclose(models[nf].model.pos_embed[0].cpu(), models[16].model.pos_embed[0, :TOKENS[nf]].cpu(),
                                atol=0, rtol=0) for nf in (8, 4))
    res["G2"] = {"learned_tensors_identical": bool(same), "pos_embed_tokens": toks,
                 "tokens_ok": toks == TOKENS, "pos_embed_is_prefix_of_16": bool(prefix)}
    log(f"G2 {res['G2']}")

    clips = build_clips()
    log(f"{len(clips)} UNBC clips")

    def emb(m, paths, order):
        x = torch.from_numpy(np.stack([_prep224(paths[i]) for i in order])).to(dev)
        with torch.no_grad():
            o = m(x.permute(1, 0, 2, 3).unsqueeze(0))
            t = o if hasattr(o, "shape") else o.last_hidden_state
            return t.squeeze(0).float().cpu().numpy()

    out = {"full": [], **{f"K{k}_up": [] for k in KS}, **{f"K{k}_native": [] for k in KS}}
    g3 = True
    for j, c in enumerate(clips):
        p = c["paths"]
        out["full"].append(emb(models[16], p, list(range(T))))
        for k in KS:
            out[f"K{k}_up"].append(np.stack([emb(models[16], p, o) for o in _window_orders(k, upsample_to=T)]))
            w = T // k; nat = []
            for i in range(k):
                order = list(range(i * w, (i + 1) * w))
                g3 &= (len(order) == w and len(set(order)) == w and order == sorted(order))
                nat.append(emb(models[w], p, order))
            out[f"K{k}_native"].append(np.stack(nat))
        if j % 50 == 0:
            log(f"{j + 1}/{len(clips)}")
    arr = {k: np.stack(v) for k, v in out.items()}
    finite = all(bool(np.isfinite(v).all()) for v in arr.values())
    res["G2"]["outputs_finite"] = finite
    res["G3"] = {"native_windows_exact_frames": bool(g3)}

    ref = np.load(PUBLISHED); n = len(clips); g1 = {}
    for k in ("full", "K2_up", "K4_up"):
        r = ref[k][:n].astype(np.float64); gk = arr[k].astype(np.float64)
        rel = np.linalg.norm(gk - r, axis=-1) / np.maximum(np.linalg.norm(r, axis=-1), 1e-12)
        g1[k] = {"max_rel_l2": float(rel.max()), "median_rel_l2": float(np.median(rel))}
    res["G1"] = {"arrays": g1, "n_clips": n, "tol": TOL, "pass": all(v["max_rel_l2"] <= TOL for v in g1.values())}
    log(f"G1 {res['G1']}")
    res["all_gates_pass"] = bool(res["G1"]["pass"] and same and toks == TOKENS and prefix and finite and g3)
    np.savez_compressed(f"{OUT}/videomaev2_native{tag}.npz", **arr)
    json.dump(res, open(f"{OUT}/extract{tag}.json", "w"), indent=2)
    log(f"all gates pass: {res['all_gates_pass']} -> {OUT}/videomaev2_native{tag}.npz")


if __name__ == "__main__":
    main()
