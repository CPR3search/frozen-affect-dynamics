"""Video-encoder features for the Aff-Wild2 video subset.

Usage:  python -m affectdyn.features.affwild_video_features                    # all five, in cost order
        python -m affectdyn.features.affwild_video_features xclip videomae     # named models only
        python -m affectdyn.features.affwild_video_features --subset 2500
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np

from affectdyn.data.unbc_clip_pipeline import T, _prep224, _prep256
from affectdyn.features.temporal_aggregation import _device, _window_orders
from affectdyn.data.affwild_valence_clips import build_clips
from affectdyn.features.affwild_image_features import stratified_subset
from affectdyn.provenance import run_provenance

OUT = "runs/affwild_clips_features"
K = 16
# Process cheaper encoders first to maximize completed extractions.
ORDER = ["xclip", "timesformer", "videomae", "videomaev2", "internvideo2", "vjepa"]


def _loader(name):
    """Return (forward_fn, prep_fn, n_frames)."""
    import torch
    dev = _device()
    if name == "vjepa":
        from transformers import VJEPA2Model
        m, info = VJEPA2Model.from_pretrained("facebook/vjepa2-vitl-fpc64-256",
                                              dtype=torch.float32, output_loading_info=True)
        _guard(name, info)
        m = m.eval().to(dev)
        def fwd(x):
            with torch.no_grad():
                return m(pixel_values_videos=x.unsqueeze(0)).last_hidden_state.mean(1).squeeze(0)
        return fwd, _prep256, T
    if name == "videomae":
        from affectdyn.features.videomae_loader import load_videomae
        m = load_videomae(device=dev)
        def fwd(x):
            with torch.no_grad():
                return m(pixel_values=x.unsqueeze(0)).last_hidden_state.mean(1).squeeze(0)
        return fwd, _prep224, T
    if name in ("timesformer", "xclip"):
        from affectdyn.features.video_window_features import _load as w2_load, MODELS, _prep as w2_prep
        m = w2_load(name)
        nf = MODELS[name]["frames"]
        norm = MODELS[name]["norm"]
        def fwd(x):
            with torch.no_grad():
                if name == "xclip":
                    return m.get_video_features(pixel_values=x.unsqueeze(0)).pooler_output.squeeze(0)
                return m(pixel_values=x.unsqueeze(0)).last_hidden_state.mean(1).squeeze(0)
        return fwd, (lambda p: w2_prep(p, norm)), nf
    if name == "videomaev2":
        from affectdyn.features.videomae_loader import load_videomae_v2
        m = load_videomae_v2(device=dev)
        def fwd(x):
            with torch.no_grad():
                o = m(x.permute(1, 0, 2, 3).unsqueeze(0))
            return (o if hasattr(o, "shape") else o.last_hidden_state).squeeze(0)
        return fwd, _prep224, T
    if name == "internvideo2":
        from affectdyn.features.internvideo2_window_features import _load as iv_load, NFRAMES
        m = iv_load()
        def fwd(x):
            with torch.no_grad():
                out = m(x.permute(1, 0, 2, 3).unsqueeze(0), mask=None, use_image=False)
            return (out[1] if isinstance(out, (tuple, list)) else out).squeeze(0)
        return fwd, _prep224, NFRAMES
    raise KeyError(name)


def _guard(name, info):
    missing = list(info.get("missing_keys", []))
    if missing:
        raise RuntimeError(f"{name}: {len(missing)} randomly initialised parameters, e.g. "
                           f"{missing[:5]}. Refusing to extract from a partly random encoder.")
    print(f"[w3/{name}] loaded with 0 missing keys", flush=True)


def _orders(nframes):
    """K=16 windows over T frames, each fitted to the model's own frame budget."""
    out = []
    for w in _window_orders(K):
        base = w
        if len(base) >= nframes:
            idx = np.linspace(0, len(base) - 1, nframes).round().astype(int)
            base = [base[j] for j in idx]
        else:
            rep = int(np.ceil(nframes / len(base)))
            base = [f for f in base for _ in range(rep)][:nframes]
        out.append(base)
    return out


def extract(name, clips):
    import torch
    cache = f"{OUT}/{name}_windows.npz"
    if os.path.exists(cache):
        print(f"[w3/{name}] cache hit, skipping", flush=True)
        return cache
    fwd, prep, nf = _loader(name)
    dev = _device()
    full_order = list(np.linspace(0, T - 1, nf).round().astype(int))
    win_orders = _orders(nf)
    full, wins = [], []
    t0 = time.time()
    for j, c in enumerate(clips):
        def emb(order):
            x = torch.from_numpy(np.stack([prep(c["paths"][i]) for i in order])).to(dev)
            return fwd(x).float().cpu().numpy()
        full.append(emb(full_order))
        wins.append(np.stack([emb(o) for o in win_orders]))
        if j % 100 == 0:
            el = time.time() - t0
            eta = el / max(j + 1, 1) * (len(clips) - j - 1)
            print(f"[w3/{name}] {j + 1}/{len(clips)}  {el:.0f}s, ~{eta / 60:.0f} min left",
                  flush=True)
    np.savez_compressed(cache, full=np.stack(full), K16_up=np.stack(wins))
    print(f"[w3/{name}] wrote {cache} ({time.time() - t0:.0f}s)", flush=True)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=None)
    ap.add_argument("--subset", type=int, default=2500)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    clips = build_clips()
    idx_path = f"{OUT}/video_subset_idx.npy"
    if os.path.exists(idx_path):
        idx = np.load(idx_path).tolist()
        print(f"[w3] reusing subset of {len(idx)} clips", flush=True)
    else:
        idx = stratified_subset(clips, args.subset)
        np.save(idx_path, np.array(idx))
    sub_clips = [clips[i] for i in idx]
    lab = np.load(f"{OUT}/labels.npz")
    npk = int((lab["traj"][idx] == 2).sum())
    print(f"[w3] subset: {len(sub_clips)} clips, {npk} peaked, "
          f"{len({c['subject'] for c in sub_clips})} subjects | K={K}, "
          f"{1 + K} forwards per clip", flush=True)
    json.dump({"provenance": run_provenance(), "n_subset": len(idx), "n_peaked": npk, "K": K,
               "subset_index_file": idx_path},
              open(f"{OUT}/video_subset.json", "w"), indent=2)

    for name in (args.models or ORDER):
        try:
            extract(name, sub_clips)
        except Exception as e:                                     # noqa: BLE001
            print(f"[w3/{name}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
