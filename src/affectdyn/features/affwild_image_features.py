"""Image-encoder features for the Aff-Wild2 clip sets.

Usage:  python -m affectdyn.features.affwild_image_features image
        python -m affectdyn.features.affwild_image_features video [--subset N]
"""
from __future__ import annotations
import argparse, json, os, sys, time

import numpy as np

from affectdyn.features.unbc_pretraining_comparison import NORMS, _prep as _prep_img_norm
from affectdyn.features.temporal_aggregation import _device
from affectdyn.data.affwild_valence_clips import build_clips, T
from affectdyn.provenance import run_provenance

OUT = "runs/affwild_clips_features"
K_DENSE = 16                 # Extract 16 windows per clip.


def _clips():
    clips = build_clips()
    print(f"[w3] {len(clips)} clips, {len({c['subject'] for c in clips})} subjects", flush=True)
    return clips


def extract_image(clips, cache=f"{OUT}/img_frames.npz", bs=64):
    """Per-frame features [N, 16, D] for each image encoder, the same layout the UNBC arm uses."""
    if os.path.exists(cache):
        print(f"[w3/img] cache hit {cache}")
        return cache
    import torch
    from transformers import AutoModel, CLIPVisionModel, SiglipVisionModel
    dev = _device()
    encs = {
        "dinov2": AutoModel.from_pretrained("facebook/dinov2-base", dtype=torch.float32),
        "clip": CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16", dtype=torch.float32),
        "siglip": SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224",
                                                   dtype=torch.float32),
    }
    encs = {k: v.eval().to(dev) for k, v in encs.items()}
    store = {e: [] for e in encs}
    t0 = time.time()
    for j, c in enumerate(clips):
        for e, m in encs.items():
            mean, std = NORMS[e]
            x = torch.from_numpy(np.stack([_prep_img_norm(p, mean, std) for p in c["paths"]])).to(dev)
            with torch.no_grad():
                outs = []
                for i in range(0, len(x), bs):
                    o = m(pixel_values=x[i:i + bs])
                    f = (o.pooler_output if getattr(o, "pooler_output", None) is not None
                         else o.last_hidden_state.mean(1))
                    outs.append(f.float().cpu().numpy())
            store[e].append(np.concatenate(outs, 0))
        if j % 250 == 0:
            el = time.time() - t0
            eta = el / max(j + 1, 1) * (len(clips) - j - 1)
            print(f"[w3/img] {j + 1}/{len(clips)}  {el:.0f}s elapsed, ~{eta / 60:.0f} min left",
                  flush=True)
    np.savez_compressed(cache, **{e: np.stack(v) for e, v in store.items()})
    print(f"[w3/img] wrote {cache}")
    return cache


def stratified_subset(clips, n, seed=0):
    rng = np.random.default_rng(seed)
    peaked = [i for i, c in enumerate(clips) if c["traj"] == 2]
    rest = [i for i, c in enumerate(clips) if c["traj"] != 2]
    by_subj = {}
    for i in rest:
        by_subj.setdefault(clips[i]["subject"], []).append(i)
    order = []
    subs = sorted(by_subj)
    rng.shuffle(subs)
    depth = 0
    while len(order) < max(0, n - len(peaked)):
        added = False
        for s in subs:
            if depth < len(by_subj[s]):
                order.append(by_subj[s][depth]); added = True
                if len(order) >= n - len(peaked):
                    break
        if not added:
            break
        depth += 1
    idx = sorted(set(peaked) | set(order))
    print(f"[w3] subset {len(idx)} clips ({len(peaked)} peaked kept in full), "
          f"{len({clips[i]['subject'] for i in idx})} subjects", flush=True)
    return idx


def extract_video(clips, subset_n, cache_dir=OUT):
    import torch
    from affectdyn.features.video_window_features import _orders, _full_order, MODELS as W2_MODELS, _prep as _prep_w2
    idx = stratified_subset(clips, subset_n)
    np.save(f"{cache_dir}/video_subset_idx.npy", np.array(idx))
    sub_clips = [clips[i] for i in idx]
    print(f"[w3/video] K={K_DENSE} only, {1 + K_DENSE} forwards per clip", flush=True)
    print("[w3/video] not yet wired: run per-model extraction as an overnight job", flush=True)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["image", "video"])
    ap.add_argument("--subset", type=int, default=2500)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    clips = _clips()
    if args.stage == "image":
        cache = extract_image(clips)
        z = np.load(cache)
        meta = {"provenance": run_provenance(), "n_clips": len(clips),
                "shapes": {k: list(z[k].shape) for k in z.files}, "K_dense": K_DENSE}
        json.dump(meta, open(f"{OUT}/extract_image.json", "w"), indent=2)
        print({k: z[k].shape for k in z.files})
    else:
        extract_video(clips, args.subset)


if __name__ == "__main__":
    main()
