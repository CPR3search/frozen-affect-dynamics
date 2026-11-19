"""Feature extraction for the video encoders at every granularity."""
from __future__ import annotations
import os, sys, time
import numpy as np

from affectdyn.data.unbc_clip_pipeline import build_clips, T, _prep224
from affectdyn.features.unbc_pretraining_comparison import NORMS, _prep as _prep_img_norm
from affectdyn.features.temporal_aggregation import _device, KS

OUT = "runs/pain_encoder_maps"

# Map encoder identifiers to loader keys, frame counts, and preprocessing.
MODELS = {
    "timesformer": {"ckpt": "facebook/timesformer-base-finetuned-k400", "frames": 8, "norm": "imagenet"},
    "xclip":       {"ckpt": "microsoft/xclip-base-patch32",             "frames": 8, "norm": "clip"},
}


def _prep(path, norm):
    if norm == "imagenet":
        return _prep224(path)
    mean, std = NORMS["clip"]
    return _prep_img_norm(path, mean, std)


def _orders(k, nframes):
    """K contiguous windows over T frames, each repeat-upsampled to the model's native length."""
    w = T // k
    out = []
    for i in range(k):
        base = list(range(i * w, (i + 1) * w))
        if len(base) >= nframes:                       # Subsample windows that meet the frame budget.
            idx = np.linspace(0, len(base) - 1, nframes).round().astype(int)
            base = [base[j] for j in idx]
        else:                                          # Repeat frames to fill shorter windows.
            rep = int(np.ceil(nframes / len(base)))
            base = [f for f in base for _ in range(rep)][:nframes]
        out.append(base)
    return out


def _full_order(nframes):
    idx = np.linspace(0, T - 1, nframes).round().astype(int)
    return list(idx)


def _load(name):
    """Load a checkpoint and REFUSE it if any parameter was randomly initialised."""
    import torch
    from transformers import TimesformerModel, XCLIPModel
    ckpt = MODELS[name]["ckpt"]
    if name == "timesformer":
        m, info = TimesformerModel.from_pretrained(ckpt, dtype=torch.float32,
                                                   output_loading_info=True)
    elif name == "xclip":
        # Load the full XCLIPModel to obtain compatible checkpoint parameters.
        m, info = XCLIPModel.from_pretrained(ckpt, dtype=torch.float32, output_loading_info=True)
    else:
        raise KeyError(name)
    missing = list(info.get("missing_keys", []))
    if missing:
        raise RuntimeError(
            f"{name} ({ckpt}) loaded with {len(missing)} randomly initialised parameters, "
            f"e.g. {missing[:5]}. Refusing to extract features from a partly random encoder.")
    print(f"[w2/{name}] loaded {ckpt} with 0 missing keys", flush=True)
    return m.eval().to(_device())


def extract(name, clips, cache=None):
    import torch
    cache = cache or f"{OUT}/{name}_windows.npz"
    if os.path.exists(cache):
        print(f"[w2/{name}] cache hit {cache}")
        return cache
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    cfg = MODELS[name]
    nf, norm = cfg["frames"], cfg["norm"]
    m = _load(name)
    dev = _device()

    def emb(paths, order):
        x = torch.from_numpy(np.stack([_prep(paths[i], norm) for i in order])).unsqueeze(0).to(dev)
        with torch.no_grad():
            if name == "xclip":
                # Use X-CLIP's video interface for temporal aggregation.
                h = m.get_video_features(pixel_values=x).pooler_output
            else:
                h = m(pixel_values=x).last_hidden_state.mean(1)
        return h.squeeze(0).float().cpu().numpy()

    store = {f"K{k}_up": [] for k in KS}
    full = []
    t0 = time.time()
    for j, c in enumerate(clips):
        full.append(emb(c["paths"], _full_order(nf)))
        for k in KS:
            store[f"K{k}_up"].append(np.stack([emb(c["paths"], o) for o in _orders(k, nf)]))
        if j % 50 == 0:
            print(f"[w2/{name}] {j + 1}/{len(clips)} clips  {time.time() - t0:.0f}s", flush=True)
    np.savez_compressed(cache, full=np.stack(full), **{k: np.stack(v) for k, v in store.items()})
    print(f"[w2/{name}] wrote {cache}")
    return cache


if __name__ == "__main__":
    which = sys.argv[1:] or list(MODELS)
    clips = build_clips()
    print(f"[w2] {len(clips)} clips | device {_device()} | models: {', '.join(which)}", flush=True)
    for name in which:
        extract(name, clips)
