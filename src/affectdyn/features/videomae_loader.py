"""VideoMAE and VideoMAEv2 loaders that verify every weight is loaded."""
from __future__ import annotations

CKPT_V1 = "MCG-NJU/videomae-base"
CKPT_V2 = "OpenGVLab/VideoMAEv2-Base"


def _bias_remap(state_dict, n_layers=12, hidden=768):
    """q_bias -> query.bias, v_bias -> value.bias, key.bias -> zeros (correct for this family)."""
    import torch
    out = {}
    for k, v in state_dict.items():
        kk = k[len("videomae."):] if k.startswith("videomae.") else k
        if not kk.startswith("encoder."):
            continue          # Exclude unused pretraining-decoder attention biases.
        if kk.endswith(".attention.attention.q_bias"):
            out[kk.replace(".q_bias", ".query.bias")] = v
        elif kk.endswith(".attention.attention.v_bias"):
            out[kk.replace(".v_bias", ".value.bias")] = v
    for i in range(n_layers):
        out[f"encoder.layer.{i}.attention.attention.key.bias"] = torch.zeros(hidden)
    return out


def load_videomae(ckpt=CKPT_V1, device=None, verbose=True):
    """VideoMAE with its attention biases actually loaded."""
    import torch
    from transformers import VideoMAEModel
    from huggingface_hub import hf_hub_download

    model, info = VideoMAEModel.from_pretrained(ckpt, dtype=torch.float32,
                                                output_loading_info=True)
    dropped = [k for k in info.get("missing_keys", []) if "attention.attention" in k]
    if dropped:
        try:
            from safetensors.torch import load_file
            sd = load_file(hf_hub_download(ckpt, "model.safetensors"))
        except Exception:                                              # noqa: BLE001
            sd = torch.load(hf_hub_download(ckpt, "pytorch_model.bin"), map_location="cpu",
                            weights_only=True)
        cfg = model.config
        remap = _bias_remap(sd, cfg.num_hidden_layers, cfg.hidden_size)
        res = model.load_state_dict(remap, strict=False)
        still = [k for k in res.missing_keys if "attention.attention" in k and k.endswith("bias")]
        if still:
            raise RuntimeError(
                f"{ckpt}: {len(still)} attention biases still unloaded after remap, e.g. "
                f"{still[:4]}. Refusing to extract features from a model missing trained weights.")
        if verbose:
            print(f"[videomae] restored {len(remap)} attention-bias tensors that transformers "
                  f"had zeroed ({len(dropped)} were missing)", flush=True)
    elif verbose:
        print(f"[videomae] {ckpt} loaded with no missing attention biases", flush=True)

    model = model.eval()
    return model.to(device) if device else model


def load_videomae_v2(ckpt=CKPT_V2, device=None, verbose=True, num_frames=None):
    """VideoMAEv2 through its own remote code, with three incompatibilities worked around."""
    import importlib
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_videomaev2.VideoMAEv2", ckpt)
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    mod = importlib.import_module(cls.__module__)

    kw = {}
    if num_frames is not None:
        cfg = cls.config_class.from_pretrained(ckpt)
        cfg.model_config = dict(cfg.model_config, num_frames=int(num_frames))
        kw["config"] = cfg
    model, info = cls.from_pretrained(ckpt, dtype=torch.float32, output_loading_info=True, **kw)
    missing = sorted(info.get("missing_keys", set()))
    unexpected = sorted(info.get("unexpected_keys", set()))
    if unexpected:
        raise RuntimeError(f"{ckpt}: {len(unexpected)} unexpected checkpoint keys, e.g. {unexpected[:5]}")
    if missing:
        raise RuntimeError(
            f"{ckpt}: {len(missing)} randomly initialised parameters, e.g. {missing[:5]}. "
            f"Refusing to extract features from a partly random encoder.")

    regenerated = 0
    for _, sm in model.named_modules():
        pe = getattr(sm, "pos_embed", None)
        if isinstance(pe, torch.Tensor) and pe.is_meta:
            sm.pos_embed = mod.get_sinusoid_encoding_table(pe.shape[1], pe.shape[2])
            regenerated += 1
    if regenerated and verbose:
        print(f"[videomaev2] regenerated {regenerated} sinusoidal pos_embed table(s) left on meta",
              flush=True)
    if verbose:
        print(f"[videomaev2] loaded {ckpt} with 0 missing keys", flush=True)

    model = model.eval()
    return model.to(device) if device else model
