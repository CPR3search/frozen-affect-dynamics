# Evaluating frozen image and video representations of facial affect dynamics

Code accompanying the article "Evaluating Frozen Image and Video Representations of Facial
Affect Dynamics".

The study asks what frozen image and video foundation models encode about how facial affect changes over time. It
takes 16-frame face clips from UNBC-McMaster (shoulder pain) and Aff-Wild2 (valence in the wild) and trains linear
probes on frozen features for two labels: the **direction** of a change (does pain or valence go up or down across the
clip?) and the **peak task** (does the trajectory peak inside the clip, or change without an interior peak?).

- **Encoders (9).** Image: CLIP, DINOv2, SigLIP. Video: V-JEPA-2, VideoMAE, VideoMAEv2, TimeSformer, X-CLIP,
  InternVideo2. All are used frozen, as feature extractors.
- **One temporal aggregation for both families.** Each clip is split into K windows (K = 2, 4, 8 or 16). The probe sees
  an appearance term and the difference between the last and first window summaries, formed the same way for image and
  video encoders.
- **Three probe fits.** A fixed-step fit (300 Adam steps, weak L2 penalty), a converged fit (same objective, solved to
  convergence) and a cross-validated fit (penalty selected by nested, subject-grouped inner cross-validation).
- **Controls and inference.** Time-reversed, frame-shuffled and appearance-only inputs, a label-only baseline,
  recording-position diagnostics, and statistics that resample whole subjects or videos rather than single clips.

## Repository layout

| Path | Contents |
|---|---|
| `src/affectdyn/data/` | Clip construction, label rules, the position-matched subset, subject-grouped splits |
| `src/affectdyn/features/` | Frozen-encoder feature extraction and the temporal aggregation |
| `src/affectdyn/probes/` | The fixed-step, converged and cross-validated probe fits |
| `src/affectdyn/inference/` | Cluster bootstrap, clustered AUC test, permutations, power benchmarks, Monte Carlo bounds |
| `src/affectdyn/analyses/` | Analyses by topic: `direction`, `peak`, `position`, `cross_dataset`, `sensitivity` |
| `src/affectdyn/simulations/` | Pipeline simulations that refit the probes under a null of exchangeable simulated encoders |
| `envs/` | Pinned package lists for the analysis environment and the two extraction environments |

## Data

The datasets are available from their maintainers under data-use agreements.

- **UNBC-McMaster.** Provide `data/unbc/manifest.json`, a JSON list of per-frame records with the fields `crop`,
  `subject`, `seq`, `frame` and `pspi`, and 112x112 aligned face crops under `data/unbc/aligned_112/`. `crop` must
  contain `aligned_112/` followed by the crop's path below that folder, `frame` must end in the frame number, and
  `pspi` is the numeric pain score.
- **Aff-Wild2.** Provide the cropped and aligned frames as `data/cropped_aligned/<video>/00001.jpg`, `00002.jpg`, ... and
  the valence annotations as `data/8th_ABAW_Annotations/VA_Estimation_Challenge/<split>/<video>.txt` (a header line,
  then one `valence,arousal` row per frame).

## Analyses

**Environments.** Use three separate environments, each listed with exact versions in `envs/`. The two extraction
environments need Linux with an NVIDIA GPU; the analysis environment runs on any platform with Python 3.13.

| File | Python | Used for |
|---|---|---|
| `envs/requirements-analysis.txt` | 3.13, CPU | Every probe, inference, simulation and reporting module |
| `envs/requirements-extract-hf.txt` | 3.12, CUDA 13, transformers 5.16.1 | CLIP, DINOv2, SigLIP, V-JEPA-2, VideoMAE (Aff-Wild2), TimeSformer and X-CLIP features |
| `envs/requirements-extract-timm.txt` | 3.13, transformers 5.9.0 with timm, einops and easydict | InternVideo2 and VideoMAEv2 features, and the UNBC VideoMAE features |

```bash
python3.12 -m venv .venv-hf && source .venv-hf/bin/activate
pip install -r envs/requirements-extract-hf.txt --extra-index-url https://download.pytorch.org/whl/cu130
pip install -e .
deactivate
python3.13 -m venv .venv-timm && source .venv-timm/bin/activate
pip install -r envs/requirements-extract-timm.txt
pip install -e .
```

`pip install -e .` only makes the `affectdyn` package importable; install the environment's list first, and run every
command from the repository root, since modules read and write `runs/`, `data/` and `out/` relative to it.

Match the listed versions: a different transformers version can change extracted features. InternVideo2 also needs
the model code from its authors' repository at a fixed commit:

```bash
git clone --filter=blob:none --sparse https://github.com/OpenGVLab/InternVideo.git _local/InternVideo
git -C _local/InternVideo sparse-checkout set InternVideo2/multi_modality
git -C _local/InternVideo checkout 3965eef16e2dadd0ea6c8d0cc29c8a3039df52e3
```

Its weights (`InternVideo2-stage2_1b-224p-f4.pt` from `OpenGVLab/InternVideo2-Stage2_1B-224p-f4`) and the other
encoders' weights are downloaded from the Hugging Face Hub on first use.

1. UNBC clips and base features (HF environment): `python -m affectdyn.data.unbc_clip_pipeline` builds the UNBC clips and the first
   feature cache (`runs/unbc_clip_pipeline/stage1_feats.npz`). Then `python -m affectdyn.features.temporal_aggregation`
   writes the per-frame image features and V-JEPA-2 window features that the pain analyses
   read (`runs/temporal_aggregation/img_frames.npz`, `vjepa_windows.npz`).
2. Aff-Wild2 clips: `python -m affectdyn.data.affwild_valence_clips` writes the clip manifest
   (`runs/affwild_clips_features/manifest.json`).
3. Feature extraction (GPU). In the HF environment:
   `python -m affectdyn.features.video_window_features` (TimeSformer and X-CLIP on UNBC),
   `python -m affectdyn.features.affwild_image_features image` (the image encoders on Aff-Wild2) and
   `python -m affectdyn.features.affwild_video_features xclip timesformer videomae vjepa`. In the timm environment:
   `python -m affectdyn.features.internvideo2_window_features`, `python -m affectdyn.features.videomae_bias_correction`
   (VideoMAE and VideoMAEv2 on UNBC) and `python -m affectdyn.features.affwild_video_features videomaev2 internvideo2`.
   `affwild_video_features` reports an encoder that fails to load and continues, so check its output for every
   encoder. The `video` mode of `affwild_image_features` only writes the video-subset index
   (`runs/affwild_clips_features/video_subset_idx.npy`); it extracts no video features.
4. Position-matched subset, after the subset index exists. The released subset definition
   (`runs/position_matched_peak_analysis/matched_subset.json`) is frozen: the module refuses to run while it exists, so
   move it aside to rebuild. Then write the frame permutations with
   `python -m affectdyn.analyses.peak.affwild_peak_order_controls perms` and extract each video encoder with
   `python -m affectdyn.features.position_matched_video_features <encoder> --force`.
5. Probe training. Every probe is an L2-penalized logistic regression on the aggregated features, fitted
   leave-one-subject-out on UNBC and with 5 folds split by video group on Aff-Wild2.
   - Fixed-step fit (300 full-batch Adam steps at learning rate 0.01 with weight decay 0.001, logits averaged over
     5 seeded fits, or 3 for the Aff-Wild2 image encoders):
     `python -m affectdyn.analyses.direction.pain_direction_encoder_map` (pain direction, all 9 encoders),
     `python -m affectdyn.analyses.peak.pain_peak_encoder_map` (pain peak task),
     `python -m affectdyn.probes.affwild_image_fixed_step` and `python -m affectdyn.probes.affwild_video_fixed_step`
     (Aff-Wild2 peak task and direction).
   - Converged fit (the same objective solved by Newton's method to a gradient tolerance of 1e-8), after the
     fixed-step records exist: `python -m affectdyn.probes.converged_probe_fits --force`.
   - Cross-validated fit (penalty chosen from 9 values between 1e-4 and 1e4 by held-out deviance over 5 grouped inner
     folds, then refitted on each outer training fold): `python -m affectdyn.probes.extended_nested_probe_fits --force`.
     `affectdyn.probes.nested_probe_fits` is an earlier 5-penalty version kept for comparison.
6. Inference and analysis modules read these caches and probe scores. `affectdyn.simulations.direction_null_replay` writes the
   reference null that the partition and no-repetition analyses and both pipeline simulations read, so it runs before
   them. The pipeline simulations are `python -m affectdyn.simulations.pipeline_simulation_resampled --run 1500`
   (followed by `python -m affectdyn.simulations.pipeline_simulation_summary`) and
   `python -m affectdyn.simulations.pipeline_simulation_distinct --arm all --run 1500`. Both use 18 worker processes by
   default (`--procs`).
   

## Licence

MIT for the code in this repository (`LICENSE`). The datasets, pretrained weights and third-party code are governed
by their own terms.
