"""Frozen-feature evaluation of facial affect dynamics.

Article implementation map:
- Eq. (1): features.temporal_aggregation.image_readout / video_readout.
- Fixed-step probes: inference.loso_permutation_tests._oof_scores
  (UNBC), probes.affwild_image_fixed_step.oof (Aff-Wild2).
- Converged probes: probes.converged_probe_fits.newton.
- Article nested probes: probes.extended_nested_probe_fits.nested_scores.
- Single-AUC and cross-dataset cluster bootstrap:
  inference.peak_cluster_inference.bootstrap / diff_bootstrap.
- Paired cluster bootstrap:
  inference.pain_direction_cluster_inference.paired_bootstrap.
- Eqs. (2)-(4): inference.calibrated_clustered_auc.
- Shared Holm adjustment: inference.aggregation_permutation_tests.holm.
- Labels: data.unbc_clip_pipeline.build_clips and
  data.affwild_valence_clips._label.

These modules also contain analysis drivers; importing some inference
modules changes CUDA visibility and thread environment variables.
"""
