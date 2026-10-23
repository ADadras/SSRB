# Reference data and reproduction results

These files contain aggregate metrics, figure inputs and fingerprints. They contain no medical images, identifying manifests, feature tensors or trained weights.

- `paper_results.csv`: metric means, available confidence intervals and seed counts from the paper.
- `paper_*_effects.csv`: control, DINOv2 and Prob-U effects and intervals calculated from the paper's evaluations.
- `paper_reference/`: aggregate inputs for the representation, mask-budget and Prob-U figures.
- `reproduction/tables/`: the 25 tables from the complete reproduction grid.
- `reproduction/`: endpoint summaries, metric/statistical comparisons, runtime measurements and plot inputs for the reproduced models.
- `sam2_proposal_results.csv`: evaluation of the paper's saved SAM2 proposals.
- `reference_feature_hashes.json`: tensor and ordered-image fingerprints for DINOv2 and DINOv3. Use with `ssrb verify-features` before fitting.
- `probunet_reference_execution.json`: the runtime settings checked by `scripts/check_probunet_profile.py`.

The reproduction tables include RN/Capacity models trained on A40/A100. The separate Blackwell procedure recovers the paper's LIDC-hard comparison; its scope and numerical limits are described in [HARDWARE.md](../docs/HARDWARE.md). Paper values and reproduction results remain distinct.

Generate outputs for your own runs with the commands in [FULL_REPRODUCTION.md](../docs/FULL_REPRODUCTION.md) and [PAPER_ANALYSIS.md](../docs/PAPER_ANALYSIS.md). Several table scopes reuse the same evaluations and therefore do not represent additional independent fits.
