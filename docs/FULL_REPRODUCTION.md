# Reproduce all reported results

Start with [installation](INSTALL.md), [datasets](DATASETS.md) and [numerical profiles](NUMERICAL_PROFILES.md). Obtain external materials under their own terms. The repository includes no dataset, trained checkpoint or medical example image. [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md) provides a worked BSDS example; this guide specifies the full experiment family.

Select hardware separately for each stage using [HARDWARE.md](HARDWARE.md). The commands below use the core model configuration. For the accepted LIDC RN/Capacity comparison, use its separate Blackwell recipe and preflight; an A40/A100 run can complete successfully while selecting different weights. Nondeterministic wider-model, Prob-U and CIMD fits are not certified as exact reruns by completing this grid.

Prepare BSDS, IMA++, LIDC and RIGA with the dataset converters. For each task, run `make-visible-manifests` with visible seeds 0, 1 and 2. BSDS/IMA++/LIDC converters write `all_annotations.json`; RIGA writes separate `riga_disc.json` and `riga_cup.json`. Keep one prepared directory per task and visible choice. LIDC-hard is calculated during LIDC evaluation and has no separate fit.

Recreate and verify features before training. The DINOv3 reference profile uses A40, ViT-B/16 blocks 5/8/11, image size 448, FP16 storage and separate batches within each split. Use batch 12 for IMA++ and 16 otherwise. RIGA extraction uses the shared `feature_images.json`, containing both tasks. Separate extraction for disc and cup changes the final batch. Import the matching task rows into each prepared directory with `import-features`, including the test cache. Do not pass the joint RIGA cache directly to an inference command expecting one task. Run `verify-features` against `artifacts/reference_feature_hashes.json` and seal the training inputs.

## Required families

[COMPARISONS.md](COMPARISONS.md) gives each baseline's commands and external dependencies. These are the task/seed grids used by the final paper:

| Family | Required grid | Fits |
|---|---|---:|
| SSRB | Five tasks × three visible choices × three optimizer seeds; semantic then residual | 45 pairs |
| Raw Capacity-WTA, wider raw, ordinary residual initialization | Each method: the same grid | 45 per method |
| Matched addition and four-plus-four union | Corresponding semantic/raw pairs; alignment uses visible validation data | No new network fits |
| Prob-U, CIMD | Each method: five tasks × three visible choices, optimizer seed zero | 15 per method |
| WTA, Soft-WTA .05, RN | Each method: five tasks × three visible choices; three optimizer seeds for LIDC, seed zero elsewhere | 21 per method |
| Single, MC-dropout | Each method: five tasks × three visible choices, optimizer seed zero | 15 per method |
| SAM2 automatic proposals and full-pool oracle | Five tasks × three visible choices | No fitting |
| SAM2 with a point from Single | Five tasks × three visible choices, matching the Single model | No additional fitting |
| DINOv2 | BSDS/LIDC × three visible choices × three optimizer seeds; semantic then residual | 18 pairs |
| Last DINOv3 block | BSDS/LIDC × three visible choices × three optimizer seeds | 18 |
| Correspondence controls | Four methods × BSDS/LIDC × three visible choices × three optimizer seeds | 72 |

The five tasks are `bsds`, `ima`, `lidc`, `riga_disc` and `riga_cup`. Set `PREPARED` to the directory containing the prepared task/visible-choice folders and `RUNS` to a new output directory. Run the 45-cell core grid from the repository root:

```bash
for endpoint in bsds ima lidc riga_disc riga_cup; do
  for visible in 0 1 2; do
    for optimizer in 0 1 2; do
      task="$PREPARED/$endpoint/vs$visible"
      run="$RUNS/ssrb/$endpoint/vs$visible/os$optimizer"
      ssrb train-semantic --root "$task/trainer" --manifest "$task/trainer/visible.json" --output "$run/semantic" --config configs/paper.json --seed "$optimizer" --device cuda --isolate --cache-inputs
      ssrb train-ssrb --root "$task/trainer" --manifest "$task/trainer/visible.json" --output "$run/residual" --config configs/paper.json --seed "$optimizer" --device cuda --anchor "$run/semantic/checkpoint.pt" --isolate --cache-inputs
      ssrb infer --checkpoint "$run/residual/checkpoint.pt" --root "$task/inference" --manifest "$task/inference/images.json" --feature-file "$task/inference/test_features.pt" --output "$run/predictions" --device cuda --batch-size 12
      ssrb evaluate --predictions "$run/predictions" --root "$task/evaluator" --manifest "$task/evaluator/hidden.json" --output "$run/evaluation"
      ssrb infer --checkpoint "$run/semantic/checkpoint.pt" --root "$task/inference" --manifest "$task/inference/images.json" --feature-file "$task/inference/test_features.pt" --output "$run/semantic_predictions" --device cuda --batch-size 32 --deterministic
      ssrb evaluate --predictions "$run/semantic_predictions" --root "$task/evaluator" --manifest "$task/evaluator/hidden.json" --output "$run/semantic_evaluation"
    done
  done
done
```

Also infer and evaluate each semantic checkpoint to obtain DINO Capacity-WTA results. Semantic inference uses batch 32; residual inference uses batch 12. The same semantic checkpoint supplies the matching residual and ordinary-initialization control. After fitting the raw bank, fit each alignment and infer both addition and union. Keep the full matrices; a single completed seed does not reproduce a reported average.

The correspondence methods are `sa_dmb`, `capacity_sa_dmb`, `sa_dmb_shuffled` and `capacity_sa_dmb_shuffled`. Use `train-sharing`, followed by `infer-comparison` and `evaluate`. The raw controls, frozen-DINO comparisons, correspondence controls, Prob-U, CIMD and last-block campaigns use deterministic CUDA. The core configuration uses default CUDA settings; its numerical differences are described in REPRODUCIBILITY.md. Preserve the execution settings with each result. DINOv2 reference training used CPU with eight threads; last-block training requires A40 and the verified DINOv3 cache. The exact paper-specific MC-dropout inference seed was not recorded, so new runs record a seed without claiming to recover those unknown draws.

Use `python scripts/infer_sam2_fp32.py --retain-full-pool` to retain the oracle inputs while generating the automatic banks. `evaluate-sam2-pool` evaluates those frozen pools. For the point experiment, supply the corresponding frozen Single predictions with `--single-predictions`; keep disc and cup separate.

## Analyses and publication outputs

[EVALUATION.md](EVALUATION.md) explains the explicit optimizer-seed sets and the two accepted averaging rules. The CIMD/SAM comparator tables use equal means of the three visible-choice runs; the paired Prob-U/SSRB summaries average visible choices within each image first. This distinction affects LIDC-hard.

[PAPER_ANALYSIS.md](PAPER_ANALYSIS.md) specifies complete evaluation lists, all paired analyses, residual diagnostics and seven publication figure layouts. Runtime procedures are in [RUNTIME.md](RUNTIME.md). The overview diagram is `figures/overview.tex`.

Use new output directories and retain their hashes, selected epochs, settings and actual metrics. Distinguish new fits, evaluation of accepted weights and analysis of archived evaluations. Paper values in `artifacts/paper_results.csv` are reference values. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) reports the observed agreement and remaining paper-specific differences.
