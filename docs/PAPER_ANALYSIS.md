# Tables, intervals and figures

The table commands in EVALUATION.md export individual runs or average the full SSRB seed matrix. To calculate paired intervals for the other accepted comparisons, create an evaluation list:

```json
[
  {"evaluation": "runs/ssrb/lidc/vs0/os0/evaluation", "method": "ssrb", "optimizer_seed": 0},
  {"evaluation": "runs/wide_raw/lidc/vs0/os0/evaluation", "method": "wide_raw", "optimizer_seed": 0}
]
```

This abbreviated list illustrates the format; the analysis requires every member of its comparison family. Paths are relative to the list file or absolute. Each evaluation directory must contain `per_image.json`, produced by the evaluator. Supply all three visible choices and all three optimizer seeds where used in the accepted family. LIDC-hard is derived from LIDC evaluation.

```bash
ssrb analyze-paper --evaluations controls.json --profile controls --output outputs/control_analysis
ssrb analyze-paper --evaluations probunet.json --profile probunet --output outputs/probunet_analysis
ssrb analyze-paper --evaluations dinov2.json --profile dinov2 --output outputs/dinov2_analysis
ssrb analyze-paper --evaluations partial.json --profile partial --output outputs/partial_analysis
ssrb analyze-representation --evaluations representation.json --output outputs/representation_analysis
ssrb analyze-assignment --evaluations assignments.json --profile visible --output outputs/assignment_visible
ssrb analyze-assignment --evaluations assignments_lidc.json --profile lidc --output outputs/assignment_lidc
ssrb analyze-assignment --evaluations assignments_riga.json --profile capacity --output outputs/assignment_riga
ssrb analyze-sharing --evaluations sharing.json --output outputs/sharing_analysis
```

- The controls profile requires `ssrb`, `posthoc_additive`, `native_union_4_4`, `conventional_residual`, `wide_raw`, `raw_capacity` and `dino_capacity`. It uses 10,000 draws and seed 20260815.
- The Prob-U profile requires `raw_residual_fusion` and `probunet`. There is one Prob-U optimizer run per visible choice; pair it with each SSRB optimizer run. It uses 10,000 draws and seed 20260814.
- The DINOv2 profile requires `dinov2_fusion` and `dinov2_capacity` on LIDC, LIDC-hard and BSDS, with 10,000 draws and seed 20260814.
- The last-block profile requires `ssrb`, `partial_dino_capacity` and `frozen_capacity` on BSDS/LIDC with all nine seed combinations. It exports 30 contrasts using the original resampling procedure and seed 20260817.
- The representation command requires `raw_residual_fusion`, `dino_capacity` and `raw_capacity`, each with all 45 cells. It initializes the original bootstrap separately for the BSDS/LIDC and IMA++/RIGA groups, exporting 24 effects and 18 descriptive rows in total.
- Assignment `visible` requires `dmb_rn`, `dmb_wta` and `soft_tau005`, all five tasks and three visible choices at optimizer seed zero. It exports 112 effects, including the secondary pooled RIGA comparison. The `lidc` profile requires those three methods plus `capacity_wta`, all three visible and optimizer seeds; it exports 48 effects. The `capacity` profile compares `dmb_rn` with `cb_wta` on both RIGA tasks and three visible choices at optimizer seed zero; it exports 24 effects. These preserve seeds 3450, 3664 and 3633 respectively.
- The correspondence command requires `sa_dmb`, `capacity_sa_dmb`, `sa_dmb_shuffled` and `capacity_sa_dmb_shuffled` on BSDS/LIDC, with all nine seed combinations. Its 72 evaluations produce six effects with seed 20260814. Infer these checkpoints with `infer-comparison`.

These routines preserve the accepted order of random draws and simultaneous intervals. The paper's analysis samples LIDC and LIDC-hard separately. RIGA disc and cup share source draws. For a new analysis that links LIDC-hard to its parent patient draws, use the separate helpers in `statistics.py`; changing the grouping changes the numerical profile and must be stated. Do not silently replace the published intervals. `bootstrap-representation` supplies the accepted representation profile.

## Figures

```bash
ssrb plot-results --summary outputs/control_analysis/method_summary.csv --output outputs/plots
ssrb plot-banks --predictions "$PREDICTIONS" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --image-id "$CASE" --output outputs/bank_plots
```

The first command exports vector plots for coverage, waste and mask budgets. The second displays saved mask banks in their preserved order. Image assets are read from the user's authorized dataset and are not bundled. `figures/overview.tex` is the standalone TikZ source for the method diagram; compile it with a LaTeX installation including TikZ and standalone.

```bash
ssrb plot-paper --profile representation --source artifacts/paper_reference/branch.csv --output outputs/representation.pdf
ssrb plot-paper --profile budget --source artifacts/paper_reference/budget.json --output outputs/budget.pdf
ssrb plot-paper --profile probunet --source artifacts/paper_reference/probunet.csv --output outputs/probunet.pdf
```

These supplied inputs are the paper's aggregate values. For new plots, supply the same CSV/JSON structure with values from the corresponding completed analysis. `representation` reads the branch table columns `regime`, `method` and `MLQ8`; `budget` reads three regime objects (`LIDC`, `LIDC-hard`, `IMA++`), each mapping the four method labels to eight prefix values; `probunet` reads the full effect CSV containing MLQ@1 through MLQ@8 and Waste@8 with their interval columns.

The four layouts using images take explicit JSON inputs:

| Profile | Input |
|---|---|
| `mechanism` | Object with `summary` (the residual summary CSV) and two `cases`, for LIDC and BSDS. Each case provides `endpoint`, image-only `root` and `manifest`, `checkpoint`, `feature_file`, `image_id`, zero-based `head_id` and `selection_basis`. Use `visual_clarity` only when the source image and predictions were used to choose that case. Rendering is on CPU and needs no reference masks. |
| `qualitative` | List of four cases, for LIDC, IMA++, RIGA disc and BSDS. Each provides `endpoint`, `diagnostics` directory, frozen `predictions` directory, image `root`, `reference_root` and the reference `manifest`. `image_manifest` defaults to `root/images.json`. Selection uses the median anchor/fused Dice from visible-seed-zero, optimizer-seed-zero diagnostics before opening references. Diagnostic and prediction checkpoint hashes must match. |
| `assignment-examples` | List of RN/WTA evaluation specifications, with `evaluation`, `method` (`dmb_rn` or `dmb_wta`), `optimizer_seed` (zero), `predictions`, image `root`, `reference_root` and reference `manifest`. Include all three visible choices and both RIGA tasks. `image_manifest` defaults to `root/images.json`. The original deterministic selector uses held-out MLQ differences to choose illustrative sources. It does not choose training settings or select a model. |
| `bsds-construction` | Object with `root`, converter `manifest` (`all_annotations.json`) and `image_id`. The first five masks are shown in their preserved converter order. |

Paths may be absolute or relative to the input JSON. The reference manifests have no image path; image and reference identities are matched explicitly. Output records preserve the selection rule and input/output hashes. The paper's assignment examples use evaluation results for selection; the SSRB qualitative and mechanism selections do not.

## Residual diagnostics

```bash
ssrb diagnose-residuals --checkpoint "$SSRB" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --feature-file "$TEST_FEATURES" --output "$RUN/diagnostics" --device cuda
```

This measures residual magnitude, area and component changes, boundary displacement, and anchor/output overlap at the model's 128×128 grid. It reads images and frozen features. It does not read reference masks or estimate hidden-mask accuracy. The scalar calculations have numerical regression tests. A separate renderer can use the saved RLE masks; original medical images remain external.

To calculate the full residual intervals, supply a JSON list of 45 diagnostic directories, each containing `head_diagnostics.json` and `provenance.json`:

```bash
ssrb analyze-residuals --diagnostics diagnostic_directories.json --output outputs/residual_analysis
```

Reproduction plot inputs are in `artifacts/reproduction/plot_inputs/` and use the same structures. Image-based layouts need separately obtained dataset images. Runtime procedures are in [RUNTIME.md](RUNTIME.md).
