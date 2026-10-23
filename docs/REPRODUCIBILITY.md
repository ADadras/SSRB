# Reproducing the results

Use the pinned dependencies, fixed dataset splits, visible-mask choices and stage-specific hardware in [HARDWARE.md](HARDWARE.md). The complete experiment grid has been run across BSDS, IMA++, LIDC and the separate RIGA disc/cup tasks. LIDC-hard is evaluated from LIDC predictions.

| Experiment | Evaluated coverage |
| --- | ---: |
| SSRB semantic/residual stages | 45 pairs |
| Raw Capacity-WTA, wider raw and ordinary residual initialization | 45 fits per method |
| Prob-U and CIMD | 15 fits per method |
| WTA, Soft-WTA and RN | 21 fits per method |
| Single and MC-dropout | 15 fits per method |
| DINOv2 semantic/residual stages | 18 pairs |
| Last DINOv3 block | 18 fits |
| Correspondence controls | 72 fits |
| Addition and union | 45 pairs of evaluations |
| SAM2 automatic proposals, full pools and point proposals | 15 evaluations per method |

This covers 474 model fits, 25 table exports and 648 statistical comparisons. Aggregate outputs are in [artifacts/reproduction/](../artifacts/reproduction/). Paper reference values and inputs for three aggregate figures are supplied separately in [artifacts/](../artifacts/README.md).

## Numerical agreement

- **Features:** all 6,259 DINOv3 and 2,377 DINOv2 feature entries match their references on the specified A40 profile. Use `verify-features` before training.
- **SSRB:** reevaluating all 45 paper checkpoints preserves the 18 MLQ/Waste/AUC means at four decimals. Repeated training changes endpoint means by up to 0.00094206 for MLQ@8 and 0.00169470 for AUC.
- **DINOv2:** all 18 CPU semantic/residual pairs match the 36 reference selected states and six paired interval rows.
- **RN:** the Blackwell profile recovers the paper's LIDC-hard effect, intervals and inconclusive decision exactly. Four affected RN/Capacity models were retrained for this check; it is not a second run of all 18 models or every secondary metric. A40/A100 float32 calculations produce different training trajectories for this comparison.
- **Wider model:** default cuDNN convolution gradients can vary despite fixed inputs and seeds. Deterministic cuDNN gives repeatable training but does not recreate the paper's exact selected weights. The reproduced LIDC-hard interval differs in whether it crosses the −0.010 margin.
- **Prob-U and CIMD:** deterministic training is repeatable on the specified setup, but the paper's nondeterministic training trajectories are not recovered. Their reproduced means remain separate from paper values.
- **SAM2:** the specified FP32 setup preserves published automatic MLQ means and point MLQ/Waste means at four decimals. IMA++ retains 60 changed pixels across 47 images, and some scores differ.
- **MC-dropout:** the paper's inference RNG seed was not recorded. The command records an explicit seed, with 3470 as its default.
- **Timing:** all eight timing procedures have been run on one A40. Latency is measured again on each setup and should be reported with its observed variation.

The comparison CSVs describe the complete reproduction run, including its A40/A100 RN result. They have not been replaced with the separate Blackwell result. See [HARDWARE.md](HARDWARE.md) for the RN recipe and the exact scope of that recovery.

## Tests

The installed package passes 66 CPU tests and lint, with no skipped tests on the tested Linux kernel. Tests cover model and evaluator regression, integration, synthetic training, hidden-data isolation and mismatched hardware profiles. The Blackwell preflights and complete synthetic GPU example also pass. The CPU tests use a clean installation; the GPU check uses the pinned CUDA runtime. Standard CI requires no dataset, original project or GPU. Eighteen Pillow deprecation warnings remain.

Small synthetic checks establish software behavior; the full grids above establish experiment coverage. Follow [FULL_REPRODUCTION.md](FULL_REPRODUCTION.md) to run those grids with authorized datasets and weights.
