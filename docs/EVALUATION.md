# Evaluation

```bash
python -m ssrb.cli infer --checkpoint "$RUN/residual/checkpoint.pt" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --feature-file "$TEST_IMAGES/test_features.pt" --output "$RUN/predictions" --device cuda
python -m ssrb.cli evaluate --predictions "$RUN/predictions" --root "$REFERENCES" --manifest "$REFERENCES/hidden.json" --output "$RUN/evaluation"
python -m ssrb.cli make-paper-tables --evaluation "$RUN/evaluation" --status new-training-reproduction --output "$RUN/results.csv"
```

Inference sorts by sigmoid rank score, descending, preserving head order at ties. Predictions are saved before evaluation. The exported probabilities, ranks and their hashes let the evaluator run independently. Evaluation produces per-image JSON and endpoint summaries, including LIDC-hard where applicable. `make-paper-tables` exports each supplied run without silently averaging seeds. Repeat `--evaluation` to include several runs; retain their provenance and status labels.

Dice has additive smoothing one, including empty masks. MLQ is the mean, over references, of the best prediction Dice in a ranked prefix. Worst takes the minimum over references. AUC averages prefixes 1 through 8. Waste is the fraction of predictions whose best-reference Dice is below 0.30. PairDice averages unordered prediction pairs. Unique95 counts connected components after linking pairs with Dice at least 0.95.

For the accepted representation analysis, supply the paired analysis-unit CSV to `bootstrap-representation --units units.csv --output bootstrap.json`. Columns are `regime`, `optimizer_seed`, `image_id`, `source_id`, `method_canonical`, `MLQ@8`, `Waste@8`; average visible choices within each image/optimizer/method first. Method names are `raw_residual_fusion`, `dino_capacity`, `raw_capacity`. This reproduces the paper's 10,000 draws with seed 20260814; its grouping caveat is documented in PROTOCOL.md. It is not a replacement for all paper-specific analysis scripts.

For a complete seed matrix, create a JSON list containing `evaluation` directories and `optimizer_seed` values. Run `aggregate-paper-table --evaluations evaluations.json --status verified-original-checkpoint --output table.csv`. It averages visible choices within image/source/optimizer first, then averages those units for each endpoint. This matters for LIDC-hard, where membership depends on the visible annotation. It rejects duplicate visible memberships, duplicate cells, missing visible choices and incomplete optimizer-seed sets. The default requires optimizer seeds 0, 1 and 2; `--optimizer-seeds 0` explicitly selects baselines fitted once per visible choice. It does not calculate confidence intervals or imply a new training campaign.

Inference defaults to batch size 12, matching the accepted evaluator. `--batch-size` can change it. CPU/GPU kernels and batch shape can cause tiny floating-point differences, which may cross a printed rounding boundary. The prediction manifest records the device and batch size; HARDWARE.md specifies the verified profiles.

## Averaging rules used by the accepted tables

The representation study, its controls and the paired Prob-U analyses average visible choices within each image and optimizer seed before averaging those units. This is `--averaging visible-first`, the default. The older comparator summaries average the complete run means with equal weight; use `--averaging run-means` for the CIMD/SAM tables and the Single/MC-dropout comparator summaries. The two rules can differ on LIDC-hard because its membership depends on the visible annotation.

For example, export the 15 CIMD evaluations with:

```bash
ssrb aggregate-paper-table --evaluations cimd_evaluations.json --optimizer-seeds 0 --averaging run-means --status new-training-reproduction --output cimd_table.csv
```

For the Prob-U medical values used alongside SSRB's paired analysis, use the same declared optimizer seed with `--averaging visible-first`. The older Prob-U comparator table also remains reproducible with `run-means`; its LIDC-hard value is a different summary of the same per-image data. Use the rule belonging to the table being checked. The output records the chosen rule.
