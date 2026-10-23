# Run the complete BSDS experiment

Use absolute paths for the task variables below. Obtain BSDS500 and the external DINOv3 checkout/weights first. These commands create new directories and refuse to overwrite earlier runs.

```bash
SOURCE=/path/to/data/converted_bsds
PREPARED=/path/to/data/bsds_visible0
RUN=/path/to/data/bsds_v0_o0
DINO_REPO=/path/to/external/dinov3
DINO_WEIGHTS=/path/to/authorized/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth

python -m ssrb.cli prepare-bsds --data-root /path/to/BSR/BSDS500/data --output "$SOURCE"
python -m ssrb.cli make-visible-manifests --source "$SOURCE/all_annotations.json" --data-root "$SOURCE" --output "$PREPARED" --seed 0
python -m ssrb.cli extract-features --root "$PREPARED/trainer" --manifest "$PREPARED/trainer/visible.json" --output "$PREPARED/features_visible" --dino-repo "$DINO_REPO" --weights "$DINO_WEIGHTS" --device cuda --batch-size 16
python -m ssrb.cli extract-features --root "$PREPARED/inference" --manifest "$PREPARED/inference/images.json" --output "$PREPARED/features_test" --dino-repo "$DINO_REPO" --weights "$DINO_WEIGHTS" --device cuda --batch-size 16
for split in train val; do
  python -m ssrb.cli import-features --source "$PREPARED/features_visible/${split}_features.pt" --output "$PREPARED/trainer/${split}_features.pt" --manifest "$PREPARED/trainer/visible.json" --split "$split"
done
python -m ssrb.cli import-features --source "$PREPARED/features_test/test_features.pt" --output "$PREPARED/inference/test_features.pt" --manifest "$PREPARED/inference/images.json" --split test
python -m ssrb.cli seal-inputs --root "$PREPARED/trainer" --manifest "$PREPARED/trainer/visible.json"
python -m ssrb.cli train-semantic --root "$PREPARED/trainer" --manifest "$PREPARED/trainer/visible.json" --output "$RUN/semantic" --config configs/paper.json --seed 0 --device cuda --isolate --cache-inputs
python -m ssrb.cli train-ssrb --root "$PREPARED/trainer" --manifest "$PREPARED/trainer/visible.json" --output "$RUN/residual" --config configs/paper.json --seed 0 --device cuda --anchor "$RUN/semantic/checkpoint.pt" --isolate --cache-inputs
python -m ssrb.cli infer --checkpoint "$RUN/residual/checkpoint.pt" --root "$PREPARED/inference" --manifest "$PREPARED/inference/images.json" --feature-file "$PREPARED/inference/test_features.pt" --output "$RUN/predictions" --device cuda
python -m ssrb.cli evaluate --predictions "$RUN/predictions" --root "$PREPARED/evaluator" --manifest "$PREPARED/evaluator/hidden.json" --output "$RUN/evaluation"
python -m ssrb.cli make-paper-tables --evaluation "$RUN/evaluation" --status new-training-reproduction --output "$RUN/results.csv"
```

For IMA++ and LIDC, use the corresponding converter and the same stages. The verified DINOv3 extraction uses A40 and batch 12 for IMA++, batch 16 for LIDC. RIGA uses the shared extraction below before the per-task training sequence. Model/visible seed combinations use separate run directories. Rerunning all three visible seeds and three optimizer seeds means nine SSRB runs per task. LIDC-hard is derived during LIDC evaluation. Disc and cup require separate RIGA models.

The table command emits run rows and their status. Recreating every accepted figure/table also requires the corresponding baseline/control outputs and analysis recipe. REPRODUCIBILITY.md describes numerical agreement and limitations. Do not relabel a newly trained single seed as reproduction of the full paper average.

When importing an existing cache, supply the matching manifest and split as above. The shared RIGA cache contains both task entries; the importer selects the requested endpoint from its manifest and rejects missing or duplicated identities. It preserves the tensor dtype and drops unrelated metadata.

## RIGA shared feature cache

`prepare-riga` creates `feature_images.json` containing both disc and cup. Keep the shared image batches for extraction, then select each task with its own manifest. Models and visible masks remain separate. The exact profile uses A40, PyTorch 2.9.1+cu128 and batch 16.

```bash
ssrb extract-features --root "$SOURCE" --manifest "$SOURCE/feature_images.json" --output "$RIGA_FEATURES" --dino-repo "$DINO_REPO" --weights "$DINO_WEIGHTS" --device cuda --batch-size 16 --backbone dinov3
for split in train val test; do
  ssrb verify-features --source "$RIGA_FEATURES/${split}_features.pt" --reference artifacts/reference_feature_hashes.json --endpoint riga --split "$split" --backbone dinov3
done
for split in train val; do
  ssrb import-features --source "$RIGA_FEATURES/${split}_features.pt" --output "$PREPARED/trainer/${split}_features.pt" --manifest "$PREPARED/trainer/visible.json" --split "$split"
done
ssrb import-features --source "$RIGA_FEATURES/test_features.pt" --output "$PREPARED/inference/test_features.pt" --manifest "$PREPARED/inference/images.json" --split test
```

Repeat the imports and model stages for each task and visible choice using separate prepared directories. The same frozen image features serve all visible choices. [NUMERICAL_PROFILES.md](NUMERICAL_PROFILES.md) describes the exact checks for both backbones.
