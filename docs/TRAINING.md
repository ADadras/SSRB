# Training

Use `configs/paper.json` for paper hyperparameters: eight heads; semantic MLP width 256, no dropout; raw U-Net base width 24; image size 128 for the raw branch. Semantic training uses AdamW, learning rate 0.0003, weight decay 0.0001, batch 32, at most 40 epochs, patience 8. Residual training uses Adam, learning rate 0.001, batch 12, at most 80 epochs, patience 12. There is no augmentation.

The loss is mean assigned BCE plus soft Dice (smoothing one), plus 0.30 times mean binary cross entropy for ranks. The Hungarian assignment is detached. The residual corrects corresponding mask/rank heads with coefficient one. Only its final mask and rank output layers start at zero. The semantic bank remains frozen and in evaluation mode during residual fitting.

```bash
python -m ssrb.cli train-semantic --root "$VISIBLE" --manifest "$VISIBLE/visible.json" --output "$RUN/semantic" --config configs/paper.json --seed 0 --device cuda --isolate
python -m ssrb.cli train-ssrb --root "$VISIBLE" --manifest "$VISIBLE/visible.json" --output "$RUN/residual" --config configs/paper.json --seed 0 --device cuda --anchor "$RUN/semantic/checkpoint.pt" --isolate
```

Use a new output directory for every run. The trainer refuses to overwrite one. Repeat preprocessing for visible seeds 0, 1, 2 and each training stage for optimizer seeds 0, 1, 2. A residual uses the semantic bank with the corresponding visible/optimizer seeds. Keep the same preprocessing, split IDs and features across optimizer seeds. Do not select a seed using test scores.

DINOv3 features are FP16 tensors of shape images x 3 x 784 x 768, from blocks 5, 8, 11 of ViT-B/16 at 448 x 448. The bank casts these to FP32. Extracted features can be imported from an authorized cache with `import-features`; that command strips unrelated metadata. A cache-based check alone does not certify raw-image feature extraction.

DINOv2 uses ViT-S/14, blocks 3, 6 and 11, with FP32 caches of shape images × 3 × 1024 × 384. Manifest filtering in `import-features` selects a task and split from a shared cache.

The optional `--deterministic` flag on core/comparison training and inference requests deterministic operations and records the numerical settings. It does not alter the objective or select a seed. Default CUDA settings are retained unless the flag is requested. See PROBUNET_REPRODUCIBILITY.md for that baseline's numerical settings.

For large images, add `--cache-inputs` to both training commands. This keeps the decoded, resized images and masks in memory for that stage. It calls the same preprocessing functions and preserves the original file hashes and checks. The cache does not include frozen feature tensors a second time. A residual cache needs about 256 KiB per image at size 128; the training record reports its actual size.

Before DINOv2 training, run `verify-features` for every split using the supplied reference digests. The accepted DINOv2 semantic and residual stages used CPU with eight threads. See [NUMERICAL_PROFILES.md](NUMERICAL_PROFILES.md) for the matched feature and training settings.
