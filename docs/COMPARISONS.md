# Baselines and controls

All commands below read the same portable manifests as SSRB. Paths beginning with `$` are variables you set locally. Run `ssrb --help` for the full interface. See HARDWARE.md for execution settings and REPRODUCIBILITY.md for numerical limits.

## Train and evaluate

Before the Prob-U commands, run the [execution-profile check](NUMERICAL_PROFILES.md) in the same environment.

```bash
ssrb train-comparison --method probunet --root "$TRAINER" --manifest "$TRAINER/visible.json" --output "$RUN/probunet" --seed 0 --device cuda --isolate --deterministic
ssrb infer-comparison --checkpoint "$RUN/probunet/checkpoint.pt" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --output "$RUN/probunet_predictions" --device cuda --deterministic --batch-size 12 --sampling-seed 3470
ssrb evaluate --predictions "$RUN/probunet_predictions" --root "$REFERENCES" --manifest "$REFERENCES/hidden.json" --output "$RUN/probunet_evaluation" --ged
```

`probunet` is the authors' implementation used in this comparison: U-Net base 24, latent size 8, BCE + Dice + 0.01 KL, Adam 0.001, batch 12, 80 epochs maximum, patience 12, and validation with the posterior mean. Inference draws eight samples from the prior. Its default sampling seed is 3470 and batch size is 12.

Other `--method` choices are:

| Method | Configuration and behavior |
|---|---|
| `raw_capacity` | Raw U-Net, base 24, eight heads, Capacity-WTA, Adam 0.001, batch 12, 80 epochs, patience 12 |
| `wide_raw` | Same raw control with base 29 |
| `conventional_residual` | Raw residual with ordinary PyTorch initialization; requires `--anchor` pointing to the semantic checkpoint |
| `dino_wta` | Frozen features, eight heads, hard WTA |
| `dino_soft_wta` | Frozen features, eight heads, soft WTA with temperature 0.05 |
| `dino_rank_neighborhood` | Four warm-up epochs of hard WTA; rank temperature falls linearly from 1 to 0.1; selection begins at epoch 5 |
| `dino_single` | One head over frozen features |
| `dino_mc_dropout` | One head, dropout 0.2; eight stochastic passes at inference, sorted by scores |
| `cimd` | Adaptation of the external CIMD implementation, described below |

The DINO comparisons use AdamW 0.0003, weight decay 0.0001, batch 32, at most 40 epochs and patience 8. Their inference batch is also 32. Masks and rank logits use the same losses as the released semantic bank. Raw controls validate with hard WTA. The raw Capacity-WTA loader draws from the global Torch RNG after model initialization; the wider and conventional controls use a separate generator seeded with the optimizer seed. The raw and wider objectives omit the diversity branch entirely. Even a term multiplied by zero can change CUDA gradient rounding. The semantic Capacity-WTA anchor is trained with `train-semantic`, rather than with `dino_wta`.

Train eight `dino_single` models with independent optimizer seeds for the ensemble. Infer each model separately, then supply all eight prediction directories:

```bash
ssrb combine-ensemble --predictions "$P0" --predictions "$P1" --predictions "$P2" --predictions "$P3" --predictions "$P4" --predictions "$P5" --predictions "$P6" --predictions "$P7" --output "$RUN/ensemble_predictions"
ssrb make-extent-banks --predictions "$SINGLE_PREDICTIONS" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --output "$RUN/extent_predictions"
```

The extent command creates separate banks for fixed thresholds, fixed logit biases and fixed morphology offsets. It uses the original image dimensions, without reading reference masks. Each subdirectory can be evaluated independently.

## Matched addition and four-plus-four union

Train `raw_capacity` with the same visible seed and optimizer seed as the semantic anchor. Fit the head permutation using visible validation masks:

```bash
ssrb align-heads --semantic-checkpoint "$SEMANTIC" --raw-checkpoint "$RAW" --root "$TRAINER" --manifest "$TRAINER/visible.json" --output "$RUN/alignment.json" --device cuda
ssrb infer-posthoc --semantic-checkpoint "$SEMANTIC" --raw-checkpoint "$RAW" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --feature-file "$TEST_IMAGES/test_features.pt" --alignment "$RUN/alignment.json" --output "$RUN/additive_predictions" --device cuda
```

Omit `--alignment` for the union. It concatenates the first four raw predictions with the first four semantic predictions. It preserves that order and does not compare scores across the two models. Matched addition sums mask and rank logits and then sorts the combined scores. Matching minimizes mean validation BCE + Dice over all 8! permutations, retaining the first lexicographic optimum within 1e-12.

## External CIMD and SAM2

Obtain upstream code and weights separately under their licenses. The release includes the authors' CIMD adapter, rather than the third-party implementation. Use the official [CIMD repository](https://github.com/aimansnigdha/Ambiguous-Medical-Image-Segmentation-using-Diffusion-Models) at commit `a9677afe4eedb163db478cd01ea8228161448b33`. The adapter checks the commit. Use the tested PyTorch 2.9.1 / CUDA 12.8 environment described in [NUMERICAL_PROFILES.md](NUMERICAL_PROFILES.md), then install `pip install '.[cimd]'`. This adds blobfile 3.0.0 and matplotlib 3.10.7. The upstream requirements file targets a different PyTorch generation; it is not the environment used by this adapter. Do not substitute the unmodified upstream training loop.

Run the supplied helpers from the release checkout to restrict file reads during both fitting and sampling:

```bash
python scripts/train_cimd_isolated.py --root "$TRAINER" --manifest "$TRAINER/visible.json" --output "$RUN/cimd" --cimd-root "$CIMD_REPO" --device cuda --deterministic
python scripts/infer_cimd_isolated.py --checkpoint "$RUN/cimd/checkpoint.pt" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --output "$RUN/cimd_predictions" --cimd-root "$CIMD_REPO" --device cuda --deterministic
ssrb evaluate --predictions "$RUN/cimd_predictions" --root "$REFERENCES" --manifest "$REFERENCES/hidden.json" --output "$RUN/cimd_evaluation" --ged
```

These Linux helpers require Landlock. They allow the pinned upstream checkout alongside the declared inputs, runtime, scratch and output. Each helper checks that an outside control file becomes unreadable. Add `--check-denied-read PATH` to check a known hidden manifest or another file that must be inaccessible. Choose separate trainer and image-only test roots.

CIMD converts RGB to luminance and repeats it across four conditioning channels. Both ambiguity networks are optimized and checkpointed. The clean target remains unchanged. The loss is MSE + 0.001 VLB + 0.001 ambiguity KL. Training uses float32, AdamW 0.0001, weight decay zero, batch 20, at most 80 epochs, patience 12 and fixed validation noise with seed 3480. Inference uses eight samples, 1000 reverse steps, sampling seed 3490 and batch 1.

For SAM2, use the official [repository](https://github.com/facebookresearch/sam2) at commit `2b90b9f5ceec907a1c18123530e92e794ad901a4` and the SAM2.1 Hiera-Large checkpoint. With the tested PyTorch 2.9.1 / torchvision 0.24.1 CUDA 12.8 wheels already installed, install the runtime pins and the upstream package separately:

```bash
pip install -r requirements/sam2-runtime.txt setuptools==80.9.0 wheel==0.45.1
SAM2_BUILD_CUDA=0 pip install --no-build-isolation --no-deps "$SAM2_REPO"
```

Use the upstream Python path without its optional compiled extension. Then run:

```bash
python scripts/infer_sam2_fp32.py --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --sam-repo "$SAM2_REPO" --weights "$SAM2_WEIGHTS" --output "$RUN/sam2_predictions" --device cuda
```

This uses the official automatic generator with its defaults and explicitly disables both TF32 settings. See SAM2_REPRODUCIBILITY.md for precision settings and numerical agreement. It retains up to eight proposals, ordered by predicted IoU, stability score and original index. Evaluation preserves its native number of masks, including zero, and resizes binary masks with nearest-neighbor interpolation. Native masks are stored as lossless run-length encodings, which avoids very large dense files for full-resolution images. The evaluator and bank plotter also accept older dense prediction files.

Add `--retain-full-pool` to freeze every automatic proposal alongside the top-eight bank. Evaluate the oracle separately:

```bash
ssrb evaluate-sam2-pool --predictions "$RUN/sam2_predictions" --root "$REFERENCES" --manifest "$REFERENCES/hidden.json" --output "$RUN/sam2_pool_oracle"
```

The full-pool oracle uses held-out references to find the closest proposal for each annotation. It is a diagnostic, and has no eight-mask budget. As in the paper, this calculation reports full LIDC; it does not add a LIDC-hard result.

For the experiment using a point from a trained model, first train and infer `dino_single` for the corresponding task and visible choice. Pass its frozen predictions with `python scripts/infer_sam2_fp32.py --single-predictions "$SINGLE_PREDICTIONS"`. The command checks their hashes and test-image manifest. It bilinearly resizes the probability map to the image grid, chooses the point at the maximum interior distance within the thresholded mask, and falls back to the probability maximum if the mask is empty. SAM2 returns three masks, sorted stably by its score. This mode adds learned localization and is reported separately in the paper.

The ordinary `--isolate` switch supports the core, raw, semantic and Prob-U trainers. For CIMD, use the helpers above to include its required upstream imports. A container with the same input restrictions is another option.

## DINOv2 and training the last DINOv3 block

`extract-features --backbone dinov2` uses the verified DINOv2 ViT-S/14 checkpoint, blocks 3, 6 and 11, image size 448, a 32x32 token grid and dimension 384 and FP32 cached features. Use `train-semantic` and `train-ssrb` on those caches. The initial weight SHA-256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. The tested DINOv2 source is commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; all 157 Python files match the source used for these extraction checks. Obtain that checkout and the weights separately.

`train-partial` and `infer-partial` accept `--dino-repo` and `--weights` along with the usual root, manifest and output paths. The trainer updates only encoder block 11 and the decoder. Encoder forwards remain float32 in evaluation mode. It converts features through FP16, uses ordered encoder batches of 16 and concatenates them before one decoder pass and one assignment over the global batch of 32. AdamW learning rates are 0.000003 for the encoder and 0.0003 for the decoder, with weight decay 0.0001. Selection begins after epoch 4, with patience 8 and a maximum of 40 epochs. The first 32 training images in sorted order must match the supplied training cache bit for bit, and a backward check must give four assignments per head with gradients in the decoder and encoder block. The trainer stops if either check fails. The accepted execution profile requires NVIDIA A40 and PyTorch 2.9.1+cu128. The full pretrained encoder is not included in the trained checkpoint; inference requires the same original weights. Both commands accept `--deterministic`. Use --deterministic for repeatable training; the paper's exact selected weights are not recovered for every fit.

DINOv3 loading uses a temporary checkpoint cache containing only the supplied weight file. An existing Torch Hub checkpoint with the same filename is not substituted for that file.

## Correspondence controls

For BSDS and LIDC, repeat these commands for three visible choices and optimizer seeds 0, 1 and 2. They train the four reported controls using the original fixed settings:

```bash
for method in sa_dmb capacity_sa_dmb sa_dmb_shuffled capacity_sa_dmb_shuffled; do
  ssrb train-sharing --method "$method" --root "$TRAINER" --manifest "$TRAINER/visible.json" --output "$RUN/$method/training" --seed "$OPTIMIZER_SEED" --device cuda --isolate --deterministic
  ssrb infer-comparison --checkpoint "$RUN/$method/training/checkpoint.pt" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --feature-file "$TEST_FEATURES" --output "$RUN/$method/predictions" --device cuda --batch-size 32 --deterministic
  ssrb evaluate --predictions "$RUN/$method/predictions" --root "$REFERENCES" --manifest "$REFERENCES/hidden.json" --output "$RUN/$method/evaluation"
done
```

The methods use real or shuffled support correspondence, with or without the capacity constraint. They share the 40-epoch DINO schedule, four warm-up epochs, patience eight, batch 32 and AdamW learning rate 0.0003. The paper uses 72 fits across these methods and seed combinations.


## Numerical execution

Use the Blackwell recipe in [HARDWARE.md](HARDWARE.md) for the LIDC RN/Capacity comparison. The repeatable wider-model setup uses deterministic cuDNN; it does not recreate the paper's exact nondeterministic weights.

Prob-U training depends on the CUDA execution profile. See [Prob-U numerical settings](PROBUNET_REPRODUCIBILITY.md) for the repeatable profile and remaining differences from the paper.

For MC-dropout, the original evaluator did not set or save an inference RNG seed. The release records an explicit seed (3470 by default), with the original batch size of 32. Exact reconstruction of those unrecorded draws is unavailable.
