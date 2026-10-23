# Runtime measurements

Use an A40, the pinned PyTorch 2.9.1 CUDA 12.8 installation, eight CPU threads and an image-only LIDC test manifest. Retain the environment, device and checkpoint hashes in each output. The command loads the model once, synchronizes CUDA around each timed call and records latency, dispersion, peak allocated memory and parameter counts.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
ssrb benchmark --method ssrb --checkpoint "$SSRB" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS" --device cuda --output "$RUN/runtime_ssrb"
```

The paper's CIMD timing seed is 3670. The default procedure uses the first eight sorted test images, 100 warm-up calls and 500 timed calls cycling through those images. Methods are `semantic`, `ssrb`, `raw_capacity`, `wide_raw`, `probunet`, `cimd`, `sam2_amg` and `sam2_point`. Supply the corresponding checkpoint and the required external `--cimd-root`, `--sam-repo` and `--sam-weights` arguments. The DINO methods require the pinned DINO repository and weights. CIMD draws eight masks with all 1000 reverse steps per call.

Semantic DINO, raw banks and SSRB use BF16 autocast on CUDA in this timing procedure. Prob-U, CIMD and automatic SAM2 use FP32. The separate SAM point procedure uses BF16 for Single/DINO and FP32 for SAM2, with **one warm-up and 24 timed calls**:

```bash
ssrb benchmark --method sam2_point --checkpoint "$SINGLE" --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --dino-repo "$DINO_REPO" --dino-weights "$DINO_WEIGHTS" --sam-repo "$SAM2_REPO" --sam-weights "$SAM2_WEIGHTS" --warmup-runs 1 --timed-runs 24 --device cuda --output "$RUN/runtime_point"
```

These are the procedures behind the reported runtime rows; they differ from FP32 training and cached-feature accuracy evaluation. `--smoke` performs a shorter functional check and labels it accordingly. Do not report that check as the full timing experiment. Latency is a new hardware measurement, so report its observed value and variation rather than expecting bytewise equality with an earlier timing run.
