# SAM2 numerical settings

Use Blackwell, Python 3.12.12, PyTorch 2.9.1+cu128, CUDA 12.8, cuDNN 91002 and eight CPU threads. [COMPARISONS.md](COMPARISONS.md) specifies the source commit, weights and installation.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python scripts/infer_sam2_fp32.py --root "$TEST_IMAGES" --manifest "$TEST_IMAGES/images.json" --sam-repo "$SAM2_REPO" --weights "$SAM2_WEIGHTS" --retain-full-pool --output "$RUN/sam2_predictions" --device cuda
```

The helper disables matmul TF32 and cuDNN TF32. Use `--single-predictions "$SINGLE_PREDICTIONS"` for the point-based comparison. Timing has its own precision settings in [RUNTIME.md](RUNTIME.md).

All automatic masks match the paper's reference masks on BSDS, LIDC and RIGA. IMA++ retains 60 changed pixels across 47 of 479 images, and predicted-IoU scores can differ. All six automatic MLQ means and twelve point MLQ/Waste means match at four decimals. Full-pool counts match for every image; the maximum per-image oracle change is 9.54e-7. The precision settings used for the paper were not fully recorded, so exact agreement for every tensor is not claimed.

Full pools are compared as multisets. Top-eight and point banks retain score order. Evaluation begins after proposal generation and uses the same original annotation grids as the other methods.
