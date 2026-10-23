# Numerical settings

Fix the hardware, dependency versions, batch sizes and numerical flags before training. [HARDWARE.md](HARDWARE.md) lists the verified setup for each stage and the LIDC RN commands. Deterministic execution does not promise identical results across GPU architectures.

## Blackwell model training

The verified setup uses NVIDIA RTX PRO 6000 Blackwell Server Edition, AMD EPYC 9475F, Python 3.12.12, PyTorch 2.9.1+cu128, CUDA 12.8, cuDNN 91002 and eight CPU threads. Set the variables before starting Python:

```bash
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
ssrb check-profile --profile lidc-dino-reference
```

Pass `--deterministic` to each training/inference command requiring this profile. It enables deterministic PyTorch and cuDNN execution and disables cuDNN benchmarking. Matmul TF32 is off and cuDNN TF32 is on. `wide-repeatable` checks the same setup for repeatable wider-model training; it does not promise the paper's exact selected weights.

Prob-U additionally uses inference batch 12 and sampling seed 3470. Run `python scripts/check_probunet_profile.py` under the variables above. It checks driver 595.91.07, compute capability and the 11 CUDA-library fingerprints in `artifacts/probunet_reference_execution.json`. See [Prob-U settings](PROBUNET_REPRODUCIBILITY.md) for numerical limits.

## Frozen DINOv2 features

Use A40, the pinned CUDA environment, ViT-S/14 blocks 3/6/11, image size 448, batch 16 and FP32 storage. Keep batches separate for each split, even when train and validation share a manifest. Matmul TF32 is off, cuDNN TF32 is on and deterministic execution is not enabled for this feature extraction. Source and weight fingerprints are in [COMPARISONS.md](COMPARISONS.md).

```bash
ssrb verify-features --source "$FEATURES/train_features.pt" --reference artifacts/reference_feature_hashes.json --endpoint bsds --split train --backbone dinov2
ssrb verify-features --source "$FEATURES/val_features.pt" --reference artifacts/reference_feature_hashes.json --endpoint bsds --split val --backbone dinov2
ssrb verify-features --source "$TEST_FEATURES/test_features.pt" --reference artifacts/reference_feature_hashes.json --endpoint bsds --split test --backbone dinov2
```

Use `--endpoint lidc` for LIDC. These checks include ordered image fingerprints, shape, dtype, grid and tensor bytes. A mismatch exits unsuccessfully. DINOv2 semantic/residual training uses the CUDA 12.8 PyTorch wheel on an AMD CPU with eight threads. All 18 pairs match the paper's reference selected states on that profile.

## Frozen DINOv3 features

Use A40 with the pinned CUDA environment, ViT-B/16 blocks 5/8/11, image size 448 and FP16 storage. Keep batches within each split. Use batch 12 for IMA++ and 16 for the other datasets. RIGA disc and cup share the converter's `feature_images.json`; extracting them separately changes the final batch. Import each task's entries after extracting the shared cache.

Use the same `verify-features` commands with `--backbone dinov3` and endpoint `bsds`, `ima`, `lidc` or `riga`. Verify the shared RIGA cache before importing task-specific entries. [REPRODUCE_PAPER.md](REPRODUCE_PAPER.md) gives the full sequence.

## CIMD and SAM2

CIMD timestep frequencies depend on CPU arithmetic. Keep the CPU model and relevant library settings fixed as well as the GPU. Execution metadata includes CPU model/capability/thread counts, `ATEN_CPU_CAPABILITY`, `MKL_CBWR` and `MKL_ENABLE_INSTRUCTIONS`. See [CIMD settings](CIMD_REPRODUCIBILITY.md).

SAM2 quality evaluation uses Blackwell, eight CPU threads, FP32, matmul TF32 off and cuDNN TF32 off. `scripts/infer_sam2_fp32.py` sets both flags explicitly. [SAM2 settings](SAM2_REPRODUCIBILITY.md) describes the remaining small pixel and score differences. Runtime measurements use the separate precision settings in [RUNTIME.md](RUNTIME.md).
