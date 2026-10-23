# Hardware requirements

Use the hardware and numerical settings for each stage. A40 and Blackwell are not interchangeable for all paper comparisons.

| Stage | Verified setup | Agreement with paper references |
| --- | --- | --- |
| Tests and synthetic example | Linux, Python 3.12, pinned CPU wheels | 66 tests and the complete CPU example; no dataset or GPU needed |
| DINOv3 features | NVIDIA A40, batch 12 for IMA++ and 16 otherwise, separate split batches, FP16 storage | All 6,259 feature entries exact |
| DINOv2 features | NVIDIA A40, batch 16 within each split, FP32 storage | All 2,377 feature entries exact |
| DINOv2 model training | AMD CPU, eight threads, CUDA 12.8 PyTorch wheel used on CPU | All 18 semantic/residual pairs match reference selected states |
| LIDC RN/Capacity comparison | NVIDIA RTX PRO 6000 Blackwell Server Edition, AMD EPYC 9475F, eight CPU threads | Tested affected models and the LIDC-hard comparison exact |
| Wider raw model | Same Blackwell setup with deterministic cuDNN | Repeatable training; paper's nondeterministic weights not recovered |
| Core SSRB and other controls | Recorded A40/A100 setups with the paper configuration | Complete grid; numerical differences in reproduction tables |
| Prob-U and CIMD | Deterministic Blackwell, eight CPU threads | Repeatable training; paper's exact training trajectories not recovered |
| Last DINOv3 block | A40 and verified DINOv3 features | Complete deterministic grid; paper's selected weights not all recovered |
| SAM2 quality | Blackwell, FP32, matmul and cuDNN TF32 off | Published means match at four decimals; small pixel/score differences remain |
| Runtime measurements | One A40, eight CPU threads | All eight timing procedures; measure latency again on your setup |

Use Linux with Landlock enabled for `--isolate`. Training uses one GPU per fit, including the full Capacity-WTA batch on that device. Multiple GPUs can run independent fits. Distributed training is not implemented. Minimum training VRAM and host RAM have not been measured; runtime peak allocation does not establish a minimum training specification.

The CUDA environment uses Python 3.12.12, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, CUDA 12.8 and cuDNN 91002. Install `requirements.lock.txt`. The tested Blackwell driver is 595.91.07. CPU-only tests use `requirements-cpu.lock.txt`; DINOv2's CPU training profile uses the CUDA wheel. [NUMERICAL_PROFILES.md](NUMERICAL_PROFILES.md) specifies batch grouping, backbone fingerprints and numerical flags.

## LIDC RN comparison

Prepare all three visible choices of LIDC. Extract and verify the DINOv3 features on A40, import them into each task directory and seal the training inputs. Use those same bytes on the Blackwell setup above.

`check-profile` verifies GPU, CPU, threads, package versions and numerical flags, and exits unsuccessfully if they differ. It configures only its own process. Export the variables and pass `--deterministic` to each training command:

```bash
set -e
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8
ssrb check-profile --profile lidc-dino-reference
for visible in 0 1 2; do
  for optimizer in 0 1 2; do
    task="$PREPARED/lidc/vs$visible"
    run="$RUNS/lidc/vs$visible/os$optimizer"
    ssrb train-semantic --root "$task/trainer" --manifest "$task/trainer/visible.json" --output "$run/capacity" --config configs/paper.json --seed "$optimizer" --device cuda --isolate --deterministic
    ssrb train-comparison --method dino_rank_neighborhood --root "$task/trainer" --manifest "$task/trainer/visible.json" --output "$run/rn" --seed "$optimizer" --device cuda --isolate --deterministic
    ssrb infer --checkpoint "$run/capacity/checkpoint.pt" --root "$task/inference" --manifest "$task/inference/images.json" --feature-file "$task/inference/test_features.pt" --output "$run/capacity_predictions" --device cpu --batch-size 32 --deterministic
    ssrb infer-comparison --checkpoint "$run/rn/checkpoint.pt" --root "$task/inference" --manifest "$task/inference/images.json" --feature-file "$task/inference/test_features.pt" --output "$run/rn_predictions" --device cpu --batch-size 32 --deterministic
    ssrb evaluate --predictions "$run/capacity_predictions" --root "$task/evaluator" --manifest "$task/evaluator/hidden.json" --output "$run/capacity_evaluation"
    ssrb evaluate --predictions "$run/rn_predictions" --root "$task/evaluator" --manifest "$task/evaluator/hidden.json" --output "$run/rn_evaluation"
  done
done
```

Use the complete evaluation list with `analyze-assignment --profile lidc` and the seed specified in [PAPER_ANALYSIS.md](PAPER_ANALYSIS.md).

The recipe contains 18 fits. The exact recovery check retrained the four cases whose hard MLQ differed: visible choices 0 and 1, optimizer seed 1, for both models. Their weights, 8,544 masks and scores match the paper references. Substituting those cases in the complete reproduction inputs recovers effect 0.0047016723702351, 95% interval [0.0007276293601953, 0.0093958484364004] and decision **inconclusive** exactly. This does not establish a second 18-model run or every secondary metric.

## Numerical limits

With identical inputs and initial weights, float32 linear outputs differ from Blackwell by about 3.22e-6 on A40 and 1.43e-6 on A100. Those differences affect the first optimizer update and later checkpoint selection. A40 RN selects epoch 33 instead of 38 in the tested cell; A100 Capacity-WTA selects 24 instead of 32. Individual CUDA kernels and reduction orders were not profiled.

The wider model has a separate sensitivity: default-cuDNN convolution gradients can vary with fixed inputs and weights. `check-profile --profile wide-repeatable` and `train-comparison --method wide_raw --deterministic` select the repeatable setup, without promising the paper's exact selected weights. [REPRODUCIBILITY.md](REPRODUCIBILITY.md) describes the remaining Prob-U, CIMD, SAM2 and MC-dropout limits.
