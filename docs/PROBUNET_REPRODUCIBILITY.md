# Prob-U numerical settings

Use NVIDIA RTX PRO 6000 Blackwell Server Edition, Python 3.12.12, PyTorch 2.9.1+cu128, CUDA 12.8, cuDNN 91002 and eight CPU threads. Set `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Use `--deterministic` for training and inference, inference batch 12 and sampling seed 3470.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/check_probunet_profile.py
```

The preflight checks the GPU, runtime, driver, library fingerprints and numerical flags against `artifacts/probunet_reference_execution.json`. It configures its own process; training and inference still need the environment variables and flags above. [COMPARISONS.md](COMPARISONS.md) specifies the model, loss and commands.

Default cuDNN backward operations can produce different gradients with fixed inputs and seeds. Deterministic execution makes the tested full fits repeat exactly, but does not recover the paper's nondeterministic training trajectory. For example, reproduced BSDS MLQ@8 is 0.13338541 versus the paper's 0.06001964. All 15 task/visible-choice fits are included in the reproduction results. Use the supplied comparison CSVs when assessing numerical agreement; the preflight alone does not certify the paper's means.
