# Installation

Python 3.12, PyTorch 2.9.1, torchvision 0.24.1, NumPy 2.2.6, SciPy 1.16.3 and Pillow 11.3.0 are pinned in the package. Start with a new virtual environment; do not expose another project's site-packages or set its source directory on PYTHONPATH.

Install `.[test,datasets,analysis]` for tests and native dataset preparation. Install `.[backbone]` for DINO feature extraction, and obtain the external DINOv3 checkout and weights under Meta's license. The tested checkout is `54694f7627fd815f62a5dcc82944ffa6153bbb76`. See the exact environment record for additional packages imported by its hub configuration.

CPU installation and tests are in the README. CUDA 12.8 wheels:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install '.[test,datasets,analysis,backbone]'
python -m ssrb.cli doctor
```

`--isolate` requires Linux x86_64/aarch64 with Landlock enabled. It fails if kernel enforcement is unavailable. The test reports a skip on unsupported kernels; such a skip does not certify isolation. No administrator privileges are needed. The implementation handles read rights only, as documented in PROTOCOL.md.

The environments are pinned in `requirements-cpu.lock.txt` and `requirements.lock.txt` (CUDA 12.8, including feature-extraction dependencies). Install the PyTorch wheels from the matching official index, then the lock and this project:

```bash
# CPU environment
python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.lock.txt
python -m pip install --no-deps .
```

For the CUDA environment, use `torch==2.9.1+cu128`, `torchvision==0.24.1+cu128`, the index `https://download.pytorch.org/whl/cu128` and `requirements.lock.txt` instead. The CPU lock includes tests and dataset converters; DINO feature extraction uses the CUDA/backbone profile. Both profiles use Python 3.12 on Linux; other platforms may produce different numerical results.

Before a full experiment, select the stage's hardware from [HARDWARE.md](HARDWARE.md). For the recovered LIDC RN comparison, export `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, then run `ssrb check-profile --profile lidc-dino-reference`. A nonzero exit means the checked setup differs. The preflight configures only its own process; training commands still require `--deterministic` and the exported environment variables.

For DINOv2, the tested upstream commit is `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. See COMPARISONS.md for the ViT-S/14 weight checksum and the A40 feature-extraction profile.
