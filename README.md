# Semantic-Spatial Residual Banks (SSRB)

Code for *Single-Annotation, Multi-Hypothesis Segmentation with Semantic-Spatial Residual Banks*, accepted at WACV 2027, Algorithms Track. Armin Dadras, Robert Sablatnig, Franziska Proksa and Markus Seidl.

SSRB returns eight segmentation masks from one fixed annotation per training image. A bank over frozen DINOv3 features supplies semantic hypotheses. A U-Net learns a spatial residual for each hypothesis; the branches add their mask and ranking logits head by head.

![LIDC CT example: the image, visible annotation in gold, held-out annotation in pink, and four SSRB predictions in blue.](figures/lidc_examples.png)

*Example from the paper: one LIDC nodule crop, its visible and held-out annotations, and the first four of eight SSRB masks in score order. The predictions illustrate different possible extents of the nodule. Image and annotations: LIDC-IDRI/TCIA, CC BY 3.0. [Figure details and attribution](figures/README.md).*

## Install and try the example

Use Python 3.12 on Linux. The synthetic example and tests need no dataset, pretrained model, GPU or credentials.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.lock.txt
python -m pip install --no-deps .
python -m ssrb.cli doctor
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 make lint test
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m ssrb.cli smoke --output outputs/demo
```

The example prepares synthetic data, trains both stages, saves eight predictions per test image and evaluates the withheld annotations. Its smaller networks and short schedule are for checking the installation. [INSTALL.md](docs/INSTALL.md) covers CUDA installation.

## Reproduce the paper

[HARDWARE.md](docs/HARDWARE.md) lists the hardware and numerical settings for each stage. Use A40 for the verified feature extraction and Blackwell for the LIDC RN/Capacity comparison. Some nondeterministic training trajectories cannot be recreated exactly; [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) describes the measured agreement and remaining limits.

1. Obtain the datasets and weights listed in [DATASETS.md](docs/DATASETS.md).
2. Prepare the fixed splits and visible annotations.
3. Extract and verify frozen features, then seal the training inputs.
4. Train the semantic bank, freeze it, and train the residual.
5. Save test predictions before running the reference-mask evaluator.
6. Aggregate the complete seed grid and generate the tables and figures.

Start with the [BSDS example](docs/REPRODUCE_PAPER.md), then use [FULL_REPRODUCTION.md](docs/FULL_REPRODUCTION.md) for all tasks, seed combinations and comparisons. [COMPARISONS.md](docs/COMPARISONS.md) gives baseline commands. [PAPER_ANALYSIS.md](docs/PAPER_ANALYSIS.md) describes the intervals and figures; [RUNTIME.md](docs/RUNTIME.md) describes timing.

Training uses one GPU per fit. Capacity assignment covers the full batch on that device. Independent fits may run on separate GPUs. Supported training commands offer `--isolate` to enforce a Linux read allowlist; keep evaluator annotations outside the trainer directory. See [PROTOCOL.md](docs/PROTOCOL.md).

MLQ@8 measures whether the mask bank contains a close match to each reference annotation. It uses references during evaluation. Model scores rank the masks for inspection; they are not probabilities of annotator choices.

## Results and citation

[artifacts/](artifacts/README.md) contains aggregate paper values, reproduction results, plot inputs and feature fingerprints. Full datasets, identifying manifests and trained checkpoints must be obtained separately. The illustration above includes one LIDC crop under the terms in [figures/README.md](figures/README.md).

The MIT license covers the SSRB code. External datasets, pretrained models and baseline implementations retain their own terms; see [THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md). Use [CITATION.cff](CITATION.cff) to cite the paper. Questions can be raised through the repository's Issues page.
