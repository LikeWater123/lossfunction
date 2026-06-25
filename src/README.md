# lossfunction

Research codebase for validating the **MBA loss family** (Monotone Bounded Amplification)
and baseline losses on image classification benchmarks.

## Purpose

Validate the MBA loss family and a set of baseline losses on CIFAR-10 and CIFAR-100,
using two backbone architectures:

- **ResNet-56** (CNN adapted for 32x32 CIFAR inputs)
- **ViT** (Vision Transformer)

The goal is to study how the proposed monotone bounded amplification losses compare
against standard cross-entropy and existing LACE variants in terms of accuracy,
calibration, and robustness.

## Directory Layout

```
src/
  __init__.py            Package marker.
  README.md             This file.
  requirements.txt      Pinned (>=) Python dependencies.
  train.py              Training entry point (stub).
  evaluate.py           Evaluation entry point (stub).
  methods/             Loss function implementations.
    baselines.py        Baseline losses (CE, focal, etc.).
    lace_variants.py    LACE loss variants.
    mba.py              MBA loss family.
  models/              Model architectures.
    cnn/resnet.py       ResNet-56 for CIFAR.
    vit/vit.py          Vision Transformer.
  datasets/
    cifar.py            CIFAR-10/100 loaders.
  configs/
    defaults.yaml       Default training configuration.
  utils/
    metrics.py          Evaluation metrics.
    plot.py             Plotting helpers.
```

## How to Run

> **Note:** Not yet implemented. The skeleton is in place so parallel agents can
> fill in the logic. Once `train.py` is implemented, run:

```bash
python src/train.py --config src/configs/defaults.yaml
```

For evaluation (once implemented):

```bash
python src/evaluate.py --config src/configs/defaults.yaml --checkpoint <path>
```

## Dependencies

Install with:

```bash
pip install -r src/requirements.txt
```
