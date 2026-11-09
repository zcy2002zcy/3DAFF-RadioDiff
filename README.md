# 3DAFF-RadioDiff

Official code for **3DAFF-RadioDiff: Geometry-Guided Residual Diffusion for Measurement-Free 3D Path-Loss Prediction**.

3DAFF-RadioDiff jointly predicts path-loss maps at receiver heights of 1, 2, 3, and 4 m from only the transmitter location and the building geometry. No received signal strength (RSS) measurement is used at inference.

This is release **v1**. It contains source code only: no dataset, no pretrained weights, and no experiment outputs.

## Method overview

- **Voxel-wise adaptive LoS--depth fusion (AFF).** A lightweight 3D convolutional module estimates normalized fusion weights for the LoS mask and the depth descriptor at every voxel.
- **3D residual U-Net with per-height spatial Transformer.** 3D convolutions couple the receiver-height layers, horizontal-only downsampling preserves the height dimension, and a Transformer at the bottleneck (2 blocks, 4 heads, dimension 128) models long-range dependencies within each height.
- **Two-stage prediction.**
  - Stage I: a deterministic coarse predictor regresses the four-height path-loss volume.
  - Stage II: the coarse predictor is frozen, and a conditional diffusion corrector generates the scaled residual (cosine schedule, $T=200$, $v$-prediction, residual scale 0.5, terminal-timestep oversampling, geometric projection).
- **Inference.** The residual is sampled from Gaussian noise with deterministic DDIM and added to the coarse prediction.

## Repository layout

```text
models/        AFF, 3D residual blocks, Transformer, coarse predictor and residual corrector
datasets/      four-height NPZ manifest dataset
diffusion/     noise schedules and forward process
scripts/       training / calibration / evaluation engine
configs/       base configuration
tests/         unit tests for the model, runner and launcher
tools/         CPU verification and source-manifest tools
reference/     immutable provenance copies verified by the engine (not imported)
run.py         entry point (check / preflight / train / calibrate / eval / all)
analyze_diffusion_value.py, make_paper_figs.py, plot_diffusion_value.py
               uncertainty analysis and figure scripts
```

## Environment

- Python >= 3.8, PyTorch >= 1.12 with CUDA
- Other dependencies: `pip install -r requirements.txt`

The launcher in this release keeps the safeguards used for the paper experiments. Formal runs require:
- an NVIDIA RTX 3090 that is idle and has at least 20 GB of free memory;
- a conda environment named `radiodiffzcy`;
- a successful full-size GPU preflight.

## Data

The experiments use the four-height (1--4 m) subset of Urban3D-RadioMap with a scene-disjoint split of 108,200 / 12,000 / 10,000 training / validation / test samples. The dataset root is expected to contain:

```text
<dataset-root>/
  metadata.json
  train.csv  val.csv  test.csv
  geometry/scene_####.npz
  packed/{train,val,test}/scene_####/<sample_id>.npz
```

Each sample NPZ contains `target_u8`, `los_u8`, and `depth_f16`; each geometry NPZ contains the building heights. The dataset itself is not redistributed here. Please obtain it from the original release.

## Usage

```bash
conda activate radiodiffzcy
# Read-only check of the code and data (CPU, no CUDA)
python -B run.py --action check --dataset-root /path/to/dataset
# Full-size GPU preflight (weights are discarded)
python -B -u run.py --action preflight --dataset-root /path/to/dataset --gpu 0
# Train Stage I (10 epochs) and Stage II (10 epochs), calibrate on validation, then evaluate
python -B -u run.py --action all --dataset-root /path/to/dataset --gpu 0
```

Fixed training settings: AdamW (learning rate 1e-4, weight decay 1e-4), batch size 2, EMA 0.999, seed 42, automatic mixed precision. All metrics (RMSE, NMSE, PSNR with data range 2, and SSIM) are computed on the normalized path loss in $[-1,1]$, only at valid receiver locations.

Uncertainty analysis (K samples per input) and paper figures:

```bash
python analyze_diffusion_value.py --split test --samples 200 --K 8 --steps 200 --output outputs_diffusion_value
python make_paper_figs.py --input outputs_diffusion_value
```

## Citation

If you find this code useful, please cite our paper (citation will be added upon publication).
