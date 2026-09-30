# Results

## Reported comparison

Overall path-loss reconstruction performance under 10% random measurements.
Down arrows indicate lower is better; the PSNR up arrow indicates higher is
better.

| Method | NMSE ↓ | RMSE ↓ | PSNR ↑ (dB) |
|---|---:|---:|---:|
| Kriging | 0.0777 | 0.1431 | 23.1646 |
| Kriging + topology | 0.0533 | 0.1201 | 24.8400 |
| IDW | 0.0340 | 0.0942 | 26.7898 |
| Pix2Pix | 0.0389 | 0.1013 | 26.3333 |
| ResNet | 0.0322 | 0.0897 | 27.3161 |
| U-Net | 0.0273 | 0.0846 | 27.8260 |
| PEFNet | 0.0234 | 0.0780 | 28.6051 |
| 3D-RadioDiff | 0.0840 | 0.1391 | 24.7667 |
| UniRM | 0.0247 | 0.0773 | 28.5691 |
| **3DAFF-RadioDiff** | **0.0179** | **0.0686** | **29.5560** |

These are the manuscript-level headline values supplied with the project. They
should not be silently replaced by metrics from a different input contract or
evaluation protocol.

## Included training and evaluation evidence

The `baselines/` directory contains the runnable baseline bundle and available
checkpoint-free logs/results. Model weights and datasets are intentionally not
committed.

Two newer building-input runs currently have complete 40,000-sample evaluation
summaries in the bundle:

| Included run | NMSE | RMSE | SSIM | PSNR (dB) | Samples |
|---|---:|---:|---:|---:|---:|
| Building-input U-Net | 0.00312278 | 0.02875459 | 0.98117117 | 37.1130 | 40,000 |
| Building-input ResNet | 0.00371320 | 0.03126805 | 0.97980418 | 36.4408 | 40,000 |

These newer values are retained as run evidence, not substituted into the
reported comparison table: their building/altitude inputs and post-prediction
measurement-consistency behavior differ from the historical table protocol.

Relevant files:

- `baselines/BASELINE_STATUS.csv`: implementation and manuscript-value status.
- `baselines/Academic_IO_Audit/MODEL_IO_CONTRACT.md`: audited model I/O contracts.
- `baselines/UNet/outputs/pathloss_unet_buildingmask_random10_matched/`: U-Net
  training logs, per-sample metrics, summary, and saved visual examples.
- `baselines/ResNet/outputs/pathloss_resnet_buildingmask_random10_matched/`:
  ResNet training logs, per-sample metrics, summary, and saved visual examples.

For a fair updated table, retrain every learned baseline with the same split,
authorized inputs, sampling mask, seed policy, metric domain, and measurement
consistency rule. PEFNet and UniRM in the bundle are aligned adapters and still
require retraining before their historical values can be treated as results of
the included implementations.
