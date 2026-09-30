# PEFNet-inspired baseline adaptation protocol

## Reporting status

The reference method is **Physics-informed Electromagnetic Field Network
(PEFNet)** from:

F. Jiang, T. Li, X. Lv, H. Rui, and D. Jin, "Physics-Informed Neural
Networks for Path Loss Estimation by Solving Electromagnetic Integral
Equations," *IEEE Transactions on Wireless Communications*, vol. 23, no. 10,
pp. 15380--15393, 2024, doi:10.1109/TWC.2024.3429196.

The verified BibTeX record is stored in `docs/baselines.bib` under the key
`Jiang2024PEFNet`.

The paper does not provide an official PEFNet repository or checkpoint. It
states that the method was implemented in PyTorch 1.12, but several quantities
needed for an exact reproduction are not disclosed. This project therefore
reports its implementation only as **PEFNet-inspired VIE-regularized U-Net
(adapted)**. It must not be described as official PEFNet or as a faithful
reproduction of the authors' implementation.

Published numbers from the paper are not copied into the comparison table. The
adapted model is retrained from scratch and evaluated with the same data and
metric protocol as every other local baseline.

## Paper-guided structure

PEFNet uses a field-estimation U-Net followed by a residual U-Net. In the paper,
the first network maps transmitter/building masks and the real and imaginary
parts of the incident field to a two-channel complex total field. Its magnitude
is converted to a coarse path-loss map. The second network predicts a residual
that is added to that map. The two networks are optimized jointly with the
paper's Eq. (18),

```text
L = L_PHY + lambda_data * L_DAT
```

and the paper reports `lambda_data = 1` as its best tested setting in Table VI.
The local implementation retains that two-part prediction and joint-loss logic,
but operates under the explicit adaptations below.

## UrbanRadio3D adaptation and assumptions

The local data expose building-height PNGs, transmitter coordinates and height,
receiver height, and raw path-loss PNG values. They do not expose carrier
frequency, metric pixel spacing, building conductivity, complex field labels,
transmit power, or a verified conversion from PNG values to dB. Consequently,
the integral equation is used as a dimensionless effective-medium regularizer,
not as a calibrated electromagnetic solution at a claimed physical frequency.

The five adapted input channels are, in order:

1. transmitter source mask;
2. binary building occupancy in the horizontal receiver-height slice;
3. real part of the dimensionless incident field;
4. imaginary part of the dimensionless incident field;
5. normalized receiver-height plane.

The fifth channel is an UrbanRadio3D adaptation; the paper's field network uses
four input channels. The checked-in formal configuration records every numerical
assumption:

| Quantity | Formal value | Status |
| --- | ---: | --- |
| Air relative permittivity | 1.0 | From the paper |
| Building relative permittivity | 4.0 | From the paper |
| Conductivity | 0 S/m | Lossless implementation assumption; not disclosed |
| Grid spacing | 1 dimensionless pixel | Implementation assumption |
| Wavenumber | 0.35 rad/pixel | Dimensionless implementation assumption |
| Physics grid | 64 x 64 | Computational adaptation |
| Data-loss weight | 1.0 | Paper Table VI |

The paper writes both residual losses as `||r||_2 / N`. Because this adaptation
evaluates the VIE on a configurable coarse grid while the data residual remains
on 256 x 256 pixels, both terms are implemented as RMS residuals. This keeps the
relative loss scale invariant to the selected physics-grid resolution. It is an
explicit numerical adaptation, not an undocumented claim of exact Eq. (15)
reproduction.

These values are not evidence that the raw dataset represents a specific carrier
frequency or material calibration. Any future verified dataset metadata must be
introduced as a new configuration and experiment tag rather than silently
changing this protocol.

## Fair comparison protocol

- scene-disjoint project split;
- fixed data-selection seed 42;
- 4,096 train maps, 128 validation maps, and 128 test maps selected with
  `balanced_scene`;
- model/training seeds 42, 43, and 44;
- at most 1,000 optimizer steps per epoch, matching the checked-in dense-neural
  baseline budget;
- no sparse path-loss observations (`sample_rate = 0`);
- validation checkpoint selection by NMSE;
- test metrics on valid receiver pixels only;
- per-map metrics followed by a macro mean;
- normalized primary domain `x / 127.5 - 1.0`, with raw encoded-domain metrics
  carrying the `raw_` prefix.

The target remains the stored PNG value. Neither normalized nor raw encoded
metrics may be labelled dB until the official PNG-to-dB mapping is verified.

## Numerical-stability revision v2

The formal dense protocol uses batch size one. An earlier BatchNorm/FP16 trial
developed exploding running variances in the refinement decoder and terminated
with a non-finite data loss; that incomplete run is audit-only and is excluded
from every result table. The frozen v2 protocol applies the same settings to all
three seeds and makes the following adaptations explicit:

- GroupNorm replaces BatchNorm in both U-Nets, using the largest divisor up to
  32 that leaves at least four channels per group;
- the bounded knowledge U-Net uses CUDA BF16 autocast, while field-to-path-loss
  conversion and the complete refinement U-Net run in FP32;
- AdamW uses a peak learning rate of `1e-4`, five linear warm-up epochs starting
  at 10%, and cosine decay to `1e-6`;
- non-finite losses, gradients, validation predictions, parameters, and buffers
  fail immediately and cannot produce `run_complete.json`;
- evaluation requires a valid completion marker and matching effective-config
  hashes, checkpoint epoch, seed, and data seed.

These are stability adaptations for the local batch-size-one implementation,
not claims about the unpublished official PEFNet training code. Failed v1
checkpoints must not be resumed or reported.

## Reproduction commands

One seed can be trained and evaluated with:

```powershell
python scripts\train_pefnet_inspired_baseline.py `
  --config configs\pefnet_inspired_adapted.yaml --seed 42 --data-seed 42
python scripts\evaluate_pefnet_inspired_baseline.py `
  --config configs\pefnet_inspired_adapted.yaml --seed 42 --data-seed 42
```

The ICC suite runs all three seeds and performs protocol validation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_pefnet_inspired_baseline.ps1
```

The complete ICC suite (all methods and ablations) is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_icc_repro_suite.ps1 `
  -SkipGeometryBuild
```

Formal outputs use
`outputs/pefnet_inspired_adapted_stable_v2_random0_seed{42,43,44}`. Each run must retain
the resolved configuration, provenance, training history, best/latest
checkpoints, a completion marker, per-map test metrics, and the test summary.
The completion marker must distinguish early stopping from reaching the maximum
epoch count; either is a valid termination when recorded truthfully.

## Permitted manuscript wording

Use wording such as:

> We implemented a paper-inspired, VIE-regularized two-stage U-Net based on
> PEFNet and retrained it under our common UrbanRadio3D protocol. Because the
> dataset and paper do not expose all quantities required for a calibrated VIE
> reproduction, we report it as an adapted baseline rather than official PEFNet.

Do not use wording such as "official PEFNet," "exact PEFNet reproduction," or
"Maxwell-accurate 5.9 GHz solution" for this implementation.
