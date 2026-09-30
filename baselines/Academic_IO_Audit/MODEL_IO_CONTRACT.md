# Model Input/Output Contract

| Method | Authorized model input | Forbidden input | Output |
| --- | --- | --- | --- |
| 3DAFF-RadioDiff | `q`, `depth`, `los`, sparse `s`, `s_mask`, altitude `z`, noisy state `x_t`, step `t` | complete clean `psi` as a condition; unobserved target pixels | one normalized PL slice |
| U-Net | `q`, building mask, sparse `s`, `s_mask`, altitude `z` | `depth`, `los`, complete `psi` as an input | one normalized PL slice |
| ResNet | `q`, building mask, sparse `s`, `s_mask`, altitude `z` | `depth`, `los`, complete `psi` as an input | one normalized PL slice |
| Pix2Pix | `q`, building mask, sparse `s`, `s_mask`, altitude `z` | `depth`, `los`, complete `psi` in generator input | one normalized PL slice |
| IDW | sampled coordinates and sampled PL values | unobserved PL values | one interpolated PL slice |
| Kriging | sampled coordinates and sampled PL values | unobserved PL values | one kriged PL slice |
| Kriging + topology | sampled coordinates/values and building mask | unobserved PL values | one topology-aware kriged PL slice |
| 3D-RadioDiff reimplementation | `q`, building mask/height, sparse PL volume, altitude/noisy state/step | `depth`, `los`, unobserved PL values | one normalized PL volume |
| UniRM adapter | `q`, building occupancy/height, sparse PL, altitude | `depth`, `los`, unobserved PL values | one normalized PL slice |
| PEFNet-inspired adapter | `q`, receiver-height building occupancy, sparse `s`, `s_mask`, altitude `z` | `depth`, `los`, complete `psi` as an input | one normalized PL slice |

## Training-only use of the target

Using `psi` as the supervised target is required and is not leakage. In a
diffusion model, using `psi` to construct `x_t` during training is also standard.
Leakage occurs if clean unobserved `psi` values enter the condition at test time.

For Pix2Pix, the discriminator may receive the real target while optimizing the
adversarial objective. The generator must not receive the real target as input.

## Evaluation

1. Load the fixed mask associated with `(sample_id, altitude)`.
2. Predict without exposing unobserved target values.
3. For every sampled method, apply the same final projection:

   `prediction = where(s_mask, s, prediction)`

4. Compute full-slice NMSE, RMSE, SSIM, and PSNR with the same normalization and
   `data_range`; macro-average per-slice values over all 40,000 test slices.
5. Save per-slice metrics and report the unobserved-outdoor region separately.

Do not use the target to select a test-time hyperparameter, choose a checkpoint,
or tune a post-processing coefficient.

The aligned UniRM adapter maps the upstream seven-channel interface to building
occupancy slices, normalized building height, transmitter map `q`, and sparse
PL. It does not read NPZ `depth` or `los`. Checkpoints trained with the earlier
LoS/depth channel mapping are incompatible and cannot be evaluated as the new
baseline.

The aligned PEFNet-inspired adapter rasterizes raw building polygons and uses
receiver-height occupancy as a documented susceptibility proxy. It is an
adaptation because material permittivity and field phase are unavailable. It
must be named `PEFNet-inspired (building-input adaptation, our implementation)`
rather than official PEFNet.
