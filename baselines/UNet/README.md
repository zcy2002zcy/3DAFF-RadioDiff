# U-Net PL baseline

Input: q + binary building mask + random-10% sparse PL + receiver altitude.

The model does not receive the NPZ `depth`, NPZ `los`, or rasterized building-height
values. Its four channels are `[q, building_mask, sparse_PL, altitude_map]`.

This is the parameter-matched variant: `base_channels=19` gives approximately
2.739M parameters, within 4.5% of 3DAFF-RadioDiff's 2.868M parameters.

```powershell
.\run_check_unet_local.ps1 -Gpu 0
.\run_train_unet_local.ps1 -Gpu 0
.\run_eval_unet_local.ps1 -Gpu 0
```

See `../README.md` for the complete publication protocol.
