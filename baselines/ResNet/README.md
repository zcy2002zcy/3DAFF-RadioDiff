# ResNet PL baseline

Input: q + binary building mask + random-10% sparse PL + receiver altitude.
No depth, LoS, or building-height values are used. `base_channels=22` gives
approximately 2.931M parameters (2.2% from 3DAFF-RadioDiff).

```powershell
.\run_check_resnet_local.ps1 -Gpu 0
.\run_train_resnet_local.ps1 -Gpu 0
.\run_eval_resnet_local.ps1 -Gpu 0
```

See `../README.md` for the complete publication protocol.
