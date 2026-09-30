# Pix2Pix PL baseline

Input: q + binary building mask + random-10% sparse PL + receiver altitude.
No depth, LoS, or building-height values are used. The inference generator uses
`base_channels=15` and approximately 2.992M parameters (4.3% from
3DAFF-RadioDiff); the training-only discriminator has approximately 0.155M.
Evaluation uses the native direct generator, not a diffusion-noise input.

```powershell
.\run_check_p2p_local.ps1 -Gpu 0
.\run_train_p2p_local.ps1 -Gpu 0
.\run_eval_p2p_local.ps1 -Gpu 0
```

See `../README.md` for the complete publication protocol.
