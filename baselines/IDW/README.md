# IDW PL baseline

IDW uses only the fixed random-10% sparse observations. It has no trainable weights.

```powershell
.\run_check_idw_local.ps1
.\run_train_idw_local.ps1
.\run_eval_idw_local.ps1 -Gpu 1
```

See `../README.md` for the complete publication protocol.
