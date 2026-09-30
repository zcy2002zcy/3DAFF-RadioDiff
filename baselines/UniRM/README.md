# UniRM Adapter Reference

This directory contains the local UniRM adapter and its training/evaluation
support code. The official UniRM source is not redistributed because the
pinned upstream revision does not declare a license.

Review `external/BASELINE_SOURCES.json`, then obtain the pinned upstream source
with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_external_baselines.ps1
```

The adapter originated in a separate UrbanRadio3D no-sampling protocol. It is
included for auditing and must be aligned to the common random-10% protocol and
retrained before its output can be used to substantiate the manuscript table.

Official repository: https://github.com/Shirleyue/UniRM

