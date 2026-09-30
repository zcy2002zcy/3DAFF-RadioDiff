# External baseline sources

This directory stores only the machine-readable source registry. Third-party
repositories are intentionally not redistributed.

The active UniRM adapter expects a local checkout at `external/UniRM` pinned to
the commit recorded in `BASELINE_SOURCES.json`. Run
`scripts/setup_external_baselines.ps1` to create it. The pinned UniRM revision
does not declare a license; review the upstream terms and do not commit or
redistribute the checkout.

The RadioUNet-, RMTransformer-, PEFNet-, and RadioDiff-3D-style implementations
used by this project are local adaptations and do not import vendored upstream
source code.
