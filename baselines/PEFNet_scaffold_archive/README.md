# PEFNet-Inspired Adapted Reference

This directory contains an independent, paper-inspired VIE-regularized
two-stage U-Net adaptation. It is not official PEFNet code and must not be
reported as an exact reproduction.

The implementation was developed under a separate UrbanRadio3D experiment
protocol. It is included here for source preservation and auditing. Before it
can support the random-10% comparison table, its dataset, sparse-input path,
split, measurement projection, metric region, and aggregation must be aligned
with the common baseline contract and the model must be retrained.

See `docs/PEFNET_ADAPTATION.md` for all physical and numerical assumptions.

Paper: F. Jiang et al., "Physics-Informed Neural Networks for Path Loss
Estimation by Solving Electromagnetic Integral Equations," IEEE TWC, 2024,
doi:10.1109/TWC.2024.3429196.

