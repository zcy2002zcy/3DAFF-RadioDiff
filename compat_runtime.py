"""Explicit API compatibility for this package, without global monkey-patches.

Target: Python >= 3.8 and PyTorch >= 1.12. Actual compatibility must still be
verified by the package's CPU tests and the server preflight on the target
environment. API feature detection is not a claim that every target GPU/runtime
combination has been tested. No model, objective, input or training budget is
changed here. Full checkpoints must be trusted: they contain optimizer/RNG state
and use PyTorch's pickle loader, not a weights-only exchange format.

Older checkpoint implementations use reentrant recomputation. They preserve RNG
and parameter gradients with normal ``loss.backward()`` (the training path used
here), but do not offer the modern non-reentrant ``autograd.grad`` semantics.
"""
from __future__ import annotations

from contextlib import nullcontext
import inspect
import re
import sys

import torch
from torch.utils.checkpoint import checkpoint as _torch_checkpoint


def _has_parameter(function, name):
    """Do not confuse **kwargs with support for a newer named argument."""
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _modern_scaler():
    scaler = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    return scaler if callable(scaler) and _has_parameter(scaler, 'device') else None


def make_grad_scaler(enabled=True):
    """Use the matching CUDA AMP API; ``enabled=False`` is CPU-safe."""
    modern = _modern_scaler()
    if modern is not None:
        return modern('cuda', enabled=bool(enabled))
    legacy = getattr(getattr(getattr(torch, 'cuda', None), 'amp', None), 'GradScaler', None)
    if not callable(legacy):
        raise RuntimeError('This PyTorch runtime has no compatible CUDA GradScaler API')
    return legacy(enabled=bool(enabled))


def autocast(device_type, enabled=True):
    """CUDA FP16 AMP, or an unchanged FP32 CPU context when disabled."""
    if not enabled:
        return nullcontext()
    if device_type != 'cuda':
        raise ValueError('This experiment permits AMP only on CUDA; CPU must remain FP32')
    generic = getattr(torch, 'autocast', None)
    if callable(generic) and _has_parameter(generic, 'device_type'):
        return generic(device_type='cuda', dtype=torch.float16, enabled=True)
    legacy = getattr(getattr(getattr(torch, 'cuda', None), 'amp', None), 'autocast', None)
    if not callable(legacy):
        raise RuntimeError('This PyTorch runtime has no compatible CUDA autocast API')
    return legacy(dtype=torch.float16, enabled=True)


def load_checkpoint(path, map_location='cpu'):
    """Load a trusted full checkpoint exactly once, without exception retries."""
    kwargs = {'map_location': map_location}
    if _has_parameter(torch.load, 'weights_only'):
        kwargs['weights_only'] = False
    return torch.load(path, **kwargs)


def activation_checkpoint(function, *tensor_args):
    """Recompute with preserved RNG and gradients on old/new checkpoint APIs.

    Reentrant APIs require a grad-bearing input even when the trainable weights
    are captured by ``function``. A scalar dummy supplies this bookkeeping only
    when necessary. It does not enter the model, consume RNG or add parameters.
    The wrapped transformer receives precisely the original arguments.
    """
    if not tensor_args or not all(isinstance(value, torch.Tensor) for value in tensor_args):
        raise TypeError('activation_checkpoint requires one or more flat Tensor arguments')
    if not torch.is_grad_enabled():
        return function(*tensor_args)
    if _has_parameter(_torch_checkpoint, 'use_reentrant'):
        return _torch_checkpoint(function, *tensor_args, use_reentrant=False,
                                 preserve_rng_state=True)
    if any(value.requires_grad for value in tensor_args):
        return _torch_checkpoint(function, *tensor_args, preserve_rng_state=True)
    first = tensor_args[0]
    dtype = first.dtype if first.is_floating_point() else torch.float32
    dummy = torch.zeros((), device=first.device, dtype=dtype, requires_grad=True)

    def with_grad_sentinel(*arguments):
        return function(*arguments[:-1])

    return _torch_checkpoint(with_grad_sentinel, *tensor_args, dummy,
                             preserve_rng_state=True)


def validate_runtime():
    """Validate the supported API floor without initializing/querying CUDA."""
    if sys.version_info[:2] < (3, 8):
        raise RuntimeError('This package requires Python >= 3.8')
    match = re.match(r'^(\d+)\.(\d+)', str(torch.__version__))
    if match is None or tuple(int(value) for value in match.groups()) < (1, 12):
        raise RuntimeError('This package requires PyTorch >= 1.12; detected %s' % torch.__version__)
    return runtime_compatibility()


def runtime_compatibility():
    """Provenance only: selected API paths, not GPU capability or test claims."""
    generic = getattr(torch, 'autocast', None)
    return {
        'python': sys.version.split()[0],
        'torch': str(torch.__version__),
        'torch_cuda_build': torch.version.cuda,
        'minimum_python': '3.8',
        'minimum_torch': '1.12',
        'grad_scaler_api': ('torch.amp.GradScaler(cuda)' if _modern_scaler() is not None
                            else 'torch.cuda.amp.GradScaler'),
        'autocast_api': ('torch.autocast(cuda, float16)' if callable(generic) and
                         _has_parameter(generic, 'device_type') else 'torch.cuda.amp.autocast(float16)'),
        'checkpoint_load': ('explicit_weights_only_false' if _has_parameter(torch.load, 'weights_only')
                            else 'legacy_full_checkpoint_default'),
        'activation_checkpoint': ('non_reentrant_preserve_rng' if
                                   _has_parameter(_torch_checkpoint, 'use_reentrant') else
                                   'legacy_reentrant_preserve_rng_with_grad_sentinel_if_needed'),
        'checkpoint_backward_contract': 'normal loss.backward(); no autograd.grad requirement',
        'cpu_precision': 'FP32; CPU AMP disabled',
        'gpu_runtime_validation': 'required on target server; API detection alone is not GPU validation',
    }
