"""Disposable full-size TRAIN-only server preflight; import never probes CUDA.

The launcher owns environment/GPU/idleness checks and the global operation lock.
This module creates only the explicitly supplied, new proof file after success.
No checkpoint, training counter, validation score or test prediction is created.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from compat_runtime import autocast, make_grad_scaler, runtime_compatibility

EXPECTED_CONDITION_SHAPE = (2, 9, 4, 256, 256)
EXPECTED_TARGET_SHAPE = (2, 1, 4, 256, 256)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _logical_cuda_index(device):
    # The launcher masks the inspected physical GPU by UUID. Within this
    # process that selected GPU is always logical 0, even for CLI --gpu 1.
    # set_device requires an explicit index; torch.device('cuda') is not one.
    device = torch.device(device)
    if device.type != 'cuda' or device.index not in (None, 0):
        raise ValueError('Preflight requires the UUID-bound logical cuda:0 device')
    return 0


def _gpu_begin(device):
    requested_index = _logical_cuda_index(device)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; preflight never falls back to CPU')
    torch.cuda.set_device(requested_index)
    index = torch.cuda.current_device()
    if index != requested_index:
        raise RuntimeError('Current CUDA device differs from UUID-bound logical cuda:0')
    properties = torch.cuda.get_device_properties(index)
    torch.cuda.reset_peak_memory_stats(index)
    return {'gpu_name': properties.name, 'cuda_device_index': index,
            'gpu_total_memory_bytes': int(properties.total_memory),
            'gpu_compute_capability': [properties.major, properties.minor]}


def _gpu_end(device):
    index = _logical_cuda_index(device)
    torch.cuda.synchronize(index)
    return {'peak_cuda_allocated_bytes': int(torch.cuda.max_memory_allocated(index)),
            'peak_cuda_reserved_bytes': int(torch.cuda.max_memory_reserved(index))}


def _gpu_cleanup():
    torch.cuda.empty_cache()


def _move(tensor, device):
    return tensor.to(device)


def _finite(value, label):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Nonfinite ' + label)


def _validate_batch(condition, target, valid):
    if tuple(condition.shape) != EXPECTED_CONDITION_SHAPE:
        raise ValueError('Preflight requires condition [2,9,4,256,256]; no automatic downsizing')
    if tuple(target.shape) != EXPECTED_TARGET_SHAPE or tuple(valid.shape) != EXPECTED_TARGET_SHAPE:
        raise ValueError('Preflight requires target/mask [2,1,4,256,256]')
    _finite(condition, 'training condition')
    _finite(target, 'training target')
    if not condition.is_floating_point() or not target.is_floating_point():
        raise TypeError('Expected floating-point condition and target')
    if not bool(valid.bool().reshape(2, 4, -1).any(-1).all()):
        raise ValueError('Every preflight sample-height must have valid voxels')


def _update(loss, model, optimizer, scaler):
    _finite(loss, 'preflight loss')
    scaler.scale(loss).backward()
    # Match formal training: finite checks and clipping must see the original
    # gradient units, not GradScaler's enlarged backward gradients.
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError('No gradients produced')
    for parameter in parameters:
        _finite(parameter.grad, 'preflight gradient')
    return _finish_update(loss, model, optimizer, scaler, parameters)


def _finish_update(loss, model, optimizer, scaler, parameters):
    """Finish one update only AFTER backward, unscale and finite checks."""
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    _finite(norm, 'preflight gradient norm')
    # Pick a parameter that actually has a gradient, instead of relying on a
    # particular first parameter receiving gradients in all real scenes.
    candidates = [p for p in parameters if bool(p.grad.abs().max() > 0)]
    if not candidates:
        raise RuntimeError('All preflight gradients are zero')
    parameter = max(candidates, key=lambda p: float(p.grad.abs().max()))
    before = parameter.detach().clone()
    old_scale = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    if float(scaler.get_scale()) < old_scale:
        raise FloatingPointError('AMP skipped the preflight optimizer update; no passed proof is written')
    change = float((parameter.detach()-before).abs().max())
    if not change > 0:
        raise RuntimeError('Preflight optimizer did not change the checked parameter')
    for parameter in model.parameters():
        _finite(parameter, 'updated preflight parameter')
    return {'loss': float(loss.detach()), 'gradient_norm': float(norm),
            'parameter_update_max_abs': change, 'successful_optimizer_updates': 1,
            'loss_scale': float(scaler.get_scale())}


def _calibrated_update(model, optimizer, scaler, forward, stage,
                       max_attempts=17, min_scale=1.):
    """Bounded loss-scale calibration of a disposable, fresh optimizer only.

    Each attempt recomputes the SAME forward with restored RNG/buffers. Only
    nonfinite gradients with enabled AMP may back off, never nonfinite losses,
    outputs, components or norms. No training weights/scaler are returned.
    """
    if max_attempts < 1 or min_scale < 1 or optimizer.state:
        raise ValueError('Calibration requires a fresh optimizer and positive bounded attempts/scale')
    named = list(model.named_parameters())
    if not named:
        raise RuntimeError('No preflight model parameters')
    # CPU snapshots avoid keeping a second large parameter copy on the GPU.
    original = [p.detach().cpu().clone() for _, p in named]
    original_groups = copy.deepcopy(optimizer.state_dict()['param_groups'])
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    cpu_rng = torch.get_rng_state()
    device = named[0][1].device
    cuda_rng = torch.cuda.get_rng_state(_logical_cuda_index(device)) if device.type == 'cuda' else None
    attempts = []
    for attempt in range(1, max_attempts+1):
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, _logical_cuda_index(device))
        with torch.no_grad():
            for buffer, value in buffers:
                buffer.copy_(value)
        optimizer.zero_grad(set_to_none=True)
        old_scale = float(scaler.get_scale())
        if not math.isfinite(old_scale) or old_scale < min_scale:
            raise FloatingPointError('Invalid preflight loss scale: '+str(old_scale))
        loss, components = forward()
        _finite(loss, 'preflight loss')
        for name, value in components.items():
            _finite(value, stage+' component '+name)
        record = {'attempt': attempt, 'scale_before': old_scale,
                  'loss': float(loss.detach()),
                  'loss_components': {k: float(v.detach()) for k, v in components.items()}}
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        parameters, bad = [], []
        for name, parameter in named:
            if parameter.grad is not None:
                parameters.append(parameter)
                count = int((~torch.isfinite(parameter.grad)).sum())
                if count:
                    bad.append({'name': name, 'nonfinite_elements': count})
        if not parameters:
            raise RuntimeError('No gradients produced')
        if not bad:
            proof = _finish_update(loss, model, optimizer, scaler, parameters)
            record.update(state='updated', scale_after=float(scaler.get_scale()))
            attempts.append(record)
            proof.update(stage=stage, amp_attempts=attempts, amp_backoffs=attempt-1,
                         loss_components=record['loss_components'])
            return proof
        # Never clip or update EMA on an overflow attempt. GradScaler must
        # demonstrably skip AdamW and reduce its scale before another forward.
        record.update(nonfinite_parameter_count=len(bad), nonfinite_parameters=bad[:8])
        if not scaler.is_enabled():
            raise FloatingPointError('Nonfinite preflight gradient without enabled AMP: '+json.dumps(record))
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        if optimizer.state or optimizer.state_dict()['param_groups'] != original_groups:
            raise RuntimeError('Overflow attempt changed preflight optimizer state')
        for (_, parameter), before in zip(named, original):
            if not torch.equal(parameter.detach().cpu(), before):
                raise RuntimeError('Overflow attempt changed preflight model parameters')
        if not math.isfinite(new_scale) or not 0 < new_scale < old_scale:
            raise FloatingPointError('AMP failed to reduce preflight loss scale')
        record.update(state='overflow_skipped', scale_after=new_scale)
        attempts.append(record)
        print('[server preflight] '+stage+' AMP backoff '+json.dumps(record, allow_nan=False), flush=True)
        del loss, components
        if attempt == max_attempts or old_scale <= min_scale or new_scale < min_scale:
            raise FloatingPointError('Persistent nonfinite '+stage+' preflight gradients; '
                                     'calibration limit reached, no training started: '+json.dumps(attempts))
    raise AssertionError('Unreachable preflight calibration state')


def _ema_update(ema, model):
    with torch.no_grad():
        for average, value in zip(ema.parameters(), model.parameters()):
            average.lerp_(value, .001)


def run_preflight(runner, args, proof_path):
    """Execute guarded CUDA preflight and return the same JSON proof written.

    The caller must already hold the global operation lock and reject the wrong
    conda environment/GPU or other active workloads. The proof parent must exist.
    Only bounded AMP loss-scale calibration is allowed on disposable weights.
    Persistent numerical errors or OOM propagate; no smaller recipe is tried.
    """
    proof_path = Path(proof_path).absolute()
    runner.reject_reparse(proof_path)
    runner.reject_reparse(proof_path.parent)
    if proof_path.exists() or not proof_path.parent.is_dir():
        raise ValueError('Preflight proof must be a NEW file in an existing directory')
    if args.output.resolve() == proof_path.resolve() or args.output.resolve() in proof_path.resolve().parents:
        raise ValueError('Preflight proof must remain outside formal training outputs')
    if args.device.type != 'cuda' or args.batch_size != 2 or args.channels != 32:
        raise ValueError('Full server preflight requires CUDA, batch2 and base_channels32')
    if args.smoke or args.train_samples != 0 or args.coarse_epochs != 10 or args.diffusion_epochs != 10:
        raise ValueError('A smoke/reduced-budget configuration cannot pass the formal preflight')
    engine = runner.make_engine()
    audit = engine.audit(args.dataset_root)  # Manifests/metadata only, not target arrays.
    protocol = engine.build_protocol(args, audit)
    protocol_hash = engine.content_hash(protocol)
    sources = engine.source_hashes()
    if protocol.get('source_sha256') != sources:
        raise ValueError('Preflight source/protocol binding differs')
    script_hash = _sha256(__file__)
    compatibility = runtime_compatibility()  # Version inspection only.
    gpu = _gpu_begin(args.device)
    # Disposable preflight randomness is reproducible. The formal engine resets
    # its own RNG streams again before training; these weights are never reused.
    torch.manual_seed(args.seed)
    print('[server preflight] full batch2, 256x256, TRAIN only; disposable weights', flush=True)
    # Direct indexing avoids worker processes and does not advance a formal
    # training loader. Selecting two TRAIN examples is a resource check only.
    data = engine.dataset(args, 'train', 2)
    if len(data) != 2:
        raise ValueError('Preflight requires exactly two TRAIN examples')
    examples = [data[0], data[1]]
    condition = _move(torch.stack([b['condition'] for b in examples]), args.device).float()
    target = _move(torch.stack([b['target'] for b in examples]), args.device).float()
    valid = _move(torch.stack([b['valid_mask'] for b in examples]), args.device).bool()
    _validate_batch(condition, target, valid)
    engine.ensure_geometry(condition, valid)
    alpha = engine.schedule(condition.device)
    _finite(alpha, 'diffusion schedule')
    coarse = engine.Predictor(args.channels).to(condition.device).train()
    coarse_ema = copy.deepcopy(coarse).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(coarse.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = make_grad_scaler(enabled=True)
    def coarse_forward():
        with autocast('cuda', enabled=True):
            prediction = coarse(condition)
            if tuple(prediction.shape) != EXPECTED_TARGET_SHAPE:
                raise ValueError('Coarse output shape mismatch')
            _finite(prediction, 'coarse output')
            loss = engine.masked_mse(prediction.float(), target, valid)
        return loss, {'masked_mse': loss}
    coarse_proof = _calibrated_update(coarse, optimizer, scaler, coarse_forward, 'coarse')
    _ema_update(coarse_ema, coarse)
    del coarse_forward, coarse, optimizer, scaler
    coarse = coarse_ema.float().eval().requires_grad_(False)
    del coarse_ema
    print('[server preflight] coarse forward/backward/update passed; frozen coarse FP32', flush=True)
    # Precisely the production recipe: frozen coarse forward OUTSIDE autocast.
    with torch.no_grad():
        mu = coarse(condition.float()).float().clamp(-1, 1)
    if mu.dtype != torch.float32 or any(p.requires_grad for p in coarse.parameters()):
        raise RuntimeError('Frozen coarse FP32 contract failed')
    _finite(mu, 'frozen coarse output')
    clean = torch.where(valid, (target-mu)/args.scale, torch.zeros_like(target))
    timestep = engine.draw_timesteps(2, alpha, condition.device, terminal_probability=.25)
    a = alpha[timestep].view(-1, 1, 1, 1, 1)
    noise = torch.randn_like(clean)
    xt = a.sqrt()*clean+(1-a).sqrt()*noise
    v_target = a.sqrt()*noise-(1-a).sqrt()*clean
    corrector = engine.Corrector(args.channels).to(condition.device).train()
    corrector_ema = copy.deepcopy(corrector).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = make_grad_scaler(enabled=True)
    def diffusion_forward():
        with autocast('cuda', enabled=True):
            v = corrector(xt, condition, timestep, mu).float()
            if tuple(v.shape) != EXPECTED_TARGET_SHAPE:
                raise ValueError('Diffusion output shape mismatch')
            _finite(v, 'diffusion output')
            return engine.diffusion_losses(v, xt, v_target, a, mu, target, valid, args.scale,
                invalid_weight=.1, high_noise_weight=1., high_noise_threshold=.2)
    diffusion_proof = _calibrated_update(corrector, optimizer, scaler, diffusion_forward, 'diffusion')
    _ema_update(corrector_ema, corrector)
    del diffusion_forward, corrector, optimizer, scaler, clean, noise, xt, v_target
    corrector = corrector_ema.float().eval().requires_grad_(False)
    del corrector_ema
    print('[server preflight] diffusion forward/backward/update passed; checking Gaussian sampling', flush=True)
    with torch.no_grad():
        # No target or evaluation-mask argument enters Gaussian sampling.
        delta = engine.sample_residual(corrector, condition, mu, alpha, args.scale,
            steps=2, seed=args.seed+987654, project_geometry=True)
    if tuple(delta.shape) != EXPECTED_TARGET_SHAPE:
        raise ValueError('Gaussian sampler output shape mismatch')
    _finite(delta, 'sampled residual')
    available = engine.available_from_condition(condition)
    if not torch.equal(available, valid):
        raise ValueError('Condition-only geometry does not match dataset support')
    if bool((delta[~available] != 0).any()):
        raise ValueError('Building-interior residual projection is not zero')
    peaks = _gpu_end(args.device)
    if engine.source_hashes() != sources or _sha256(__file__) != script_hash:
        raise ValueError('Source changed during server preflight')
    if engine.build_protocol(args, engine.audit(args.dataset_root)) != protocol:
        raise ValueError('Protocol/data changed during server preflight')
    proof = dict(state='passed', time_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=protocol_hash, protocol=protocol, source_sha256=sources,
        preflight_script_sha256=script_hash, weights_discarded=True, train_only=True,
        test_labels_used=False, validation_labels_used=False, formal_optimizer_steps=0,
        condition_shape=list(EXPECTED_CONDITION_SHAPE), target_shape=list(EXPECTED_TARGET_SHAPE),
        training_sample_ids=[str(b['id']) for b in examples],
        coarse=coarse_proof, diffusion=diffusion_proof, gaussian_sampling_steps=2,
        gaussian_sampling_target_access=False, geometry_support_verified=True,
        python=sys.version, python_executable=sys.executable, platform=platform.platform(),
        torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
        preflight_seed=int(args.seed), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        compatibility=compatibility, **gpu, **peaks)
    del coarse, corrector, condition, target, valid, mu, delta, alpha, examples, data
    del available, a, timestep
    _gpu_cleanup()
    # One explicit immutable JSON artifact, no directory creation/checkpoints.
    with proof_path.open('x', encoding='utf-8') as stream:
        json.dump(proof, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    print('[server preflight] PASSED; weights discarded, no formal training steps', flush=True)
    return proof
