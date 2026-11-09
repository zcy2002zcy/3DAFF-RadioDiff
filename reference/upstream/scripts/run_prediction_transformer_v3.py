"""Explicit, isolated AFF+Transformer 10+10 experiment; default action is check.

This runner privately loads the unchanged v2 implementation and binds new model
factories only in that private module. Training, checkpoint selection, residual
math, validation calibration and four-metric evaluation otherwise remain v2.
No legacy script/module is modified, and no training starts without --action.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import os
import stat
import sys
from contextlib import contextmanager
from pathlib import Path

# Even read-only checks must not emit bytecode into the old project.
sys.dont_write_bytecode = True
import torch
import yaml

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE if (HERE / 'datasets/pl3d_manifest_dataset.py').is_file() else Path(
    'D:/zdata/3DAFF-RadioDiff/3d-radiodiff')
VERSION = 'q_building_prediction_transformer_10plus10_v3'
OUTPUT_NAME = 'outputs_prediction_main_transformer_10plus10_v3'
LOCK_NAME = 'transformer_v3.run.lock'
PINNED_ORIGINAL_SOURCES = {
    'models/transformer.py': '46be825a0d24e2cb18582e21da57edcc9bafb17d54811614b0873abf1b8c6414',
    'configs/base_pl_aff_random10_fullnoise.yaml': '6e0416a4634e172803288aa59551bb5ecb0e3cc9164dd909edb12b818128f698',
}
REFERENCE_PROTOCOL = ROOT / 'outputs_prediction_main_10plus10_v2/protocol.json'
sys.path.insert(0, str(ROOT))


def private_module(name, path):
    """Create an unregistered module object, not a replacement in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_engine():
    engine = private_module('_aff_transformer_v3_private_v2', ROOT / 'scripts/run_prediction_10plus10_v2.py')
    verify_original_sources(engine)
    model_path = HERE / 'models/pl3d_residual_prediction_transformer.py'
    if not model_path.is_file():
        raise FileNotFoundError(f'New Transformer model is missing: {model_path}')
    architecture = private_module('_aff_transformer_v3_private_architecture', model_path)
    original_builder = engine.build_protocol
    original_sources = engine.source_hashes

    def sources():
        result = original_sources()
        result['reused_v2_runner'] = result.pop('runner')
        result.update({
            'runner': engine.sha256(__file__),
            'transformer_model': engine.sha256(model_path),
            'original_transformer': engine.sha256(ROOT / 'models/transformer.py'),
            'original_transformer_configuration': engine.sha256(ROOT / 'configs/base_pl_aff_random10_fullnoise.yaml'),
            'reference_v2_protocol': engine.sha256(REFERENCE_PROTOCOL),
        })
        return result

    def protocol(args, report):
        verify_original_sources(engine)
        validate_original_transformer_config(architecture.ARCHITECTURE_SPEC)
        reference = engine.read_json(REFERENCE_PROTOCOL)
        if reference.get('version') != 'q_building_prediction_10plus10_geometry_highnoise_v2':
            raise ValueError('Unexpected reference experiment version')
        # The entire legacy dependency graph must still match the experiment
        # being extended. Do not silently borrow a later altered metric/data code.
        if original_sources() != reference.get('source_sha256'):
            raise ValueError('Current v2 sources differ from the recorded reference experiment')
        planned = copy.copy(args)
        if args.action == 'check':
            # This is metadata only: torch.device does not initialize CUDA.
            planned.device = torch.device(args.requested_device)
        config = original_builder(planned, report)
        excluded = {'version', 'source_sha256', 'architecture'}
        for name, expected in reference.items():
            if name not in excluded and config.get(name) != expected:
                raise ValueError(f'V3 would alter a non-architecture v2 setting: {name}')
        config['architecture'] = 'Existing full-3D AFF Predictor/Corrector + original-config per-height spatial Transformer'
        config['architecture_spec'] = copy.deepcopy(architecture.ARCHITECTURE_SPEC)
        config['stage_parameters_shared'] = False
        config['reference_experiment'] = {
            'version': reference['version'],
            'protocol_sha256': engine.sha256(REFERENCE_PROTOCOL),
            'comparison': 'architecture addition; data/task/loss/budget/calibration/metrics retained',
        }
        config['historical_yaml_policy'] = (
            'Only Transformer dimensions/depth/heads/FFN are reused. Historical base128, '
            'random10% measurements, dataset, optimizer and linear schedule are NOT imported.')
        config['extra_ddim200_evaluation'] = (
            'Not part of all. The legacy standalone DDIM200 evaluator is tied to old models; '
            'do not pass v3 checkpoints to it.')
        config['checkpoint_compatibility'] = 'v3 only; never resume or load v2/no-Transformer weights'
        config['architecture_compute_disclosure'] = {
            'added_parameters_both_stages': architecture.ARCHITECTURE_SPEC['transformer']['parameters_both_stages'],
            'same_epoch_budget_not_same_compute': True,
            'additional_compute': 'per-height full spatial attention and training activation recomputation',
        }
        return config

    # All old function globals belong to this *private* module object. Both
    # train_stage and load_finished_models resolve these new factories here.
    engine.Predictor = architecture.Predictor
    engine.Corrector = architecture.Corrector
    engine.VERSION = VERSION
    engine.source_hashes = sources
    engine.build_protocol = protocol
    engine.validate_bootstrap_output = validate_fresh_contents
    engine.transformer_architecture = architecture
    return engine


def verify_original_sources(engine):
    for relative, expected in PINNED_ORIGINAL_SOURCES.items():
        if engine.sha256(ROOT / relative) != expected:
            raise ValueError(f'Pinned original Transformer source changed: {relative}')


def validate_original_transformer_config(specification):
    from models.transformer import TransformerBlock
    path = ROOT / 'configs/base_pl_aff_random10_fullnoise.yaml'
    loaded = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    settings = loaded.get('model', {})
    actual = specification['transformer']
    for key, field in (('transformer_dim', 'dim'), ('transformer_heads', 'heads'),
                       ('transformer_mlp_dim', 'ffn_dim'), ('transformer_depth', 'depth')):
        if settings.get(key) != actual[field]:
            raise ValueError(f'Original YAML Transformer setting {key} differs from the v3 architecture')
    parameters = inspect.signature(TransformerBlock.__init__).parameters
    for field in ('dropout', 'activation'):
        expected = settings.get('transformer_' + field, parameters[field].default)
        if expected != actual[field]:
            raise ValueError(f'Original Transformer {field} differs from the v3 architecture')


def reject_reparse(path):
    if path.is_symlink():
        raise ValueError(f'Links are not accepted in v3 output: {path}')
    try:
        attributes = getattr(path.lstat(), 'st_file_attributes', 0)
    except FileNotFoundError:
        return
    if attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
        raise ValueError(f'Reparse/junction paths are not accepted in v3 output: {path}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--action', choices=('check', 'train', 'calibrate', 'eval', 'all'), default='check')
    p.add_argument('--dataset-root', type=Path, default=Path('D:/datas/pl_3d_h1_h4_npz_v1'))
    p.add_argument('--output', type=Path, default=ROOT / OUTPUT_NAME)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--log-every', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    if args.batch_size != 2 or args.workers != 4:
        p.error('Architecture-only v3 keeps batch-size=2 and workers=4 from the current main experiment')
    if args.log_every < 1:
        p.error('log-every must be positive')
    if args.action != 'check' and args.device != 'cuda':
        p.error('Formal v3 execution keeps CUDA AMP; CPU is only allowed for the read-only check')
    # A check executes on CPU but audits the unchanged CUDA training protocol.
    args.requested_device = 'cuda'
    args.device = torch.device('cpu' if args.action == 'check' else args.device)
    args.dataset_root = args.dataset_root.resolve()
    args.output = args.output.absolute()
    args.coarse_epochs = args.diffusion_epochs = 10
    args.channels = 32
    args.lr = 1e-4
    args.scale = .5
    args.seed = 42
    args.val_samples = 32
    args.max_samples = 200
    args.train_samples = 0
    args.repeats = 3
    args.smoke = False
    # No inherited channels4 smoke mode: Transformer dim128 must stay unchanged.
    return args


def validate_output_path(output):
    expected = ROOT.resolve() / OUTPUT_NAME
    if output.resolve() != expected:
        raise ValueError(f'V3 may write only its new dedicated output: {expected}')
    reject_reparse(output)
    if output.exists() and not output.is_dir():
        raise ValueError('Output must be a real dedicated directory, never a link/file')
    if output.resolve().parent != ROOT.resolve():
        raise ValueError('Output must be directly inside the project root')


def validate_fresh_contents(output):
    if not output.exists():
        return
    for entry in output.iterdir():
        reject_reparse(entry)
        if entry.name != LOCK_NAME or not entry.is_file() or entry.is_symlink():
            raise ValueError(f'Fresh v3 output contains unexpected content: {entry.name}')
        if entry.resolve().parent != output.resolve():
            raise ValueError('Lock file resolves outside the dedicated output directory')


def validate_output_state(engine, args, config):
    """Read-only preflight; invoked again under the lock to close launch races."""
    validate_output_path(args.output)
    protocol_path = args.output / 'protocol.json'
    reject_reparse(protocol_path)
    if not protocol_path.is_file():
        validate_fresh_contents(args.output)
        if args.resume:
            raise ValueError('--resume requires an existing v3 protocol, not a fresh/old run')
        if args.action in ('calibrate', 'eval'):
            raise ValueError('Calibration/evaluation requires completed v3 training')
        return
    existing = engine.read_json(protocol_path)
    if existing.get('version') != VERSION or existing != config:
        raise ValueError('Existing output is not this exact v3 source/architecture/protocol')
    for directory, directories, files in os.walk(args.output, followlinks=False):
        for name in directories + files:
            reject_reparse(Path(directory) / name)
    if args.action in ('train', 'all') and not args.resume:
        raise ValueError('Existing v3 training requires explicit --resume')
    coarse_best = args.output / 'coarse/best.pt'
    for stage in ('coarse', 'diffusion'):
        reject_reparse(args.output / stage)
        for name in ('last.pt', 'best.pt'):
            path = args.output / stage / name
            reject_reparse(path)
            if not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError('Checkpoint links are not accepted')
            if stage == 'diffusion' and not coarse_best.is_file():
                raise ValueError('Diffusion checkpoint lacks its frozen coarse checkpoint')
            ck = engine.load(path, 'cpu')
            engine.check_ck(ck, config, stage, engine.sha256(coarse_best) if stage == 'diffusion' else None)
            del ck


@contextmanager
def exclusive_run_lock(output):
    """OS-backed nonblocking process lock; persistent byte is not a launch claim."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / LOCK_NAME
    reject_reparse(path)
    handle = path.open('a+b')
    locked = False
    try:
        if path.stat().st_size == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('Another v3 action owns the output lock; no duplicate run started') from exc
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def main(argv=None):
    args = parse_args(argv)
    # Reject an old/wrong output before even reading audit files or loading models.
    validate_output_path(args.output)
    engine = make_engine()
    report = engine.audit(args.dataset_root)  # CSV/metadata only; no target arrays.
    config = engine.build_protocol(args, report)
    validate_output_state(engine, args, config)
    if args.action == 'check':
        print(json.dumps({'version': VERSION, 'action': 'read_only_check',
                          'execution_device': 'cpu', 'planned_training_device': args.requested_device,
                          'output': str(args.output), 'protocol': config}, indent=2), flush=True)
        print('TRANSFORMER_V3_CHECK_PASSED: no CUDA probe, no model forward, no output created', flush=True)
        return
    with exclusive_run_lock(args.output):
        validate_output_state(engine, args, config)
        if args.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no training started')
        # Only this private engine resolves parse_args/factories differently.
        # The persisted v3 protocol additionally records every reused source.
        engine.parse_args = lambda: args
        engine.main()


if __name__ == '__main__':
    main()
