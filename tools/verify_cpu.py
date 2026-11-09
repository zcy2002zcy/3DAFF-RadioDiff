"""CPU suite plus train/val tensor/protocol audit; never formal training."""
from __future__ import annotations
import argparse
import json
import os
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--record', required=True, type=Path)
    options = parser.parse_args()
    if options.record.exists():
        raise ValueError('CPU validation record already exists; do not overwrite evidence')
    import torch
    torch.set_num_threads(2)
    from compat_runtime import runtime_compatibility
    from scripts import run_prediction_transformer_v3 as runner
    from package_integrity import verify_package, file_hash
    verify_package()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_*.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {'state': 'failed', 'time_utc': datetime.now(timezone.utc).isoformat(),
              'runtime': runtime_compatibility(), 'tests_run': result.testsRun,
              'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
              'gpu_used': False, 'formal_training_started': False,
              'manifest_sha256': file_hash(ROOT / 'package_manifest.json')}
    if result.wasSuccessful() and not result.skipped:
        args = runner.parse_args(['--dataset-root', options.dataset_root])
        engine = runner.make_engine()
        report = engine.audit(args.dataset_root)
        config = engine.build_protocol(args, report)
        runner.validate_output_state(engine, args, config)
        tensors = {}
        for split in ('train', 'val'):
            dataset = engine.dataset(args, split, 1)
            one = dataset[0]
            c = one['condition'].unsqueeze(0)
            y = one['target'].unsqueeze(0)
            m = one['valid_mask'].unsqueeze(0).bool()
            engine.ensure_geometry(c, m)
            if not (torch.isfinite(c).all() and torch.isfinite(y).all()):
                raise ValueError('Nonfinite train/val tensor')
            tensors[split] = {'condition': list(c.shape), 'target': list(y.shape), 'valid_voxels': int(m.sum())}
        record.update(state='passed', data_audit=report, train_val_tensors=tensors,
                      test_labels_read=False, formal_protocol_sha256=engine.content_hash(config))
    options.record.parent.mkdir(parents=True, exist_ok=True)
    with options.record.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2), flush=True)
    if record['state'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
