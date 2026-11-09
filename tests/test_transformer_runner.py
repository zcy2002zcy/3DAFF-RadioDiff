"""CPU/read-only integration tests; never run a real dataset/model forward."""
from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

sys.dont_write_bytecode = True
BUNDLE = Path(__file__).resolve().parents[1]
RUNNER = BUNDLE / 'scripts/run_prediction_transformer_v3.py'
spec = importlib.util.spec_from_file_location('isolated_transformer_runner_test', RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class TransformerRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = runner.make_engine()
        cls.reference = cls.engine.read_json(runner.REFERENCE_PROTOCOL)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='v3_runner_test_', dir=BUNDLE)
        self.root = Path(self.temp.name)
        self.patch_root = mock.patch.object(runner, 'ROOT', self.root)

    def tearDown(self):
        self.temp.cleanup()

    def configuration(self, args=None):
        return self.engine.build_protocol(args or runner.parse_args([]), self.reference['audit'])

    def test_default_check_is_cpu_without_cuda_probe(self):
        with mock.patch.object(runner.torch.cuda, 'is_available', side_effect=AssertionError('CUDA probe')):
            args = runner.parse_args([])
        self.assertEqual(args.action, 'check')
        self.assertEqual(args.device.type, 'cpu')
        self.assertEqual(args.requested_device, 'cuda')
        self.assertFalse(args.smoke)
        self.assertEqual((args.channels, args.coarse_epochs, args.diffusion_epochs), (32, 10, 10))

    def test_private_binding_does_not_modify_canonical_v2(self):
        canonical = importlib.import_module('scripts.run_prediction_10plus10_v2')
        original = (canonical.Predictor, canonical.Corrector, canonical.VERSION, canonical.build_protocol)
        private = runner.make_engine()
        self.assertIsNot(private, canonical)
        self.assertEqual(original, (canonical.Predictor, canonical.Corrector, canonical.VERSION, canonical.build_protocol))
        self.assertIsNot(private.Predictor, canonical.Predictor)
        self.assertIsNot(private.Corrector, canonical.Corrector)
        for function in ('train_stage', 'load_finished_models', 'calibrate', 'evaluate', 'validate', 'check_ck'):
            self.assertIs(inspect.unwrap(getattr(private, function)).__globals__, private.__dict__)
        self.assertIs(private.train_stage.__globals__['Predictor'], private.Predictor)
        self.assertIs(private.load_finished_models.__globals__['Corrector'], private.Corrector)

    def test_protocol_keeps_task_budget_metric_and_adds_provenance(self):
        config = self.configuration()
        self.assertEqual(config['version'], runner.VERSION)
        for key in ('audit', 'condition_profile', 'sample_rate', 'include_idw_prior', 'seed',
                    'batch_size', 'base_channels', 'lr', 'scale', 'diffusion_steps', 'schedule',
                    'diffusion_loss', 'training_precision', 'checkpoint_selection', 'inference_calibration',
                    'coarse_epochs', 'diffusion_epochs', 'train_samples', 'val_samples', 'test'):
            self.assertEqual(config[key], self.reference[key], key)
        self.assertEqual(config['sample_rate'], 0)
        self.assertEqual(config['test']['psnr_data_range'], 2)
        self.assertEqual(config['architecture_compute_disclosure']['added_parameters_both_stages'], 529920)
        self.assertTrue(config['architecture_compute_disclosure']['same_epoch_budget_not_same_compute'])
        for key in ('reused_v2_runner', 'runner', 'transformer_model', 'original_transformer',
                    'original_transformer_configuration', 'reference_v2_protocol'):
            self.assertEqual(len(config['source_sha256'][key]), 64)
        self.assertIn('Not part of all', config['extra_ddim200_evaluation'])

    def test_original_yaml_must_match_transformer_spec(self):
        wrong = copy.deepcopy(self.engine.transformer_architecture.ARCHITECTURE_SPEC)
        wrong['transformer']['heads'] = 8
        with self.assertRaisesRegex(ValueError, 'Transformer setting'):
            runner.validate_original_transformer_config(wrong)

    def test_check_performs_no_cuda_dataset_forward_or_output_write(self):
        config = self.configuration()
        with self.patch_root:
            output = self.root / runner.OUTPUT_NAME
            with mock.patch.object(runner, 'make_engine', return_value=self.engine), \
                 mock.patch.object(self.engine, 'audit', return_value=self.reference['audit']), \
                 mock.patch.object(self.engine, 'build_protocol', return_value=config), \
                 mock.patch.object(self.engine, 'dataset', side_effect=AssertionError('Dataset target access')), \
                 mock.patch.object(self.engine, 'Predictor', side_effect=AssertionError('Model construction')), \
                 mock.patch.object(runner.torch.cuda, 'is_available', side_effect=AssertionError('CUDA probe')), \
                 mock.patch.object(runner.torch.cuda, 'manual_seed_all', side_effect=AssertionError('CUDA seed')), \
                 redirect_stdout(io.StringIO()):
                runner.main([])
            self.assertFalse(output.exists())
            self.assertEqual(list(self.root.iterdir()), [])

    def test_old_path_rejected_before_engine_or_any_write(self):
        old = self.root / 'outputs_prediction_main_10plus10_v2'
        old.mkdir()
        marker = old / 'untouched.txt'
        marker.write_text('original', encoding='utf-8')
        with self.patch_root, mock.patch.object(runner, 'make_engine', side_effect=AssertionError('Too late')):
            with self.assertRaisesRegex(ValueError, 'dedicated output'):
                runner.main(['--action', 'train', '--output', str(old)])
        self.assertEqual(marker.read_text(), 'original')
        self.assertEqual([p.name for p in old.iterdir()], ['untouched.txt'])

    def test_noncheck_cpu_and_hyperparameter_changes_are_rejected(self):
        for values in (['--action', 'train', '--device', 'cpu'], ['--batch-size', '1'], ['--workers', '2'], ['--smoke']):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(values)

    def test_synthetic_cpu_two_stage_checkpoint_roundtrip_uses_new_models(self):
        # This deliberately calls helpers directly with a TEST-ONLY protocol;
        # the formal CLI still forbids CPU training and altered budgets.
        engine = runner.make_engine()
        torch = runner.torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            args = SimpleNamespace(output=self.root / 'synthetic_not_formal', channels=32,
                                   device=torch.device('cpu'), lr=1e-4, coarse_epochs=1,
                                   diffusion_epochs=1, resume=False, scale=.5, seed=42, log_every=1)
            args.output.mkdir()
            config = {'test_only': True, 'version': runner.VERSION,
                      'architecture_spec': engine.transformer_architecture.ARCHITECTURE_SPEC}
            c = torch.zeros(1, 9, 4, 16, 16)
            c[:, 2, :, 8, 8] = 1
            y = torch.linspace(-.8, .8, 1024).reshape(1, 1, 4, 16, 16)
            batch = {'condition': c, 'target': y, 'valid_mask': torch.ones_like(y, dtype=torch.bool)}
            alpha = engine.schedule(args.device)
            with redirect_stdout(io.StringIO()):
                coarse = engine.train_stage(args, 'coarse', None, [batch], [batch], alpha, config)
                corrector = engine.train_stage(args, 'diffusion', coarse, [batch], [batch], alpha, config)
                coarse_loaded, corrector_loaded = engine.load_finished_models(args, config)
            self.assertIsInstance(coarse_loaded, engine.Predictor)
            self.assertIsInstance(corrector_loaded, engine.Corrector)
            self.assertTrue(any('bottleneck.1.blocks.0.attn' in name for name in coarse_loaded.state_dict()))
            for before, after in ((coarse, coarse_loaded), (corrector, corrector_loaded)):
                for key, tensor in before.state_dict().items():
                    self.assertTrue(torch.equal(tensor, after.state_dict()[key]), key)
            for stage in ('coarse', 'diffusion'):
                ck = engine.load(args.output / stage / 'last.pt')
                self.assertEqual(ck['version'], runner.VERSION)
                self.assertEqual(ck['epoch'], 1)
                self.assertIn('optimizer', ck)
                self.assertIn('ema', ck)
                del ck
            # A v2 checkpoint label is rejected before model-state construction.
            ck = engine.load(args.output / 'diffusion/best.pt')
            ck['version'] = self.reference['version']
            with self.assertRaisesRegex(ValueError, 'protocol/source mismatch'):
                engine.check_ck(ck, config, 'diffusion', engine.sha256(args.output / 'coarse/best.pt'))
        finally:
            torch.set_num_threads(previous_threads)

    def test_resume_requires_new_exact_protocol_and_no_unknown_bootstrap(self):
        config = self.configuration()
        with self.patch_root:
            args = runner.parse_args(['--action', 'train'])
            args.output.mkdir()
            unknown = args.output / 'last.pt'
            unknown.write_bytes(b'old checkpoint')
            with self.assertRaisesRegex(ValueError, 'unexpected content'):
                runner.validate_output_state(self.engine, args, config)
            unknown.unlink()
            self.engine.dump(args.output / 'protocol.json', self.reference)
            with self.assertRaisesRegex(ValueError, 'exact v3'):
                runner.validate_output_state(self.engine, args, config)
            self.engine.dump(args.output / 'protocol.json', config)
            with self.assertRaisesRegex(ValueError, 'explicit --resume'):
                runner.validate_output_state(self.engine, args, config)
            args.resume = True
            runner.validate_output_state(self.engine, args, config)
            changed = copy.deepcopy(config)
            changed['architecture_spec']['transformer']['depth'] = 1
            with self.assertRaisesRegex(ValueError, 'exact v3'):
                runner.validate_output_state(self.engine, args, changed)

    def test_old_checkpoint_version_is_rejected_without_model_loading(self):
        config = self.configuration()
        with self.patch_root:
            args = runner.parse_args(['--resume'])
            (args.output / 'coarse').mkdir(parents=True)
            self.engine.dump(args.output / 'protocol.json', config)
            old = {'version': self.reference['version'], 'stage': 'coarse', 'protocol': config,
                   'coarse_sha256': None, 'epoch': 10}
            self.engine.save(args.output / 'coarse/last.pt', old)
            with self.assertRaisesRegex(ValueError, 'protocol/source mismatch'):
                runner.validate_output_state(self.engine, args, config)

    def test_output_resolve_cannot_follow_junction_to_old_directory(self):
        with self.patch_root:
            output = self.root / runner.OUTPUT_NAME
            old = self.root / 'outputs_prediction_main_10plus10_v2'
            real_resolve = Path.resolve

            def redirected(path, *a, **kw):
                return old if path == output else real_resolve(path, *a, **kw)

            with mock.patch.object(Path, 'resolve', redirected), self.assertRaisesRegex(ValueError, 'dedicated output'):
                runner.validate_output_path(output)
        attributes = type('WinStat', (), {'st_file_attributes': getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)})()
        with mock.patch.object(Path, 'is_symlink', return_value=False), \
             mock.patch.object(Path, 'lstat', return_value=attributes), self.assertRaisesRegex(ValueError, 'Reparse'):
            runner.reject_reparse(self.root / 'junction')

    def test_cross_process_lock_blocks_duplicate_then_releases(self):
        output = self.root / runner.OUTPUT_NAME
        child = (
            "import importlib.util,pathlib,sys;"
            f"s=importlib.util.spec_from_file_location('lock_test',{str(RUNNER)!r});"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            f"p=pathlib.Path({str(output)!r});"
            "\ntry:\n with m.exclusive_run_lock(p): pass\nexcept RuntimeError:\n sys.exit(23)\n"
        )
        with runner.exclusive_run_lock(output):
            process = subprocess.run([sys.executable, '-B', '-c', child], capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 23, process.stdout + process.stderr)
        process = subprocess.run([sys.executable, '-B', '-c', child], capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
