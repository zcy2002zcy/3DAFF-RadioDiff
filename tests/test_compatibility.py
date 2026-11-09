"""CPU-only compatibility checks, including simulated legacy API signatures.

Legacy mocks establish dispatch/gradient contracts, not an actual PyTorch 1.12
GPU certification. Run this file on the destination environment as well.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as real_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compat_runtime as compat

torch.set_num_threads(2)


class CompatibilityTests(unittest.TestCase):
    def test_old_loader_does_not_receive_weights_only_through_kwargs(self):
        calls = []

        def old_load(path, map_location='cpu', **kwargs):
            calls.append((path, map_location, kwargs))
            return {'loaded': True}

        with patch.object(compat.torch, 'load', old_load):
            self.assertEqual(compat.load_checkpoint('trusted.pt'), {'loaded': True})
        self.assertEqual(calls, [('trusted.pt', 'cpu', {})])

    def test_modern_loader_explicit_full_checkpoint(self):
        calls = []

        def modern_load(path, map_location=None, *, weights_only=None):
            calls.append((path, map_location, weights_only))
            return 7

        with patch.object(compat.torch, 'load', modern_load):
            self.assertEqual(compat.load_checkpoint('trusted.pt', 'cpu'), 7)
        self.assertEqual(calls, [('trusted.pt', 'cpu', False)])

    def test_loader_failure_is_not_retried(self):
        calls = []

        def broken_load(path, map_location=None, *, weights_only=None):
            calls.append(path)
            raise RuntimeError('actual load failure')

        with patch.object(compat.torch, 'load', broken_load):
            with self.assertRaisesRegex(RuntimeError, 'actual load failure'):
                compat.load_checkpoint('trusted.pt')
        self.assertEqual(calls, ['trusted.pt'])

    def test_actual_cpu_checkpoint_round_trip(self):
        data = io.BytesIO()
        torch.save({'weights': torch.arange(4), 'epoch': 3, 'rng': torch.get_rng_state()}, data)
        data.seek(0)
        restored = compat.load_checkpoint(data)
        self.assertEqual(restored['epoch'], 3)
        self.assertTrue(torch.equal(restored['weights'], torch.arange(4)))
        self.assertTrue(torch.equal(restored['rng'], torch.get_rng_state()))

    def test_modern_grad_scaler_selected(self):
        def modern_scaler(device='cuda', enabled=True):
            return ('modern', device, enabled)

        fake = SimpleNamespace(amp=SimpleNamespace(GradScaler=modern_scaler))
        with patch.object(compat, 'torch', fake):
            self.assertEqual(compat.make_grad_scaler(False), ('modern', 'cuda', False))

    def test_legacy_grad_scaler_selected(self):
        def old_scaler(enabled=True):
            return ('legacy', enabled)

        fake = SimpleNamespace(amp=SimpleNamespace(),
                               cuda=SimpleNamespace(amp=SimpleNamespace(GradScaler=old_scaler)))
        with patch.object(compat, 'torch', fake):
            self.assertEqual(compat.make_grad_scaler(False), ('legacy', False))

    def test_actual_disabled_scaler_requires_no_cuda(self):
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('no CUDA query')):
            scaler = compat.make_grad_scaler(False)
            self.assertFalse(scaler.is_enabled())
            value = torch.tensor(2., requires_grad=True)
            scaler.scale(value * value).backward()
            self.assertEqual(value.grad.item(), 4.)

    def test_autocast_dispatch_and_cpu_policy(self):
        def modern_autocast(device_type, dtype=None, enabled=True):
            return (device_type, dtype, enabled)

        fake = SimpleNamespace(autocast=modern_autocast, float16='half')
        with patch.object(compat, 'torch', fake):
            self.assertEqual(compat.autocast('cuda'), ('cuda', 'half', True))
            with compat.autocast('cpu', enabled=False):
                pass
            with self.assertRaisesRegex(ValueError, 'CPU must remain FP32'):
                compat.autocast('cpu')

    def test_old_autocast_dispatch(self):
        fake = SimpleNamespace(float16='half', cuda=SimpleNamespace(amp=SimpleNamespace(
            autocast=lambda dtype, enabled: ('legacy', dtype, enabled))))
        with patch.object(compat, 'torch', fake):
            self.assertEqual(compat.autocast('cuda'), ('legacy', 'half', True))

    def test_modern_checkpoint_uses_non_reentrant_and_preserves_rng(self):
        calls = []

        def modern(function, *args, use_reentrant=None, **kwargs):
            calls.append((use_reentrant, kwargs, len(args)))
            return function(*args)

        with patch.object(compat, '_torch_checkpoint', modern):
            self.assertEqual(compat.activation_checkpoint(lambda x: x + 1, torch.tensor(2.)).item(), 3.)
        self.assertEqual(calls, [(False, {'preserve_rng_state': True}, 1)])

    def _check_legacy_gradients_and_rng(self, input_requires_grad):
        reference = nn.Sequential(nn.Linear(4, 6), nn.Dropout(.25), nn.GELU(), nn.Linear(6, 2))
        recomputed = copy.deepcopy(reference)
        source = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 7
        ordinary_input = source.clone().requires_grad_(input_requires_grad)
        checkpoint_input = source.clone().requires_grad_(input_requires_grad)
        torch.manual_seed(9812)
        expected = reference(ordinary_input)
        expected.square().sum().backward()
        expected_next = torch.rand(5)
        calls = []

        # No explicit use_reentrant in the signature: emulate the old API,
        # executing its real reentrant implementation in the installed runtime.
        def legacy(function, *args, **kwargs):
            calls.append((len(args), [value.requires_grad for value in args], kwargs))
            self.assertNotIn('use_reentrant', kwargs)
            return real_checkpoint(function, *args, use_reentrant=True, **kwargs)

        torch.manual_seed(9812)
        with patch.object(compat, '_torch_checkpoint', legacy):
            actual = compat.activation_checkpoint(recomputed, checkpoint_input)
            actual.square().sum().backward()
        actual_next = torch.rand(5)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(expected_next, actual_next))
        for left, right in zip(reference.parameters(), recomputed.parameters()):
            self.assertIsNotNone(right.grad)
            self.assertTrue(torch.equal(left.grad, right.grad))
            self.assertTrue(torch.isfinite(right.grad).all())
        if input_requires_grad:
            self.assertTrue(torch.equal(ordinary_input.grad, checkpoint_input.grad))
            self.assertEqual(calls[0][0], 1)
        else:
            self.assertIsNone(checkpoint_input.grad)
            self.assertEqual(calls[0][0], 2)
            self.assertEqual(calls[0][1], [False, True])
        self.assertEqual(calls[0][2], {'preserve_rng_state': True})

    def test_legacy_checkpoint_preserves_input_parameter_gradients_rng(self):
        self._check_legacy_gradients_and_rng(True)

    def test_legacy_checkpoint_retains_parameter_gradients_without_grad_input(self):
        self._check_legacy_gradients_and_rng(False)

    def test_checkpoint_no_grad_and_input_contract(self):
        with torch.no_grad(), patch.object(compat, '_torch_checkpoint', side_effect=AssertionError('unused')):
            self.assertEqual(compat.activation_checkpoint(lambda x: x + 2, torch.tensor(3.)).item(), 5.)
        with self.assertRaises(TypeError):
            compat.activation_checkpoint(lambda x: x, [torch.tensor(1.)])

    def test_actual_dataset_import_and_transformer_cpu_backward(self):
        from datasets.pl3d_manifest_dataset import PL3DManifestDataset
        from models.pl3d_residual_prediction_transformer import PerHeightSpatialTransformer
        self.assertTrue(issubclass(PL3DManifestDataset, torch.utils.data.Dataset))
        ordinary = PerHeightSpatialTransformer(activation_checkpoint=False).train()
        recomputed = copy.deepcopy(ordinary)
        recomputed.activation_checkpoint = True
        source = torch.randn(1, 128, 4, 2, 2)
        ordinary_input = source.clone().requires_grad_(True)
        checkpoint_input = source.clone().requires_grad_(True)
        expected, actual = ordinary(ordinary_input), recomputed(checkpoint_input)
        expected.square().mean().backward()
        actual.square().mean().backward()
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(ordinary_input.grad, checkpoint_input.grad))
        for left, right in zip(ordinary.parameters(), recomputed.parameters()):
            self.assertIsNotNone(right.grad)
            self.assertTrue(torch.equal(left.grad, right.grad))

    def test_runtime_provenance_no_cuda_queries(self):
        with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('no CUDA query')):
            result = compat.validate_runtime()
        self.assertEqual(result['cpu_precision'], 'FP32; CPU AMP disabled')
        self.assertEqual(result['torch'], str(torch.__version__))
        self.assertIn('activation_checkpoint', result)

    def test_unsupported_runtime_rejected(self):
        with patch.object(torch, '__version__', '1.11.0'):
            with self.assertRaisesRegex(RuntimeError, 'PyTorch >= 1.12'):
                compat.validate_runtime()


if __name__ == '__main__':
    unittest.main()
