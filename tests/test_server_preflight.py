"""CPU-only mocked server preflight; never probes or initializes CUDA."""
import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

import server_preflight as preflight


class Scaler:
    def scale(self, value):
        return value

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def get_scale(self):
        return 1.

    def is_enabled(self):
        return False


class ScaledScaler(Scaler):
    """CPU arithmetic double with real non-unit backward scaling."""
    def __init__(self, events, scale=32.):
        self.events = events
        self.factor = scale

    def scale(self, value):
        self.events.append('scale')
        return value*self.factor

    def unscale_(self, optimizer):
        self.events.append('unscale')
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.factor)

    def step(self, optimizer):
        self.events.append('step')
        optimizer.step()

    def update(self):
        self.events.append('update')

    def get_scale(self):
        return self.factor


class TinyPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.1))

    def forward(self, condition):
        return self.weight.expand(condition.shape[0], 1, 4, 8, 8)


class TinyCorrector(TinyPredictor):
    def forward(self, noisy, condition, timestep, coarse):
        return self.weight*noisy


def available(condition):
    heights = condition.new_tensor([1., 2., 3., 4.]).view(1, 1, 4, 1, 1)
    return condition[:, 1:2] < heights/20.


class ServerPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gpu_begin_passes_explicit_zero_through_real_torch_index_parser(self):
        from torch.cuda._utils import _get_device_index
        properties = SimpleNamespace(name='NVIDIA GeForce RTX 3090', total_memory=24*1024**3, major=8, minor=6)
        # This is the real argument parser from the failing remote stack, not a
        # permissive setter mock. It never queries CUDA for an explicit integer.
        with self.assertRaises(ValueError):
            _get_device_index(torch.device('cuda'))
        def strict_set_device(argument):
            self.assertIs(type(argument), int)
            self.assertEqual(_get_device_index(argument), 0)
        for device in (torch.device('cuda'), torch.device('cuda:0')):
            with self.subTest(device=str(device)), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(torch.cuda, 'set_device', side_effect=strict_set_device) as setter, \
                 patch.object(torch.cuda, 'current_device', return_value=0), \
                 patch.object(torch.cuda, 'get_device_properties', return_value=properties) as props, \
                 patch.object(torch.cuda, 'reset_peak_memory_stats') as reset:
                record = preflight._gpu_begin(device)
                setter.assert_called_once_with(0)
                props.assert_called_once_with(0)
                reset.assert_called_once_with(0)
                self.assertEqual(record['cuda_device_index'], 0)

    def test_gpu_helpers_reject_other_logical_devices_before_cuda(self):
        for device in (torch.device('cpu'), torch.device('cuda:1')):
            for function in (preflight._gpu_begin, preflight._gpu_end):
                with self.subTest(device=str(device), function=function.__name__), \
                     patch.object(torch.cuda, 'is_available', side_effect=AssertionError('must not probe')):
                    with self.assertRaisesRegex(ValueError, 'logical cuda:0'):
                        function(device)

    def test_gpu_begin_refuses_unavailable_or_wrong_current_device(self):
        with patch.object(torch.cuda, 'is_available', return_value=False), patch.object(torch.cuda, 'set_device') as setter:
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                preflight._gpu_begin(torch.device('cuda'))
            setter.assert_not_called()
        with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(torch.cuda, 'set_device'), \
             patch.object(torch.cuda, 'current_device', return_value=1), patch.object(torch.cuda, 'get_device_properties') as props:
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                preflight._gpu_begin(torch.device('cuda'))
            props.assert_not_called()

    def test_gpu_end_uses_explicit_logical_zero(self):
        with patch.object(torch.cuda, 'synchronize') as sync, \
             patch.object(torch.cuda, 'max_memory_allocated', return_value=123) as allocated, \
             patch.object(torch.cuda, 'max_memory_reserved', return_value=456) as reserved:
            record = preflight._gpu_end(torch.device('cuda'))
        sync.assert_called_once_with(0)
        allocated.assert_called_once_with(0)
        reserved.assert_called_once_with(0)
        self.assertEqual(record, {'peak_cuda_allocated_bytes': 123, 'peak_cuda_reserved_bytes': 456})

    def fixture(self, directory, fail_geometry=False, nan_loss=False):
        args = SimpleNamespace(device=torch.device('cuda'), batch_size=2, channels=32,
            smoke=False, train_samples=0, coarse_epochs=10, diffusion_epochs=10,
            output=directory/'formal', dataset_root=directory/'data', seed=42, scale=.5, lr=1e-4)
        self.dataset_calls, self.sample_calls, self.geometry_calls = [], [], []
        config = {'version': 'mock', 'source_sha256': {'model': 'fixed'}}
        def content_hash(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
        def dataset(unused_args, split, maximum):
            self.dataset_calls.append((split, maximum))
            self.assertEqual((split, maximum), ('train', 2))
            c = torch.zeros(9, 4, 8, 8)
            c[1, :, 0, 0] = .1
            mask = available(c[None])[0]
            return [dict(condition=c.clone(), target=torch.full((1, 4, 8, 8), .25),
                         valid_mask=mask.clone(), id='train_'+str(i)) for i in range(2)]
        def geometry(c, m):
            self.geometry_calls.append(True)
            if fail_geometry or not torch.equal(available(c), m):
                raise ValueError('geometry mismatch')
        def masked_mse(p, y, m):
            result = (p-y).square()[m].mean()
            return result*float('nan') if nan_loss else result
        def losses(v, xt, vt, a, mu, y, m, scale, **kwargs):
            result = (v-vt).square()[m].mean()
            return result, {'test_component': result}
        def sample(net, c, mu, alpha, scale, **kwargs):
            self.sample_calls.append(kwargs)
            self.assertFalse(net.training)
            self.assertFalse(any(p.requires_grad for p in net.parameters()))
            self.assertEqual(mu.dtype, torch.float32)
            return torch.zeros_like(mu)
        engine = SimpleNamespace(audit=lambda root: {'manifest': 'fixed'},
            build_protocol=lambda args, audit: config, content_hash=content_hash,
            source_hashes=lambda: config['source_sha256'].copy(), dataset=dataset,
            ensure_geometry=geometry, available_from_condition=available,
            schedule=lambda device: torch.linspace(.999, .001, 200),
            Predictor=lambda channels: TinyPredictor(), Corrector=lambda channels: TinyCorrector(),
            masked_mse=masked_mse, diffusion_losses=losses,
            draw_timesteps=lambda count, alpha, device, terminal_probability: torch.tensor([49, 199]),
            sample_residual=sample)
        runner = SimpleNamespace(make_engine=lambda: engine, reject_reparse=lambda path: None)
        return runner, args, config, content_hash

    @contextlib.contextmanager
    def no_cuda(self):
        # Only the wrapper calls that intentionally require a real server GPU
        # are mocked; autograd/AdamW/update/finiteness remain real CPU operations.
        with contextlib.ExitStack() as stack:
            # Newer AdamW checks CUDA graph status even for CPU-only parameters;
            # bypass that administrative check rather than touching the driver.
            stack.enter_context(patch.object(torch.optim.Optimizer,
                '_cuda_graph_capture_health_check', return_value=None, create=True))
            # torch.manual_seed seeds CPU normally while its CUDA branch is
            # intercepted; no CUDA lazy initialization is scheduled by tests.
            stack.enter_context(patch.object(torch.cuda, 'manual_seed_all'))
            for name in ('is_available', 'set_device', 'get_device_properties', 'current_device',
                         'reset_peak_memory_stats', 'synchronize', 'max_memory_allocated',
                         'max_memory_reserved', 'empty_cache', 'get_rng_state', 'init'):
                stack.enter_context(patch.object(torch.cuda, name,
                    side_effect=AssertionError('CUDA API forbidden in CPU tests: '+name)))
            stack.enter_context(patch.object(preflight, '_gpu_begin', return_value={
                'gpu_name': 'mock RTX3090', 'cuda_device_index': 0, 'gpu_total_memory_bytes': 24*1024**3}))
            stack.enter_context(patch.object(preflight, '_gpu_end', return_value={
                'peak_cuda_allocated_bytes': 1, 'peak_cuda_reserved_bytes': 1}))
            cleanup = stack.enter_context(patch.object(preflight, '_gpu_cleanup'))
            stack.enter_context(patch.object(preflight, '_move', side_effect=lambda x, device: x))
            stack.enter_context(patch.object(preflight, 'autocast', return_value=contextlib.nullcontext()))
            stack.enter_context(patch.object(preflight, 'make_grad_scaler', side_effect=lambda enabled: Scaler()))
            stack.enter_context(patch.object(preflight, 'runtime_compatibility', return_value={'mock': True}))
            stack.enter_context(patch.object(preflight, 'EXPECTED_CONDITION_SHAPE', (2, 9, 4, 8, 8)))
            stack.enter_context(patch.object(preflight, 'EXPECTED_TARGET_SHAPE', (2, 1, 4, 8, 8)))
            yield cleanup

    def test_production_shape_is_fullsize(self):
        self.assertEqual(preflight.EXPECTED_CONDITION_SHAPE, (2, 9, 4, 256, 256))
        with self.assertRaisesRegex(ValueError, 'no automatic downsizing'):
            preflight._validate_batch(torch.zeros(2, 9, 4, 8, 8),
                torch.zeros(2, 1, 4, 8, 8), torch.ones(2, 1, 4, 8, 8, dtype=torch.bool))

    def test_success_writes_only_explicit_bound_proof(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_cuda() as cleanup, \
                patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': 'GPU-unit-test-uuid'}):
            directory = Path(tmp)
            runner, args, config, content_hash = self.fixture(directory)
            path = directory/'proof.json'
            proof = preflight.run_preflight(runner, args, path)
            self.assertEqual(proof, json.loads(path.read_text(encoding='utf-8')))
            self.assertEqual(proof['state'], 'passed')
            self.assertEqual(proof['protocol_sha256'], content_hash(config))
            self.assertEqual(proof['source_sha256'], config['source_sha256'])
            self.assertEqual(proof['formal_optimizer_steps'], 0)
            self.assertEqual(proof['preflight_seed'], 42)
            self.assertEqual(proof['cuda_visible_devices'], 'GPU-unit-test-uuid')
            self.assertTrue(proof['weights_discarded'] and proof['train_only'])
            self.assertFalse(proof['test_labels_used'] or proof['validation_labels_used'])
            self.assertEqual(proof['coarse']['successful_optimizer_updates'], 1)
            self.assertEqual(proof['diffusion']['successful_optimizer_updates'], 1)
            for stage in ('coarse', 'diffusion'):
                self.assertEqual(proof[stage]['amp_backoffs'], 0)
                self.assertEqual(len(proof[stage]['amp_attempts']), 1)
                self.assertEqual(proof[stage]['amp_attempts'][0]['state'], 'updated')
            self.assertEqual(self.dataset_calls, [('train', 2)])
            self.assertEqual(self.sample_calls, [dict(steps=2, seed=42+987654, project_geometry=True)])
            self.assertEqual({x.name for x in directory.iterdir()}, {'proof.json'})
            cleanup.assert_called_once()
            with self.assertRaisesRegex(ValueError, 'NEW file'):
                preflight.run_preflight(runner, args, path)

    def test_scaled_gradients_unscale_before_clip_and_match_fp32_update(self):
        with self.no_cuda():
            reference = torch.nn.Linear(1, 1, bias=False)
            candidate = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                reference.weight.fill_(.25)
                candidate.weight.copy_(reference.weight)
            reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.1)
            optimizer = torch.optim.SGD(candidate.parameters(), lr=.1)
            x = torch.ones(2, 1)
            reference_loss = (reference(x)-1).square().mean()
            reference_loss.backward()
            reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.)
            reference_optimizer.step()

            events, backward_gradients = [], []
            def backward_hook(gradient):
                events.append('backward')
                backward_gradients.append(gradient.detach().clone())
                return gradient
            candidate.weight.register_hook(backward_hook)
            scaler = ScaledScaler(events, scale=32.)
            original_clip = torch.nn.utils.clip_grad_norm_
            def checked_clip(parameters, maximum, *args, **kwargs):
                events.append('clip')
                # Original gradient is -1.5; a missing unscale would give -48.
                self.assertAlmostEqual(float(candidate.weight.grad), -1.5, places=6)
                return original_clip(parameters, maximum, *args, **kwargs)
            loss = (candidate(x)-1).square().mean()
            with patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=checked_clip):
                result = preflight._update(loss, candidate, optimizer, scaler)
            self.assertEqual(events, ['scale', 'backward', 'unscale', 'clip', 'step', 'update'])
            self.assertAlmostEqual(float(backward_gradients[0]), -48., places=5)
            self.assertAlmostEqual(result['gradient_norm'], float(reference_norm), places=6)
            self.assertTrue(torch.equal(candidate.weight, reference.weight))
            self.assertEqual(result['successful_optimizer_updates'], 1)

    def test_geometry_failure_has_no_passed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_cuda():
            directory = Path(tmp)
            runner, args, _, _ = self.fixture(directory, fail_geometry=True)
            with self.assertRaisesRegex(ValueError, 'geometry mismatch'):
                preflight.run_preflight(runner, args, directory/'proof.json')
            self.assertEqual(list(directory.iterdir()), [])

    def test_nonfinite_loss_has_no_passed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_cuda():
            directory = Path(tmp)
            runner, args, _, _ = self.fixture(directory, nan_loss=True)
            with self.assertRaisesRegex(FloatingPointError, 'Nonfinite preflight loss'):
                preflight.run_preflight(runner, args, directory/'proof.json')
            self.assertEqual(list(directory.iterdir()), [])

    def test_reduced_config_refused_before_engine_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_cuda():
            directory = Path(tmp)
            runner, args, _, _ = self.fixture(directory)
            args.batch_size = 1
            with self.assertRaisesRegex(ValueError, 'batch2'):
                preflight.run_preflight(runner, args, directory/'proof.json')
            self.assertEqual(self.dataset_calls, [])

    def test_failed_diffusion_gradient_never_updates_ema_or_writes_proof(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_cuda(), \
                patch.object(preflight, '_ema_update', wraps=preflight._ema_update) as ema:
            directory = Path(tmp)
            runner, args, _, _ = self.fixture(directory)
            engine = runner.make_engine()
            def bad_corrector(channels):
                net = TinyCorrector()
                net.weight.register_hook(lambda grad: torch.full_like(grad, float('inf')))
                return net
            engine.Corrector = bad_corrector
            with self.assertRaisesRegex(FloatingPointError, 'gradient without enabled AMP'):
                preflight.run_preflight(runner, args, directory/'proof.json')
            self.assertEqual(ema.call_count, 1)  # Successful coarse update only.
            self.assertEqual(self.sample_calls, [])
            self.assertEqual(list(directory.iterdir()), [])

    def test_nonfinite_diffusion_component_fails_before_backward(self):
        with tempfile.TemporaryDirectory() as tmp, self.no_cuda(), \
                patch.object(preflight, '_ema_update', wraps=preflight._ema_update) as ema:
            directory = Path(tmp)
            runner, args, _, _ = self.fixture(directory)
            engine = runner.make_engine()
            original = engine.diffusion_losses
            def bad_component(*values, **kwargs):
                loss, components = original(*values, **kwargs)
                components['invalid_component'] = loss.detach()*float('nan')
                return loss, components
            engine.diffusion_losses = bad_component
            with self.assertRaisesRegex(FloatingPointError, 'diffusion component invalid_component'):
                preflight.run_preflight(runner, args, directory/'proof.json')
            self.assertEqual(ema.call_count, 1)
            self.assertEqual(self.sample_calls, [])
            self.assertEqual(list(directory.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
