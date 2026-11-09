"""CPU-only numerical tests for disposable preflight AMP calibration.

No real dataset, model weights, proof file or CUDA API is used. A real CPU
GradScaler is exercised when that public API exists; older PyTorch versions
exercise the same arithmetic with an explicit dynamic-scaling test double.
"""
from __future__ import annotations

import contextlib
import inspect
import io
import math
import unittest
from unittest.mock import patch

import torch

import server_preflight as preflight


class CountingAdamW(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=.01, weight_decay=.01)
        self.actual_updates = 0

    def step(self, closure=None):
        self.actual_updates += 1
        return super().step(closure)


class ArithmeticScaler:
    """Dynamic scaler with real scaled backward and cached unscale finiteness."""

    def __init__(self, scale=65536., enabled=True, backoff=.5):
        self.factor = float(scale) if enabled else 1.
        self.enabled = enabled
        self.backoff = backoff
        self.found_inf = False
        self.events = []

    def is_enabled(self):
        return self.enabled

    def get_scale(self):
        return self.factor

    def scale(self, loss):
        self.events.append('scale')
        return loss*self.factor

    def unscale_(self, optimizer):
        self.events.append('unscale')
        self.found_inf = False
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.div_(self.factor)
                    self.found_inf |= not bool(torch.isfinite(parameter.grad).all())

    def step(self, optimizer):
        self.events.append('step_skip' if self.found_inf and self.enabled else 'step')
        if not (self.found_inf and self.enabled):
            optimizer.step()

    def update(self):
        self.events.append('update')
        if self.found_inf and self.enabled:
            self.factor *= self.backoff


def real_cpu_scaler_or_arithmetic(scale=65536.):
    factory = getattr(getattr(torch, 'amp', None), 'GradScaler', None)
    if factory is not None and 'device' in inspect.signature(factory).parameters:
        return factory('cpu', init_scale=scale, growth_interval=2000)
    # PyTorch 2.2 and older do not expose a CPU GradScaler. This is an actual
    # arithmetic test in those versions, not a skipped test or CUDA fallback.
    return ArithmeticScaler(scale)


class ScalarModel(torch.nn.Module):
    def __init__(self, size=1):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((size,), .25))
        self.register_buffer('forward_count', torch.tensor(0, dtype=torch.int64))


class AlwaysBadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return value.clone()

    @staticmethod
    def backward(ctx, incoming):
        (value,) = ctx.saved_tensors
        return torch.full_like(value, float('inf'))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # CPU AdamW may make an administrative CUDA-graph check on new torch;
        # intercept that check, then make every GPU access an explicit failure.
        self.stack.enter_context(patch.object(torch.optim.Optimizer,
            '_cuda_graph_capture_health_check', return_value=None, create=True))
        for name in ('is_available', 'init', '_lazy_init', 'get_rng_state',
                     'get_rng_state_all', 'set_rng_state', 'set_rng_state_all',
                     'manual_seed', 'manual_seed_all', 'set_device', 'current_device',
                     'synchronize', 'get_device_properties', 'empty_cache'):
            self.stack.enter_context(patch.object(torch.cuda, name,
                side_effect=AssertionError('CUDA forbidden in CPU calibration test: '+name)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def model_optimizer(self, size=1):
        model = ScalarModel(size)
        return model, CountingAdamW(model.parameters())

    def call(self, model, optimizer, scaler, forward, **kwargs):
        return preflight._calibrated_update(model, optimizer, scaler, forward,
                                            'diffusion', **kwargs)

    def half_forward(self, model, observations=None, scaler=None, random=False):
        def forward():
            model.forward_count.add_(1)
            draw = float(torch.rand(())) if random else 0.
            if observations is not None:
                observations.append((float(scaler.get_scale()), draw,
                                     model.weight.detach().clone()))
            # The forward/loss are finite, but gradient=65536 cannot cross the
            # FP16 cast on the first backward. At scale32768 it is representable.
            loss = model.weight.half().float().sum()
            return loss, {'residual': loss*.25}
        return forward

    def bad_forward(self, model, scales, scaler):
        def forward():
            scales.append(float(scaler.get_scale()))
            loss = AlwaysBadGradient.apply(model.weight).sum()
            return loss, {'residual': loss}
        return forward

    def assert_one_update(self, model, optimizer, proof):
        self.assertEqual(optimizer.actual_updates, 1)
        self.assertEqual(proof['successful_optimizer_updates'], 1)
        self.assertEqual(int(optimizer.state[model.weight]['step']), 1)
        self.assertTrue(bool(torch.isfinite(model.weight).all()))
        self.assertGreater(proof['parameter_update_max_abs'], 0.)

    def test_actual_cpu_scaler_or_legacy_arithmetic_backoff_matches_low_scale(self):
        candidate, optimizer = self.model_optimizer()
        reference, reference_optimizer = self.model_optimizer()
        scaler = real_cpu_scaler_or_arithmetic()
        reference_scaler = real_cpu_scaler_or_arithmetic(32768.)
        initial = candidate.weight.detach().clone()
        calls = []
        proof = self.call(candidate, optimizer, scaler,
            self.half_forward(candidate, calls, scaler))
        reference_proof = self.call(reference, reference_optimizer, reference_scaler,
            self.half_forward(reference))
        self.assertEqual([x[0] for x in calls], [65536., 32768.])
        self.assertTrue(all(torch.equal(x[2], initial) for x in calls))
        self.assertEqual(proof['amp_backoffs'], 1)
        self.assertEqual(reference_proof['amp_backoffs'], 0)
        self.assert_one_update(candidate, optimizer, proof)
        self.assertTrue(torch.equal(candidate.weight, reference.weight))
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            self.assertTrue(torch.equal(optimizer.state[candidate.weight][key],
                                       reference_optimizer.state[reference.weight][key]))
        self.assertEqual(proof['amp_attempts'][0]['nonfinite_parameters'][0]['name'], 'weight')
        self.assertEqual(proof['amp_attempts'][0]['state'], 'overflow_skipped')
        self.assertEqual(proof['amp_attempts'][-1]['state'], 'updated')

    def test_retry_replays_torch_rng_and_rolls_back_buffers(self):
        candidate, optimizer = self.model_optimizer()
        reference, reference_optimizer = self.model_optimizer()
        scaler, low = ArithmeticScaler(), ArithmeticScaler(32768.)
        calls, reference_calls = [], []
        rng = torch.get_rng_state().clone()
        proof = self.call(candidate, optimizer, scaler,
            self.half_forward(candidate, calls, scaler, random=True))
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        self.call(reference, reference_optimizer, low,
            self.half_forward(reference, reference_calls, low, random=True))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[0][1], reference_calls[0][1])
        self.assertTrue(torch.equal(final_rng, torch.get_rng_state()))
        self.assertEqual(int(candidate.forward_count), 1)
        self.assertEqual(int(reference.forward_count), 1)
        self.assert_one_update(candidate, optimizer, proof)

    def test_backoff_does_not_clip_bad_gradients(self):
        model, optimizer = self.model_optimizer()
        scaler = ArithmeticScaler()
        original_clip = torch.nn.utils.clip_grad_norm_
        clipped = []
        def checked_clip(parameters, maximum, *args, **kwargs):
            parameters = list(parameters)
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in parameters))
            self.assertAlmostEqual(float(model.weight.grad), 1.)
            clipped.append(True)
            return original_clip(parameters, maximum, *args, **kwargs)
        with patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=checked_clip):
            proof = self.call(model, optimizer, scaler, self.half_forward(model))
        self.assertEqual(len(clipped), 1)
        self.assertEqual(scaler.events,
                         ['scale', 'unscale', 'step_skip', 'update',
                          'scale', 'unscale', 'step', 'update'])
        self.assert_one_update(model, optimizer, proof)

    def test_persistent_bad_gradient_exactly_seventeen_attempts_no_subunit_forward(self):
        model, optimizer = self.model_optimizer()
        before = model.weight.detach().clone()
        scaler, scales = ArithmeticScaler(), []
        with self.assertRaisesRegex(FloatingPointError, 'Persistent nonfinite'):
            self.call(model, optimizer, scaler, self.bad_forward(model, scales, scaler))
        self.assertEqual(scales, [float(2**power) for power in range(16, -1, -1)])
        self.assertEqual(optimizer.actual_updates, 0)
        self.assertFalse(optimizer.state)
        self.assertTrue(torch.equal(model.weight, before))
        # An overflow update may lower scaler state to .5 before throwing, but
        # that state is discarded; no forward at .5 is allowed or performed.
        self.assertGreaterEqual(min(scales), 1.)

    def test_smaller_attempt_bound_stops_without_update(self):
        model, optimizer = self.model_optimizer()
        scaler, scales = ArithmeticScaler(), []
        with self.assertRaisesRegex(FloatingPointError, 'calibration limit'):
            self.call(model, optimizer, scaler, self.bad_forward(model, scales, scaler),
                      max_attempts=2)
        self.assertEqual(scales, [65536., 32768.])
        self.assertEqual(optimizer.actual_updates, 0)

    def test_disabled_amp_bad_gradient_fails_without_step_or_scale_update(self):
        model, optimizer = self.model_optimizer()
        scaler, scales = ArithmeticScaler(enabled=False), []
        with self.assertRaisesRegex(FloatingPointError, 'without enabled AMP'):
            self.call(model, optimizer, scaler, self.bad_forward(model, scales, scaler))
        self.assertEqual(scales, [1.])
        self.assertEqual(scaler.events, ['scale', 'unscale'])
        self.assertEqual(optimizer.actual_updates, 0)

    def test_disabled_amp_finite_gradient_can_update_once(self):
        model, optimizer = self.model_optimizer()
        proof = self.call(model, optimizer, ArithmeticScaler(enabled=False),
                          self.half_forward(model))
        self.assertEqual(proof['amp_backoffs'], 0)
        self.assert_one_update(model, optimizer, proof)

    def test_nonfinite_loss_is_not_retried(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value):
                model, optimizer = self.model_optimizer()
                scaler, calls = ArithmeticScaler(), []
                def forward():
                    calls.append(True)
                    return model.weight.sum()*value, {}
                with self.assertRaisesRegex(FloatingPointError, 'Nonfinite preflight loss'):
                    self.call(model, optimizer, scaler, forward)
                self.assertEqual(len(calls), 1)
                self.assertEqual(scaler.events, [])
                self.assertEqual(optimizer.actual_updates, 0)

    def test_nonfinite_component_is_not_retried(self):
        model, optimizer = self.model_optimizer()
        scaler = ArithmeticScaler()
        with self.assertRaisesRegex(FloatingPointError, 'component poisoned'):
            self.call(model, optimizer, scaler,
                lambda: (model.weight.sum(), {'poisoned': torch.tensor(float('nan'))}))
        self.assertEqual(scaler.events, [])
        self.assertEqual(optimizer.actual_updates, 0)

    def test_nonfinite_output_guard_propagates_before_backward(self):
        model, optimizer = self.model_optimizer()
        scaler = ArithmeticScaler()
        def forward():
            preflight._finite(torch.tensor(float('inf')), 'diffusion output')
            return model.weight.sum(), {}
        with self.assertRaisesRegex(FloatingPointError, 'diffusion output'):
            self.call(model, optimizer, scaler, forward)
        self.assertEqual(scaler.events, [])
        self.assertEqual(optimizer.actual_updates, 0)

    def test_scaler_that_does_not_reduce_scale_fails(self):
        model, optimizer = self.model_optimizer()
        scaler, scales = ArithmeticScaler(backoff=1.), []
        with self.assertRaisesRegex(FloatingPointError, 'failed to reduce'):
            self.call(model, optimizer, scaler, self.bad_forward(model, scales, scaler))
        self.assertEqual(len(scales), 1)
        self.assertEqual(optimizer.actual_updates, 0)

    def test_overflow_parameter_or_optimizer_mutation_is_rejected(self):
        for mutation in ('parameter', 'state', 'group'):
            with self.subTest(mutation=mutation):
                model, optimizer = self.model_optimizer()
                class TamperingScaler(ArithmeticScaler):
                    def step(self, opt):
                        super().step(opt)
                        if mutation == 'parameter':
                            with torch.no_grad():
                                model.weight.add_(1.)
                        elif mutation == 'state':
                            opt.state[model.weight]['step'] = torch.tensor(1.)
                        else:
                            opt.param_groups[0]['lr'] *= .5
                scaler, scales = TamperingScaler(), []
                with self.assertRaisesRegex(RuntimeError, 'Overflow attempt changed'):
                    self.call(model, optimizer, scaler, self.bad_forward(model, scales, scaler))
                self.assertEqual(len(scales), 1)
                self.assertEqual(optimizer.actual_updates, 0)

    def test_finite_gradient_elements_but_overflowing_norm_fail_without_update(self):
        model, optimizer = self.model_optimizer(size=2)
        scaler = ArithmeticScaler(1.)
        def forward():
            loss = model.weight.sum()*2.e38
            self.assertTrue(bool(torch.isfinite(loss)))
            return loss, {}
        with self.assertRaisesRegex(FloatingPointError, 'gradient norm'):
            self.call(model, optimizer, scaler, forward)
        self.assertEqual(scaler.events, ['scale', 'unscale'])
        self.assertEqual(optimizer.actual_updates, 0)

    def test_no_model_gradients_or_all_zero_gradients_fail(self):
        for mode in ('none', 'zero'):
            with self.subTest(mode=mode):
                model, optimizer = self.model_optimizer()
                scaler = ArithmeticScaler(1.)
                def forward():
                    loss = (torch.tensor(.2, requires_grad=True) if mode == 'none'
                            else model.weight.sum()*0.)
                    return loss, {}
                with self.assertRaisesRegex(RuntimeError, 'No gradients|All preflight gradients'):
                    self.call(model, optimizer, scaler, forward)
                self.assertEqual(optimizer.actual_updates, 0)

    def test_invalid_scale_and_attempt_limit_fail_before_forward(self):
        for scale in (.5, float('nan'), float('inf')):
            with self.subTest(scale=scale):
                model, optimizer = self.model_optimizer()
                called = []
                def forward():
                    called.append(True)
                    return model.weight.sum(), {}
                with self.assertRaisesRegex(FloatingPointError, 'Invalid preflight loss scale'):
                    self.call(model, optimizer, ArithmeticScaler(scale), forward)
                self.assertEqual(called, [])
        model, optimizer = self.model_optimizer()
        with self.assertRaises(ValueError):
            self.call(model, optimizer, ArithmeticScaler(), self.half_forward(model), max_attempts=0)

    def test_nonfresh_optimizer_is_rejected_before_forward(self):
        model, optimizer = self.model_optimizer()
        optimizer.state[model.weight]['step'] = torch.tensor(1.)
        calls = []
        def forward():
            calls.append(True)
            return model.weight.sum(), {}
        with self.assertRaisesRegex(ValueError, 'fresh optimizer'):
            self.call(model, optimizer, ArithmeticScaler(), forward)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
