"""Synthetic CPU checks only; no dataset/checkpoint load, CUDA, or training run.

Run with ``python -B -m unittest discover -s tests -p test_transformer_model.py``.
All dependencies are imported from this portable package, never the old project.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn


HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE
if not (PROJECT / "models/pl3d_residual_prediction.py").is_file():
    PROJECT = HERE
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location(
    "_transformer_model_under_test", HERE / "models/pl3d_residual_prediction_transformer.py"
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

from models.pl3d_residual_prediction import Predictor as OriginalPredictor
from models.pl3d_residual_prediction import Corrector as OriginalCorrector
from models.radiodiff_aff_full3d import ResidualBlock3D
from models.transformer import TransformerBlock as OriginalTransformerBlock


def original_block():
    return OriginalTransformerBlock(dim=128, num_heads=4, mlp_dim=256,
                                    dropout=0.0, activation="gelu")


class TransformerModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(772)

    def test_original_block_weight_mapping_forward_and_gradient(self):
        old = original_block().double().train()
        new = model.OriginalConfigTransformerBlock().double().train()
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        new.load_state_dict(old.state_dict(), strict=True)
        x_old = torch.randn(2, 128, 2, 3, dtype=torch.float64, requires_grad=True)
        x_new = x_old.detach().clone().requires_grad_(True)
        y_old, y_new = old(x_old), new(x_new)
        torch.testing.assert_close(y_new, y_old, rtol=1e-11, atol=1e-12)
        weights = torch.randn_like(y_old)
        (y_old * weights).sum().backward()
        (y_new * weights).sum().backward()
        torch.testing.assert_close(x_new.grad, x_old.grad, rtol=1e-10, atol=1e-11)
        for (old_name, old_param), (new_name, new_param) in zip(
            old.named_parameters(), new.named_parameters()
        ):
            self.assertEqual(old_name, new_name)
            self.assertIsNotNone(new_param.grad)
            torch.testing.assert_close(new_param.grad, old_param.grad, rtol=1e-10, atol=1e-11)

    def test_original_block_eval_and_single_spatial_token(self):
        old = original_block().eval()
        new = model.OriginalConfigTransformerBlock().eval()
        new.load_state_dict(old.state_dict(), strict=True)
        with torch.no_grad():
            for shape in [(1, 128, 1, 1), (2, 128, 2, 3)]:
                x = torch.randn(shape)
                actual = new(x)
                self.assertEqual(actual.shape, x.shape)
                self.assertTrue(torch.isfinite(actual).all())
                torch.testing.assert_close(actual, old(x), rtol=3e-5, atol=3e-6)

    def test_per_height_exact_configuration_and_no_extra_projection(self):
        stack = model.PerHeightSpatialTransformer(activation_checkpoint=False)
        self.assertEqual(len(stack.blocks), 2)
        self.assertEqual(sum(p.numel() for p in stack.parameters()), 264960)
        self.assertEqual(len(list(stack.children())), 1)
        for block in stack.blocks:
            self.assertEqual(block.dim, 128)
            self.assertEqual(block.num_heads, 4)
            self.assertEqual(block.mlp_dim, 256)
            self.assertEqual(block.attn.dropout, 0.0)
            self.assertTrue(block.attn.batch_first)
            self.assertIsNotNone(block.attn.in_proj_bias)
            self.assertIsNotNone(block.attn.out_proj.bias)
            self.assertEqual(block.norm1.eps, 1e-5)
            self.assertEqual(block.norm2.eps, 1e-5)
            self.assertIsInstance(block.ffn.act, nn.GELU)
            self.assertEqual(block.ffn.act.approximate, "none")
            self.assertEqual(block.ffn.drop1.p, 0.0)
            self.assertEqual(block.ffn.drop2.p, 0.0)
            self.assertEqual(block.ffn.fc1.in_features, 128)
            self.assertEqual(block.ffn.fc1.out_features, 256)
            self.assertEqual(block.ffn.fc2.out_features, 128)
        decoded = json.loads(json.dumps(model.ARCHITECTURE_SPEC))
        self.assertEqual(decoded["transformer"]["parameters_both_stages"], 529920)
        self.assertIsNone(decoded["transformer"]["position_encoding"])

    def test_per_height_matches_original_and_no_cross_height_attention(self):
        stack = model.PerHeightSpatialTransformer(activation_checkpoint=False).double().eval()
        originals = nn.Sequential(original_block(), original_block()).double().eval()
        for old, new in zip(originals, stack.blocks):
            old.load_state_dict(new.state_dict(), strict=True)
        x = torch.randn(1, 128, 4, 2, 3, dtype=torch.float64)
        with torch.no_grad():
            expected = torch.stack([originals(x[:, :, h]) for h in range(4)], dim=2)
            actual = stack(x)
            torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
            perturbed = x.clone()
            perturbed[:, :, 2] += torch.randn_like(perturbed[:, :, 2])
            changed = stack(perturbed)
            for height in (0, 1, 3):
                torch.testing.assert_close(changed[:, :, height], actual[:, :, height], rtol=0, atol=0)
            self.assertFalse(torch.equal(changed[:, :, 2], actual[:, :, 2]))
            repeated = x[:, :, 0:1].repeat(1, 1, 4, 1, 1)
            result = stack(repeated)
            for height in range(1, 4):
                torch.testing.assert_close(result[:, :, height], result[:, :, 0], rtol=0, atol=0)

    def test_checkpoint_recomputation_preserves_values_and_gradients(self):
        plain = model.PerHeightSpatialTransformer(activation_checkpoint=False).double().train()
        recomputed = model.PerHeightSpatialTransformer(activation_checkpoint=True).double().train()
        recomputed.load_state_dict(plain.state_dict(), strict=True)
        x = torch.randn(1, 128, 4, 2, 2, dtype=torch.float64, requires_grad=True)
        twin = x.detach().clone().requires_grad_(True)
        expected = plain(x)
        with patch.object(model, "activation_checkpoint", wraps=model.activation_checkpoint) as spy:
            actual = recomputed(twin)
            self.assertEqual(spy.call_count, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(twin.grad, x.grad, rtol=0, atol=0)
        for p, q in zip(plain.parameters(), recomputed.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        with patch.object(model, "activation_checkpoint", side_effect=AssertionError("eval must not checkpoint")):
            recomputed.eval()(x.detach())
            recomputed.train()
            with torch.no_grad():
                recomputed(x.detach())

    def test_stages_are_independent_and_attention_has_real_gradients(self):
        predictor = model.Predictor()
        corrector = model.Corrector()
        stage_a = predictor.bottleneck[1]
        stage_b = corrector.bottleneck[1]
        self.assertTrue(set(map(id, stage_a.parameters())).isdisjoint(set(map(id, stage_b.parameters()))))
        self.assertTrue(issubclass(model.Predictor, OriginalPredictor))
        self.assertTrue(issubclass(model.Corrector, OriginalCorrector))
        self.assertIsInstance(predictor.bottleneck[0], ResidualBlock3D)
        self.assertIsInstance(predictor.bottleneck[2], ResidualBlock3D)
        self.assertEqual(predictor.input_block.conv1.in_channels, 5)
        self.assertEqual(corrector.input_block.conv1.in_channels, 6)
        self.assertEqual(predictor.input_block.conv1.out_channels, 32)
        self.assertEqual(corrector.enc2.down.out_channels, 128)
        calls = []
        handles = []
        def record(module, args, kwargs):
            calls.append((id(module), args[0].shape, kwargs.get("need_weights")))
        for stage in (stage_a, stage_b):
            for block in stage.blocks:
                handles.append(block.attn.register_forward_pre_hook(record, with_kwargs=True))
        try:
            condition = torch.randn(1, 9, 4, 8, 8)
            coarse = predictor(condition)
            residual = corrector(torch.randn_like(coarse), condition, torch.tensor([199]), coarse.detach())
            self.assertEqual(coarse.shape, (1, 1, 4, 8, 8))
            self.assertEqual(residual.shape, coarse.shape)
            self.assertTrue(torch.isfinite(coarse).all())
            self.assertTrue(torch.isfinite(residual).all())
            self.assertEqual(len(calls), 16)
            self.assertEqual(len({entry[0] for entry in calls}), 4)
            for _, shape, need_weights in calls:
                self.assertEqual(shape, torch.Size([1, 4, 128]))
                self.assertIs(need_weights, False)
            (coarse.square().mean() + residual.square().mean()).backward()
            for stage in (stage_a, stage_b):
                for block in stage.blocks:
                    for name, parameter in block.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(float(block.attn.in_proj_weight.grad.abs().sum()), 0)
                    self.assertGreater(float(block.attn.out_proj.weight.grad.abs().sum()), 0)
                    self.assertGreater(float(block.ffn.fc1.weight.grad.abs().sum()), 0)
        finally:
            for handle in handles:
                handle.remove()

    def test_forbidden_channels_cannot_affect_either_stage(self):
        predictor = model.Predictor().eval()
        corrector = model.Corrector().eval()
        condition = torch.randn(1, 9, 4, 8, 8)
        poisoned = condition.clone()
        poisoned[:, [3, 4, 7, 8]] = float("nan")
        noisy = torch.randn(1, 1, 4, 8, 8)
        t = torch.tensor([37])
        with torch.no_grad():
            coarse = predictor(condition)
            poisoned_coarse = predictor(poisoned)
            torch.testing.assert_close(coarse, poisoned_coarse, rtol=0, atol=0)
            first = corrector(noisy, condition, t, coarse)
            second = corrector(noisy, poisoned, t, poisoned_coarse)
            self.assertTrue(torch.isfinite(second).all())
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertEqual(list(inspect.signature(model.Predictor.forward).parameters), ["self", "condition"])
        self.assertEqual(list(inspect.signature(model.Corrector.forward).parameters),
                         ["self", "noisy_residual", "condition", "t", "coarse"])

    def test_inherited_cnn_and_rng_unchanged_when_added_stack_bypassed(self):
        condition = torch.randn(1, 9, 4, 8, 8)
        signal = torch.randn(1, 1, 4, 8, 8)
        t = torch.tensor([71])
        for old_class, new_class in ((OriginalPredictor, model.Predictor),
                                     (OriginalCorrector, model.Corrector)):
            torch.manual_seed(419)
            old = old_class(base_channels=32).eval()
            old_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(419)
            new = new_class(base_channels=32).eval()
            torch.testing.assert_close(torch.random.get_rng_state(), old_rng, rtol=0, atol=0)
            expected_keys = set()
            for key, value in old.state_dict().items():
                mapped = key.replace("bottleneck.1.", "bottleneck.2.")
                expected_keys.add(mapped)
                torch.testing.assert_close(new.state_dict()[mapped], value, rtol=0, atol=0)
            new_only = set(new.state_dict()) - expected_keys
            self.assertTrue(new_only)
            self.assertTrue(all(key.startswith("bottleneck.1.blocks.") for key in new_only))
            self.assertEqual(sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()), 264960)
            new.bottleneck[1] = nn.Identity()
            with torch.no_grad():
                if old_class is OriginalPredictor:
                    expected, actual = old(condition), new(condition)
                else:
                    expected, actual = old(signal, condition, t, signal), new(signal, condition, t, signal)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_old_checkpoint_strictly_rejected_and_new_roundtrip(self):
        for old_class, new_class in ((OriginalPredictor, model.Predictor),
                                     (OriginalCorrector, model.Corrector)):
            old, new = old_class(), new_class()
            with self.assertRaises(RuntimeError):
                new.load_state_dict(old.state_dict(), strict=True)
            self.assertFalse(new.load_state_dict(new.state_dict(), strict=True).missing_keys)
            with self.assertRaises(RuntimeError):
                old.load_state_dict(new.state_dict(), strict=True)

    def test_optimizer_and_ema_include_all_new_parameters(self):
        net = model.Predictor()
        ema = copy.deepcopy(net).eval().requires_grad_(False)
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer_parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
        transformer_parameters = {id(p) for p in net.bottleneck[1].parameters()}
        self.assertTrue(transformer_parameters <= optimizer_parameters)
        self.assertEqual(set(net.state_dict()), set(ema.state_dict()))
        condition = torch.randn(1, 9, 4, 8, 8)
        old_parameter = net.bottleneck[1].blocks[0].attn.in_proj_weight.detach().clone()
        net(condition).square().mean().backward()
        optimizer.step()
        parameter = net.bottleneck[1].blocks[0].attn.in_proj_weight
        self.assertFalse(torch.equal(parameter, old_parameter))
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), net.parameters()):
                ep.lerp_(p, .001)
        actual = ema.bottleneck[1].blocks[0].attn.in_proj_weight
        torch.testing.assert_close(actual, torch.lerp(old_parameter, parameter.detach(), .001), rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad for p in ema.parameters()))

    def test_minimum_1x1_bottleneck_and_non_square_shapes(self):
        predictor = model.Predictor(activation_checkpoint=False).eval()
        corrector = model.Corrector(activation_checkpoint=False).eval()
        with torch.no_grad():
            for h, w in ((4, 4), (7, 9)):
                condition = torch.randn(1, 9, 4, h, w)
                coarse = predictor(condition)
                output = corrector(torch.zeros_like(coarse), condition, torch.tensor([0]), coarse)
                self.assertEqual(output.shape, (1, 1, 4, h, w))
                self.assertTrue(torch.isfinite(output).all())

    def test_invalid_shapes_and_configuration_fail_closed(self):
        for cls in (model.Predictor, model.Corrector):
            for width in (4, 64, True, "32"):
                with self.assertRaises(ValueError):
                    cls(base_channels=width)
            with self.assertRaises(TypeError):
                cls(transformer_heads=8)
        with self.assertRaises(TypeError):
            model.PerHeightSpatialTransformer(activation_checkpoint="yes")
        with self.assertRaises(TypeError):
            model.OriginalConfigTransformerBlock(dim=64)
        stack = model.PerHeightSpatialTransformer(activation_checkpoint=False)
        for shape in ((1, 128, 3, 2, 2), (1, 64, 4, 2, 2), (1, 128, 2, 2), (1, 128, 4, 0, 2)):
            with self.assertRaises(ValueError):
                stack(torch.empty(shape))
        predictor, corrector = model.Predictor(), model.Corrector()
        for shape in ((1, 8, 4, 8, 8), (1, 9, 3, 8, 8), (1, 9, 4, 3, 8), (1, 9, 8, 8)):
            with self.assertRaises(ValueError):
                predictor(torch.empty(shape))
        c = torch.randn(1, 9, 4, 8, 8)
        x = torch.randn(1, 1, 4, 8, 8)
        t = torch.tensor([5])
        with self.assertRaises(TypeError):
            predictor(c.long())
        for noisy, coarse, times in ((x.repeat(1, 2, 1, 1, 1), x, t),
                                     (x, x[:, :, :3], t), (x, x, torch.tensor([1, 2]))):
            with self.assertRaises(ValueError):
                corrector(noisy, c, times, coarse)
        with self.assertRaises(TypeError):
            corrector(x, c, t.float(), x)


if __name__ == "__main__":
    unittest.main()
