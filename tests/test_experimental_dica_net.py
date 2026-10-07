"""Phase 5.4.1 CPU tensor/numerical tests; no pretrained artifact or data access."""

import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import efficientnet_v2_s

from src.ingredient_selection.runtime import PROJECTION_ID, resolve_projection
from src.models.dica_net import DICANetSCore, DICAReadoutS
from src.models.experimental_contract import ExperimentalModelContract


def _reference_block(block, queries, memory):
    """Direct route-Q equations, independent of module forward and SDPA."""
    b, labels, width = queries.shape
    normalized = F.layer_norm(queries, (width,), block.query_norm.weight, block.query_norm.bias, 1e-5)
    q = F.linear(normalized, block.q_projection.weight, block.q_projection.bias).reshape(b, labels, 4, 32)
    k = F.linear(memory, block.k_projection.weight, block.k_projection.bias).reshape(b, 245, 4, 32)
    v = F.linear(memory, block.v_projection.weight, block.v_projection.bias).reshape(b, 245, 4, 32)
    scores = torch.einsum("blhd,bnhd->bhln", q, k) / math.sqrt(32)
    probabilities = scores.softmax(dim=-1)
    values = torch.einsum("bhln,bnhd->blhd", probabilities, v).reshape(b, labels, width)
    residual = queries + F.linear(values, block.attention_output.weight, block.attention_output.bias)
    normalized = F.layer_norm(residual, (width,), block.ffn_norm.weight, block.ffn_norm.bias, 1e-5)
    hidden = F.gelu(F.linear(normalized, block.ffn_input.weight, block.ffn_input.bias), approximate="none")
    return residual + F.linear(hidden, block.ffn_output.weight, block.ffn_output.bias)


def _shared_rows(source, indices):
    """Test-only algebraic copy, never a selected-task training/resume API."""
    target = DICAReadoutS(len(indices), head_seed=91).to(dtype=source.ingredient_queries.dtype)
    indexed = {"ingredient_queries", "classwise_readout.weight", "classwise_readout.bias",
               "pooled_context_readout.weight", "pooled_context_readout.bias"}
    target.load_state_dict({key: value[indices].clone() if key in indexed else value.clone()
                            for key, value in source.state_dict().items()}, strict=True)
    return target.eval()


class DICANetTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.rng = torch.random.fork_rng(devices=[])
        cls.rng.__enter__()
        torch.manual_seed(541)

    @classmethod
    def tearDownClass(cls):
        cls.rng.__exit__(None, None, None)
        torch.set_num_threads(cls.previous_threads)

    @staticmethod
    def _features(batch=1, dtype=torch.float32, requires_grad=False):
        return (torch.randn(batch, 160, 14, 14, dtype=dtype, requires_grad=requires_grad),
                torch.randn(batch, 1280, 7, 7, dtype=dtype, requires_grad=requires_grad))

    def test_real_encoder_shapes_counts_once_only_traversal_all_output_widths(self):
        features = efficientnet_v2_s(weights=None).features.eval()
        self.assertEqual(sum(p.numel() for p in features.parameters()), 20_177_488)
        for width in (1, 50, 59, 165):
            with self.subTest(width=width):
                model = DICANetSCore(features, width).eval()
                self.assertIs(model.features, features)
                calls = [0] * 8
                maps = {}
                def observe(index):
                    def hook(module, inputs, output):
                        calls[index] += 1
                        maps[index] = tuple(output.shape)
                    return hook
                handles = [stage.register_forward_hook(observe(i)) for i, stage in enumerate(features)]
                try:
                    with torch.no_grad():
                        logits = model(torch.randn(1, 3, 224, 224))
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(calls, [1] * 8)
                self.assertEqual(maps[5], (1, 160, 14, 14))
                self.assertEqual(maps[7], (1, 1280, 7, 7))
                self.assertEqual(tuple(logits.shape), (1, width))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(sum(p.numel() for p in model.readout.parameters()), 383_616 + 1538 * width)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 20_561_104 + 1538 * width)
                self.assertFalse(any("classifier" in name for name, _ in model.named_parameters()))

    def test_memory_row_major_scale_ranges_and_affine_normalization(self):
        readout = DICAReadoutS(3)
        f16, f32 = self._features(batch=2)
        with torch.no_grad():
            readout.f16_token_norm.weight.copy_(torch.linspace(.5, 1.5, 128))
            readout.f16_token_norm.bias.fill_(.3)
            readout.f16_scale_embedding.copy_(torch.linspace(-.2, .2, 128))
            readout.f32_scale_embedding.copy_(torch.linspace(.7, 1.2, 128))
            memory = readout.build_memory(f16, f32)
            reference = []
            for features, projection, norm, embedding in (
                    (f16, readout.f16_projection, readout.f16_token_norm, readout.f16_scale_embedding),
                    (f32, readout.f32_projection, readout.f32_token_norm, readout.f32_scale_embedding)):
                projected = F.conv2d(features, projection.weight)
                for row in range(projected.shape[2]):
                    for col in range(projected.shape[3]):
                        token = F.layer_norm(projected[:, :, row, col], (128,), norm.weight, norm.bias, 1e-5)
                        reference.append(token + embedding)
            torch.testing.assert_close(memory, torch.stack(reference, dim=1), rtol=1e-6, atol=1e-6)
        self.assertEqual(tuple(memory.shape), (2, 245, 128))
        self.assertEqual(readout.TOKEN_RANGES, ((0, 196), (196, 245)))
        self.assertIsNone(readout.f16_projection.bias)
        self.assertIsNone(readout.f32_projection.bias)

    def test_attention_reference_forward_and_backward_fp32_fp64(self):
        for dtype, rtol, atol in ((torch.float32, 2e-4, 3e-6), (torch.float64, 1e-9, 1e-11)):
            for labels in (1, 59, 165):
                with self.subTest(dtype=dtype, labels=labels):
                    block = DICAReadoutS(labels).blocks[0].to(dtype=dtype)
                    queries = torch.randn(2, labels, 128, dtype=dtype, requires_grad=True)
                    memory = torch.randn(2, 245, 128, dtype=dtype, requires_grad=True)
                    cotangent = torch.randn_like(queries) / queries.numel()
                    actual = block(queries, memory)
                    variables = (queries, memory, *block.parameters())
                    actual_grads = torch.autograd.grad((actual * cotangent).sum(), variables)
                    reference = _reference_block(block, queries, memory)
                    reference_grads = torch.autograd.grad((reference * cotangent).sum(), variables)
                    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
                    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
                        torch.testing.assert_close(actual_grad, reference_grad, rtol=rtol, atol=atol)
                        self.assertTrue(torch.isfinite(actual_grad).all())

    def test_sdpa_is_unmasked_noncausal_and_zero_dropout_in_train_and_eval(self):
        readout = DICAReadoutS(3)
        features = self._features()
        for training in (True, False):
            with self.subTest(training=training), patch(
                    "src.models.dica_net.F.scaled_dot_product_attention",
                    wraps=F.scaled_dot_product_attention) as attention:
                readout.train(training)(*features)
                attention.assert_called_once()
                self.assertEqual(attention.call_args.kwargs,
                                 {"attn_mask": None, "dropout_p": 0.0, "is_causal": False})
                q, k, v = attention.call_args.args
                self.assertEqual(tuple(q.shape), (1, 4, 3, 32))
                self.assertEqual(tuple(k.shape), (1, 4, 245, 32))
                self.assertEqual(tuple(v.shape), (1, 4, 245, 32))

    def test_class_rows_permute_and_subset_without_changing_other_logits(self):
        source = DICAReadoutS(165).eval()
        features = self._features(batch=2)
        projection = resolve_projection(PROJECTION_ID)
        self.assertEqual(len(projection.class_order), 59)
        index_sets = (torch.randperm(165), torch.tensor(projection.base_class_indices),
                      torch.arange(50), torch.tensor([73]))
        with torch.no_grad():
            original_branches = source.forward_branches(*features)
            original = source(*features)
            for indices in index_sets:
                with self.subTest(labels=len(indices)):
                    target = _shared_rows(source, indices)
                    for branch, expected in zip(target.forward_branches(*features), original_branches):
                        torch.testing.assert_close(branch, expected[:, indices], rtol=2e-5, atol=3e-6)
                    actual = target(*features)
                    torch.testing.assert_close(actual, original[:, indices], rtol=2e-5, atol=3e-6)
                    if len(indices) == 59:
                        projected = projection.project_columns(original, projection.base_class_order)
                        torch.testing.assert_close(actual, projected, rtol=2e-5, atol=3e-6)

    def test_one_label_output_has_no_gradient_to_other_label_rows(self):
        readout = DICAReadoutS(5)
        readout(*self._features()).select(1, 2).sum().backward()
        for parameter in (readout.ingredient_queries, readout.classwise_readout.weight,
                          readout.classwise_readout.bias, readout.pooled_context_readout.weight,
                          readout.pooled_context_readout.bias):
            self.assertGreater(parameter.grad[2].abs().sum().item(), 0)
            self.assertEqual(parameter.grad[[0, 1, 3, 4]].abs().sum().item(), 0)

    def test_both_branches_reach_their_features_and_trainable_parameters(self):
        readout = DICAReadoutS(59)
        f16, f32 = self._features(requires_grad=True)
        query, context = readout.forward_branches(f16, f32)
        q_grads = torch.autograd.grad(query.square().mean(),
                                     (f16, f32, readout.ingredient_queries,
                                      readout.f16_projection.weight, readout.f32_projection.weight),
                                     retain_graph=True)
        c_grads = torch.autograd.grad(context.square().mean(),
                                     (f16, f32, readout.pooled_context_readout.weight),
                                     allow_unused=True, retain_graph=True)
        self.assertIsNone(c_grads[0])
        for gradient in (*q_grads, *c_grads[1:]):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        (query + context).square().mean().backward()
        for parameter in readout.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_real_encoder_full_forward_backward_and_input_gradient(self):
        features = efficientnet_v2_s(weights=None).features
        model = DICANetSCore(features, 59).train()
        image = torch.randn(1, 3, 224, 224, requires_grad=True)
        logits = model(image)
        target = torch.arange(59).remainder(2).float().unsqueeze(0)
        F.binary_cross_entropy_with_logits(logits, target).backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(image.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for parameter in (features[0][0].weight, model.readout.ingredient_queries,
                          model.readout.pooled_context_readout.weight):
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_fixed_unit_logit_addition_has_no_probability_or_label_normalization(self):
        readout = DICAReadoutS(3)
        with torch.no_grad():
            readout.classwise_readout.weight.zero_()
            readout.classwise_readout.bias.copy_(torch.tensor([-4., .5, 1.]))
            readout.pooled_context_readout.weight.zero_()
            readout.pooled_context_readout.bias.copy_(torch.tensor([1., -.5, 3.]))
            logits = readout(*self._features(batch=2))
        torch.testing.assert_close(logits, torch.tensor([[-3., 0., 4.], [-3., 0., 4.]]), rtol=0, atol=0)

    def test_memory_permutation_invariance_and_batch_independence(self):
        readout = DICAReadoutS(5).eval()
        f16, f32 = self._features(batch=2)
        with torch.no_grad():
            memory = readout.build_memory(f16, f32)
            queries = readout.ingredient_queries.unsqueeze(0).expand(2, -1, -1)
            block = readout.blocks[0]
            torch.testing.assert_close(block(queries, memory), block(queries, memory[:, torch.randperm(245)]),
                                       rtol=2e-5, atol=3e-6)
            together = readout(f16, f32)
            individual = torch.cat([readout(f16[i:i+1], f32[i:i+1]) for i in range(2)])
            torch.testing.assert_close(together, individual, rtol=2e-5, atol=3e-6)

    def test_core_preserves_supplied_encoder_and_matches_declared_s_structure(self):
        features = efficientnet_v2_s(weights=None).features.eval()
        before = {key: value.clone() for key, value in features.state_dict().items()}
        model = DICANetSCore(features, 165)
        for key, value in features.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(features.training)
        architecture = ExperimentalModelContract("p2_s", 165).to_config()["architecture"]
        self.assertEqual(model.readout.WIDTH, architecture["width"])
        self.assertEqual(model.readout.HEADS, architecture["heads"])
        self.assertEqual(len(model.readout.blocks), architecture["query_blocks"])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 20_814_874)
        self.assertEqual(sum(p.numel() for p in model.readout.parameters()), 637_386)
        self.assertFalse(any(isinstance(m, (nn.Dropout, nn.MultiheadAttention)) for m in model.readout.modules()))
        block = model.readout.blocks[0]
        self.assertEqual(len({m.weight.data_ptr() for m in
                             (block.q_projection, block.k_projection, block.v_projection)}), 3)
        self.assertEqual(block.ffn_input.out_features, 4 * architecture["width"])
        for module in model.readout.modules():
            if isinstance(module, nn.LayerNorm):
                self.assertEqual(module.eps, 1e-5)
                self.assertEqual(module.normalized_shape, (128,))

    def test_invalid_count_seed_input_and_feature_boundaries(self):
        for width in (True, 0, -1, 1.0):
            with self.assertRaises(ValueError):
                DICAReadoutS(width)
        for seed in (True, -1, 2**63, 1.0):
            with self.assertRaises(ValueError):
                DICAReadoutS(3, seed)
        with self.assertRaises(ValueError):
            DICANetSCore(nn.Sequential(nn.Identity()), 3)
        model = DICANetSCore(nn.Sequential(*[nn.Identity() for _ in range(8)]), 3)
        for image in (torch.zeros(3, 224, 224), torch.zeros(0, 3, 224, 224),
                      torch.zeros(1, 3, 384, 384), torch.zeros(1, 1, 224, 224),
                      torch.zeros(1, 3, 224, 224, dtype=torch.uint8)):
            with self.assertRaises(ValueError):
                model(image)
        fine, coarse = self._features()
        for f16, f32 in ((fine[:, :159], coarse), (fine, coarse[:, :, :6]),
                         (fine, coarse.repeat(2, 1, 1, 1)), (fine, coarse.double())):
            with self.assertRaises(ValueError):
                model.readout(f16, f32)


if __name__ == "__main__":
    unittest.main()
