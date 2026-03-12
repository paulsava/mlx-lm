import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.hooks import wrap_model_with_hooks, HookPoint
from mlx_lm.hooks.sdpa import manual_scaled_dot_product_attention
from mlx_lm.models.base import scaled_dot_product_attention


class TestHookPoint(unittest.TestCase):
    """Unit tests for HookPoint class."""

    def test_hook_point_creation(self):
        """Test basic HookPoint creation."""
        # Create a simple linear layer
        linear = nn.Linear(10, 10)
        hook_point = HookPoint("test.hook", linear)

        self.assertEqual(hook_point.name, "test.hook")
        self.assertIsNotNone(hook_point.base)
        self.assertEqual(len(hook_point.hooks), 0)

    def test_hook_point_without_base(self):
        """Test HookPoint for intermediate activations (no base module)."""
        hook_point = HookPoint("test.intermediate", base_module=None)

        x = mx.random.normal((2, 10))
        output = hook_point(x)

        # Should pass through unchanged when no hooks
        self.assertTrue(mx.array_equal(output, x))

    def test_add_hook(self):
        """Test adding hooks to HookPoint."""
        hook_point = HookPoint("test.hook", base_module=None)

        def simple_hook(name, activation, context):
            return activation * 2

        hook_point.add_hook(simple_hook)
        self.assertEqual(len(hook_point.hooks), 1)

        x = mx.array([1.0, 2.0, 3.0])
        output = hook_point(x)

        expected = x * 2
        self.assertTrue(mx.allclose(output, expected))

    def test_multiple_hooks(self):
        """Test that multiple hooks are applied sequentially."""
        hook_point = HookPoint("test.hook", base_module=None)

        def add_one(name, activation, context):
            return activation + 1

        def multiply_two(name, activation, context):
            return activation * 2

        hook_point.add_hook(add_one)
        hook_point.add_hook(multiply_two)

        x = mx.array([1.0, 2.0, 3.0])
        output = hook_point(x)

        # Should be (x + 1) * 2
        expected = (x + 1) * 2
        self.assertTrue(mx.allclose(output, expected))

    def test_clear_hooks(self):
        """Test clearing all hooks."""
        hook_point = HookPoint("test.hook", base_module=None)

        def simple_hook(name, activation, context):
            return activation * 2

        hook_point.add_hook(simple_hook)
        self.assertEqual(len(hook_point.hooks), 1)

        hook_point.clear_hooks()
        self.assertEqual(len(hook_point.hooks), 0)

        x = mx.array([1.0, 2.0, 3.0])
        output = hook_point(x)

        # Should pass through unchanged
        self.assertTrue(mx.array_equal(output, x))

    def test_hook_with_context(self):
        """Test that hooks receive context correctly."""
        hook_point = HookPoint("test.hook", base_module=None)

        received_context = {}

        def context_hook(name, activation, context):
            received_context.update(context)
            return activation

        hook_point.add_hook(context_hook)

        x = mx.array([1.0, 2.0, 3.0])
        output = hook_point(x, cache="test_cache", mask="test_mask")

        self.assertEqual(received_context['cache'], "test_cache")
        self.assertEqual(received_context['mask'], "test_mask")


class TestModelWrapping(unittest.TestCase):
    """Integration tests for model wrapping."""

    def test_wrap_simple_model(self):
        """Test wrapping a simple model structure."""
        # Create a minimal model structure similar to MLX-LM
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 2
                self.n_kv_heads = 2
                self.scale = 0.125
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)
                self.v_proj = nn.Linear(64, 64, bias=False)
                self.o_proj = nn.Linear(64, 64, bias=False)
                self.rope = nn.RoPE(32, traditional=False)

            def __call__(self, x, mask=None, cache=None):
                B, L, D = x.shape
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                q = self.rope(q)
                k = self.rope(k)

                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 128, bias=False)
                self.up_proj = nn.Linear(64, 128, bias=False)
                self.down_proj = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

        class SimpleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SimpleAttention()
                self.mlp = SimpleMLP()
                self.input_layernorm = nn.RMSNorm(64)
                self.post_attention_layernorm = nn.RMSNorm(64)

            def __call__(self, x, mask=None, cache=None):
                r = self.self_attn(self.input_layernorm(x), mask, cache)
                h = x + r
                r = self.mlp(self.post_attention_layernorm(h))
                return h + r

        class SimpleModelCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [SimpleBlock() for _ in range(2)]

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleModelCore()

            def __call__(self, x, cache=None):
                for layer in self.model.layers:
                    x = layer(x, cache=cache)
                return x

        # Create and wrap model
        model = SimpleModel()
        hooked_model = wrap_model_with_hooks(model, use_manual_sdpa=False)

        # Check that hook points were created
        hook_points = hooked_model.hook_points()
        self.assertGreater(len(hook_points), 0)

        # Check expected hook points exist
        expected_hooks = [
            "blocks.0.attn.hook_q",
            "blocks.0.attn.hook_k",
            "blocks.0.attn.hook_v",
            "blocks.0.mlp.hook_pre",
            "blocks.0.hook_resid_pre",
        ]
        for hook_name in expected_hooks:
            self.assertIn(hook_name, hook_points)

    def test_wrap_model_with_attention_norms(self):
        """Ensure wrapping works when attention uses per-head norms (e.g., Qwen3)."""

        class NormedAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 2
                self.n_kv_heads = 2
                self.scale = 0.125
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)
                self.v_proj = nn.Linear(64, 64, bias=False)
                self.o_proj = nn.Linear(64, 64, bias=False)
                self.rope = nn.RoPE(32, traditional=False)
                self.q_norm = nn.RMSNorm(32)
                self.k_norm = nn.RMSNorm(32)

            def __call__(self, x, mask=None, cache=None):
                B, L, _ = x.shape
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = self.q_norm(q.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
                k = self.k_norm(k.reshape(B, L, self.n_kv_heads, -1)).transpose(
                    0, 2, 1, 3
                )
                v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                q = self.rope(q)
                k = self.rope(k)

                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

        class NormedMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 128, bias=False)
                self.up_proj = nn.Linear(64, 128, bias=False)
                self.down_proj = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

        class NormedBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = NormedAttention()
                self.mlp = NormedMLP()
                self.input_layernorm = nn.RMSNorm(64)
                self.post_attention_layernorm = nn.RMSNorm(64)

            def __call__(self, x, mask=None, cache=None):
                h = self.self_attn(self.input_layernorm(x), mask, cache) + x
                return self.mlp(self.post_attention_layernorm(h)) + h

        class NormedCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [NormedBlock()]

        class NormedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = NormedCore()

            def __call__(self, x, cache=None):
                for layer in self.model.layers:
                    x = layer(x, cache=cache)
                return x

        mx.random.seed(0)
        reference_model = NormedModel()
        mx.random.seed(0)
        model_to_wrap = NormedModel()
        inputs = mx.random.normal((1, 4, 64))

        expected = reference_model(inputs)
        hooked_model = wrap_model_with_hooks(model_to_wrap, use_manual_sdpa=False)
        actual = hooked_model(inputs)

        self.assertTrue(mx.allclose(expected, actual))

    def test_run_with_cache(self):
        """Test run_with_cache functionality."""
        # Create a minimal model (reusing from above)
        class SimpleAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 2
                self.n_kv_heads = 2
                self.scale = 0.125
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)
                self.v_proj = nn.Linear(64, 64, bias=False)
                self.o_proj = nn.Linear(64, 64, bias=False)
                self.rope = nn.RoPE(32, traditional=False)

            def __call__(self, x, mask=None, cache=None):
                B, L, D = x.shape
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = q.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                k = k.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                v = v.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                q = self.rope(q)
                k = self.rope(k)

                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scale, mask=mask
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.o_proj(output)

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 128, bias=False)
                self.up_proj = nn.Linear(64, 128, bias=False)
                self.down_proj = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

        class SimpleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SimpleAttention()
                self.mlp = SimpleMLP()
                self.input_layernorm = nn.RMSNorm(64)
                self.post_attention_layernorm = nn.RMSNorm(64)

            def __call__(self, x, mask=None, cache=None):
                r = self.self_attn(self.input_layernorm(x), mask, cache)
                h = x + r
                r = self.mlp(self.post_attention_layernorm(h))
                return h + r

        class SimpleModelCore(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [SimpleBlock()]

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = SimpleModelCore()

            def __call__(self, x, cache=None):
                for layer in self.model.layers:
                    x = layer(x, cache=cache)
                return x

        model = SimpleModel()
        hooked_model = wrap_model_with_hooks(model, use_manual_sdpa=False)

        # Test run_with_cache
        x = mx.random.normal((1, 10, 64))
        output, cache = hooked_model.run_with_cache(x)

        # Check that cache contains activations
        self.assertGreater(len(cache), 0)
        self.assertIn("blocks.0.attn.hook_q", cache)
        self.assertIsInstance(cache["blocks.0.attn.hook_q"], mx.array)


class TestHybridAttentionWrapping(unittest.TestCase):
    """Tests for wrapping models with hybrid attention (e.g., Qwen3.5)."""

    def _make_hybrid_model(self):
        """Create a minimal model with both linear_attn and self_attn layers."""

        class SimpleLinearAttn(nn.Module):
            """Minimal stand-in for GatedDeltaNet."""
            def __init__(self):
                super().__init__()
                self.num_v_heads = 4
                self.num_k_heads = 2
                self.head_k_dim = 16
                self.head_v_dim = 16
                self.key_dim = self.head_k_dim * self.num_k_heads
                self.value_dim = self.head_v_dim * self.num_v_heads
                self.conv_dim = self.key_dim * 2 + self.value_dim
                self.conv_kernel_size = 4
                self.sharding_group = None

                self.in_proj_qkv = nn.Linear(64, self.key_dim * 2 + self.value_dim, bias=False)
                self.in_proj_z = nn.Linear(64, self.value_dim, bias=False)
                self.in_proj_b = nn.Linear(64, self.num_v_heads, bias=False)
                self.in_proj_a = nn.Linear(64, self.num_v_heads, bias=False)

                self.conv1d = nn.Conv1d(
                    in_channels=self.conv_dim,
                    out_channels=self.conv_dim,
                    bias=False,
                    kernel_size=self.conv_kernel_size,
                    groups=self.conv_dim,
                    padding=0,
                )
                self.dt_bias = mx.ones(self.num_v_heads)
                self.A_log = mx.log(mx.random.uniform(low=0, high=16, shape=(self.num_v_heads,)))

                from mlx_lm.models.qwen3_next import Qwen3NextRMSNormGated
                self.norm = Qwen3NextRMSNormGated(self.head_v_dim, eps=1e-6)
                self.out_proj = nn.Linear(self.value_dim, 64, bias=False)

            def __call__(self, inputs, mask=None, cache=None):
                B, S, _ = inputs.shape
                qkv = self.in_proj_qkv(inputs)
                z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
                b = self.in_proj_b(inputs)
                a = self.in_proj_a(inputs)

                conv_state = mx.zeros(
                    (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
                )
                conv_input = mx.concatenate([conv_state, qkv], axis=1)
                conv_out = nn.silu(self.conv1d(conv_input))

                q, k, v = [
                    t.reshape(B, S, h, d)
                    for t, h, d in zip(
                        mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                        [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                        [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                    )
                ]

                inv_scale = k.shape[-1] ** -0.5
                q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
                k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

                from mlx_lm.models.gated_delta import gated_delta_update
                out, _ = gated_delta_update(
                    q, k, v, a, b, self.A_log, self.dt_bias, None, None,
                    use_kernel=False,
                )

                out = self.norm(out, z)
                return self.out_proj(out.reshape(B, S, -1))

        class SimpleSoftmaxAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_heads = 2
                self.n_kv_heads = 2
                self.scale = 0.125
                self.q_proj = nn.Linear(64, 64, bias=False)
                self.k_proj = nn.Linear(64, 64, bias=False)
                self.v_proj = nn.Linear(64, 64, bias=False)
                self.o_proj = nn.Linear(64, 64, bias=False)
                self.rope = nn.RoPE(32, traditional=False)

            def __call__(self, x, mask=None, cache=None):
                B, L, D = x.shape
                q = self.q_proj(x).reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                k = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                v = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                q = self.rope(q)
                k = self.rope(k)
                output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
                return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))

        class SimpleMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(64, 128, bias=False)
                self.up_proj = nn.Linear(64, 128, bias=False)
                self.down_proj = nn.Linear(128, 64, bias=False)

            def __call__(self, x):
                return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))

        class LinearBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_linear = True
                self.linear_attn = SimpleLinearAttn()
                self.mlp = SimpleMLP()
                self.input_layernorm = nn.RMSNorm(64)
                self.post_attention_layernorm = nn.RMSNorm(64)

            def __call__(self, x, mask=None, cache=None):
                r = self.linear_attn(self.input_layernorm(x), mask, cache)
                h = x + r
                return h + self.mlp(self.post_attention_layernorm(h))

        class SoftmaxBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_linear = False
                self.self_attn = SimpleSoftmaxAttn()
                self.mlp = SimpleMLP()
                self.input_layernorm = nn.RMSNorm(64)
                self.post_attention_layernorm = nn.RMSNorm(64)

            def __call__(self, x, mask=None, cache=None):
                r = self.self_attn(self.input_layernorm(x), mask, cache)
                h = x + r
                return h + self.mlp(self.post_attention_layernorm(h))

        class HybridCore(nn.Module):
            def __init__(self):
                super().__init__()
                # Pattern: 3 linear + 1 softmax (like Qwen3.5 with interval=4)
                self.layers = [
                    LinearBlock(),
                    LinearBlock(),
                    LinearBlock(),
                    SoftmaxBlock(),
                ]

        class HybridModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = HybridCore()

            def __call__(self, x, cache=None):
                for layer in self.model.layers:
                    x = layer(x, cache=cache)
                return x

        return HybridModel()

    def test_wrap_hybrid_model(self):
        """Test that wrapping detects both linear_attn and self_attn layers."""
        model = self._make_hybrid_model()
        hooked_model = wrap_model_with_hooks(model, use_manual_sdpa=False)

        hook_points = hooked_model.hook_points()

        # Linear attention layers (0, 1, 2) should have linear_attn hooks
        for i in range(3):
            self.assertIn(f"blocks.{i}.linear_attn.hook_qkv", hook_points)
            self.assertIn(f"blocks.{i}.linear_attn.hook_conv_out", hook_points)
            self.assertIn(f"blocks.{i}.linear_attn.hook_recurrence_out", hook_points)
            self.assertIn(f"blocks.{i}.linear_attn.hook_z", hook_points)

        # Softmax attention layer (3) should have standard attn hooks
        self.assertIn("blocks.3.attn.hook_q", hook_points)
        self.assertIn("blocks.3.attn.hook_k", hook_points)
        self.assertIn("blocks.3.attn.hook_v", hook_points)

        # All layers should have block and MLP hooks
        for i in range(4):
            self.assertIn(f"blocks.{i}.hook_resid_pre", hook_points)
            self.assertIn(f"blocks.{i}.hook_resid_mid", hook_points)
            self.assertIn(f"blocks.{i}.hook_resid_post", hook_points)
            self.assertIn(f"blocks.{i}.mlp.hook_pre", hook_points)

    def test_hybrid_model_forward_pass(self):
        """Test that hooked hybrid model produces same output as original."""
        mx.random.seed(42)
        reference = self._make_hybrid_model()
        mx.random.seed(42)
        model = self._make_hybrid_model()

        x = mx.random.normal((1, 4, 64))
        expected = reference(x)

        hooked = wrap_model_with_hooks(model, use_manual_sdpa=False)
        actual = hooked(x)

        self.assertTrue(mx.allclose(expected, actual, atol=1e-5))

    def test_hybrid_run_with_cache(self):
        """Test run_with_cache captures activations from both layer types."""
        mx.random.seed(42)
        model = self._make_hybrid_model()
        hooked = wrap_model_with_hooks(model, use_manual_sdpa=False)

        x = mx.random.normal((1, 4, 64))
        output, cache = hooked.run_with_cache(x)

        # Linear layer activations should be cached
        self.assertIn("blocks.0.linear_attn.hook_qkv", cache)
        self.assertIn("blocks.0.linear_attn.hook_recurrence_out", cache)

        # Softmax layer activations should be cached
        self.assertIn("blocks.3.attn.hook_q", cache)

        # Residual stream should be cached for all layers
        self.assertIn("blocks.0.hook_resid_pre", cache)
        self.assertIn("blocks.3.hook_resid_post", cache)


class TestManualSDPA(unittest.TestCase):
    """Tests for the manual scaled dot-product attention implementation."""

    def test_matches_fused_kernel_with_causal_mask(self):
        B, n_heads, n_kv_heads, seq_len, head_dim = 1, 4, 2, 5, 8
        queries = mx.random.normal((B, n_heads, seq_len, head_dim))
        keys = mx.random.normal((B, n_kv_heads, seq_len + 2, head_dim))
        values = mx.random.normal((B, n_kv_heads, seq_len + 2, head_dim))
        scale = head_dim ** -0.5
        mask = "causal"

        manual_output, attn_pattern = manual_scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
        fused_output = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=scale, mask=mask
        )

        self.assertTrue(mx.allclose(manual_output, fused_output, atol=1e-5))
        self.assertIsNone(attn_pattern)

    def test_supports_boolean_mask(self):
        B, n_heads, seq_len, head_dim = 1, 2, 4, 4
        queries = mx.random.normal((B, n_heads, seq_len, head_dim))
        keys = mx.random.normal((B, n_heads, seq_len, head_dim))
        values = mx.random.normal((B, n_heads, seq_len, head_dim))
        scale = head_dim ** -0.5
        mask = mx.tril(mx.ones((seq_len, seq_len), dtype=mx.bool_))

        output, attn_pattern = manual_scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )

        self.assertEqual(output.shape, queries.shape)
        self.assertIsNone(attn_pattern)


if __name__ == '__main__':
    unittest.main()
