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
