from .attention import inject_attention_hooks
from .mlp import inject_mlp_hooks
from .block import inject_block_hooks

__all__ = [
    'inject_attention_hooks',
    'inject_mlp_hooks',
    'inject_block_hooks',
]
