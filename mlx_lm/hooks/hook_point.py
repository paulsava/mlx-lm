import mlx.core as mx
import mlx.nn as nn
from typing import Callable, Optional, Dict, Any, List


class HookPoint(nn.Module):
    """Wrap a module and run hook functions on its output.

    Hooks run sequentially after the wrapped module executes. If no base module
    is provided the hook operates directly on the passed activation.
    """

    def __init__(self, name: str, base_module: Optional[nn.Module] = None):
        super().__init__()
        self.name = name
        self.base = base_module
        self.hooks: List[Callable] = []

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        """Apply the wrapped module (if present) and then run hooks."""
        if self.base is not None:
            output = self.base(x, *args, **kwargs)
        else:
            output = x

        context = {
            'cache': kwargs.get('cache'),
            'mask': kwargs.get('mask'),
        }

        for hook_fn in self.hooks:
            output = hook_fn(self.name, output, context)

        return output

    def add_hook(self, hook_fn: Callable[[str, mx.array, Dict[str, Any]], mx.array]):
        """Register a hook function with signature (name, activation, context)."""
        self.hooks.append(hook_fn)

    def clear_hooks(self):
        """Remove every registered hook."""
        self.hooks.clear()

    def __repr__(self) -> str:
        return f"HookPoint(name='{self.name}', num_hooks={len(self.hooks)})"
