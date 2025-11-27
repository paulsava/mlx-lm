"""Utilities for wrapping MLX-LM models with activation hooks."""

from .wrapper import wrap_model_with_hooks, get_hook_schema
from .hook_point import HookPoint
from .hooked_module import HookedRootModule

__version__ = "0.1.0"

__all__ = [
    'wrap_model_with_hooks',
    'get_hook_schema',
    'HookPoint',
    'HookedRootModule',
]
