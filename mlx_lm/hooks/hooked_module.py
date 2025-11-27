import mlx.core as mx
from typing import Callable, Dict, Any, List, Tuple
from .hook_point import HookPoint


class HookedRootModule:
    """Mixin that adds hook management utilities to a model."""

    def _init_hooks(self):
        """Initialise the hook registry on first use."""
        if not hasattr(self, '_hook_points'):
            self._hook_points: Dict[str, HookPoint] = {}

    def register_hook_point(self, name: str, hook_point: HookPoint):
        """Register a hook point so users can attach hooks later."""
        if not hasattr(self, '_hook_points'):
            self._init_hooks()
        self._hook_points[name] = hook_point

    def add_hook(
        self,
        name: str,
        hook_fn: Callable[[str, mx.array, Dict[str, Any]], mx.array]
    ):
        """Attach `hook_fn` to the named hook point."""
        if not hasattr(self, '_hook_points'):
            self._init_hooks()

        if name not in self._hook_points:
            raise ValueError(
                f"Hook point '{name}' not found. "
                f"Available hook points: {list(self._hook_points.keys())}"
            )
        self._hook_points[name].add_hook(hook_fn)

    def reset_hooks(self):
        """Remove every hook from each hook point."""
        if not hasattr(self, '_hook_points'):
            return

        for hp in self._hook_points.values():
            hp.clear_hooks()

    def run_with_cache(
        self,
        *args,
        **kwargs
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """Run the model once while caching the output of every hook point."""
        if not hasattr(self, '_hook_points'):
            self._init_hooks()

        cache_dict = {}

        def cache_hook(name: str, activation: mx.array, context: Dict[str, Any]) -> mx.array:
            """Store the activation and forward it unchanged."""
            cache_dict[name] = activation
            return activation

        # Add cache hook to all points
        for name in self._hook_points:
            self.add_hook(name, cache_hook)

        # Run model
        output = self(*args, **kwargs)

        # Clean up cache hooks
        self.reset_hooks()

        return output, cache_dict

    def hook_points(self) -> List[str]:
        """Return the sorted list of available hook point names."""
        if not hasattr(self, '_hook_points'):
            self._init_hooks()

        return sorted(self._hook_points.keys())

    def get_hook_point(self, name: str) -> HookPoint:
        """Return the HookPoint identified by `name`."""
        if not hasattr(self, '_hook_points'):
            self._init_hooks()

        if name not in self._hook_points:
            raise ValueError(
                f"Hook point '{name}' not found. "
                f"Available hook points: {list(self._hook_points.keys())}"
            )
        return self._hook_points[name]
