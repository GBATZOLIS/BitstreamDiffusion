# callbacks/base.py
from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar
from functools import wraps

F = TypeVar("F", bound=Callable[..., Any])


def maybe_disable_dynamo(fn: Optional[F] = None) -> Callable[[F], F] | F:
    """
    Returns torch._dynamo.disable(fn) when available, otherwise returns fn.

    Safe as a decorator in both forms:
        @maybe_disable_dynamo
        def f(...): ...

        @maybe_disable_dynamo()
        def g(...): ...
    """
    try:
        import torch._dynamo as dynamo  # type: ignore
        disable = dynamo.disable
    except Exception:
        disable = None  # type: ignore[assignment]

    def _decorator(f: F) -> F:
        if disable is None:
            return f

        disabled_f = disable(f)

        # Preserve metadata even if dynamo returns a wrapped function.
        @wraps(f)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return disabled_f(*args, **kwargs)

        return _wrapped  # type: ignore[return-value]

    if fn is None:
        return _decorator
    return _decorator(fn)


class Callback:
    run_on_all_ranks: bool = False

    def on_train_begin(self, trainer: Any):
        ...

    def on_epoch_end(self, trainer: Any, epoch: int):
        ...
