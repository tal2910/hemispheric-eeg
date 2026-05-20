"""Timing decorator for instrumenting hot paths.

The @timed decorator logs how long the decorated function takes to execute.
It transparently handles both sync and async functions (detects coroutines and
returns an async wrapper that awaits the wrapped call).

Use sparingly — applied to a per-chunk function it would log thousands of
times. Apply to top-level orchestration phases (load, plan, run) instead.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

log = logging.getLogger(__name__)


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    """Log how long the decorated function takes to execute.

    Works on both sync and async functions. Logs at INFO level with the
    function's qualified name and elapsed wall-clock time in milliseconds
    for sub-second runs, seconds otherwise.

    Example:
        @timed
        def load_all_visits(data_dir):
            ...

        @timed
        async def run_provider(shards, ports, ...):
            ...
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                _log_elapsed(func, time.perf_counter() - start)

        return async_wrapper

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            _log_elapsed(func, time.perf_counter() - start)

    return sync_wrapper


def _log_elapsed(func: Callable[..., Any], elapsed_s: float) -> None:
    """Format and log the elapsed time at an appropriate scale."""
    if elapsed_s < 1.0:
        log.info("%s took %.1f ms", func.__qualname__, elapsed_s * 1000)
    else:
        log.info("%s took %.2f s", func.__qualname__, elapsed_s)
