"""Idle-stream timeout for the Claude backend.

A stalled API response (an ESTABLISHED-but-silent connection that streams
nothing and never errors) must become a transient TimeoutError so the existing
retry_on_transient retries it — rather than hanging the run forever. The timeout
is an IDLE timeout (reset per message) and CONSERVATIVE, so a legitimately long
single-message generation is never false-positived.
"""
from __future__ import annotations

import asyncio

import pytest

from f3dasm._src.agentic.backends.claude import (
    _anext_or_done,
    _STREAM_DONE,
    _stream_with_idle_timeout,
)


class _CleanGen:
    """Yields n messages then stops — a normal stream."""

    def __init__(self, n: int) -> None:
        self._n, self._i = n, 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= self._n:
            raise StopAsyncIteration
        self._i += 1
        return f"m{self._i}"


class _StallGen:
    """Yields one message, then stalls forever (the hang we observed)."""

    def __init__(self) -> None:
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._sent:
            self._sent = True
            return "first"
        await asyncio.sleep(3600)  # silent ESTABLISHED stream


def test_clean_stream_completes():
    async def run():
        return [m async for m in _stream_with_idle_timeout(_CleanGen(3), 5.0)]
    assert asyncio.run(run()) == ["m1", "m2", "m3"]


def test_stall_raises_transient_timeout():
    async def run():
        got = []
        with pytest.raises(TimeoutError):
            async for m in _stream_with_idle_timeout(_StallGen(), 0.2):
                got.append(m)
        return got
    got = asyncio.run(run())
    assert got == ["first"]  # delivered what arrived, then tripped on the stall


def test_timeouterror_is_transient_so_retry_catches_it():
    # The whole point: the raised error must be retryable by retry_on_transient.
    from f3dasm._src.agentic.backends.base import is_transient_error
    assert is_transient_error(TimeoutError("Anthropic stream stalled"))


def test_anext_or_done_sentinel_on_exhaustion():
    async def run():
        return await _anext_or_done(_CleanGen(0))
    assert asyncio.run(run()) is _STREAM_DONE
