"""Report pipeline steps from a worker thread without blocking the event loop."""

import asyncio
from collections.abc import Callable

from app.assistant.deps import StatusReport


def threadsafe_report(report: StatusReport) -> Callable[[str, str], None]:
    """Return a sync callback for code running in ``asyncio.to_thread``.

    The event loop must stay free to run ``report``. Calling this callback on
    the loop thread waits forever, because the loop cannot service itself.
    """
    loop = asyncio.get_running_loop()

    def progress(phase: str, detail: str) -> None:
        asyncio.run_coroutine_threadsafe(report(phase, detail), loop).result()

    return progress
