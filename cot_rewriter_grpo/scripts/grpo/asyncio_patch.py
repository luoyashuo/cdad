"""
Asyncio compatibility patch for Python 3.10+

This patch fixes the issue where asyncio.get_event_loop() raises RuntimeError
when there is no running event loop in the current thread.

This is required for SGLang support in VERL on Python 3.10+.

Usage:
    import asyncio_patch  # Import before any other imports
"""

import asyncio

_original_get_event_loop = asyncio.get_event_loop


def _patched_get_event_loop():
    """
    Get the event loop for the current thread.

    If there is no running event loop, create a new one and set it as the current loop.
    This is compatible with Python 3.10+ behavior.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, check if there's one set for the thread
        try:
            loop = _original_get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
            return loop
        except RuntimeError:
            # No event loop, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop


# Apply the patch
asyncio.get_event_loop = _patched_get_event_loop

print("[asyncio_patch] Applied asyncio.get_event_loop() compatibility patch for Python 3.10+")
