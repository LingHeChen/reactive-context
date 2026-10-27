"""Async bounded execution; no graph lock is held across a plugin await."""

import asyncio
import inspect


async def invoke(hook, *args):
    result = hook(*args)
    return await result if inspect.isawaitable(result) else result


class Scheduler:
    def __init__(self, concurrency: int = 16):
        if concurrency < 1:
            raise ValueError("concurrency must be positive")
        self.semaphore = asyncio.Semaphore(concurrency)

    async def call(self, hook, deadline_seconds, *args):
        async with asyncio.timeout(deadline_seconds):
            async with self.semaphore:
                return await invoke(hook, *args)
