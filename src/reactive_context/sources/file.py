"""Read-only file source: openat + O_NOFOLLOW on every path component."""

import asyncio
import os
import stat
from hashlib import sha256
from pathlib import Path

from ..core.part import ChangeSignal, source_part
from ..errors import AccessRevoked, SourceUnavailable


class FileSource:
    def __init__(self, key, path, *, root, max_bytes=1_048_576, poll_interval=None, **kwargs):
        self.key = key
        self.root = Path(root).absolute()
        path = Path(path)
        self.path = path if path.is_absolute() else self.root / path
        try:
            self.relative = self.path.relative_to(self.root)
        except ValueError:
            raise AccessRevoked(key, "file outside allowed root") from None
        if not self.relative.parts or any(p in {".", ".."} for p in self.relative.parts):
            raise AccessRevoked(key, "unsafe path")
        if max_bytes < 1 or (poll_interval is not None and poll_interval <= 0):
            raise ValueError("invalid file size/poll interval")
        self.max_bytes = max_bytes
        self.poll_interval = poll_interval
        self.part = source_part(
            key=key,
            schema=str,
            resolve=self.resolve,
            revision=self.revision,
            subscribe=self.subscribe if poll_interval else None,
            resource_key=self.path.as_uri(),
            **kwargs,
        )

    def _read(self):
        # Root is an explicitly trusted host directory. Descendant symlinks are forbidden.
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for component in self.relative.parts[:-1]:
                next_dir = os.open(
                    component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
                )
                os.close(directory)
                directory = next_dir
            fd = os.open(
                self.relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
            )
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise AccessRevoked(self.key, "not a regular file")
                with os.fdopen(fd, "rb", closefd=False) as stream:
                    data = stream.read(self.max_bytes + 1)
                if len(data) > self.max_bytes:
                    raise SourceUnavailable(self.key, "file exceeds limit")
                return data
            finally:
                os.close(fd)
        except OSError:
            raise AccessRevoked(self.key, "file unavailable or unsafe path") from None
        finally:
            os.close(directory)

    async def revision(self, ctx):
        return sha256(await asyncio.to_thread(self._read)).hexdigest()

    async def resolve(self, ctx):
        return (await asyncio.to_thread(self._read)).decode("utf-8")

    def subscribe(self, ctx, notify):
        async def watch():
            last = None
            while True:
                try:
                    current = await self.revision(ctx)
                except Exception:
                    current = "unavailable"
                if last is not None and current != last:
                    notify(ChangeSignal(self.key))
                last = current
                await asyncio.sleep(self.poll_interval or 1.0)

        task = asyncio.create_task(watch(), name=f"file-watch:{self.key}")

        async def unsubscribe():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return unsubscribe
