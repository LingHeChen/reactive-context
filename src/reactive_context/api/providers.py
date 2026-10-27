"""Host-bound file provider. Declaration and construction perform no filesystem IO."""

import asyncio
from pathlib import Path

from ..core.snapshot import digest
from ..errors import AccessRevoked
from .protocol import PartMaterialization


class FilePartProvider:
    """Expose only a host-selected directory; resource is a relative UTF-8 file name.

    The host owns this directory. This is not a sandbox for hostile concurrent filesystem writers.
    Content polling detects missed notifications. Skills still need a separate compiler grant.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()

    def _read(self, resource):
        root = self.root.resolve()
        path = (root / resource).resolve()
        if not path.is_relative_to(root):
            raise AccessRevoked(resource, "resource outside provider root")
        return path.read_text(encoding="utf-8")

    async def revision(self, reader, resource, **options):
        return digest(await asyncio.to_thread(self._read, resource))

    async def resolve(self, reader, resource, **options):
        value = await asyncio.to_thread(self._read, resource)
        return PartMaterialization(value, sources=(f"file:{resource}",))
