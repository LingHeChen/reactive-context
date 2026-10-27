"""Read-only Skiva distribution/v1 connector; no backend database coupling."""

import asyncio
import ipaddress
import json
import re
import socket
import unicodedata
from hashlib import sha256
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ..core.part import ChangeSignal, source_part
from ..errors import AccessRevoked, SourceUnavailable


class ManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    type: str
    size: int = Field(ge=0)
    digest: str
    mode: int


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schemaVersion: int
    snapshotId: str
    manifestDigest: str
    generation: int
    profileId: str
    files: list[ManifestFile]


def canonical_files(files):
    # Match Go encoding/json's HTML and U+2028/2029 escaping exactly.
    text = json.dumps(
        [f.model_dump() for f in sorted(files, key=lambda f: f.path)],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    for before, after in [
        ("&", r"\u0026"),
        ("<", r"\u003c"),
        (">", r"\u003e"),
        (" ", r"\u2028"),
        (" ", r"\u2029"),
    ]:
        text = text.replace(before, after)
    return text.encode()


class SkivaSkillSource:
    def __init__(
        self,
        key,
        *,
        base_url,
        token,
        skill_path,
        client=None,
        allow_private=False,
        max_bytes=1_048_576,
        poll_interval=None,
        **kwargs,
    ):
        url = urlsplit(base_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("Skiva requires a credential-free HTTPS base URL")
        if not self._safe_path(skill_path):
            raise ValueError("unsafe skill path")
        if max_bytes < 1 or (poll_interval is not None and poll_interval <= 0):
            raise ValueError("invalid size or polling interval")
        self.key, self.path = key, skill_path
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.allow_private = allow_private
        self.max_bytes, self.poll_interval = max_bytes, poll_interval
        self._client = client or httpx.AsyncClient(timeout=10, follow_redirects=False)
        self._owned_client = client is None
        self.part = source_part(
            key=key,
            schema=str,
            resolve=self.resolve,
            revision=self.revision,
            acl_check=self.acl_check,
            subscribe=self.subscribe if poll_interval else None,
            resource_key=f"skiva://{url.hostname}/{skill_path}",
            **kwargs,
        )

    @staticmethod
    def _safe_path(path):
        return (
            bool(path)
            and not path.startswith("/")
            and "\\" not in path
            and not any(c in path for c in "\x00\r\n")
            and str(PurePosixPath(path)) == path
            and ".." not in PurePosixPath(path).parts
            and unicodedata.normalize("NFC", path) == path
        )

    async def _get(self, endpoint, *, params=None):
        host = urlsplit(self.base_url).hostname
        if not self.allow_private:
            addresses = await asyncio.to_thread(socket.getaddrinfo, host, 443)
            if not addresses or any(
                not ipaddress.ip_address(item[4][0]).is_global for item in addresses
            ):
                raise AccessRevoked(self.key, "private network target forbidden")
        # Fixed base and endpoints; no server-supplied URLs and no redirects.
        async with self._client.stream(
            "GET",
            self.base_url + endpoint,
            params=params,
            headers={"Authorization": f"Bearer {self._token}"},
            follow_redirects=False,
        ) as response:
            if response.status_code in {401, 403}:
                raise AccessRevoked(self.key)
            if response.status_code != 200:
                raise SourceUnavailable(self.key)
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > self.max_bytes:
                    raise SourceUnavailable(self.key, "response exceeds limit")
            return bytes(data)

    async def acl_check(self, ctx):
        # Auth is rechecked for pinned and cached scopes too.
        await self._get("/distribution/v1/profile")
        manifest = await self._manifest()
        if not any(entry.path == self.path for entry in manifest.files):
            raise AccessRevoked(self.key, "skill no longer authorized")
        return True

    async def _manifest(self):
        m = Manifest.model_validate_json(await self._get("/distribution/v1/manifest"))
        if m.schemaVersion != 1:
            raise SourceUnavailable(self.key, "unsupported manifest version")
        paths = set()
        for entry in sorted(m.files, key=lambda e: e.path):
            if (
                not self._safe_path(entry.path)
                or entry.path in paths
                or entry.type != "file"
                or entry.mode not in {292, 365}
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", entry.digest)
                or any(str(parent) in paths for parent in PurePosixPath(entry.path).parents)
            ):
                raise SourceUnavailable(self.key, "invalid manifest entry")
            paths.add(entry.path)
        actual = "sha256:" + sha256(canonical_files(m.files)).hexdigest()
        if actual != m.manifestDigest:
            raise SourceUnavailable(self.key, "manifest integrity mismatch")
        return m

    async def revision(self, ctx):
        manifest = await self._manifest()
        return f"{manifest.snapshotId}:{manifest.manifestDigest}"

    async def resolve(self, ctx):
        manifest = await self._manifest()
        revision = f"{manifest.snapshotId}:{manifest.manifestDigest}"
        if ctx.revision and revision != ctx.revision:
            # Core reconciles revisions before committing, never accepts this as current.
            from ..errors import ConsistencyUnavailable

            raise ConsistencyUnavailable(self.key)
        entry = next((f for f in manifest.files if f.path == self.path), None)
        if entry is None:
            raise AccessRevoked(self.key, "skill not in authorized manifest")
        data = await self._get(
            "/distribution/v1/objects/" + entry.digest, params={"snapshotId": manifest.snapshotId}
        )
        if len(data) != entry.size or "sha256:" + sha256(data).hexdigest() != entry.digest:
            raise SourceUnavailable(self.key, "object integrity mismatch")
        return data.decode("utf-8")

    def subscribe(self, ctx, notify):
        async def poll():
            previous = None
            while True:
                try:
                    revision = await self.revision(ctx)
                except AccessRevoked:
                    notify(ChangeSignal(self.key, revoked=True))
                    return
                except Exception:
                    revision = None
                if previous is not None and revision != previous:
                    notify(ChangeSignal(self.key))
                previous = revision
                await asyncio.sleep(self.poll_interval or 1.0)

        task = asyncio.create_task(poll(), name=f"skiva-watch:{self.key}")

        async def unsubscribe():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        return unsubscribe

    async def close(self):
        if self._owned_client:
            await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
