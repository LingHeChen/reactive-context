"""Skiva HTTP-protocol A→B fixture. Replace transport/credentials for a real deployment."""

import asyncio
from hashlib import sha256

import httpx

from reactive_context import Context
from reactive_context.sources.skiva import ManifestFile, SkivaSkillSource, canonical_files


async def main():
    release = {"id": "A", "body": b"SELECT legacy_column FROM records;"}

    def serve(request):
        if request.url.path.endswith("/profile"):
            return httpx.Response(200, json={"activeSnapshot": release["id"]})
        if request.url.path.endswith("/manifest"):
            entry = ManifestFile(
                path="sql-query/SKILL.md",
                type="file",
                size=len(release["body"]),
                digest="sha256:" + sha256(release["body"]).hexdigest(),
                mode=292,
            )
            return httpx.Response(
                200,
                json={
                    "schemaVersion": 1,
                    "snapshotId": release["id"],
                    "profileId": "demo",
                    "generation": 1,
                    "files": [entry.model_dump()],
                    "manifestDigest": "sha256:" + sha256(canonical_files([entry])).hexdigest(),
                },
            )
        return httpx.Response(200, content=release["body"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        source = SkivaSkillSource(
            "sql",
            base_url="https://skiva.example",
            token="fixture-only",
            skill_path="sql-query/SKILL.md",
            client=client,
            allow_private=True,
        )
        async with source, Context([source.part]) as context:
            first = await context.next_context("session")
            release.update(id="B", body=b"SELECT current_column FROM records;")
            second = await context.next_context("session")
            assert "legacy_column" not in str(context.compile(second))
            assert "current_column" in str(context.compile(second))
            print("A:", first.part("sql").revision)
            print("B:", second.part("sql").revision)
            print(second.part("sql").value)


if __name__ == "__main__":
    asyncio.run(main())
