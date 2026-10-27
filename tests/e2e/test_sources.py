import asyncio
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from reactive_context import Context
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
from reactive_context.errors import AccessRevoked, SourceUnavailable
from reactive_context.sources.file import FileSource
from reactive_context.sources.skiva import ManifestFile, SkivaSkillSource, canonical_files


def fixture_manifest(version, body):
    entry = ManifestFile(
        path="sql-query/SKILL.md",
        type="file",
        size=len(body),
        digest="sha256:" + sha256(body).hexdigest(),
        mode=292,
    )
    return {
        "schemaVersion": 1,
        "snapshotId": version,
        "manifestDigest": "sha256:" + sha256(canonical_files([entry])).hexdigest(),
        "generation": 1,
        "profileId": "profile-1",
        "files": [entry.model_dump()],
    }


class SkivaServer:
    def __init__(self):
        self.version = "A"
        self.body = b"RULE_A_ONLY"
        self.revoked = False
        self.remove_skill = False
        self.corrupt_blob = False
        self.corrupt_manifest = False
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        assert request.headers["Authorization"] == "Bearer sensitive-credential"
        if self.revoked:
            return httpx.Response(403)
        if request.url.path.endswith("/profile"):
            return httpx.Response(200, json={"activeSnapshot": self.version})
        if request.url.path.endswith("/manifest"):
            manifest = fixture_manifest(self.version, self.body)
            if self.remove_skill:
                manifest["files"] = []
                manifest["manifestDigest"] = "sha256:" + sha256(b"[]").hexdigest()
            if self.corrupt_manifest:
                manifest["manifestDigest"] = "sha256:" + "0" * 64
            return httpx.Response(200, json=manifest)
        assert request.url.params["snapshotId"] == self.version
        return httpx.Response(200, content=b"corrupt" if self.corrupt_blob else self.body)


def make_source(server, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    source = SkivaSkillSource(
        "skill",
        base_url="https://skiva.test",
        token="sensitive-credential",
        skill_path="sql-query/SKILL.md",
        client=client,
        allow_private=True,
        **kwargs,
    )
    return source, client


async def test_skiva_a_to_b_during_actual_agent_run_without_read_tool():
    server = SkivaServer()
    source, client = make_source(server)
    observed = []
    async with client, Context([source.part]) as context:

        def respond(messages, info):
            observed.append(deepcopy(messages))
            if len(observed) == 1:
                assert "RULE_A_ONLY" in str(messages)
                return ModelResponse(parts=[ToolCallPart("publish", {}, tool_call_id="publish-1")])
            assert "RULE_B_ONLY" in str(messages)
            assert "RULE_A_ONLY" not in str(messages)
            return ModelResponse(parts=[TextPart("updated")])

        agent = Agent(
            FunctionModel(respond),
            name="skiva_update",
            capabilities=[ReactiveContextCapability(context)],
        )

        @agent.tool_plain
        def publish() -> str:
            server.version, server.body = "B", b"RULE_B_ONLY"
            return "release changed externally"

        result = await agent.run("Use the current rules")
        assert result.output == "updated"
        assert "RULE_A_ONLY" not in str(result.all_messages())
        assert len(observed) == 2
        assert "sensitive-credential" not in context.export_trace()


@pytest.mark.parametrize("failure", ["corrupt_blob", "corrupt_manifest"])
async def test_skiva_integrity_failure(failure):
    server = SkivaServer()
    setattr(server, failure, True)
    source, client = make_source(server)
    async with client, Context([source.part]) as context:
        with pytest.raises(SourceUnavailable):
            await context.next_context()


@pytest.mark.parametrize("failure", ["revoked", "remove_skill"])
async def test_skiva_pinned_revocation(failure):
    server = SkivaServer()
    source, client = make_source(server)
    async with client, Context([source.part]) as context:
        snapshot = await context.next_context(consistency="pinned")
        setattr(server, failure, True)
        with pytest.raises(AccessRevoked):
            await context.next_context(consistency="pinned", on_unavailable="cached")
        with pytest.raises(AccessRevoked):
            context.compile(snapshot)


async def test_skiva_polling_marks_dirty_and_unsubscribes():
    server = SkivaServer()
    source, client = make_source(server, poll_interval=0.005)
    async with client, Context([source.part]) as context:
        await context.next_context()
        await asyncio.sleep(0.015)
        server.version, server.body = "B", b"RULE_B_ONLY"
        for _ in range(100):
            if context.inspect("skill")["state"] == "DIRTY":
                break
            await asyncio.sleep(0.005)
        assert context.inspect("skill")["state"] == "DIRTY"
        assert (await context.next_context()).part("skill").value == "RULE_B_ONLY"
        await context.close_scope()
        assert not any(
            t.get_name().startswith("skiva-watch:") for t in asyncio.all_tasks() if not t.done()
        )


async def test_file_update_poll_frozen_inputs_and_unsubscribe(tmp_path):
    path = tmp_path / "rules.txt"
    path.write_text("FILE_A_ONLY")
    source = FileSource("file", path, root=tmp_path, poll_interval=0.005)
    async with Context([source.part]) as context:
        first = await context.next_context()
        compiled = context.compile(first)
        await asyncio.sleep(0.015)
        path.write_text("FILE_B_ONLY")
        for _ in range(100):
            if context.inspect("file")["state"] == "DIRTY":
                break
            await asyncio.sleep(0.005)
        second = await context.next_context()
        assert first.part("file").value == "FILE_A_ONLY"
        assert second.part("file").value == "FILE_B_ONLY"
        assert "FILE_B_ONLY" not in str(compiled)
        assert "FILE_A_ONLY" not in str(context.compile(second))
    assert not any(
        t.get_name().startswith("file-watch:") for t in asyncio.all_tasks() if not t.done()
    )


async def test_file_paths_symlink_and_size_limits(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-secret")
    outside.write_text("sensitive")
    with pytest.raises(AccessRevoked):
        FileSource("bad", "../secret", root=tmp_path)
    link = tmp_path / "link"
    link.symlink_to(outside)
    async with Context([FileSource("link", link, root=tmp_path).part]) as context:
        with pytest.raises(AccessRevoked):
            await context.next_context()
    (tmp_path / "big").write_text("x" * 100)
    async with Context([FileSource("big", "big", root=tmp_path, max_bytes=10).part]) as context:
        with pytest.raises(SourceUnavailable):
            await context.next_context()
    outside.unlink()


@pytest.mark.parametrize(
    "url", ["http://skiva.test", "file:///etc/passwd", "https://token:secret@skiva.test"]
)
def test_skiva_url_safety(url):
    with pytest.raises(ValueError):
        SkivaSkillSource("skill", base_url=url, token="x", skill_path="sql/SKILL.md")


async def test_private_network_rejected_before_http():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: (_ for _ in ()).throw(AssertionError("must not reach HTTP"))
        )
    ) as client:
        source = SkivaSkillSource(
            "skill",
            base_url="https://127.0.0.1",
            token="x",
            skill_path="sql/SKILL.md",
            client=client,
        )
        async with Context([source.part]) as context:
            with pytest.raises(AccessRevoked):
                await context.next_context()


def test_skiva_canonical_matches_actual_repository_fixture():
    fixture_path = Path(__file__).parents[1] / "fixtures/skiva-manifest-v1.json"
    fixture = json.loads(fixture_path.read_text())
    files = [ManifestFile.model_validate(f) for f in fixture["files"]]
    assert "sha256:" + sha256(canonical_files(files)).hexdigest() == fixture["manifestDigest"]
