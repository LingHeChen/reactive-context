"""Protocol regressions only; these fixtures are never benchmark quality scores."""

import json

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("openai")

from pydantic_ai.messages import ModelResponse, TextPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402

from reactive_context.benchmarking.config import BenchmarkConfig  # noqa: E402
from reactive_context.benchmarking.contracts import (  # noqa: E402
    BenchmarkCase,
    InferenceCase,
)
from reactive_context.benchmarking.datasets import (  # noqa: E402
    LongMemEvalDataset,
    pilot_split,
)
from reactive_context.benchmarking.runtime import (  # noqa: E402
    BudgetExceeded,
    BudgetLedger,
    CallTrace,
    generate,
)


def test_public_input_excludes_answer_flags():
    dataset = object.__new__(LongMemEvalDataset)
    dataset._rows = [
        {
            "question_id": "q",
            "question": "Question",
            "question_date": "today",
            "question_type": "single-session-user",
            "answer": "SECRET_LABEL",
            "answer_session_ids": ["answer_1"],
            "haystack_session_ids": ["answer_1"],
            "haystack_dates": ["yesterday"],
            "haystack_sessions": [[{"role": "user", "content": "public", "has_answer": True}]],
        }
    ]
    case = next(dataset.cases())
    assert "has_answer" not in case.inference.documents[0].text
    assert "SECRET_LABEL" not in case.inference.model_dump_json()
    assert case.evaluator_labels["answer"] == "SECRET_LABEL"


def test_pilot_is_reproducible_and_disjoint():
    cases = [
        BenchmarkCase(
            InferenceCase(case_id=str(i), benchmark="fixture", question="q", documents=[]),
            {},
            subgroup=str(i % 3),
        )
        for i in range(25)
    ]
    first, heldout = pilot_split(cases, 10, 17)
    again, _ = pilot_split(cases, 10, 17)
    ids = lambda rows: {r.inference.case_id for r in rows}  # noqa: E731
    assert ids(first) == ids(again)
    assert not ids(first) & ids(heldout)
    assert len(first) + len(heldout) == 25


async def test_shared_budget_counts_requests_and_survives_restart(tmp_path):
    config = BenchmarkConfig(max_requests=1)
    path = tmp_path / "ledger.json"
    ledger = BudgetLedger(config, path)
    reservation = await ledger.reserve(100, 10)
    await ledger.settle(reservation, RequestUsage(input_tokens=100, output_tokens=10))
    restored = BudgetLedger(config, path)
    assert restored.spent > 0
    with pytest.raises(BudgetExceeded):
        await restored.reserve(100, 10)
    assert json.loads(path.read_text())["requests"] == 1


async def test_concurrent_reservations_cannot_exceed_shared_cap():
    import asyncio

    ledger = BudgetLedger(BenchmarkConfig(max_requests=2))
    results = await asyncio.gather(
        *(ledger.reserve(100, 10) for _ in range(10)), return_exceptions=True
    )
    assert sum(isinstance(result, float) for result in results) == 2
    assert sum(isinstance(result, BudgetExceeded) for result in results) == 8


async def test_unfinished_reservation_remains_charged_on_restart(tmp_path):
    path = tmp_path / "ledger.json"
    config = BenchmarkConfig()
    ledger = BudgetLedger(config, path)
    reservation = await ledger.reserve(100, 10)
    restored = BudgetLedger(config, path)
    assert restored.spent == pytest.approx(reservation)
    assert restored.requests == 1


async def test_reactive_bootstrap_matches_real_baseline_request():
    from reactive_context import CompilerPolicy, Context, ContextIRSegment
    from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
    from reactive_context.sources.config import ConfigSource

    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    config = BenchmarkConfig()
    ledger = BudgetLedger(config)
    baseline, reactive = CallTrace(), CallTrace()
    prompt = "The complete public benchmark question and evidence."
    await generate(model, ledger, baseline, "answer", prompt)

    def project(value, ctx):
        return [
            ContextIRSegment(
                "memory", "memory", ctx.revision, value, kind="history", requested_role="user"
            )
        ]

    source = ConfigSource("memory", prompt, project=project)
    async with Context([source.part], compiler_policy=CompilerPolicy(max_tokens=10000)) as context:
        await generate(
            model,
            ledger,
            reactive,
            "answer",
            capabilities=[ReactiveContextCapability(context, scope="fixture", cache=None)],
        )
    assert baseline.calls[0].input_digest == reactive.calls[0].input_digest


def test_scorer_rejects_silent_empty_gold(tmp_path):
    from reactive_context.benchmarking.official import CACHE, ContextBenchScorer

    if not (CACHE / "repos/contextbench").exists():
        pytest.skip("pinned public scorer is not downloaded")
    (tmp_path / "example.py").write_text("def example():\n    return 1\n")
    scorer = ContextBenchScorer()
    metrics = scorer.score(
        {
            "instance_id": "fixture",
            "gold_context": json.dumps([{"file": "example.py", "start_line": 1, "end_line": 2}]),
        },
        [{"source_uri": "example.py", "source_span": [1, 2]}],
        tmp_path,
    )
    assert metrics["file"]["gold_size"] == 1
    assert metrics["line"]["coverage"] == 1
    assert metrics["symbol"]["gold_size"] > 0


def test_secret_keys_are_not_in_public_config():
    config = BenchmarkConfig(api_key="SECRET", judge_key="SECRET")
    assert "SECRET" not in repr(config)
    assert "SECRET" not in json.dumps(config.public())


async def test_cache_experiment_namespace_precedes_every_skill():
    from reactive_context.benchmarking.cache_pilot import SkillPrefix

    observed = []

    def respond(messages, info):
        observed.extend(messages)
        return ModelResponse(parts=[TextPart("OK")])

    await generate(
        FunctionModel(respond),
        BudgetLedger(BenchmarkConfig()),
        CallTrace(),
        "cache-probe",
        "Reply OK only.",
        capabilities=[SkillPrefix("isolated-arm", tuple("ABC"))],
    )
    assert observed[0].parts[0].content == "Experiment isolated-arm."
    assert "skill_id" in observed[0].parts[2].content


async def test_request_response_capture_preserves_repeated_attempts(tmp_path):
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))
    ledger = BudgetLedger(BenchmarkConfig())
    traces = [CallTrace(capture_directory=tmp_path) for _ in range(2)]
    for trace in traces:
        await generate(model, ledger, trace, "answer", "same request")
    assert traces[0].calls[0].request_file != traces[1].calls[0].request_file
    assert traces[0].calls[0].response_file != traces[1].calls[0].response_file
    assert len(list(tmp_path.glob("*.json"))) == 4


async def test_capture_write_failure_does_not_double_charge_response(tmp_path, monkeypatch):
    from pathlib import Path

    def respond(messages, info):
        return ModelResponse(
            parts=[TextPart("ok")], usage=RequestUsage(input_tokens=100, output_tokens=10)
        )

    original = Path.write_bytes

    def write(path, payload):
        if path.name.endswith("response.json"):
            raise OSError("capture disk full")
        return original(path, payload)

    monkeypatch.setattr(Path, "write_bytes", write)
    config = BenchmarkConfig()
    ledger = BudgetLedger(config)
    with pytest.raises(OSError, match="capture disk full"):
        await generate(
            FunctionModel(respond), ledger, CallTrace(capture_directory=tmp_path), "answer", "q"
        )
    expected = (100 * config.input_miss_usd_per_million + 10 * config.output_usd_per_million) / 1e6
    assert ledger.spent == pytest.approx(expected)
    assert ledger.reserved == pytest.approx(0)


async def test_deepseek_output_limit_uses_supported_wire_field(monkeypatch):
    import httpx
    from openai import AsyncOpenAI

    from reactive_context.benchmarking import runtime

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    client = AsyncOpenAI(
        api_key="test",
        base_url="https://example.test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    )
    monkeypatch.setattr(runtime, "AsyncOpenAI", lambda **kwargs: client)
    config = BenchmarkConfig(api_key="test")
    async with runtime.make_model(config) as model:
        await generate(model, BudgetLedger(config), CallTrace(), "answer", "q", max_tokens=17)
    assert requests[0]["max_tokens"] == 17
    assert "max_completion_tokens" not in requests[0]
    assert requests[0]["thinking"] == {"type": "disabled"}
