"""Scoring, metering and isolation gates; never presented as model quality scores."""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("openai")
pytest.importorskip("rank_bm25")

from pydantic_ai import BinaryContent  # noqa: E402
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart  # noqa: E402
from pydantic_ai.models.function import FunctionModel  # noqa: E402
from pydantic_ai.usage import RequestUsage  # noqa: E402

from reactive_context.benchmarking import longmem_v2 as v2  # noqa: E402
from reactive_context.benchmarking.config import BenchmarkConfig  # noqa: E402
from reactive_context.benchmarking.history_regression import (  # noqa: E402
    answer_agent,
    answer_closed,
    choose_v2,
    grade_v2,
    loss_probability_upper,
    paired_summary,
)
from reactive_context.benchmarking.runtime import BudgetLedger, CallTrace, generate  # noqa: E402


def test_trajectory_whitelist_excludes_annotations_at_every_level():
    trajectory = {
        "id": "t",
        "domain": "web",
        "environment": "site",
        "goal": "public goal",
        "outcome": "success",
        "start_url": "https://example.test",
        "answer": "SECRET_LABEL",
        "states": [
            {
                "state_index": 0,
                "accessibility_tree": "public observation",
                "screenshot": "screenshots/t/0.png",
                "has_answer": True,
                "reference_answer": "SECRET_LABEL",
            }
        ],
    }
    payload = json.dumps(list(v2.trajectory_events(trajectory)))
    assert "public observation" in payload and "screenshots/t/0.png" in payload
    assert "SECRET_LABEL" not in payload and "has_answer" not in payload


def test_v2_sample_covers_all_five_abilities_and_both_domains_without_outcome_selection():
    if not (v2.DATA_ROOT / "questions.jsonl").exists():
        pytest.skip("public V2 questions not downloaded")
    sample = choose_v2(10)
    assert [q.id for q in sample] == [q.id for q in choose_v2(10)]
    assert len({(q.category, q.domain) for q in sample}) == 10
    assert any(q.image for q in sample)


def test_image_paths_cannot_escape_dataset_root(tmp_path):
    inside = tmp_path / "root"
    inside.mkdir()
    (tmp_path / "secret.png").write_bytes(b"outside")
    with pytest.raises(ValueError):
        v2.safe_image(inside, "../secret.png")


def test_zero_observed_losses_in_a_tiny_pilot_cannot_establish_non_regression():
    assert loss_probability_upper(0, 10) == pytest.approx(1 - 0.05**0.1)
    assert loss_probability_upper(0, 10) > 0.05
    assert loss_probability_upper(0, 60) < 0.05
    assert loss_probability_upper(10, 10) == 1


def test_execution_failure_is_not_silently_excluded_from_paired_accuracy():
    common = {
        "benchmark": "fixture",
        "case_id": "one",
        "category": "recall",
        "initial_input_tokens": 1,
        "input_tokens": 1,
        "output_tokens": 1,
        "estimated_cost_usd_upper_bound": 0.0,
        "end_to_end_ms": 1,
    }
    rows = [
        {**common, "arm": "old", "score": 1.0, "status": "completed"},
        {**common, "arm": "minimal-history", "score": 0.0, "status": "failed"},
    ]
    result = paired_summary(rows, "fixture", "old")
    assert result["paired_cases"] == result["losses"] == 1
    assert result["minimal_accuracy"] == 0
    assert result["noninferiority_status"] == "not_established"


async def test_binary_image_reservation_uses_documented_image_token_bound(tmp_path):
    config = BenchmarkConfig(max_cost_usd=0.01)
    ledger = BudgetLedger(config)
    trace = CallTrace(capture_directory=tmp_path)
    image = BinaryContent(data=b"x" * 200000, media_type="image/png")
    model = FunctionModel(
        lambda messages, info: ModelResponse(
            parts=[TextPart("ok")], usage=RequestUsage(input_tokens=1100, output_tokens=2)
        )
    )
    assert await generate(model, ledger, trace, "image", ["look", image]) == "ok"
    assert trace.calls[0].input_image_count == 1
    assert trace.calls[0].input_bytes < 2000
    request = json.loads(await asyncio.to_thread(Path(trace.calls[0].request_file).read_text))
    content = request["messages"][0]["parts"][0]["content"]
    assert content[1]["binary_sha256"] and content[1]["bytes"] == 200000
    assert ledger.spent < config.max_cost_usd


@pytest.mark.parametrize("arm", ["previous-4-groups", "minimal-history"])
async def test_paired_native_runner_can_answer_old_references_using_real_archive_tools(arm):
    raw = [
        ModelRequest.user_text_prompt("orchid code SECRET_321"),
        ModelResponse(parts=[TextPart("Saved")]),
        ModelRequest.user_text_prompt("unrelated"),
        ModelResponse(parts=[TextPart("NOISE" + "x" * 3000)]),
    ]
    steps = []

    def respond(messages, info):
        steps.append(messages)
        if "SECRET_321" in str(messages):
            return ModelResponse(parts=[TextPart("SECRET_321")])
        return ModelResponse(
            parts=[ToolCallPart("context_history_search", {"intent": "orchid"}, "find")]
        )

    model = FunctionModel(respond)
    trace = CallTrace()
    result, metrics = await answer_agent(
        model, BudgetLedger(BenchmarkConfig()), trace, arm, "What was the code?", raw=raw
    )
    assert result == "SECRET_321" and metrics["archive_events"] >= 6
    if arm == "minimal-history":
        assert "NOISE" not in str(steps[0]) and len(steps) == 2


async def test_official_v2_pure_and_llm_scoring_keep_unknown_incorrect():
    if not v2.REPO_ROOT.exists():
        pytest.skip("pinned official V2 evaluator not downloaded")
    ledger, trace = BudgetLedger(BenchmarkConfig()), CallTrace()
    model = FunctionModel(
        lambda messages, info: ModelResponse(parts=[TextPart('{"label":1,"reason":"matches"}')])
    )
    question = v2.V2Question(
        "fixture",
        "web",
        "Question",
        None,
        "static-state",
        {"answer": "Alpha, Beta", "eval_function": "norm_phrase_set_match|separators=,;"},
    )
    assert await grade_v2(model, ledger, trace, question, r"\boxed{Alpha, Beta}") == 1
    assert await grade_v2(model, ledger, trace, question, r"\boxed{UNKNOWN}") == 0
    assert not trace.calls
    gotcha = v2.V2Question(
        "fixture",
        "web",
        "Question",
        None,
        "gotchas",
        {"answer": "expected insight", "eval_function": "llm_gotchas_checker"},
    )
    assert await grade_v2(model, ledger, trace, gotcha, r"\boxed{expected insight}") == 1
    assert len(trace.calls) == 1 and trace.calls[0].phase == "judge"


async def test_no_memory_baseline_preserves_official_instructions_without_archive_tools():
    if not v2.REPO_ROOT.exists():
        pytest.skip("pinned V2 prompts not downloaded")

    def respond(messages, info):
        assert not info.function_tools
        assert "customized ServiceNow" in info.instructions
        assert "Only the public question" in str(messages)
        return ModelResponse(parts=[TextPart(r"\boxed{UNKNOWN}")])

    result, metrics = await answer_closed(
        FunctionModel(respond),
        BudgetLedger(BenchmarkConfig()),
        CallTrace(),
        "Only the public question",
        domain="enterprise",
        image=None,
    )
    assert result == r"\boxed{UNKNOWN}" and metrics["historical_context_items"] == 0


async def test_real_deepseek_wire_keeps_current_question_and_tool_screenshot(tmp_path, monkeypatch):
    import base64

    import httpx
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    from reactive_context import ContextStore, HistoryGroup, HistoryManager
    from reactive_context.benchmarking.history_regression import HOST

    if not v2.REPO_ROOT.exists():
        pytest.skip("pinned V2 prompts not downloaded")
    image = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1sAAAAASUVORK5CYII="
    )
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "state.png").write_bytes(image)
    monkeypatch.setattr(v2, "DATA_ROOT", data)
    base = tmp_path / "base"
    event = {"kind": "historical_trajectory_state", "state": {"screenshot": "state.png"}}
    with ContextStore(base) as store:
        store.create_scope(HOST, "memory")
        history = HistoryManager(store, HOST)
        history.observe("memory", [HistoryGroup((event,))])
        event_id = history.event_id(event)
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "image",
                        "type": "function",
                        "function": {
                            "name": "context_history_image",
                            "arguments": json.dumps({"event_id": event_id}),
                        },
                    }
                ],
            }
            finish = "tool_calls"
        else:
            assert "CURRENT_SCREEN_QUESTION" in str(body["messages"])
            urls = [
                part["image_url"]["url"]
                for message in body["messages"]
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            ]
            assert len(urls) == 1 and base64.b64decode(urls[0].split(",", 1)[1]) == image
            message, finish = {"role": "assistant", "content": r"\boxed{UNKNOWN}"}, "stop"
        return httpx.Response(
            200,
            json={
                "id": "mock-image",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AsyncOpenAI(api_key="mock-only", base_url="https://example.test", http_client=http)
        model = OpenAIChatModel(
            "deepseek-flash",
            provider=DeepSeekProvider(openai_client=client),
            profile=OpenAIModelProfile(openai_chat_supports_max_completion_tokens=False),
        )
        trace = CallTrace()
        prediction, _ = await answer_agent(
            model,
            BudgetLedger(BenchmarkConfig()),
            trace,
            "minimal-history",
            "CURRENT_SCREEN_QUESTION",
            base_store=base,
            domain="enterprise",
        )
    assert prediction == r"\boxed{UNKNOWN}" and len(seen) == 2
    assert trace.calls[1].input_image_count == 1
