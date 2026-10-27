import json
from types import SimpleNamespace

import httpx
import pytest

from reactive_context import ContextManager, ContextStore, Objective, Principal
from reactive_context.adapters.jev import JevAdapter, JevError, JevProtocolError


def reply(answers):
    return {
        "id": "decision-1",
        "model": "typesafe/jev-1.14-verified",
        "answers": answers,
        "usage": {"input_tokens": 100, "cost": 0.0000042},
    }


async def test_decisions_wire_types_usage_and_context_scope(tmp_path):
    captured, usage = [], []

    def respond(request):
        captured.append(json.loads(request.content))
        assert request.url == "https://openrouter.ai/api/alpha/decisions"
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json=reply(
                {
                    "update": {"type": "noul", "noul": 0.9},
                    "route": {
                        "type": "choice",
                        "choice": "keep",
                        "probabilities": {"keep": 0.9, "change": 0.1},
                    },
                    "priority": {"type": "score", "score": 1.5},
                }
            ),
        )

    questions = {
        "update": {"type": "noul", "instructions": "Update?"},
        "route": {
            "type": "choice",
            "instructions": "Route?",
            "criteria": {"keep": "Keep", "change": "Change"},
        },
        "priority": {
            "type": "score",
            "instructions": "Priority?",
            "criteria": ["Low", "Medium", "High"],
        },
    }
    host, actor = Principal("host", True), Principal("agent")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        async with JevAdapter(api_key="test-key", client=client, on_usage=usage.append) as jev:
            with ContextStore(tmp_path) as store:
                store.create_scope(host, "chat")
                store.grant_scope(host, "chat", actor.id)
                objective = Objective(store, actor)
                objective.update("chat", {"goal": "CURRENT_GOAL"})
                async with ContextManager(
                    store=store, principal=actor, objective=objective
                ) as manager:
                    result = await jev.decide_context(manager, "chat", questions)
                    assert result.answers["update"].noul == 0.9
        assert not client.is_closed
    assert "CURRENT_GOAL" in str(captured[0]["state"])
    assert "messages" not in captured[0] and "questions" in captured[0]
    assert usage == [{"input_tokens": 100, "cost": 0.0000042}]


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"update_needed": {"type": "noul", "noul": 1.1}},
        {"update_needed": {"type": "choice", "choice": "yes"}},
    ],
)
async def test_invalid_jev_response_cannot_trigger_objective_update(answers):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=reply(answers)))
    ) as client:
        jev = JevAdapter(api_key="test-key", client=client)
        with pytest.raises(JevProtocolError):
            await jev.should_update_objective(
                SimpleNamespace(current=None, events=(), reason="threshold")
            )


@pytest.mark.parametrize("probability,expected", [(0.9, True), (0.2, False)])
async def test_jev_decider_returns_probability_gate_without_generating_text(probability, expected):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json=reply({"update_needed": {"type": "noul", "noul": probability}})
            )
        )
    ) as client:
        jev = JevAdapter(api_key="test-key", client=client, decision_threshold=0.8)
        result = await jev.should_update_objective(
            SimpleNamespace(current=None, events=(), reason="threshold")
        )
        assert result is expected


async def test_jev_transport_errors_are_sanitized_and_input_limit_prevents_requests():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"message": "sensitive vendor content"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        jev = JevAdapter(api_key="secret-token", client=client)
        with pytest.raises(JevError) as error:
            await jev.decide("data", {"q": {"type": "noul", "instructions": "Question?"}})
        assert "secret-token" not in str(error.value) and "sensitive" not in str(error.value)
        jev.max_request_bytes = 10
        with pytest.raises(JevError):
            await jev.decide("x" * 100, {"q": {"type": "noul", "instructions": "Question?"}})
    assert len(calls) == 1
