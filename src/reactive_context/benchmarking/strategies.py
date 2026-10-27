"""Strategy differences affect context only, never the answer model,judge or tool
permissions.
"""

import re
from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter

from rank_bm25 import BM25Okapi

from reactive_context import CompilerPolicy, Context, ContextIRSegment
from reactive_context.adapters.pydantic_ai import ReactiveContextCapability
from reactive_context.sources.config import ConfigSource

from .datasets import official_memory_prompt
from .runtime import generate


@dataclass
class PreparedRequest:
    prompt: str | None
    context: Context | None = None
    capabilities: list = field(default_factory=list)
    included: list = field(default_factory=list)
    retrieved: list = field(default_factory=list)
    ir_digest: str = ""
    preparation_ms: float = 0

    async def close(self):
        if self.context:
            await self.context.close()


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def rank_documents(question, documents):
    if not documents:
        return []
    # Corpus keys, answer IDs and reference labels never enter ranking.
    index = BM25Okapi([tokenize(document.text) or ["_empty_"] for document in documents])
    scores = index.get_scores(tokenize(question))
    return [
        document.model_copy(update={"relevance": float(scores[i])})
        for i, document in sorted(
            enumerate(documents), key=lambda pair: (-scores[pair[0]], pair[0])
        )
    ]


def reference(document):
    return {
        "id": document.id,
        "source_uri": document.source_uri,
        "source_revision": document.source_revision,
        "source_span": document.source_span,
    }


class MemoryStrategy:
    def __init__(self, name, config, model, ledger, trace):
        if name not in {"B0", "B1", "B2", "B3"}:
            raise ValueError("unknown strategy")
        self.name, self.config = name, config
        self.model, self.ledger, self.trace = model, ledger, trace

    async def prepare_request(self, invocation):
        case = invocation  # An InferenceCase, never BenchmarkCase or evaluator_labels.
        start = perf_counter()
        documents = case.documents
        retrieved = documents
        prompt = official_memory_prompt(case, documents, self.config.context_tokens_proxy)
        if self.name == "B2":
            retrieved = rank_documents(case.question, documents)[: self.config.rag_sessions]
            prompt = official_memory_prompt(case, retrieved, self.config.context_tokens_proxy)
            # A genuine additional model call; usage, latency and budget are charged to B2.
            summary = await generate(
                self.model,
                self.ledger,
                self.trace,
                "summary",
                "Extract compact evidence for the question from the "
                "history below. Preserve dates, "
                "conflicts and updates. Do not answer the question. Do not "
                "invent missing evidence.\n\n" + prompt,
                max_tokens=2048,
            )
            prompt = (
                "I will give you several history chats between you and a user. Please answer "
                "the question based on the relevant chat history.\n\n\nHistory Chats:\n\n"
                + summary
                + "\n\nCurrent Date: "
                + case.question_date
                + "\nQuestion: "
                + case.question
                + "\nAnswer:"
            )
        refs = [reference(document) for document in retrieved]
        prepared = PreparedRequest(
            prompt,
            included=refs,
            retrieved=refs,
            ir_digest=sha256(prompt.encode()).hexdigest(),
            preparation_ms=(perf_counter() - start) * 1000,
        )
        if self.name == "B3":
            # The original dataset provides immutable sessions, not supersession/version
            # events.
            # Preserve them all; B1 and B3 therefore intentionally have identical model input.
            def project(value, ctx):
                return [
                    ContextIRSegment(
                        "memory",
                        "memory",
                        ctx.revision,
                        value,
                        kind="history",
                        requested_role="user",
                    )
                ]

            source = ConfigSource("memory", prompt, project=project)
            context = Context(
                [source.part],
                compiler_policy=CompilerPolicy(
                    max_tokens=2_000_000, output_reserve=self.config.max_output_tokens
                ),
            )
            prepared.prompt = None
            prepared.context = context
            prepared.capabilities = [
                ReactiveContextCapability(context, scope=case.case_id, cache=None)
            ]
        return prepared
