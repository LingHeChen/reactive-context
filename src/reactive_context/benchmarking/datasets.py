"""Whitelist input fields before they can reach any reader, retriever or Context Part."""

import json
import os
import random
from copy import deepcopy
from hashlib import sha256

from .contracts import BenchmarkCase, Evidence, InferenceCase
from .official import CACHE, source_function


class LongMemEvalDataset:
    revision = "98d7416c24c778c2fee6e6f3006e7a073259d48f"

    def __init__(self):
        self.path = CACHE / "data/longmemeval/longmemeval_s_cleaned.json"
        self._rows = json.loads(self.path.read_text())
        if len(self._rows) != 500:
            raise ValueError("LongMemEval S must contain 500 original cases")

    def cases(self):
        for row in self._rows:
            documents = []
            for sid, date, session in zip(
                row["haystack_session_ids"],
                row["haystack_dates"],
                row["haystack_sessions"],
                strict=True,
            ):
                # has_answer and every other annotation are excluded by whitelist.
                turns = [{key: turn[key] for key in ("role", "content")} for turn in session]
                text = json.dumps({"date": date, "turns": turns}, ensure_ascii=False)
                documents.append(
                    Evidence(
                        id=sid,
                        text=text,
                        source_uri=f"session://{sid}",
                        source_revision=sha256(text.encode()).hexdigest(),
                    )
                )
            public = InferenceCase(
                case_id=row["question_id"],
                benchmark="longmemeval",
                question=row["question"],
                question_date=row["question_date"],
                documents=documents,
            )
            labels = {
                key: deepcopy(row[key])
                for key in ("question_id", "question_type", "answer", "answer_session_ids")
            }
            yield BenchmarkCase(public, labels, row["question_type"], row["question_id"])

    def iter_cases(self, split="all"):
        yield from self.cases()

    def get_official_scorer(self):
        from .official import LongMemEvalScorer

        return LongMemEvalScorer()


def official_memory_prompt(case: InferenceCase, documents, max_tokens=128000):
    import tiktoken

    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(CACHE / "tokenizers"))
    prepare = source_function(
        CACHE / "repos/longmemeval/src/generation/run_generation.py", "prepare_prompt"
    )
    dates, sessions, identifiers = [], [], []
    for document in documents:
        data = json.loads(document.text)
        dates.append(data["date"])
        sessions.append(data["turns"])
        identifiers.append(document.id)
    # No evaluation-only keys are supplied, even though the upstream function cleans some
    # flags.
    entry = {
        "question": case.question,
        "question_date": case.question_date,
        "haystack_dates": dates,
        "haystack_sessions": sessions,
        "haystack_session_ids": identifiers,
    }
    return prepare(
        entry,
        retriever_type="orig-session",
        topk_context=1000,
        useronly=False,
        history_format="json",
        cot=False,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        tokenizer_backend="openai",
        max_retrieval_length=max_tokens,
        merge_key_expansion_into_value="none",
        con=False,
    )


def pilot_split(cases, count=30, seed=17):
    """Balanced, disjoint development pilot; never call it a full benchmark score."""
    groups = {}
    for case in cases:
        groups.setdefault(case.subgroup or "other", []).append(case)
    rng = random.Random(seed)
    for key in sorted(groups):
        groups[key].sort(key=lambda c: c.inference.case_id)
        rng.shuffle(groups[key])
    selected = []
    total = sum(map(len, groups.values()))
    while len(selected) < min(count, total):
        made_progress = False
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
                made_progress = True
        if not made_progress:
            break
    ids = {c.inference.case_id for c in selected}
    heldout = [
        case for group in groups.values() for case in group if case.inference.case_id not in ids
    ]
    rng.shuffle(selected)
    return selected, heldout
