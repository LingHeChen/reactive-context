"""Pinned LongMemEval-V2 public trajectories and evaluator-only reference answers."""

import ast
import importlib.util
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..history import HistoryGroup, HistoryManager
from .official import CACHE, file_hash, git_revision

REPO_REVISION = "2cc8c540bdb87fe6761629b585e727e1c4704520"
DATA_REVISION = "f152293e235517d504809563c833d7190b8c713b"
DATA_ROOT = CACHE / "data/longmemeval-v2"
REPO_ROOT = CACHE / "repos/longmemeval-v2"


@dataclass(frozen=True)
class V2Question:
    id: str
    domain: str
    question: str
    image: str | None
    category: str
    labels: dict[str, Any] = field(repr=False)


def ability(question_type):
    if question_type.endswith("-abs"):
        return "premise-awareness"
    return {
        "static-environment": "static-state",
        "dynamic-environment": "dynamic-state",
        "procedure": "workflow",
        "errors-gotchas": "gotchas",
    }[question_type]


def load_questions(root=DATA_ROOT):
    rows = []
    with (root / "questions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            rows.append(
                V2Question(
                    row["id"],
                    row["domain"],
                    row["question"],
                    row["image"],
                    ability(row["question_type"]),
                    {
                        k: row[k]
                        for k in ("id", "question", "question_type", "answer", "eval_function")
                    },
                )
            )
    if len(rows) != 451 or len({q.id for q in rows}) != 451:
        raise ValueError("expected the complete 451-question V2 release")
    return rows


def safe_image(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("missing or out-of-root benchmark screenshot")
    return path


def trajectory_events(trajectory):
    # Do not copy arbitrary future dataset fields (especially annotations).
    common = {
        k: trajectory[k] for k in ("id", "domain", "environment", "goal", "outcome", "start_url")
    }
    for state in trajectory["states"]:
        yield {
            "kind": "historical_trajectory_state",
            "trajectory": common,
            "state": {
                k: state.get(k)
                for k in (
                    "state_index",
                    "step",
                    "url",
                    "action",
                    "thought",
                    "accessibility_tree",
                    "screenshot",
                )
            },
        }


def archive_haystack(store, principal, domain, root=DATA_ROOT):
    questions = load_questions(root)
    mapping = json.loads((root / "haystacks/lme_v2_small.json").read_text())
    sets = {tuple(mapping[q.id]) for q in questions if q.domain == domain}
    if len(sets) != 1:
        raise ValueError("small tier must share one haystack per domain")
    wanted = set(next(iter(sets)))
    if len(wanted) != 100:
        raise ValueError("small haystack must contain exactly 100 complete trajectories")
    history = HistoryManager(store, principal)
    found, count = set(), 0
    with (root / "trajectories.jsonl").open() as stream:
        for line in stream:
            trajectory = json.loads(line)
            if trajectory["id"] not in wanted:
                continue
            if trajectory["domain"] != domain:
                raise ValueError("cross-domain haystack")
            events = list(trajectory_events(trajectory))
            history.observe("memory", (HistoryGroup((event,)) for event in events))
            count += len(events)
            found.add(trajectory["id"])
    if found != wanted:
        raise ValueError("incomplete trajectory haystack")
    return {"domain": domain, "trajectories": len(found), "states": count}


def load_scorer(repo=REPO_ROOT):
    if git_revision(repo) != REPO_REVISION:
        raise ValueError("LongMemEval-V2 scorer revision differs from pinned source")
    path = repo / "evaluation/qa_eval_metrics.py"
    spec = importlib.util.spec_from_file_location("_pinned_lme_v2_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load official V2 evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def domain_prompt(domain, repo=REPO_ROOT):
    tree = ast.parse((repo / "evaluation/harness.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DOMAIN_SYSTEM_PROMPTS" for t in n.targets)
    )
    return ast.literal_eval(node.value)[domain]


def describe(root=DATA_ROOT):
    questions = load_questions(root)
    return {
        "dataset_revision": DATA_REVISION,
        "repo_revision": REPO_REVISION,
        "question_count": len(questions),
        "abilities": dict(Counter(q.category for q in questions)),
        "domains": dict(Counter(q.domain for q in questions)),
        "image_questions": sum(bool(q.image) for q in questions),
        "questions_sha256": file_hash(root / "questions.jsonl"),
        "small_sha256": file_hash(root / "haystacks/lme_v2_small.json"),
        "scorer_sha256": file_hash(REPO_ROOT / "evaluation/qa_eval_metrics.py"),
    }
