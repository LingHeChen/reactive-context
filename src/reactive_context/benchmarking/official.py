"""Load pinned upstream functions without rewriting their scoring or prompt algorithms."""

import ast
import importlib
import json
import random
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / ".benchmark-cache"


def file_hash(path):
    checksum = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def git_revision(path):
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    return result.stdout.strip()


def source_function(path, name) -> Any:
    """Execute the exact upstream AST node with only its declared pure dependencies.

    This avoids importing generation CLI GPU/tokenizer dependencies. The upstream file is not
    patched or copied into this package. Its whole-file hash is frozen in the run manifest.
    """
    path = Path(path)
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"json": json, "random": random}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class LongMemEvalScorer:
    def __init__(self):
        self.repo = CACHE / "repos/longmemeval"
        self.path = self.repo / "src/evaluation/evaluate_qa.py"
        self.prompt = source_function(self.path, "get_anscheck_prompt")
        self.revision = git_revision(self.repo)
        self.source_hash = file_hash(self.path)
        tree = ast.parse(self.path.read_text())
        rules = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "label" for target in node.targets
            )
        ]
        if len(rules) != 1:
            raise RuntimeError("official judge label rule is ambiguous")
        self._label_code = compile(
            ast.Module(body=list(rules), type_ignores=[]), str(self.path), "exec"
        )

    def judge_prompt(self, question, labels, prediction):
        return self.prompt(
            labels["question_type"],
            question,
            labels["answer"],
            prediction,
            abstention="_abs" in labels["question_id"],
        )

    def parse(self, response):
        namespace = {"eval_response": response.strip()}
        exec(self._label_code, namespace)
        return float(namespace["label"])

    def official_command(self, predictions, references, judge="gpt-4o"):
        return [sys.executable, str(self.path), judge, str(predictions), str(references)]


class ContextBenchScorer:
    def __init__(self):
        self.repo = CACHE / "repos/contextbench"
        sys.path.insert(0, str(self.repo))
        self.compute = importlib.import_module("contextbench.metrics.compute")
        self.gold_module = importlib.import_module("contextbench.parsers.gold")
        self.fileio = importlib.import_module("contextbench.core.fileio")
        self.extractors = importlib.import_module("contextbench.extractors")
        self.revision = git_revision(self.repo)
        self.source_hash = file_hash(self.repo / "contextbench/metrics/compute.py")

    def score(self, labels, included, repo_dir):
        # Gold and reference patch never leave the scoring phase.
        gold = self.gold_module.Gold(
            {
                "inst_id": labels["instance_id"],
                "gold_ctx": json.loads(labels["gold_context"])
                if isinstance(labels["gold_context"], str)
                else labels["gold_context"],
            }
        )
        files = {item["source_uri"] for item in included}
        lines = {}
        spans = {}
        for item in included:
            path = item["source_uri"]
            first, last = item["source_span"]
            lines.setdefault(path, []).append((first, last))
            span = self.fileio.line_to_byte(str(Path(repo_dir) / path), first, last)
            if span:
                spans.setdefault(path, []).append(span)
        pred_symbols = self.extractors.extract_def_set_in_spans(spans, str(repo_dir))
        gold_spans = gold.byte_spans(str(repo_dir))
        gold_symbols = self.extractors.extract_def_set_in_spans(gold_spans, str(repo_dir))
        result = self.compute.compute_granularity_metrics(
            files,
            pred_symbols,
            spans,
            set(gold.files()),
            gold_symbols,
            gold_spans,
            pred_lines=lines,
            gold_lines=gold.line_spans_init(),
        )
        for granularity in result.values():
            p, r = granularity["precision"], granularity["coverage"]
            granularity["derived_f1"] = 2 * p * r / (p + r) if p + r else 0.0
        return result
