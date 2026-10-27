"""Pinned MemoryArena travel tasks, unmodified tools, parser and scoring functions."""

import ast
import csv
import difflib
import importlib
import json
import os
import random
import re
import sys
import types
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .official import CACHE, file_hash, git_revision

REPO = CACHE / "repos/memoryarena"
REVISION = "6cd9de14b71915e39ac742a20dc33785e14b6aab"
DATA = CACHE / "data/memoryarena/group_travel_planner.jsonl"
TRAVEL = REPO / "env/env_systems/travel_planner_env"


def tasks(count=3, seed=83):
    rows = [json.loads(line) for line in DATA.read_text().split("\n") if line.strip()]
    if len(rows) != 270:
        raise ValueError("expected 270 official travel groups")
    rows.sort(key=lambda row: row["id"])
    random.Random(seed).shuffle(rows)
    return rows[:count]


def public_task(row):
    # Future queries remain under the runner's control; the actor/writer receive one at a time.
    return {"id": row["id"], "base_person": row["base_person"], "questions": row["questions"]}


def pure_functions(path, names, constants=(), namespace=None):
    tree = ast.parse(Path(path).read_text())
    nodes: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        )
    ]
    found = {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
    if found != set(names):
        raise ValueError("upstream pure-function boundary changed")
    env = {
        "json": json,
        "csv": csv,
        "difflib": difflib,
        "os": os,
        "re": re,
        "datetime": datetime,
        "defaultdict": defaultdict,
        **(namespace or {}),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def parser():
    return pure_functions(TRAVEL / "combination.py", {"parse_plan_text"})["parse_plan_text"]


def system_prompt():
    tree = ast.parse((TRAVEL / "prompts.py").read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "AGENT_SYSTEM_PROMPT" for t in n.targets)
    )
    return ast.literal_eval(node.value)


def executor():
    if git_revision(REPO) != REVISION:
        raise ValueError("MemoryArena source differs from pinned revision")
    root = CACHE / "data/memoryarena/travel_db"
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "restaurants",
        "accommodations",
        "attractions",
        "googleDistanceMatrix",
        "background",
    ):
        target = root / name
        if not target.exists():
            target.symlink_to(TRAVEL / "database" / name, target_is_directory=True)
    flights = CACHE / "data/memoryarena/clean_Flights_2022.csv"
    if not flights.exists() or flights.stat().st_size != 304807007:
        raise ValueError("official flight database is not ready")
    folder = root / "flights"
    folder.mkdir(exist_ok=True)
    target = folder / flights.name
    if not target.exists():
        target.symlink_to(flights)
    # Import only the upstream tool package, avoiding optional web/GPU/server entry points.
    name = "_reactive_context_memoryarena_travel"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(TRAVEL)]
        sys.modules[name] = module
    return importlib.import_module(name + ".tool_executor").ToolExecutor(db_path=str(root))


def score(rows, submission):
    ground, base, counts = {}, {}, {}
    for row in rows:
        base[row["id"]] = row["base_person"]["daily_plans"]
        counts[row["id"]] = len(row["questions"])
        for index, plan in enumerate(row["answers"], 1):
            ground[(row["id"], index)] = plan
    names = {
        "similarity",
        "_read_jsonl",
        "load_submissions",
        "get_day",
        "find_constraint_slots",
        "check_slot_pass",
        "check_person_full_pass",
        "evaluate",
    }
    env = pure_functions(
        TRAVEL / "eval.py",
        names,
        constants={"SIM_TH", "SLOTS"},
        namespace={"load_ground_truth": lambda: (ground, base, counts)},
    )
    return env["evaluate"](
        str(submission), model_name="deepseek-flash", memory_system="reactive-context"
    )


def per_person_scores(row, index, predicted):
    names = {
        "similarity",
        "get_day",
        "find_constraint_slots",
        "check_slot_pass",
        "check_person_full_pass",
    }
    env = pure_functions(TRAVEL / "eval.py", names, constants={"SIM_TH", "SLOTS"})
    ground = {(row["id"], j): plan for j, plan in enumerate(row["answers"], 1)}
    plan = ground[(row["id"], index)]
    slots = env["find_constraint_slots"](
        ground, {row["id"]: row["base_person"]["daily_plans"]}, row["id"], index
    )
    inherited = {
        (day.get("days") or day.get("day"), slot) for day in plan for slot in env["SLOTS"]
    } - slots
    return {
        "full_pass": bool(env["check_person_full_pass"](plan, predicted)),
        "constraint_count": len(slots),
        "constraints_passed": sum(
            bool(env["check_slot_pass"](plan, predicted, day, slot)) for day, slot in slots
        ),
        "inherited_count": len(inherited),
        "inherited_passed": sum(
            bool(env["check_slot_pass"](plan, predicted, day, slot)) for day, slot in inherited
        ),
    }


def manifest():
    return {
        "repository_revision": git_revision(REPO),
        "dataset_sha256": file_hash(DATA),
        "tool_source_sha256": {
            str(p.relative_to(REPO)): file_hash(p) for p in (TRAVEL / "tools").glob("*.py")
        },
        "executor_sha256": file_hash(TRAVEL / "tool_executor.py"),
        "parser_sha256": file_hash(TRAVEL / "combination.py"),
        "scorer_sha256": file_hash(TRAVEL / "eval.py"),
    }
