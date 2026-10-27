"""Exercise the runnable example through its CLI and inspect actual model/tool evidence."""

import json
import subprocess
import sys
from pathlib import Path


def test_three_independent_parts_change_model_input_and_sqlite_results(tmp_path):
    root = Path(__file__).resolve().parents[2]
    artifact = tmp_path / "three-parts.json"
    completed = subprocess.run(
        [sys.executable, str(root / "examples/three_parts/main.py"), "--json", str(artifact)],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(artifact.read_text())
    assert report["registered_parts"] == ["skill", "tool", "dbschema"]
    stages = report["stages"]
    assert len(stages) == 4
    expected_resolves = [
        {"skill": 1, "tool": 1, "dbschema": 1},
        {"skill": 1, "tool": 0, "dbschema": 0},
        {"skill": 0, "tool": 1, "dbschema": 0},
        {"skill": 0, "tool": 0, "dbschema": 1},
    ]
    for stage, resolves in zip(stages, expected_resolves, strict=True):
        assert {part["part"]: part["resolves"] for part in stage["parts"]} == resolves
        assert all(part["status"] == "CURRENT" for part in stage["parts"])
        assert stage["input_history_unchanged"]
        assert stage["model_requests"] == 2
        assert [frame["phase"] for frame in stage["request_frames"]] == ["tool_call", "answer"]
    for stage in stages:
        for frame in stage["request_frames"]:
            assert frame["message_layout"][:4] == [
                "system:skill",
                "user-prefix:dbschema",
                "cache-breakpoint",
                "user:history/task",
            ]
            assert frame["active_data"]["skill://customer-report"]["role"] == "system"
    assert [stage["prefix_diagnostic"]["reason"] for stage in stages] == [
        "first_prefix",
        "changed:instruction:skill",
        "changed:tools",
        "changed:prefix-data:dbschema",
    ]
    assert stages[0]["execution"]["rows"] == [
        {"id": 1, "name": "Ada"},
        {"id": 2, "name": "Lin"},
        {"id": 3, "name": "Mei"},
    ]
    assert stages[1]["execution"]["rows"] == [{"id": 1, "name": "Ada"}]
    assert stages[1]["output"]["sql"].endswith("LIMIT 1;")
    for stage in stages[:2]:
        assert all(
            frame["visible_tools"] == ["query_database"] for frame in stage["request_frames"]
        )
    for stage in stages[2:]:
        assert all(frame["visible_tools"] == ["export_csv"] for frame in stage["request_frames"])
    assert stages[2]["output"]["result"] == "id,name\n1,Ada\n"
    assert stages[3]["output"]["sql"] == (
        "SELECT id, display_name FROM customer_accounts ORDER BY id LIMIT 1;"
    )
    assert stages[3]["output"]["result"] == "id,display_name\n1,Ada\n"
    assert stages[3]["execution"]["rows"] == [{"id": 1, "display_name": "Ada"}]
    for index, changed_key in enumerate(("skill", "tool", "dbschema"), start=1):
        before = {p["part"]: p["revision"] for p in stages[index - 1]["parts"]}
        after = {p["part"]: p["revision"] for p in stages[index]["parts"]}
        assert {key for key in before if before[key] != after[key]} == {changed_key}
