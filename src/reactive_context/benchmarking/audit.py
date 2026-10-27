"""Offline audit of pinned public inputs; missing benchmark runs stay explicit."""

import json
from pathlib import Path

from .official import CACHE, file_hash, git_revision


def run_audit(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    definitions = {
        "longmemeval": ("MIT", "development pilot", "longmemeval"),
        "contextbench": ("Apache-2.0", "bounded context-quality pilot", "contextbench"),
        "longmemeval-v2": (
            "Apache-2.0",
            "DEFERRED: full trajectories/screenshots and official reader runtime not prepared",
            "longmemeval-v2",
        ),
        "swe-bench": (
            "MIT",
            "DEFERRED: official Docker task-outcome evaluation has not run",
            "swe-bench",
        ),
    }
    license_path = CACHE / "data-licenses.json"
    data_licenses = json.loads(license_path.read_text()) if license_path.exists() else {}
    audit = {}
    for name, (license_name, status, repo) in definitions.items():
        repository = CACHE / "repos" / repo
        audit[name] = {
            "status": status,
            "repository_commit": git_revision(repository) if repository.exists() else None,
            "repository_license": license_name,
            "dataset_card_license_audit": data_licenses.get(name, {}),
            "data": [
                {
                    "path": str(p.relative_to(CACHE)),
                    "bytes": p.stat().st_size,
                    "sha256": file_hash(p),
                }
                for p in sorted((CACHE / "data" / name).glob("*"))
                if p.is_file()
            ],
            "official_leaderboard_score": None,
        }
    audit["statemembench"] = {
        "status": "BLOCKED",
        "reason": "No public official dataset/scorer located; no substitute generated",
        "paper": "https://arxiv.org/abs/2608.19652",
        "official_leaderboard_score": None,
    }
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit
