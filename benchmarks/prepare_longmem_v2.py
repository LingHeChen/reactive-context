"""Download and validate the pinned full multimodal LongMemEval-V2 release.

Run with the benchmark dependency group. Downloads ~7.1 GB, plus extracted images.
"""

import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx

from reactive_context.benchmarking.longmem_v2 import (
    DATA_REVISION,
    DATA_ROOT,
    REPO_REVISION,
    REPO_ROOT,
)
from reactive_context.benchmarking.official import git_revision
from reactive_context.benchmarking.prepare import download


def main():
    if not REPO_ROOT.exists():
        REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                "https://github.com/xiaowu0162/LongMemEval-V2.git",
                str(REPO_ROOT),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(REPO_ROOT), "fetch", "origin", REPO_REVISION], check=True)
        subprocess.run(
            ["git", "-C", str(REPO_ROOT), "checkout", "--detach", REPO_REVISION], check=True
        )
    if git_revision(REPO_ROOT) != REPO_REVISION:
        raise RuntimeError("existing LongMemEval-V2 checkout differs from pinned revision")
    with httpx.Client(follow_redirects=True, timeout=120) as client:
        response = client.get(
            "https://huggingface.co/api/datasets/xiaowu0162/longmemeval-v2/tree/" + DATA_REVISION,
            params={"recursive": "true", "expand": "false"},
        )
        response.raise_for_status()
        rows = response.json()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / "download-manifest.json").write_text(
        json.dumps(
            {
                "revision": DATA_REVISION,
                "files": rows,
            },
            indent=2,
        )
        + "\n"
    )

    def fetch(row):
        name = row["path"]
        if row["size"] >= 8 * 1024 * 1024:
            download(
                (
                    "xiaowu0162/longmemeval-v2",
                    DATA_REVISION,
                    name,
                    "longmemeval-v2/" + name,
                    row["size"],
                    row["lfs"]["oid"],
                )
            )
            return
        destination = DATA_ROOT / name
        if destination.exists():
            payload = destination.read_bytes()
        else:
            with httpx.Client(follow_redirects=True, timeout=120) as client:
                result = client.get(
                    "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/"
                    + DATA_REVISION
                    + "/"
                    + name
                )
                result.raise_for_status()
                payload = result.content
        expected = row.get("lfs", {}).get("oid")
        checksum = (
            hashlib.sha256(payload).hexdigest()
            if expected
            else hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        )
        if len(payload) != row["size"] or checksum != (expected or row["oid"]):
            raise RuntimeError("dataset checksum mismatch: " + name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    files = [row for row in rows if row["type"] == "file"]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch, files))
    for script, flags in (
        ("prepare_data.py", ["--mode", "symlink"]),
        ("validate_data.py", ["--tier", "small"]),
    ):
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "data" / script),
                "--data-root",
                str(DATA_ROOT),
                *flags,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
