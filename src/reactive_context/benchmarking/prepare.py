"""Reproduce the priority pilot inputs from pinned public upstream versions."""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import httpx

from .official import CACHE, file_hash, git_revision

REPOSITORIES = {
    "longmemeval": ("xiaowu0162/LongMemEval", "9e0b455f4ef0e2ab8f2e582289761153549043fc"),
    "contextbench": ("EuniAI/ContextBench", "1436c28a8eb95496da4ea69ad458b9f8a8eb7d61"),
}

DATA = [
    (
        "xiaowu0162/longmemeval-cleaned",
        "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "longmemeval_s_cleaned.json",
        "longmemeval/longmemeval_s_cleaned.json",
        277383467,
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    ),
    (
        "Contextbench/ContextBench",
        "c2855792b006af41c67202d33883fb9d46362853",
        "data/full.parquet",
        "contextbench/full.parquet",
        26102607,
        "2f56535bdc73eb8a68bf4ebb49789d8e9cd4f219ea60df6290b85278aee61ca8",
    ),
    (
        "Contextbench/ContextBench",
        "c2855792b006af41c67202d33883fb9d46362853",
        "data/train.parquet",
        "contextbench/train.parquet",
        15013408,
        "34482f20a65dbe1bd6cdc928aa5322bea2143a7ab26363145274581019f02e76",
    ),
]


def download(item):
    dataset, revision, remote, local, size, expected = item
    destination = CACHE / "data" / local
    if destination.exists() and file_hash(destination) == expected:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunks = destination.parent / (destination.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    url = f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{remote}"
    chunk_size = 1024 * 1024

    def fetch(index):
        start = index * chunk_size
        end = min(size, start + chunk_size) - 1
        path = chunks / str(index)
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        with httpx.Client(follow_redirects=True, timeout=120) as client:
            response = client.get(
                url, params={"chunk": index}, headers={"Range": f"bytes={start}-{end}"}
            )
            response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - start + 1:
            raise RuntimeError("public download did not honor the requested byte range")
        path.write_bytes(response.content)
        return path

    count = (size + chunk_size - 1) // chunk_size
    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = list(executor.map(fetch, range(count)))
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as stream:
        for path in paths:
            stream.write(path.read_bytes())
    if file_hash(temporary) != expected:
        raise RuntimeError("upstream data checksum mismatch; refusing to use it")
    os.replace(temporary, destination)


def prepare_inputs():
    for name, (remote, revision) in REPOSITORIES.items():
        path = CACHE / "repos" / name
        if path.exists():
            if git_revision(path) != revision:
                raise RuntimeError("existing upstream checkout differs from the pinned version")
            continue
        path.mkdir(parents=True)
        commands = [
            ["init", "-q"],
            ["remote", "add", "origin", f"https://github.com/{remote}.git"],
            ["fetch", "--depth=1", "--filter=blob:none", "origin", revision],
        ]
        if name == "contextbench":
            commands += [["sparse-checkout", "set", "contextbench", "docs"]]
        commands += [["checkout", "--detach", revision]]
        for args in commands:
            subprocess.run(["git", "-C", str(path), *args], check=True, timeout=600)
    for item in DATA:
        download(item)
