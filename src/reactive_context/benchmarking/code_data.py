"""ContextBench's public task fields and pinned source repositories; no gold in retrieval."""

import fcntl
import io
import json
import os
import re
import signal
import subprocess
import tarfile
import tempfile
from hashlib import sha256
from pathlib import Path

import pyarrow.parquet as pq

from .contracts import BenchmarkCase, Evidence, InferenceCase
from .official import CACHE
from .strategies import rank_documents


class ContextBenchDataset:
    revision = "c2855792b006af41c67202d33883fb9d46362853"

    def __init__(self):
        self.path = CACHE / "data/contextbench/full.parquet"
        self.train_path = CACHE / "data/contextbench/train.parquet"
        self._full = pq.read_table(self.path).to_pylist()
        train = pq.read_table(self.train_path).to_pylist()
        self._train_keys = {(r["original_inst_id"], r["repo"], r["base_commit"]) for r in train}
        if len(self._full) != 1136:
            raise ValueError("ContextBench full release must have 1136 tasks")

    def cases(self, split="development"):
        for row in self._full:
            key = (row["original_inst_id"], row["repo"], row["base_commit"])
            if split == "development" and key not in self._train_keys:
                continue
            public = InferenceCase(
                case_id=row["instance_id"],
                benchmark="contextbench",
                question=row["problem_statement"],
                documents=[],
                repository=row["repo"],
                base_commit=row["base_commit"],
                language=row["language"],
            )
            # Labels originate from the JSON full release, joined to public training
            # membership.
            labels = {"instance_id": row["instance_id"], "gold_context": row["gold_context"]}
            yield BenchmarkCase(public, labels, row["language"], row["repo"])

    def iter_cases(self, split="development"):
        yield from self.cases(split)

    def get_official_scorer(self):
        from .official import ContextBenchScorer

        return ContextBenchScorer()


SOURCE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".hpp",
    ".java",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".md",
    ".txt",
    ".rst",
    ".ini",
    ".cfg",
}
EXCLUDE_DIRS = {".git", "node_modules", "vendor", ".venv", "venv", "target", "dist", "build"}


def checkout_repository(case: InferenceCase):
    repo, commit = case.repository, case.base_commit
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("unsafe repository slug")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("unsafe repository commit")
    destination = CACHE / "source-archives" / repo.replace("/", "__") / commit
    marker = destination / ".benchmark-source.json"
    if marker.exists():
        return destination
    try:
        return sparse_repository(case)
    except (OSError, subprocess.SubprocessError):
        pass
    url = f"https://codeload.github.com/{repo}/legacy.tar.gz/{commit}"
    # A pinned source archive avoids orphaned partial-clone lazy-fetch processes.
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=commit + ".", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(filename)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--max-time",
            "60",
            "--max-filesize",
            "200000000",
            "--output",
            str(temporary),
            url,
        ],
        check=True,
        capture_output=True,
        timeout=70,
    )
    data = temporary.read_bytes()
    temporary.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            relative = Path(*path.parts[1:])
            if not relative.parts or ".." in relative.parts or member.issym() or member.islnk():
                continue
            if not member.isfile():
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.extractfile(member)
            if content is not None:
                target.write_bytes(content.read())
    marker.write_text(
        json.dumps({"url": url, "commit": commit, "archive_sha256": sha256(data).hexdigest()})
    )
    return destination


def sparse_repository(case):
    """Share pinned git objects across tasks; materialize only the common readable corpus."""
    repo, commit = case.repository, case.base_commit
    base = CACHE / "sparse-repositories" / repo.replace("/", "__")
    base.mkdir(parents=True, exist_ok=True)
    mirror, destination = base / "objects.git", base / commit
    marker = destination / ".benchmark-source.json"
    if marker.exists():
        return destination
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}

    def git(path, args, data=None, *, lazy=True):
        process = subprocess.Popen(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.autocrlf=false",
                "-C",
                str(path),
                *args,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env if lazy else {**env, "GIT_NO_LAZY_FETCH": "1"},
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(data.encode() if data else None, timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
        return stdout.decode()

    with (base / "fetch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if marker.exists():
            return destination
        # All writers to this private cache hold fetch.lock. A killed checkout can leave
        # its worktree index lock behind; no live checkout owns it once this lock is held.
        git_pointer = destination / ".git"
        if git_pointer.is_file():
            metadata = Path(git_pointer.read_text().removeprefix("gitdir: ").strip()).resolve()
            if metadata.is_relative_to(mirror.resolve()):
                (metadata / "index.lock").unlink(missing_ok=True)
        if not mirror.exists():
            mirror.mkdir()
            git(mirror, ["init", "--bare", "-q"])
            git(mirror, ["remote", "add", "origin", "https://github.com/" + repo + ".git"])
            git(mirror, ["config", "remote.origin.promisor", "true"])
            git(mirror, ["config", "remote.origin.partialclonefilter", "blob:none"])
        try:
            git(mirror, ["cat-file", "-e", commit + "^{commit}"], lazy=False)
        except subprocess.CalledProcessError:
            git(mirror, ["fetch", "--depth=1", "--filter=blob:none", "origin", commit])
        if not (destination / ".git").exists():
            git(mirror, ["worktree", "add", "--no-checkout", "--detach", str(destination), commit])
        patterns = "\n".join("*" + suffix for suffix in sorted(SOURCE_SUFFIXES))
        patterns += "\n" + "\n".join("!" + name + "/" for name in sorted(EXCLUDE_DIRS))
        git(destination, ["sparse-checkout", "set", "--no-cone", "--stdin"], patterns + "\n")
        git(destination, ["checkout", "--detach", "-q", commit])
        if git(destination, ["rev-parse", "HEAD"]).strip() != commit:
            raise ValueError("source checkout revision mismatch")
        marker.write_text(
            json.dumps(
                {
                    "source": "git sparse worktree",
                    "commit": commit,
                    "repository": repo,
                    "suffixes": sorted(SOURCE_SUFFIXES),
                    "note": (
                        "same extension/directory corpus filters as "
                        "retrieval; binary assets omitted"
                    ),
                }
            )
        )
        return destination


def retrieve_repository(case: InferenceCase, repo_dir: Path, config):
    documents = []
    total_bytes = 0
    for folder, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if (
                path.is_symlink()
                or path.suffix not in SOURCE_SUFFIXES
                or path.stat().st_size > 500_000
            ):
                continue
            data = path.read_bytes()
            if b"\x00" in data:
                continue
            total_bytes += len(data)
            if total_bytes > 100_000_000:
                raise RuntimeError("repository indexing exceeds 100 MB safety cap")
            relative = path.relative_to(repo_dir).as_posix()
            lines = data.decode("utf-8", errors="replace").split("\n")
            revision = sha256(data).hexdigest()
            width = config.retrieval_chunk_lines
            for first in range(0, len(lines), width):
                last = min(len(lines), first + width)
                content = "\n".join(f"{i + 1}: {lines[i]}" for i in range(first, last))
                documents.append(
                    Evidence(
                        id=f"{relative}:{first + 1}-{last}",
                        text=f"File: {relative}\nLines: {first + 1}-{last}\n{content}",
                        source_uri=relative,
                        source_revision=revision,
                        source_span=(first + 1, last),
                    )
                )
    ranked = rank_documents(case.question, documents)
    return ranked[: config.candidate_code_spans]
