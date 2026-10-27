"""Measured 100K-fact memory/throughput check. No fabricated RSS or model-quality scores."""

import argparse
import gc
import json
import os
import platform
import resource
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel

from reactive_context.store import ContextStore, Principal


class Fact(BaseModel):
    text: str
    sequence: int


def rss_bytes():
    # Current RSS rather than ru_maxrss; units are consistent across macOS and Linux ps.
    return (
        int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024
    )


def run(path, count):
    actor = Principal("host", admin=True)
    rows = []
    start = perf_counter()
    with ContextStore(path, cache_bytes=65536, scope_cache_bytes=32768) as store:
        store.register_schema("measured", Fact)
        store.create_scope(actor, "benchmark")
        evidence = store.observe_evidence(actor, "benchmark", "benchmark://generator", "v02")
        for offset in range(0, count, 1000):
            with store.batch():
                for i in range(offset, min(count, offset + 1000)):
                    proposal = store.propose_fact(
                        actor,
                        "benchmark",
                        "measured",
                        f"fact-{i}",
                        {"text": "small fact " + str(i), "sequence": i},
                        [evidence],
                    )
                    record = store.commit(actor, proposal)
            if offset + 1000 in {1000, 10000, 100000, count}:
                store.cache.clear()
                store.db.execute("PRAGMA shrink_memory")
                gc.collect()
                rows.append(
                    {
                        "facts": min(count, offset + 1000),
                        "idle_rss_bytes": rss_bytes(),
                        "hot_cache": asdict(store.cache.stats),
                        "elapsed_seconds": perf_counter() - start,
                    }
                )
            if offset % 10000 == 0:
                print(
                    json.dumps({"phase": "ingest", "facts": min(count, offset + 1000)}), flush=True
                )
        assert store.db.execute("SELECT COUNT(*) FROM parts").fetchone()[0] == count
        store.activate(actor, "benchmark", record.id)
        assert len(store.active_refs(actor, "benchmark")) == 1
        assert store.cache.stats.bytes <= 65536
        integrity = store.integrity(actor)
        assert integrity == {"sqlite": "ok", "bad_revisions": []}
        return {
            "mode": "protocol_performance",
            "model_quality_score": None,
            "environment": {"python": platform.python_version(), "platform": platform.platform()},
            "facts": count,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024),
            "measurements": rows,
            "integrity": integrity,
            "elapsed_seconds": perf_counter() - start,
            "limits": [
                "RSS includes Python allocator and SQLite; a finite-"
                "machine measurement, not an asymptotic proof",
                "atomic batches of 1000; production throughput "
                "depends on transaction grouping and durability cost",
                "all facts use original public store APIs; payloads are small and stored inline",
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=100000)
    parser.add_argument("--output", default="results/v02/store-performance.json")
    args = parser.parse_args()
    if args.facts < 1000:
        raise ValueError("use at least 1000 facts")
    with tempfile.TemporaryDirectory(prefix="reactive-v02-store-") as directory:
        result = run(directory, args.facts)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("Wrote " + str(output), flush=True)


if __name__ == "__main__":
    main()
