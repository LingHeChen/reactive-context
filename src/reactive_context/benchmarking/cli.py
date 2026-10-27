import argparse
import asyncio
from pathlib import Path

from .config import BenchmarkConfig
from .official import ROOT
from .report import write_report


def main():
    parser = argparse.ArgumentParser(
        description="Public benchmark pilots with pinned official sources"
    )
    parser.add_argument(
        "command", choices=["longmemeval", "contextbench", "report", "audit", "prepare"]
    )
    parser.add_argument("--env", default=str(ROOT / ".env"))
    parser.add_argument("--output", default=str(ROOT / "results/public-pilot"))
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    config = BenchmarkConfig.load(args.env)
    if args.command == "longmemeval":
        from .runner import run_longmemeval

        asyncio.run(run_longmemeval(Path(args.output), config, retry_failures=args.retry_failures))
    elif args.command == "report":
        write_report(args.output)
    elif args.command == "contextbench":
        from .code_track import run_contextbench

        asyncio.run(run_contextbench(Path(args.output), config))
    elif args.command == "prepare":
        from .prepare import prepare_inputs

        prepare_inputs()
    else:
        from .audit import run_audit

        run_audit(Path(args.output))


if __name__ == "__main__":
    main()
