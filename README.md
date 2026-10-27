# Reactive Context

Reactive context compilation for agents. Python 3.12+.

[Project showcase](https://linghechen.github.io/reactive-context/)

```sh
uv sync --frozen --all-extras --group benchmark
git clone https://github.com/ZexueHe/MemoryArena.git .benchmark-cache/repos/memoryarena
git -C .benchmark-cache/repos/memoryarena checkout 6cd9de14b71915e39ac742a20dc33785e14b6aab
uv run pytest -q
uv run python examples/three_parts/main.py
```

MIT licensed. The showcase uses a recorded deterministic model and real SQLite tool execution; it does not measure real-model task quality.
