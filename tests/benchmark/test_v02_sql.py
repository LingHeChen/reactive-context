import pytest

pytest.importorskip("openai")
pytest.importorskip("rank_bm25")

from pydantic_ai import ModelRetry  # noqa: E402

from reactive_context.benchmarking.dynamic_pilot import SqlWorld  # noqa: E402
from reactive_context.core.snapshot import digest  # noqa: E402


def test_real_sql_evaluator_and_execution_preconditions():
    world = SqlWorld(101, "all")
    old_revision = digest(world.schema())
    world.advance()
    with pytest.raises(ModelRetry, match="retired"):
        world.query(1, "SELECT id FROM orders", old_revision)
    with pytest.raises(ModelRetry, match="revision"):
        world.query(2, "SELECT id FROM orders", old_revision)
    current = digest(world.schema())
    values = world.query(
        2,
        "SELECT id FROM orders WHERE status='active' AND total>=35 ORDER BY id DESC LIMIT 1",
        current,
    )
    assert world.score([r["id"] for r in values]) == 1
    assert world.score([]) == 0
    with pytest.raises(ModelRetry):
        world.query(2, "DELETE FROM orders", current)
    with pytest.raises(ModelRetry):
        world.query(2, "SELECT randomblob(1000000000)", current)
    assert world.db.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 9
    world.db.close()
