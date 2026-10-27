"""Standard host-composable parts; grants still must be configured on CompilerPolicy."""

from .compiler.ir import ContextIRSegment
from .sources.catalog import ToolCatalogSource
from .sources.config import ConfigSource
from .sources.file import FileSource


def _part(key, value, *, role="user", kind="state", **kwargs):
    def project(value, ctx):
        return [
            ContextIRSegment(
                key,
                key,
                ctx.revision,
                value,
                kind=kind,
                requested_role=role,
                trust_class="application" if role != "user" else "external_data",
            )
        ]

    return ConfigSource(key, value, project=project, **kwargs).part


def system(content, *, key="standard.system"):
    return _part(key, content, role="system", kind="instruction")


def developer(content, *, key="standard.developer"):
    return _part(key, content, role="developer", kind="instruction")


def task(content, *, key="standard.task"):
    return _part(key, content)


def state(value, *, key="standard.state", schema=None):
    return _part(key, value, schema=schema)


def skills(value, *, key="standard.skills"):
    return _part(key, value, kind="evidence")


def history(events=(), *, key="standard.history"):
    # Host-owned raw history stays in the runtime. Optional supplied history is immutable data.
    return _part(key, list(events), kind="history")


def capabilities(tools, *, key="standard.capabilities"):
    return ToolCatalogSource(key, tools).part


def tool_discovery(tools, *, key="standard.tool-discovery"):
    return capabilities(tools, key=key)


def file(path, *, root, key="standard.file", **kwargs):
    return FileSource(key, path, root=root, **kwargs).part
