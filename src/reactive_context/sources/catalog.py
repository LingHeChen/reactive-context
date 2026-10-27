from ..compiler.ir import ContextIRSegment
from .config import ConfigSource


class ToolCatalogSource(ConfigSource):
    """Schema discovery only. The adapter can select existing executable host tools."""

    def __init__(self, key, tools):
        def project(value, ctx):
            return [ContextIRSegment(key, key, ctx.revision, value, kind="tool_definition")]

        super().__init__(key, tools, schema=list[dict], project=project)
