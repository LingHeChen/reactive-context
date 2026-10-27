"""Small extension protocol, normalized into the existing PartDefinition and Compiler."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.part import PartDefinition, ResolveContext
from ..core.snapshot import digest, freeze


@dataclass(frozen=True)
class PartSpec:
    key: str
    declaration_version: str = "1"
    scope: str = "instance"
    mode: str = "source"
    required: bool = True
    schema: Any = Any


@dataclass(frozen=True)
class PartMaterialization:
    value: Any
    revision: str | None = None
    sources: tuple[str, ...] = ()
    dependency_versions: Mapping = field(default_factory=dict)
    trust: str = "external_data"
    consistency: str = "best_effort"

    def __post_init__(self):
        if self.trust != "external_data":
            raise ValueError("providers cannot grant themselves instruction authority")
        object.__setattr__(self, "value", freeze(self.value))
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "dependency_versions", freeze(self.dependency_versions))

    @property
    def content_hash(self):
        return digest(self.value)


class ContextPart(Protocol):
    @property
    def spec(self) -> PartSpec: ...
    def dependencies(self) -> tuple[str, ...]: ...
    async def resolve(self, reader) -> Any: ...
    async def materialize(self, reader, value) -> PartMaterialization: ...


class LegacyPartAdapter:
    """Keep legacy hooks/ACL/lifecycle in their original runtime, without invented guarantees."""

    def __init__(self, definition: PartDefinition):
        self.definition = definition
        self.spec = PartSpec(
            definition.key,
            str(definition.schema_version),
            required=definition.required,
            schema=definition.schema,
        )

    def dependencies(self):
        return self.definition.deps

    async def resolve(self, reader):
        from ..core.scheduler import invoke

        return await invoke(self.definition.resolve, reader)

    async def materialize(self, reader: ResolveContext, value):
        from ..core.scheduler import invoke

        revision = (
            await invoke(self.definition.revision, reader) if self.definition.revision else None
        )
        return PartMaterialization(
            value,
            revision,
            (self.definition.resource_key or self.definition.key,),
            {k: v.revision for k, v in reader.dependencies.items()},
            consistency="versioned" if revision is not None else "best_effort",
        )


def normalize_part(value):
    if isinstance(value, LegacyPartAdapter):
        return value.definition
    if isinstance(value, PartDefinition):
        return value
    # Protocol plugins use declared dependencies, so the core DAG remains authoritative.
    spec = value.spec

    async def resolve(reader):
        from ..core.scheduler import invoke

        result = await invoke(value.resolve, reader)
        result = await invoke(value.materialize, reader, result)
        if not isinstance(result, PartMaterialization):
            raise TypeError("materialize must return PartMaterialization")
        return result.value

    return PartDefinition(
        spec.key,
        spec.schema,
        resolve,
        deps=tuple(value.dependencies()),
        revision=getattr(value, "revision", None),
        required=spec.required,
        algorithm_version=spec.declaration_version,
        acl_check=getattr(value, "acl_check", None),
        subscribe=getattr(value, "subscribe", None),
        dispose=getattr(value, "dispose", None),
    )
