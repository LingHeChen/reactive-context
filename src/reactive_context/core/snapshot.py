"""Deeply immutable values and deterministic snapshot identities."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from hashlib import sha256
from pathlib import PurePath
from types import MappingProxyType
from typing import Any, TypeVar
from uuid import UUID

from pydantic import BaseModel

T = TypeVar("T")


class FrozenRecord(Mapping):
    """Immutable structural view of model/dataclass values, supporting attribute access."""

    __slots__ = ("_data",)

    def __init__(self, values):
        object.__setattr__(self, "_data", MappingProxyType(dict(values)))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattribute__(self, key):
        if not key.startswith("_"):
            data = object.__getattribute__(self, "_data")
            if key in data:
                return data[key]
        return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        raise TypeError("snapshot is immutable")


def freeze(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return FrozenRecord({k: freeze(getattr(value, k)) for k in type(value).model_fields})
    if is_dataclass(value) and not isinstance(value, type):
        return FrozenRecord({f.name: freeze(getattr(value, f.name)) for f in fields(value)})
    if isinstance(value, Mapping):
        return MappingProxyType({k: freeze(v) for k, v in Mapping.items(value)})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(v) for v in value)
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return value
    if isinstance(value, (date, datetime, time, Decimal, UUID, PurePath)):
        return value
    if isinstance(value, Enum):
        return freeze(value.value)
    raise TypeError(f"unsupported mutable snapshot value: {type(value).__name__}")


def thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in Mapping.items(value)}
    if isinstance(value, (tuple, list)):
        return [thaw(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((thaw(v) for v in value), key=repr)
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, (date, datetime, time, Decimal, UUID, PurePath)):
        return {"type": type(value).__name__, "value": str(value)}
    return value


def digest(value: Any) -> str:
    return sha256(
        json.dumps(
            thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PartSnapshot[T]:
    part_id: str
    scope_id: str
    resource_id: str
    revision: str
    generation: int
    value: T
    observed_at: float
    verified_at: float
    status: str = "CURRENT"
    provenance: tuple[tuple[str, str], ...] = ()
    event_time: float | None = None
    valid_from: float | None = None
    valid_to: float | None = None


@dataclass(frozen=True)
class ContextSnapshot:
    id: str
    scope_id: str
    parts: tuple[PartSnapshot, ...]
    segments: tuple[Any, ...]
    compiler_config_revision: str
    policy_revision: str
    consistency: str
    metadata: Any = None

    @property
    def revision_vector(self) -> Mapping[str, str]:
        return MappingProxyType({p.part_id: p.revision for p in self.parts})

    def part(self, key: str) -> PartSnapshot:
        return next(p for p in self.parts if p.part_id == key)
