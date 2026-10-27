"""Explicit host authorization and append-only activation order for trusted Skills."""

import json
from dataclasses import replace

from ..compiler.ir import ContextIRSegment
from ..core.snapshot import thaw
from ..errors import AuthorityViolation, SchemaInvalid
from .config import ConfigSource


def skill_part(part, *, name=None, role="developer"):
    """Project a text source as Skill instructions. CompilerPolicy.grants is still required.

    Calling this helper does NOT grant authority. The host must independently authorize the key.
    """
    if role not in {"system", "developer"}:
        raise ValueError("Skill instruction role must be system or developer")
    name = name or part.key

    def project(value, ctx):
        if not isinstance(value, str):
            raise SchemaInvalid(part.key, "Skill instructions must be text")
        payload = json.dumps(
            {"skill_id": part.key, "name": name, "instructions": value},
            sort_keys=True,
            ensure_ascii=False,
        )
        return [
            ContextIRSegment(
                part.key,
                part.key,
                ctx.revision,
                payload,
                kind="instruction",
                requested_role=role,
                trust_class="trusted_policy",
            )
        ]

    return replace(part, project=project)


class SkillRegistry:
    """Host-owned discovery index and lazy, ordered activation in one Context.

    Inactive Skills have no core instance and are never resolved. By default first activation
    appends a permanent slot; an explicit host order reserves fixed cross-session slots.
    Source edits, deactivation and reactivation never reorder existing slots.
    This is a host API, not a loader tool exposed to untrusted model output.
    """

    def __init__(self, context, *, index_key="standard.skill-index", role="developer", order=()):
        if role not in {"system", "developer"}:
            raise ValueError("invalid Skill role")
        self.context, self.index_key, self.role = context, index_key, role
        self._fixed_order = tuple(order)
        if len(self._fixed_order) != len(set(self._fixed_order)) or index_key in self._fixed_order:
            raise ValueError("duplicate Skill slot")
        self._entries = {}
        self._metadata = []
        self._activation_order = []
        self._active = set()
        self._registered = set()

        def project(value, ctx):
            text = json.dumps(
                {
                    "available_skills": thaw(value),
                    "policy": (
                        "Only activated Skill instructions apply; "
                        "index entries are discovery metadata."
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            return [
                ContextIRSegment(
                    index_key,
                    index_key,
                    ctx.revision,
                    text,
                    kind="instruction",
                    requested_role=role,
                    trust_class="trusted_policy",
                )
            ]

        self._index = ConfigSource(index_key, [], schema=list[dict], project=project)
        context.register(self._index.part)
        self._sync_policy()

    def register(self, part, *, name, description, trusted: bool, active=False):
        if trusted is not True:
            raise AuthorityViolation(part.key, "Skill activation requires explicit host approval")
        if part.key in self._entries or part.key in self.context.registry.definitions:
            raise ValueError(f"duplicate Skill: {part.key}")
        if not isinstance(name, str) or not name or not isinstance(description, str):
            raise ValueError("Skill name and description must be host-provided text")
        self._entries[part.key] = skill_part(part, name=name, role=self.role)
        self._metadata.append({"id": part.key, "name": name, "description": description})
        positions = {key: index for index, key in enumerate(self._fixed_order)}
        self._index.set(
            sorted(
                self._metadata,
                key=lambda item: (positions.get(item["id"], len(positions)), item["id"]),
            )
        )
        if active:
            self.activate(part.key)
        return self

    def activate(self, key):
        part = self._entries[key]
        if key in self._active:
            return
        if key not in self._registered:
            self.context.register(part)
            self._registered.add(key)
        else:
            self.context.enable(key)
        if key not in self._activation_order:
            self._activation_order.append(key)
        self._active.add(key)
        self._sync_policy()
        self.context.events.emit("skill_activate", part_id=key, reason="host_authorized")

    async def deactivate(self, key):
        if key not in self._entries:
            raise KeyError(key)
        if key in self._active:
            await self.context.disable(key)
            self._active.remove(key)
            self.context.events.emit("skill_deactivate", part_id=key)

    @property
    def active_order(self):
        return tuple(key for key in self._slot_order if key in self._active)

    @property
    def _slot_order(self):
        return (
            *self._fixed_order,
            *(key for key in self._activation_order if key not in self._fixed_order),
        )

    def selected_parts(self, *, extra=()):
        return [self.index_key, *self.active_order, *extra]

    def _sync_policy(self):
        policy = self.context.compiler.policy
        managed = {self.index_key, *self._entries, *self._fixed_order}
        order = [key for key in policy.part_order if key not in managed]
        grants = dict(policy.grants)
        grants[self.index_key] = self.role
        grants.update({key: self.role for key in self._registered})
        self.context.compiler.policy = replace(
            policy,
            grants=grants,
            part_order=tuple([*order, self.index_key, *self._slot_order]),
        )
