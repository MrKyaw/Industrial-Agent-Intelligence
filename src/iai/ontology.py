"""Facility ontology: typed assets, sensors, derived tags, and relationships.

The ontology is deliberately small and boring: JSON in, validated graph out.
That keeps agent context inspectable and makes hallucinated asset names easy to
detect before any tool is called.
"""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

NODE_TYPES = {
    "site",
    "zone",
    "equipment",
    "sensor",
    "setpoint",
    "derived_tag",
    "alarm_definition",
    "loop",
}

RELATIONS = {
    "part_of",
    "measures",
    "controls",
    "serves",
    "depends_on",
    "indicates",
    "monitors",
    "located_in",
}

SEVERITIES = {"low", "medium", "high", "critical"}


class OntologyError(ValueError):
    """Raised when an ontology cannot be loaded or validated."""


class UnknownEntityError(OntologyError):
    """Raised when an id or alias is not present in the ontology."""


@dataclass
class OntologyNode:
    id: str
    type: str
    name: str
    site_id: str | None = None
    subtype: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OntologyNode":
        if not isinstance(payload, dict):
            raise OntologyError("node must be an object")
        node_id = payload.get("id")
        node_type = payload.get("type")
        name = payload.get("name")
        if not isinstance(node_id, str) or not node_id:
            raise OntologyError("node.id must be a non-empty string")
        if not isinstance(node_type, str) or not node_type:
            raise OntologyError(f"node {node_id}: type must be a non-empty string")
        if not isinstance(name, str) or not name:
            raise OntologyError(f"node {node_id}: name must be a non-empty string")
        attributes = payload.get("attributes", {})
        if not isinstance(attributes, dict):
            raise OntologyError(f"node {node_id}: attributes must be an object")
        tags = payload.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise OntologyError(f"node {node_id}: tags must be a list of strings")
        site_id = payload.get("site_id")
        subtype = payload.get("subtype")
        if site_id is not None and not isinstance(site_id, str):
            raise OntologyError(f"node {node_id}: site_id must be a string")
        if subtype is not None and not isinstance(subtype, str):
            raise OntologyError(f"node {node_id}: subtype must be a string")
        return cls(
            id=node_id,
            type=node_type,
            name=name,
            site_id=site_id,
            subtype=subtype,
            attributes=attributes,
            tags=tags,
        )

    def alias_candidates(self) -> list[str]:
        candidates = [self.id, self.name]
        point = self.attributes.get("point")
        if isinstance(point, str):
            candidates.extend([point, point.split(".")[-1]])
        candidates.append(self.id.split(".")[-1])
        aliases = self.attributes.get("aliases", [])
        if isinstance(aliases, list):
            candidates.extend(str(alias) for alias in aliases)
        return [candidate for candidate in candidates if candidate]

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "attributes": self.attributes,
            "tags": self.tags,
        }
        if self.site_id is not None:
            payload["site_id"] = self.site_id
        if self.subtype is not None:
            payload["subtype"] = self.subtype
        return payload


@dataclass
class OntologyEdge:
    src: str
    dst: str
    relation: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OntologyEdge":
        if not isinstance(payload, dict):
            raise OntologyError("edge must be an object")
        src, dst, relation = payload.get("src"), payload.get("dst"), payload.get("relation")
        if not all(isinstance(value, str) and value for value in (src, dst, relation)):
            raise OntologyError("edge requires non-empty src, dst, and relation strings")
        attributes = payload.get("attributes", {})
        if not isinstance(attributes, dict):
            raise OntologyError("edge attributes must be an object")
        return cls(src=src, dst=dst, relation=relation, attributes=attributes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "relation": self.relation,
            "attributes": self.attributes,
        }


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            preview = "; ".join(self.errors[:5])
            raise OntologyError(f"ontology validation failed ({len(self.errors)} errors): {preview}")

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def _normalize_alias(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


class Ontology:
    """Validated, queryable facility knowledge graph."""

    def __init__(
        self,
        nodes: Iterable[OntologyNode],
        edges: Iterable[OntologyEdge],
        *,
        version: str = "1.0.0",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.version = version
        self.metadata = metadata or {}
        self._nodes: dict[str, OntologyNode] = {}
        self._edges: list[OntologyEdge] = []
        self._out: dict[str, list[OntologyEdge]] = {}
        self._in: dict[str, list[OntologyEdge]] = {}
        for node in nodes:
            if node.id in self._nodes:
                raise OntologyError(f"duplicate node id: {node.id}")
            self._nodes[node.id] = node
        for edge in edges:
            self._edges.append(edge)
            self._out.setdefault(edge.src, []).append(edge)
            self._in.setdefault(edge.dst, []).append(edge)
        self._alias_index: dict[str, list[str]] = {}
        for node in self._nodes.values():
            for alias in node.alias_candidates():
                self._alias_index.setdefault(_normalize_alias(alias), []).append(node.id)
        for ids in self._alias_index.values():
            ids.sort()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Ontology":
        if not isinstance(payload, dict):
            raise OntologyError("ontology must be a JSON object")
        raw_nodes = payload.get("nodes")
        raw_edges = payload.get("edges", [])
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise OntologyError("ontology.nodes must be a non-empty list")
        if not isinstance(raw_edges, list):
            raise OntologyError("ontology.edges must be a list")
        nodes = [OntologyNode.from_dict(item) for item in raw_nodes]
        edges = [OntologyEdge.from_dict(item) for item in raw_edges]
        return cls(
            nodes,
            edges,
            version=str(payload.get("version", "1.0.0")),
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "Ontology":
        with Path(path).open(encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise OntologyError(f"invalid ontology JSON: {exc}") from exc
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "metadata": self.metadata,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [edge.to_dict() for edge in self._edges],
        }

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        if not any(node.type == "site" for node in self._nodes.values()):
            report.errors.append("ontology must contain at least one site")
        for node in self._nodes.values():
            if node.type not in NODE_TYPES:
                report.errors.append(f"{node.id}: unknown node type {node.type!r}")
            if node.type == "site":
                if node.site_id is not None:
                    report.errors.append(f"{node.id}: site nodes must not declare site_id")
            else:
                if not node.site_id:
                    report.errors.append(f"{node.id}: non-site node requires site_id")
                elif node.site_id not in self._nodes:
                    report.errors.append(f"{node.id}: unknown site_id {node.site_id!r}")
                elif self._nodes[node.site_id].type != "site":
                    report.errors.append(f"{node.id}: site_id must reference a site node")
            if node.type == "sensor":
                unit = node.attributes.get("unit")
                point = node.attributes.get("point")
                if not isinstance(unit, str) or not unit:
                    report.errors.append(f"{node.id}: sensor requires attributes.unit")
                if not isinstance(point, str) or not point:
                    report.errors.append(f"{node.id}: sensor requires attributes.point")
            if node.type == "setpoint":
                self._validate_setpoint(node, report)
            if node.type == "derived_tag":
                self._validate_derived(node, report)
            if node.type == "alarm_definition":
                self._validate_alarm(node, report)
            if node.type == "equipment":
                criticality = node.attributes.get("criticality")
                if criticality is not None and (
                    not isinstance(criticality, int) or not 1 <= criticality <= 5
                ):
                    report.errors.append(f"{node.id}: criticality must be an integer in [1, 5]")
            relation_targets = {edge.dst for edge in self._out.get(node.id, [])}
            relation_sources = {edge.src for edge in self._in.get(node.id, [])}
            if node.type == "sensor" and not (relation_targets or relation_sources):
                report.warnings.append(f"{node.id}: sensor has no relationships")
        seen: set[tuple[str, str, str]] = set()
        for edge in self._edges:
            if edge.src not in self._nodes:
                report.errors.append(f"edge references unknown src {edge.src!r}")
            if edge.dst not in self._nodes:
                report.errors.append(f"edge references unknown dst {edge.dst!r}")
            if edge.src == edge.dst:
                report.errors.append(f"self-referential edge on {edge.src}")
            if edge.relation not in RELATIONS:
                report.errors.append(f"unknown relation {edge.relation!r}")
            key = (edge.src, edge.dst, edge.relation)
            if key in seen:
                report.errors.append(f"duplicate edge {key}")
            seen.add(key)
            if edge.src in self._nodes and edge.dst in self._nodes:
                src_site = self.site_of(edge.src)
                dst_site = self.site_of(edge.dst)
                if src_site and dst_site and src_site != dst_site:
                    report.warnings.append(
                        f"cross-site edge {edge.src} -> {edge.dst} ({src_site} vs {dst_site})"
                    )
        report.errors.extend(self._derived_cycles())
        return report

    def _validate_setpoint(self, node: OntologyNode, report: ValidationReport) -> None:
        unit = node.attributes.get("unit")
        minimum = node.attributes.get("min")
        maximum = node.attributes.get("max")
        writable = node.attributes.get("writable")
        if not isinstance(unit, str) or not unit:
            report.errors.append(f"{node.id}: setpoint requires attributes.unit")
        if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
            report.errors.append(f"{node.id}: setpoint requires numeric min and max")
        elif minimum >= maximum:
            report.errors.append(f"{node.id}: setpoint min must be below max")
        if not isinstance(writable, bool):
            report.errors.append(f"{node.id}: setpoint requires boolean writable")
        default = node.attributes.get("default")
        if default is not None and isinstance(minimum, (int, float)) and isinstance(maximum, (int, float)):
            if not minimum <= float(default) <= maximum:
                report.errors.append(f"{node.id}: default setpoint outside [min, max]")
        deadband = node.attributes.get("deadband")
        if deadband is not None and (not isinstance(deadband, (int, float)) or deadband < 0):
            report.errors.append(f"{node.id}: deadband must be a non-negative number")

    def _validate_derived(self, node: OntologyNode, report: ValidationReport) -> None:
        expression = node.attributes.get("expression")
        inputs = node.attributes.get("inputs")
        if not isinstance(expression, str) or not expression.strip():
            report.errors.append(f"{node.id}: derived tag requires attributes.expression")
        if not isinstance(inputs, dict) or not inputs:
            report.errors.append(f"{node.id}: derived tag requires attributes.inputs mapping")
            return
        for variable, source in inputs.items():
            if not isinstance(variable, str) or not isinstance(source, str):
                report.errors.append(f"{node.id}: derived input keys and values must be strings")
            elif source not in self._nodes:
                report.errors.append(f"{node.id}: derived input {variable!r} references {source!r}")
        unit = node.attributes.get("unit")
        if not isinstance(unit, str) or not unit:
            report.warnings.append(f"{node.id}: derived tag has no unit")

    def _validate_alarm(self, node: OntologyNode, report: ValidationReport) -> None:
        severity = node.attributes.get("severity")
        threshold = node.attributes.get("threshold")
        tag = node.attributes.get("tag")
        if severity not in SEVERITIES:
            report.errors.append(f"{node.id}: alarm severity must be one of {sorted(SEVERITIES)}")
        if not isinstance(threshold, (int, float)):
            report.errors.append(f"{node.id}: alarm threshold must be numeric")
        if not isinstance(tag, str) or tag not in self._nodes:
            report.errors.append(f"{node.id}: alarm tag must reference an existing node")

    def _derived_cycles(self) -> list[str]:
        derived = {
            node.id: [
                source
                for source in node.attributes.get("inputs", {}).values()
                if isinstance(source, str)
                and source in self._nodes
                and self._nodes[source].type == "derived_tag"
            ]
            for node in self._nodes.values()
            if node.type == "derived_tag"
        }
        state: dict[str, int] = {}
        errors: list[str] = []

        def visit(node_id: str, stack: list[str]) -> None:
            if state.get(node_id) == 1:
                cycle = " -> ".join([*stack, node_id])
                errors.append(f"derived tag cycle: {cycle}")
                return
            if state.get(node_id) == 2:
                return
            state[node_id] = 1
            for child in derived.get(node_id, []):
                visit(child, [*stack, node_id])
            state[node_id] = 2

        for node_id in sorted(derived):
            visit(node_id, [])
        return errors

    def get(self, node_id: str) -> OntologyNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise UnknownEntityError(f"unknown entity: {node_id}") from exc

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def nodes(self) -> list[OntologyNode]:
        return list(self._nodes.values())

    def edges(self) -> list[OntologyEdge]:
        return list(self._edges)

    def find(
        self,
        *,
        node_type: str | None = None,
        site_id: str | None = None,
        tag: str | None = None,
        subtype: str | None = None,
    ) -> list[OntologyNode]:
        results = []
        for node in self._nodes.values():
            if node_type is not None and node.type != node_type:
                continue
            if site_id is not None and node.site_id != site_id:
                continue
            if tag is not None and tag.lower() not in {item.lower() for item in node.tags}:
                continue
            if subtype is not None and node.subtype != subtype:
                continue
            results.append(node)
        return sorted(results, key=lambda item: item.id)

    def sites(self) -> list[OntologyNode]:
        return self.find(node_type="site")

    def equipment(self, site_id: str | None = None) -> list[OntologyNode]:
        return self.find(node_type="equipment", site_id=site_id)

    def resolve(self, alias: str) -> OntologyNode:
        """Resolve an id, point name, short name, or alias to a node."""
        matches = self._alias_index.get(_normalize_alias(alias), [])
        if not matches:
            raise UnknownEntityError(f"unknown alias: {alias}")
        if len(matches) > 1:
            preferred = [node_id for node_id in matches if self._nodes[node_id].type != "alarm_definition"]
            matches = preferred or matches
        return self._nodes[matches[0]]

    def try_resolve(self, alias: str) -> OntologyNode | None:
        try:
            return self.resolve(alias)
        except UnknownEntityError:
            return None

    def find_in_text(self, text: str, node_type: str | None = None) -> OntologyNode | None:
        """Find the longest node alias mentioned in free text."""
        normalized = _normalize_alias(text)
        candidates: list[tuple[int, str]] = []
        for alias, node_ids in self._alias_index.items():
            if len(alias) < 3:
                continue
            if not re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", normalized):
                continue
            for node_id in node_ids:
                node = self._nodes[node_id]
                if node_type is not None and node.type != node_type:
                    continue
                candidates.append((len(alias), node_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return self._nodes[candidates[0][1]]

    def neighbors(
        self,
        node_id: str,
        *,
        relation: str | None = None,
        direction: str = "out",
        node_type: str | None = None,
    ) -> list[OntologyNode]:
        if direction not in {"out", "in", "both"}:
            raise ValueError("direction must be 'out', 'in', or 'both'")
        self.get(node_id)
        found: list[OntologyNode] = []
        edge_sets: list[OntologyEdge] = []
        if direction in {"out", "both"}:
            edge_sets.extend(self._out.get(node_id, []))
        if direction in {"in", "both"}:
            edge_sets.extend(self._in.get(node_id, []))
        for edge in edge_sets:
            if relation is not None and edge.relation != relation:
                continue
            other_id = edge.dst if edge.src == node_id else edge.src
            other = self._nodes[other_id]
            if node_type is None or other.type == node_type:
                found.append(other)
        unique = {node.id: node for node in found}
        return sorted(unique.values(), key=lambda node: node.id)

    def sensors_for(self, equipment_id: str) -> list[OntologyNode]:
        sensors = {
            node.id: node
            for node in self.neighbors(equipment_id, relation="measures", direction="both")
            if node.type == "sensor"
        }
        for child in self.neighbors(equipment_id, relation="part_of", direction="in"):
            for sensor in self.neighbors(child.id, relation="measures", direction="both"):
                if sensor.type == "sensor":
                    sensors[sensor.id] = sensor
        return sorted(sensors.values(), key=lambda node: node.id)

    def equipment_for_tag(self, tag_id: str) -> OntologyNode | None:
        node = self.get(tag_id)
        if node.type == "equipment":
            return node
        candidates = self.neighbors(tag_id, direction="both", node_type="equipment")
        if candidates:
            return candidates[0]
        if node.site_id:
            equipment = self.equipment(node.site_id)
            return equipment[0] if len(equipment) == 1 else None
        return None

    def site_of(self, node_id: str) -> str | None:
        node = self.get(node_id)
        if node.type == "site":
            return node.id
        return node.site_id

    def shortest_path(self, src: str, dst: str, *, max_depth: int = 6) -> list[str] | None:
        if src == dst:
            return [src]
        self.get(src)
        self.get(dst)
        queue: deque[tuple[str, list[str]]] = deque([(src, [src])])
        seen = {src}
        while queue:
            current, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for edge in self._out.get(current, []):
                if edge.dst in seen:
                    continue
                next_path = [*path, edge.dst]
                if edge.dst == dst:
                    return next_path
                seen.add(edge.dst)
                queue.append((edge.dst, next_path))
        return None

    def subgraph(self, node_id: str, hops: int = 1) -> list[OntologyNode]:
        if hops < 0:
            raise ValueError("hops must be non-negative")
        self.get(node_id)
        seen = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for current in sorted(frontier):
                for neighbor in self.neighbors(current, direction="both"):
                    if neighbor.id not in seen:
                        seen.add(neighbor.id)
                        next_frontier.add(neighbor.id)
            frontier = next_frontier
        return [self._nodes[node_id] for node_id in sorted(seen)]

    def derived_tags(self) -> list[OntologyNode]:
        return self.find(node_type="derived_tag")

    def alarm_definitions(self, site_id: str | None = None) -> list[OntologyNode]:
        return self.find(node_type="alarm_definition", site_id=site_id)

    def describe(self, node_id: str, *, hops: int = 0) -> str:
        node = self.get(node_id)
        parts = [f"{node.name} ({node.id}, {node.type})"]
        if node.subtype:
            parts.append(f"subtype={node.subtype}")
        if node.site_id:
            parts.append(f"site={node.site_id}")
        for key in sorted(node.attributes):
            value = node.attributes[key]
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}={value}")
        if node.tags:
            parts.append(f"tags={','.join(sorted(node.tags))}")
        if hops:
            related = self.neighbors(node_id, direction="both")
            if related:
                parts.append("related=" + ", ".join(item.id for item in related))
        return "; ".join(parts)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for node in self._nodes.values():
            counts[node.type] = counts.get(node.type, 0) + 1
        return {
            "version": self.version,
            "nodes": len(self._nodes),
            "edges": len(self._edges),
            "by_type": counts,
            "sites": [site.id for site in self.sites()],
        }


def load_ontology(path: str | Path, *, validate: bool = True) -> Ontology:
    """Load and optionally validate an ontology file."""
    ontology = Ontology.from_path(path)
    if validate:
        ontology.validate().raise_for_errors()
    return ontology
