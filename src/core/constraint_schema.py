"""
Constraint type classification and output-based verification.

- Classifies query into constraint type from entities + question text
- Verifies constraint effect in bus_schedule_after.json output
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional


class ConstraintType(Enum):
    MUST_ASSIGN = "must_assign"
    MUST_NOT_ASSIGN = "must_not_assign"
    CAPACITY = "capacity"
    ORDERING = "ordering"
    SAME_BUS = "same_bus"
    NOT_SAME_BUS = "not_same_bus"
    MAX_STOPS = "max_stops"


def classify_constraint(query: Dict[str, Any]) -> "ConstraintSpec":
    """
    Classify a query into a constraint type and build its spec.

    Args:
        query: Dict with 'question' (str) and 'entities' (dict)

    Returns:
        ConstraintSpec with type, question, and entities
    """
    question = query.get("question", "")
    entities = query.get("entities", {})
    question_lower = question.lower()

    if "max_stops" in entities:
        ctype = ConstraintType.MAX_STOPS
    elif "stop_id_1" in entities and "stop_id_2" in entities:
        if "same bus" in question_lower and "not" in question_lower:
            ctype = ConstraintType.NOT_SAME_BUS
        elif "same bus" in question_lower:
            ctype = ConstraintType.SAME_BUS
        else:
            ctype = ConstraintType.ORDERING
    elif "capacity" in entities:
        ctype = ConstraintType.CAPACITY
    elif "stop_id" in entities and "bus_id" in entities:
        if "must not" in question_lower or "must't" in question_lower:
            ctype = ConstraintType.MUST_NOT_ASSIGN
        else:
            ctype = ConstraintType.MUST_ASSIGN
    else:
        # Fallback: try to infer from question text
        if "before" in question_lower or "order" in question_lower:
            ctype = ConstraintType.ORDERING
        elif "capacity" in question_lower or "reduced" in question_lower:
            ctype = ConstraintType.CAPACITY
        elif "not" in question_lower:
            ctype = ConstraintType.MUST_NOT_ASSIGN
        else:
            ctype = ConstraintType.MUST_ASSIGN

    return ConstraintSpec(
        constraint_type=ctype,
        question=question,
        entities=entities,
    )


class ConstraintSpec:
    __slots__ = ("constraint_type", "question", "entities")

    def __init__(self, constraint_type: ConstraintType, question: str, entities: Dict[str, Any]):
        self.constraint_type = constraint_type
        self.question = question
        self.entities = entities


# ---------------------------------------------------------------------------
# Output-based verification (bus_schedule_after.json)
# ---------------------------------------------------------------------------

def verify_output(
    bus_schedule_path: str,
    spec: ConstraintSpec,
) -> Dict[str, Any]:
    """
    Verify that a constraint is actually in effect in the solver output.

    Reads bus_schedule_after.json and checks whether the constraint's
    postcondition is satisfied in the routing result.

    Returns:
        Dict with 'satisfied' (bool|None), 'detail' (str), 'evidence' (dict)
    """
    try:
        with open(bus_schedule_path, "r") as f:
            schedule = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"satisfied": None, "detail": f"Could not read schedule: {e}", "evidence": {}}

    verifiers = {
        ConstraintType.MUST_ASSIGN: _verify_must_assign,
        ConstraintType.MUST_NOT_ASSIGN: _verify_must_not_assign,
        ConstraintType.CAPACITY: _verify_capacity,
        ConstraintType.ORDERING: _verify_ordering,
        ConstraintType.SAME_BUS: _verify_same_bus,
        ConstraintType.NOT_SAME_BUS: _verify_not_same_bus,
        ConstraintType.MAX_STOPS: _verify_max_stops,
    }

    verifier = verifiers.get(spec.constraint_type)
    if verifier:
        return verifier(schedule, spec)
    return {"satisfied": None, "detail": "Unknown constraint type", "evidence": {}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_stop_on_bus(schedule: List[Dict], stop_id: int) -> Optional[int]:
    """Find which working bus_id serves a given stop_id. Skips broken buses."""
    for bus in schedule:
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        for workload in bus.get("workload_list", []):
            for stop in workload.get("stops", []):
                if stop.get("stop_id") == stop_id:
                    return bus.get("bus_id")
    return None


def _get_stop_indices_on_bus(schedule: List[Dict], bus_id: int) -> Dict[int, int]:
    """Get {stop_id: index} for all stops on a given bus (across all workloads)."""
    indices: Dict[int, int] = {}
    idx = 0
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        for workload in bus.get("workload_list", []):
            for stop in workload.get("stops", []):
                indices[stop.get("stop_id")] = idx
                idx += 1
    return indices


def _get_max_load_on_bus(schedule: List[Dict], bus_id: int) -> Optional[int]:
    """Get the maximum running_load for a working bus across all stops."""
    max_load = None
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        for workload in bus.get("workload_list", []):
            for stop in workload.get("stops", []):
                load = stop.get("running_load", 0)
                if max_load is None or load > max_load:
                    max_load = load
    return max_load


def _count_stops_on_bus(schedule: List[Dict], bus_id: int) -> int:
    """Count total stops assigned to a working bus."""
    count = 0
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        for workload in bus.get("workload_list", []):
            count += len(workload.get("stops", []))
    return count


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

def _verify_must_assign(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    stop_id = spec.entities.get("stop_id")
    bus_id = spec.entities.get("bus_id")
    actual_bus = _find_stop_on_bus(schedule, stop_id)

    if actual_bus is None:
        return {
            "satisfied": False,
            "detail": f"Stop {stop_id} not found in any bus route",
            "evidence": {"stop_id": stop_id, "expected_bus": bus_id, "actual_bus": None},
        }

    satisfied = actual_bus == bus_id
    return {
        "satisfied": satisfied,
        "detail": (
            f"Stop {stop_id} is on bus {actual_bus}"
            + ("" if satisfied else f" (expected bus {bus_id})")
        ),
        "evidence": {"stop_id": stop_id, "expected_bus": bus_id, "actual_bus": actual_bus},
    }


def _verify_must_not_assign(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    stop_id = spec.entities.get("stop_id")
    bus_id = spec.entities.get("bus_id")
    actual_bus = _find_stop_on_bus(schedule, stop_id)

    if actual_bus is None:
        return {
            "satisfied": True,
            "detail": f"Stop {stop_id} not found on any bus (constraint trivially satisfied)",
            "evidence": {"stop_id": stop_id, "excluded_bus": bus_id, "actual_bus": None},
        }

    satisfied = actual_bus != bus_id
    return {
        "satisfied": satisfied,
        "detail": (
            f"Stop {stop_id} is on bus {actual_bus}"
            + ("" if satisfied else f" (should NOT be on bus {bus_id})")
        ),
        "evidence": {"stop_id": stop_id, "excluded_bus": bus_id, "actual_bus": actual_bus},
    }


def _verify_capacity(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    bus_id = spec.entities.get("bus_id")
    capacity = spec.entities.get("capacity")
    max_load = _get_max_load_on_bus(schedule, bus_id)

    if max_load is None:
        return {
            "satisfied": None,
            "detail": f"Bus {bus_id} not found in schedule or has no stops",
            "evidence": {"bus_id": bus_id, "capacity_limit": capacity, "max_load": None},
        }

    satisfied = max_load <= capacity
    return {
        "satisfied": satisfied,
        "detail": (
            f"Bus {bus_id} max load is {max_load}"
            + (f" (<= {capacity})" if satisfied else f" (exceeds capacity {capacity})")
        ),
        "evidence": {"bus_id": bus_id, "capacity_limit": capacity, "max_load": max_load},
    }


def _verify_ordering(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    stop_1 = spec.entities.get("stop_id_1")
    stop_2 = spec.entities.get("stop_id_2")

    for bus in schedule:
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        bus_id = bus.get("bus_id")
        indices = _get_stop_indices_on_bus(schedule, bus_id)
        if stop_1 in indices and stop_2 in indices:
            idx_1 = indices[stop_1]
            idx_2 = indices[stop_2]
            satisfied = idx_1 < idx_2
            return {
                "satisfied": satisfied,
                "detail": (
                    f"On bus {bus_id}: stop {stop_1} at index {idx_1}, "
                    f"stop {stop_2} at index {idx_2}"
                    + ("" if satisfied else " (wrong order!)")
                ),
                "evidence": {
                    "bus_id": bus_id,
                    "stop_1": stop_1, "index_1": idx_1,
                    "stop_2": stop_2, "index_2": idx_2,
                },
            }

    bus_1 = _find_stop_on_bus(schedule, stop_1)
    bus_2 = _find_stop_on_bus(schedule, stop_2)
    return {
        "satisfied": None,
        "detail": (
            f"Stops not on same bus: stop {stop_1} on bus {bus_1}, "
            f"stop {stop_2} on bus {bus_2} (ordering N/A)"
        ),
        "evidence": {"stop_1": stop_1, "bus_1": bus_1, "stop_2": stop_2, "bus_2": bus_2},
    }


def _verify_same_bus(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    stop_1 = spec.entities.get("stop_id_1")
    stop_2 = spec.entities.get("stop_id_2")
    bus_1 = _find_stop_on_bus(schedule, stop_1)
    bus_2 = _find_stop_on_bus(schedule, stop_2)

    if bus_1 is None or bus_2 is None:
        missing = []
        if bus_1 is None:
            missing.append(str(stop_1))
        if bus_2 is None:
            missing.append(str(stop_2))
        return {
            "satisfied": False,
            "detail": f"Stop(s) {', '.join(missing)} not found on any working bus",
            "evidence": {"stop_1": stop_1, "bus_1": bus_1, "stop_2": stop_2, "bus_2": bus_2},
        }

    satisfied = bus_1 == bus_2
    return {
        "satisfied": satisfied,
        "detail": (
            f"Stop {stop_1} on bus {bus_1}, stop {stop_2} on bus {bus_2}"
            + (" (same bus)" if satisfied else " (different buses!)")
        ),
        "evidence": {"stop_1": stop_1, "bus_1": bus_1, "stop_2": stop_2, "bus_2": bus_2},
    }


def _verify_not_same_bus(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    stop_1 = spec.entities.get("stop_id_1")
    stop_2 = spec.entities.get("stop_id_2")
    bus_1 = _find_stop_on_bus(schedule, stop_1)
    bus_2 = _find_stop_on_bus(schedule, stop_2)

    if bus_1 is None or bus_2 is None:
        missing = []
        if bus_1 is None:
            missing.append(str(stop_1))
        if bus_2 is None:
            missing.append(str(stop_2))
        return {
            "satisfied": None,
            "detail": f"Stop(s) {', '.join(missing)} not found on any working bus",
            "evidence": {"stop_1": stop_1, "bus_1": bus_1, "stop_2": stop_2, "bus_2": bus_2},
        }

    satisfied = bus_1 != bus_2
    return {
        "satisfied": satisfied,
        "detail": (
            f"Stop {stop_1} on bus {bus_1}, stop {stop_2} on bus {bus_2}"
            + (" (different buses)" if satisfied else " (same bus!)")
        ),
        "evidence": {"stop_1": stop_1, "bus_1": bus_1, "stop_2": stop_2, "bus_2": bus_2},
    }


def _verify_max_stops(schedule: List[Dict], spec: ConstraintSpec) -> Dict[str, Any]:
    bus_id = spec.entities.get("bus_id")
    max_stops = spec.entities.get("max_stops")

    if bus_id is None or max_stops is None:
        return {
            "satisfied": None,
            "detail": "Missing bus_id or max_stops in entities",
            "evidence": {"bus_id": bus_id, "max_stops": max_stops},
        }

    actual_count = _count_stops_on_bus(schedule, bus_id)
    satisfied = actual_count <= max_stops
    return {
        "satisfied": satisfied,
        "detail": (
            f"Bus {bus_id} has {actual_count} stop(s)"
            + (f" (<= {max_stops})" if satisfied else f" (exceeds limit of {max_stops})")
        ),
        "evidence": {"bus_id": bus_id, "max_stops": max_stops, "actual_count": actual_count},
    }
