"""
Per-type constraint validators.

One function per constraint type — each works for ALL queries of that type,
parameterized by the entities dict (stop_id, bus_id, capacity, etc.).

Usage:
    from src.core.validation import validate_constraint

    result = validate_constraint(schedule, query)
    # result: {"satisfied": bool, "detail": str}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _find_stop_bus(schedule: List[Dict], stop_id: int) -> Optional[int]:
    """Return bus_id that serves stop_id, or None if not found."""
    for bus in schedule:
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        for wl in bus.get("workload_list", []):
            for stop in wl.get("stops", []):
                if stop["stop_id"] == stop_id:
                    return bus["bus_id"]
    return None


def _get_max_load_on_bus(schedule: List[Dict], bus_id: int) -> Optional[int]:
    """Return max running_load on bus_id, or None if bus not found."""
    max_load = None
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        for wl in bus.get("workload_list", []):
            for stop in wl.get("stops", []):
                load = stop.get("running_load", 0)
                if max_load is None or load > max_load:
                    max_load = load
    return max_load


def _get_stop_order_on_bus(schedule: List[Dict], bus_id: int) -> Dict[int, int]:
    """Return {stop_id: position_index} for all stops on bus_id."""
    indices: Dict[int, int] = {}
    idx = 0
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        for wl in bus.get("workload_list", []):
            for stop in wl.get("stops", []):
                indices[stop["stop_id"]] = idx
                idx += 1
    return indices


def _get_stop_count_on_bus(schedule: List[Dict], bus_id: int) -> int:
    """Return total number of stops on bus_id."""
    count = 0
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        if str(bus.get("status", "")).upper() == "BROKEN":
            continue
        for wl in bus.get("workload_list", []):
            count += len(wl.get("stops", []))
    return count


# ---------------------------------------------------------------------------
# Per-type validators
# Each returns {"satisfied": bool, "detail": str}
# ---------------------------------------------------------------------------

def validate_must_assign(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    stop_id = entities["stop_id"]
    bus_id = entities["bus_id"]
    actual = _find_stop_bus(schedule, stop_id)
    satisfied = actual == bus_id
    detail = f"stop {stop_id} on bus {actual}" + ("" if satisfied else f" (expected bus {bus_id})")
    return {"satisfied": satisfied, "detail": detail}


def validate_must_not_assign(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    stop_id = entities["stop_id"]
    bus_id = entities["bus_id"]
    actual = _find_stop_bus(schedule, stop_id)
    satisfied = actual != bus_id
    detail = f"stop {stop_id} on bus {actual}" + ("" if satisfied else f" (should NOT be on bus {bus_id})")
    return {"satisfied": satisfied, "detail": detail}


def validate_capacity(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    bus_id = entities["bus_id"]
    capacity = entities["capacity"]
    max_load = _get_max_load_on_bus(schedule, bus_id)
    if max_load is None:
        return {"satisfied": False, "detail": f"bus {bus_id} not found"}
    satisfied = max_load <= capacity
    detail = f"bus {bus_id} max load {max_load}" + (f" <= {capacity}" if satisfied else f" exceeds {capacity}")
    return {"satisfied": satisfied, "detail": detail}


def validate_ordering(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    stop_1 = entities["stop_id_1"]
    stop_2 = entities["stop_id_2"]
    bus_1 = _find_stop_bus(schedule, stop_1)
    bus_2 = _find_stop_bus(schedule, stop_2)
    if bus_1 is None or bus_2 is None:
        return {"satisfied": False, "detail": f"stop {stop_1 if bus_1 is None else stop_2} not found"}
    if bus_1 != bus_2:
        return {"satisfied": False, "detail": f"stops on different buses: {stop_1}→bus {bus_1}, {stop_2}→bus {bus_2}"}
    indices = _get_stop_order_on_bus(schedule, bus_1)
    idx_1, idx_2 = indices.get(stop_1), indices.get(stop_2)
    satisfied = idx_1 is not None and idx_2 is not None and idx_1 < idx_2
    detail = f"bus {bus_1}: stop {stop_1}@{idx_1}, stop {stop_2}@{idx_2}"
    return {"satisfied": satisfied, "detail": detail + ("" if satisfied else " (wrong order)")}


def validate_same_bus(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    stop_1 = entities["stop_id_1"]
    stop_2 = entities["stop_id_2"]
    bus_1 = _find_stop_bus(schedule, stop_1)
    bus_2 = _find_stop_bus(schedule, stop_2)
    if bus_1 is None or bus_2 is None:
        return {"satisfied": False, "detail": f"stop not found: {stop_1}→{bus_1}, {stop_2}→{bus_2}"}
    satisfied = bus_1 == bus_2
    detail = f"stop {stop_1} on bus {bus_1}, stop {stop_2} on bus {bus_2}"
    return {"satisfied": satisfied, "detail": detail + (" (same)" if satisfied else " (different)")}


def validate_not_same_bus(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    stop_1 = entities["stop_id_1"]
    stop_2 = entities["stop_id_2"]
    bus_1 = _find_stop_bus(schedule, stop_1)
    bus_2 = _find_stop_bus(schedule, stop_2)
    if bus_1 is None or bus_2 is None:
        return {"satisfied": False, "detail": f"stop not found: {stop_1}→{bus_1}, {stop_2}→{bus_2}"}
    satisfied = bus_1 != bus_2
    detail = f"stop {stop_1} on bus {bus_1}, stop {stop_2} on bus {bus_2}"
    return {"satisfied": satisfied, "detail": detail + (" (different)" if satisfied else " (same — violation)")}


def validate_max_stops(schedule: List[Dict], entities: Dict[str, Any]) -> Dict[str, Any]:
    bus_id = entities["bus_id"]
    max_stops = entities["max_stops"]
    count = _get_stop_count_on_bus(schedule, bus_id)
    satisfied = count <= max_stops
    detail = f"bus {bus_id} has {count} stops" + (f" <= {max_stops}" if satisfied else f" exceeds {max_stops}")
    return {"satisfied": satisfied, "detail": detail}


# ---------------------------------------------------------------------------
# Dispatch table and public entry point
# ---------------------------------------------------------------------------

VALIDATORS = {
    "must_assign":     validate_must_assign,
    "must_not_assign": validate_must_not_assign,
    "capacity":        validate_capacity,
    "ordering":        validate_ordering,
    "same_bus":        validate_same_bus,
    "not_same_bus":    validate_not_same_bus,
    "max_stops":       validate_max_stops,
}


def validate_constraint(
    schedule: List[Dict],
    query: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a constraint against a schedule.

    Args:
        schedule: Parsed bus_schedule_after.json
        query:    Query dict with 'type' and 'entities' fields

    Returns:
        {"satisfied": bool, "detail": str}
        satisfied is None if the constraint type is unknown.
    """
    constraint_type = query.get("type", "")
    validator = VALIDATORS.get(constraint_type)
    if validator is None:
        return {"satisfied": None, "detail": f"no validator for type '{constraint_type}'"}
    return validator(schedule, query["entities"])
