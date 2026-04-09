"""
Compare vanilla vs modified bus schedules and verify constraint satisfaction.

Reads two bus_schedule JSON files (before/after agent edit) and checks:
1. Constraint satisfaction: is the constraint enforced in the modified schedule?
2. No regressions: are all stops still served?
3. Minimal impact: only expected buses/stops changed.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ConstraintType(Enum):
    MUST_ASSIGN = "must_assign"
    MUST_NOT_ASSIGN = "must_not_assign"
    CAPACITY = "capacity"
    ORDERING = "ordering"
    SAME_BUS = "same_bus"
    NOT_SAME_BUS = "not_same_bus"
    MAX_STOPS = "max_stops"


def _classify_constraint(query: Dict[str, Any]) -> "ConstraintType":
    """Infer the constraint type from query entities and question text."""
    question = query.get("question", "").lower()
    entities = query.get("entities", {})

    if "max_stops" in entities:
        return ConstraintType.MAX_STOPS
    if "stop_id_1" in entities and "stop_id_2" in entities:
        if "same bus" in question and "not" in question:
            return ConstraintType.NOT_SAME_BUS
        if "same bus" in question:
            return ConstraintType.SAME_BUS
        return ConstraintType.ORDERING
    if "capacity" in entities:
        return ConstraintType.CAPACITY
    if "stop_id" in entities and "bus_id" in entities:
        if "must not" in question or "must't" in question:
            return ConstraintType.MUST_NOT_ASSIGN
        return ConstraintType.MUST_ASSIGN

    # Fallback from question text
    if "before" in question or "order" in question:
        return ConstraintType.ORDERING
    if "capacity" in question or "reduced" in question:
        return ConstraintType.CAPACITY
    if "not" in question:
        return ConstraintType.MUST_NOT_ASSIGN
    return ConstraintType.MUST_ASSIGN


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def _working_buses(schedule: List[Dict]) -> List[Dict]:
    return [b for b in schedule if str(b.get("status", "")).upper() != "BROKEN"]


def _all_stop_ids(schedule: List[Dict]) -> Set[int]:
    stops: Set[int] = set()
    for bus in _working_buses(schedule):
        for wl in bus.get("workload_list", []):
            for s in wl.get("stops", []):
                stops.add(s["stop_id"])
    return stops


def _stop_to_bus(schedule: List[Dict]) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for bus in _working_buses(schedule):
        bus_id = bus["bus_id"]
        for wl in bus.get("workload_list", []):
            for s in wl.get("stops", []):
                mapping[s["stop_id"]] = bus_id
    return mapping


def _bus_stop_count(schedule: List[Dict]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for bus in _working_buses(schedule):
        bus_id = bus["bus_id"]
        total = sum(len(wl.get("stops", [])) for wl in bus.get("workload_list", []))
        counts[bus_id] = total
    return counts


def _get_stop_indices_on_bus(schedule: List[Dict], bus_id: Any) -> Dict[int, int]:
    """Get {stop_id: index} for all stops on a given bus (across all workloads)."""
    indices: Dict[int, int] = {}
    idx = 0
    for bus in schedule:
        if bus.get("bus_id") != bus_id:
            continue
        for wl in bus.get("workload_list", []):
            for stop in wl.get("stops", []):
                indices[stop.get("stop_id")] = idx
                idx += 1
    return indices


def _get_max_load_on_bus(schedule: List[Dict], bus_id: int) -> Optional[int]:
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


# ---------------------------------------------------------------------------
# Constraint verifiers
# ---------------------------------------------------------------------------

def _verify_constraint(schedule: List[Dict], query: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether the constraint is satisfied in the schedule output."""
    ctype = _classify_constraint(query)
    entities = query.get("entities", {})
    stop_map = _stop_to_bus(schedule)

    if ctype == ConstraintType.MUST_ASSIGN:
        stop_id = entities.get("stop_id")
        bus_id = entities.get("bus_id")
        actual = stop_map.get(stop_id)
        satisfied = actual == bus_id
        detail = f"Stop {stop_id} on bus {actual}" + ("" if satisfied else f" (expected {bus_id})")
        return {"satisfied": satisfied, "detail": detail}

    if ctype == ConstraintType.MUST_NOT_ASSIGN:
        stop_id = entities.get("stop_id")
        bus_id = entities.get("bus_id")
        actual = stop_map.get(stop_id)
        satisfied = actual != bus_id
        detail = f"Stop {stop_id} on bus {actual}" + ("" if satisfied else f" (should NOT be on {bus_id})")
        return {"satisfied": satisfied, "detail": detail}

    if ctype == ConstraintType.CAPACITY:
        bus_id = entities.get("bus_id")
        capacity = entities.get("capacity")
        max_load = _get_max_load_on_bus(schedule, bus_id)
        if max_load is None:
            return {"satisfied": None, "detail": f"Bus {bus_id} not found"}
        satisfied = max_load <= capacity
        detail = f"Bus {bus_id} max load {max_load}" + (f" (<= {capacity})" if satisfied else f" (exceeds {capacity})")
        return {"satisfied": satisfied, "detail": detail}

    if ctype == ConstraintType.ORDERING:
        stop_1 = entities.get("stop_id_1")
        stop_2 = entities.get("stop_id_2")
        for bus in _working_buses(schedule):
            bid = bus.get("bus_id")
            indices = _get_stop_indices_on_bus(schedule, bid)
            if stop_1 in indices and stop_2 in indices:
                satisfied = indices[stop_1] < indices[stop_2]
                detail = f"Bus {bid}: stop {stop_1}@{indices[stop_1]}, stop {stop_2}@{indices[stop_2]}"
                return {"satisfied": satisfied, "detail": detail + ("" if satisfied else " (wrong order)")}
        return {"satisfied": None, "detail": f"Stops {stop_1},{stop_2} not on same bus"}

    if ctype == ConstraintType.SAME_BUS:
        stop_1 = entities.get("stop_id_1")
        stop_2 = entities.get("stop_id_2")
        bus_1 = stop_map.get(stop_1)
        bus_2 = stop_map.get(stop_2)
        if bus_1 is None or bus_2 is None:
            return {"satisfied": False, "detail": f"Stop(s) not found: {stop_1}→{bus_1}, {stop_2}→{bus_2}"}
        satisfied = bus_1 == bus_2
        detail = f"Stop {stop_1} on bus {bus_1}, stop {stop_2} on bus {bus_2}"
        return {"satisfied": satisfied, "detail": detail + (" (same)" if satisfied else " (different!)")}

    if ctype == ConstraintType.NOT_SAME_BUS:
        stop_1 = entities.get("stop_id_1")
        stop_2 = entities.get("stop_id_2")
        bus_1 = stop_map.get(stop_1)
        bus_2 = stop_map.get(stop_2)
        if bus_1 is None or bus_2 is None:
            return {"satisfied": None, "detail": f"Stop(s) not found: {stop_1}→{bus_1}, {stop_2}→{bus_2}"}
        satisfied = bus_1 != bus_2
        detail = f"Stop {stop_1} on bus {bus_1}, stop {stop_2} on bus {bus_2}"
        return {"satisfied": satisfied, "detail": detail + (" (different)" if satisfied else " (same!)")}

    if ctype == ConstraintType.MAX_STOPS:
        bus_id = entities.get("bus_id")
        max_stops = entities.get("max_stops")
        if bus_id is None or max_stops is None:
            return {"satisfied": None, "detail": "Missing bus_id or max_stops"}
        count = 0
        for bus in _working_buses(schedule):
            if bus.get("bus_id") == bus_id:
                count = sum(len(wl.get("stops", [])) for wl in bus.get("workload_list", []))
        satisfied = count <= max_stops
        detail = f"Bus {bus_id} has {count} stops" + (f" (<= {max_stops})" if satisfied else f" (exceeds {max_stops})")
        return {"satisfied": satisfied, "detail": detail}

    return {"satisfied": None, "detail": "Unknown constraint type"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compare_schedules(
    vanilla_path: str,
    modified_path: str,
    query: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare vanilla and modified bus schedules, and verify constraint satisfaction.

    Returns a dict with:
        - regression_free: bool — no stops were lost
        - stops_lost / stops_gained: list of stop IDs
        - reassignments: list of {stop_id, vanilla_bus, modified_bus}
        - buses_changed: list of bus IDs whose routes differ
        - constraint_satisfied: bool|None — is the constraint enforced?
        - constraint_detail: str — human-readable explanation
        - summary: human-readable string
    """
    try:
        vanilla = _load(vanilla_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": f"Could not load vanilla schedule: {e}", "regression_free": None}

    try:
        modified = _load(modified_path)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return {"error": f"Could not load modified schedule: {e}", "regression_free": None}

    # --- Regression check ---
    vanilla_stops = _all_stop_ids(vanilla)
    modified_stops = _all_stop_ids(modified)
    stops_lost = sorted(vanilla_stops - modified_stops)
    stops_gained = sorted(modified_stops - vanilla_stops)

    vanilla_map = _stop_to_bus(vanilla)
    modified_map = _stop_to_bus(modified)

    reassignments: List[Dict[str, Any]] = []
    for stop_id in sorted(vanilla_stops & modified_stops):
        v_bus = vanilla_map.get(stop_id)
        m_bus = modified_map.get(stop_id)
        if v_bus != m_bus:
            reassignments.append({"stop_id": stop_id, "vanilla_bus": v_bus, "modified_bus": m_bus})

    buses_changed: Set[int] = set()
    for r in reassignments:
        if r["vanilla_bus"] is not None:
            buses_changed.add(r["vanilla_bus"])
        if r["modified_bus"] is not None:
            buses_changed.add(r["modified_bus"])

    vanilla_counts = _bus_stop_count(vanilla)
    modified_counts = _bus_stop_count(modified)
    bus_stop_count_diff: Dict[int, Dict[str, int]] = {}
    for bid in sorted(set(vanilla_counts) | set(modified_counts)):
        v = vanilla_counts.get(bid, 0)
        m = modified_counts.get(bid, 0)
        if v != m:
            bus_stop_count_diff[bid] = {"vanilla": v, "modified": m, "delta": m - v}

    # --- Constraint satisfaction ---
    constraint_check: Dict[str, Any] = {"satisfied": None, "detail": "No query provided"}
    if query:
        constraint_check = _verify_constraint(modified, query)

    # --- Summary ---
    parts = [f"Stops: {len(vanilla_stops)} vanilla, {len(modified_stops)} modified"]
    if stops_lost:
        parts.append(f"LOST {len(stops_lost)} stop(s): {stops_lost}")
    if stops_gained:
        parts.append(f"Gained {len(stops_gained)} new stop(s): {stops_gained}")
    if reassignments:
        parts.append(f"{len(reassignments)} stop(s) reassigned across {len(buses_changed)} bus(es)")
    else:
        parts.append("No reassignments (schedules identical)")
    for bid, d in bus_stop_count_diff.items():
        parts.append(f"  Bus {bid}: {d['vanilla']} -> {d['modified']} stops ({d['delta']:+d})")
    parts.append(f"Constraint: {constraint_check['detail']}")

    return {
        "regression_free": len(stops_lost) == 0,
        "stops_lost": stops_lost,
        "stops_gained": stops_gained,
        "reassignments": reassignments,
        "buses_changed": sorted(buses_changed),
        "bus_stop_count_diff": bus_stop_count_diff,
        "constraint_satisfied": constraint_check["satisfied"],
        "constraint_detail": constraint_check["detail"],
        "summary": "\n".join(parts),
    }
