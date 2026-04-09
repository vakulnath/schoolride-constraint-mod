"""
Claude baseline: apply known-correct edits for all 35 queries and run toy tests.
This validates the constraint verification functions and establishes a benchmark.

Usage:
    python -m src.core.test_claude_baseline --solver insertion --queries all
    python -m src.core.test_claude_baseline --solver insertion --queries 19,20
"""

import asyncio
import json
import os
import shutil
import sys
import time

_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _project_root)

with open(os.path.join(os.path.dirname(__file__), "..", "..", ".env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from src.configs import config_parser as cfg

_core_dir = os.path.dirname(__file__)
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)
_archive_dir = os.path.join(_core_dir, "_archive")
if _archive_dir not in sys.path:
    sys.path.insert(0, _archive_dir)

from api_agent_llm import _run_toy_test, _apply_edit_to_temp_dir


# ---------------------------------------------------------------------------
# Edit generators for each constraint type (insertion heuristic)
# ---------------------------------------------------------------------------

def _gen_must_assign(entities):
    """Generate edit for must_assign: reject all buses except target for this stop."""
    stop_id = entities["stop_id"]
    bus_id = entities["bus_id"]
    return {
        "path": "insert_heuristic.py",
        "old_text": "    bus_id = bus[\"id\"]\n    bus_capacity = config.default_capacity",
        "new_text": (
            f"    bus_id = bus[\"id\"]\n"
            f"    # Constraint: stop {stop_id} must be on bus {bus_id}\n"
            f"    if stop == {stop_id} and bus_id != {bus_id}:\n"
            f"        return _fail_log(\"must_assign_{stop_id}_to_{bus_id}\", [stop])\n"
            f"    bus_capacity = config.default_capacity"
        ),
        "reasoning": {
            "search_query": "evaluate bus candidate eligibility for stop assignment",
            "function_found": "evaluate_bus_candidate in insert_heuristic.py",
            "why_this_function": "This function decides if a bus can serve a stop. It has bus_id and stop in scope. Returning _fail_log rejects the bus.",
            "scope_vars_used": ["bus_id (from bus['id'])", "stop (function parameter)"],
            "api_pattern": "return _fail_log('reason', [stop])",
            "edit_location": "After bus_id = bus['id'], before bus_capacity assignment",
            "constraint_parts": 1,
            "constraint_logic": f"If stop=={stop_id} and this bus is NOT {bus_id}, reject it. This forces the stop onto bus {bus_id} only.",
        },
    }


def _gen_must_not_assign(entities):
    """Generate edit for must_not_assign: reject target bus for this stop."""
    stop_id = entities["stop_id"]
    bus_id = entities["bus_id"]
    return {
        "path": "insert_heuristic.py",
        "old_text": "    bus_id = bus[\"id\"]\n    bus_capacity = config.default_capacity",
        "new_text": (
            f"    bus_id = bus[\"id\"]\n"
            f"    # Constraint: stop {stop_id} must NOT be on bus {bus_id}\n"
            f"    if stop == {stop_id} and bus_id == {bus_id}:\n"
            f"        return _fail_log(\"must_not_assign_{stop_id}_on_{bus_id}\", [stop])\n"
            f"    bus_capacity = config.default_capacity"
        ),
        "reasoning": {
            "search_query": "evaluate bus candidate eligibility for stop assignment",
            "function_found": "evaluate_bus_candidate in insert_heuristic.py",
            "why_this_function": "Same as must_assign — this function accepts/rejects buses for a stop.",
            "scope_vars_used": ["bus_id", "stop"],
            "api_pattern": "return _fail_log('reason', [stop])",
            "edit_location": "After bus_id = bus['id']",
            "constraint_parts": 1,
            "constraint_logic": f"If stop=={stop_id} and this bus IS {bus_id}, reject it.",
        },
    }


def _gen_capacity(entities):
    """Generate edit for capacity: override bus_capacity for target bus."""
    bus_id = entities["bus_id"]
    capacity = entities["capacity"]
    return {
        "path": "insert_heuristic.py",
        "old_text": "    bus_capacity = config.default_capacity",
        "new_text": (
            f"    # Constraint: bus {bus_id} capacity = {capacity}\n"
            f"    if bus_id == {bus_id}:\n"
            f"        bus_capacity = {capacity}\n"
            f"    else:\n"
            f"        bus_capacity = config.default_capacity"
        ),
        "reasoning": {
            "search_query": "bus capacity load check default_capacity",
            "function_found": "evaluate_bus_candidate in insert_heuristic.py",
            "why_this_function": "bus_capacity = config.default_capacity is where the capacity is set. Override it for the target bus.",
            "scope_vars_used": ["bus_id", "bus_capacity", "config.default_capacity"],
            "api_pattern": "bus_capacity = config.default_capacity",
            "edit_location": "Replace bus_capacity assignment with conditional",
            "constraint_parts": 1,
            "constraint_logic": f"Override bus_capacity to {capacity} when bus_id=={bus_id}.",
        },
    }


def _gen_ordering(entities):
    """Generate TWO edits for ordering: same-bus guard + route order fix."""
    stop_1 = entities["stop_id_1"]  # must come first
    stop_2 = entities["stop_id_2"]  # must come second
    return [
        {
            "path": "insert_heuristic.py",
            "old_text": "            for _tier_buses in _candidate_tiers:\n                for bus in _tier_buses:",
            "new_text": (
                f"            # Constraint: stop {stop_1} before stop {stop_2} (same bus)\n"
                f"            if stop == {stop_2} and {stop_1} in inserted_nodes:\n"
                f"                _candidate_tiers = [[b for b in bus_map.values() if {stop_1} in b.get('route', [])]]\n"
                f"            if stop == {stop_1} and {stop_2} in inserted_nodes:\n"
                f"                _candidate_tiers = [[b for b in bus_map.values() if {stop_2} in b.get('route', [])]]\n"
                f"            for _tier_buses in _candidate_tiers:\n"
                f"                for bus in _tier_buses:"
            ),
            "reasoning": {
                "search_query": "assign stop to bus loop candidate tiers inserted_nodes",
                "function_found": "replan_bus_routes in insert_heuristic.py",
                "why_this_function": "This is the ASSIGNMENT LOOP that iterates stops one-at-a-time and picks buses. It has inserted_nodes (set of already-assigned stops) and _candidate_tiers (list of candidate buses). Filtering _candidate_tiers forces co-assignment.",
                "scope_vars_used": ["stop", "inserted_nodes", "_candidate_tiers", "bus_map"],
                "api_pattern": "_candidate_tiers = [[b for b in bus_map.values() if X in b.get('route', [])]]",
                "edit_location": "Before the bus iteration loop (for _tier_buses in _candidate_tiers)",
                "constraint_parts": "1 of 2 (same-bus enforcement)",
                "constraint_logic": f"When inserting {stop_2} and {stop_1} is already assigned, only consider buses that have {stop_1}. Vice versa.",
                "key_insight": "Stops are inserted ONE AT A TIME. inserted_nodes tracks what's placed. If partner isn't assigned yet, no filtering needed (first stop goes anywhere, second follows).",
            },
        },
        {
            "path": "insert_heuristic.py",
            "old_text": "    last_pos = {node: i for i, node in enumerate(new_route)}",
            "new_text": (
                f"    last_pos = {{node: i for i, node in enumerate(new_route)}}\n"
                f"    # Constraint: stop {stop_1} must come before stop {stop_2}\n"
                f"    if {stop_1} in last_pos and {stop_2} in last_pos and last_pos[{stop_1}] > last_pos[{stop_2}]:\n"
                f"        return False"
            ),
            "reasoning": {
                "search_query": "validate route order insertion constraints precedence",
                "function_found": "insertion_constraints in insert_heuristic.py",
                "why_this_function": "This POST-insertion validator checks route structure. It has new_route (the proposed route) and last_pos (position map). Returning False rejects routes with wrong ordering.",
                "scope_vars_used": ["last_pos (dict: stop_id -> index)", "new_route"],
                "api_pattern": "return False",
                "edit_location": "After last_pos is built",
                "constraint_parts": "2 of 2 (route order validation)",
                "constraint_logic": f"If both {stop_1} and {stop_2} are in the route and {stop_1} comes AFTER {stop_2}, reject.",
            },
        },
    ]


def _gen_same_bus(entities):
    """Generate edit for same_bus: filter candidate tiers so both stops end up together."""
    stop_1 = entities["stop_id_1"]
    stop_2 = entities["stop_id_2"]
    return {
        "path": "insert_heuristic.py",
        "old_text": "            for _tier_buses in _candidate_tiers:\n                for bus in _tier_buses:",
        "new_text": (
            f"            # Constraint: stops {stop_1} and {stop_2} must be on same bus\n"
            f"            if stop == {stop_1} and {stop_2} in inserted_nodes:\n"
            f"                _candidate_tiers = [[b for b in bus_map.values() if {stop_2} in b.get('route', [])]]\n"
            f"            if stop == {stop_2} and {stop_1} in inserted_nodes:\n"
            f"                _candidate_tiers = [[b for b in bus_map.values() if {stop_1} in b.get('route', [])]]\n"
            f"            for _tier_buses in _candidate_tiers:\n"
            f"                for bus in _tier_buses:"
        ),
        "reasoning": {
            "search_query": "assign stop to bus loop candidate tiers inserted_nodes",
            "function_found": "replan_bus_routes in insert_heuristic.py",
            "why_this_function": "Same as ordering — the assignment loop with inserted_nodes and _candidate_tiers. Filter to force co-assignment.",
            "scope_vars_used": ["stop", "inserted_nodes", "_candidate_tiers", "bus_map"],
            "api_pattern": "_candidate_tiers = [[b for b in bus_map.values() if X in b.get('route', [])]]",
            "edit_location": "Before the bus iteration loop",
            "constraint_parts": 1,
            "constraint_logic": f"When inserting one stop, if the partner is already assigned, only consider buses that have the partner.",
            "key_insight": "Same pattern as ordering's same-bus guard, without the ordering check.",
        },
    }


def _gen_not_same_bus(entities):
    """Generate edit for not_same_bus: prevent both stops on same bus."""
    stop_1 = entities["stop_id_1"]
    stop_2 = entities["stop_id_2"]
    return {
        "path": "insert_heuristic.py",
        "old_text": "    bus_id = bus[\"id\"]\n    bus_capacity = config.default_capacity",
        "new_text": (
            f"    bus_id = bus[\"id\"]\n"
            f"    # Constraint: stops {stop_1} and {stop_2} must NOT be on same bus\n"
            f"    if stop == {stop_1} and {stop_2} in bus.get('route', []):\n"
            f"        return _fail_log(\"not_same_bus_{stop_1}_{stop_2}\", [stop])\n"
            f"    if stop == {stop_2} and {stop_1} in bus.get('route', []):\n"
            f"        return _fail_log(\"not_same_bus_{stop_2}_{stop_1}\", [stop])\n"
            f"    bus_capacity = config.default_capacity"
        ),
        "reasoning": {
            "search_query": "evaluate bus candidate check route contains stop",
            "function_found": "evaluate_bus_candidate in insert_heuristic.py",
            "why_this_function": "This function has bus['route'] in scope — can check if the partner stop is already on this bus.",
            "scope_vars_used": ["stop", "bus.get('route', [])", "bus_id"],
            "api_pattern": "return _fail_log('reason', [stop])",
            "edit_location": "After bus_id = bus['id']",
            "constraint_parts": 1,
            "constraint_logic": f"If inserting {stop_1} and {stop_2} is already on this bus's route, reject. Vice versa.",
            "key_insight": "Unlike same_bus (which needs replan_bus_routes for the assignment loop), not_same_bus can be checked in evaluate_bus_candidate because we just need to reject buses that already have the partner.",
        },
    }


def _gen_max_stops(entities):
    """Generate edit for max_stops: limit route length for target bus."""
    bus_id = entities["bus_id"]
    max_stops = entities["max_stops"]
    return {
        "path": "insert_heuristic.py",
        "old_text": "    bus_id = bus[\"id\"]\n    bus_capacity = config.default_capacity",
        "new_text": (
            f"    bus_id = bus[\"id\"]\n"
            f"    # Constraint: bus {bus_id} max {max_stops} stops\n"
            f"    if bus_id == {bus_id} and len(temp_route) + 1 > {max_stops}:\n"
            f"        return _fail_log(\"max_stops_{bus_id}_{max_stops}\", [stop])\n"
            f"    bus_capacity = config.default_capacity"
        ),
        "reasoning": {
            "search_query": "evaluate bus candidate route length stop count",
            "function_found": "evaluate_bus_candidate in insert_heuristic.py",
            "why_this_function": "Has bus_id and temp_route (copy of current route). len(temp_route) + 1 is the count after adding this stop.",
            "scope_vars_used": ["bus_id", "temp_route", "stop"],
            "api_pattern": "return _fail_log('reason', [stop])",
            "edit_location": "After bus_id = bus['id']",
            "constraint_parts": 1,
            "constraint_logic": f"If bus {bus_id} would have more than {max_stops} stops after insertion, reject.",
        },
    }


GENERATORS = {
    "must_assign": _gen_must_assign,
    "must_not_assign": _gen_must_not_assign,
    "capacity": _gen_capacity,
    "ordering": _gen_ordering,
    "same_bus": _gen_same_bus,
    "not_same_bus": _gen_not_same_bus,
    "max_stops": _gen_max_stops,
}


def run_query(q, solver_type, output_dir):
    """Apply the correct edit for a query and run the toy test."""
    qid = q["query_id"]
    qtype = q["type"]
    entities = q["entities"]

    query_dir = os.path.join(output_dir, f"query_{qid}", solver_type)
    os.makedirs(query_dir, exist_ok=True)
    temp_dir = os.path.join(query_dir, "temp_code")
    os.makedirs(temp_dir, exist_ok=True)

    # Generate the edit
    gen = GENERATORS.get(qtype)
    if not gen:
        print(f"Q{qid:2d} ({qtype:16s}): SKIP (no generator)")
        return qid, qtype, False

    edits = gen(entities)
    if isinstance(edits, dict):
        edits = [edits]

    # Setup temp policy dir
    policy_dir = os.path.join(temp_dir, "policy")
    if os.path.isdir(policy_dir):
        shutil.rmtree(policy_dir)
    import policy as _pol
    shutil.copytree(os.path.dirname(_pol.__file__), policy_dir,
                    ignore=shutil.ignore_patterns("__pycache__"))

    # Apply all edits and collect diffs
    import difflib
    env = os.environ.copy()
    all_diffs = []
    for i, edit in enumerate(edits):
        applied = _apply_edit_to_temp_dir(edit, temp_dir, env, None)
        if not applied:
            print(f"Q{qid:2d} ({qtype:16s}): FAIL (edit {i+1} apply failed)")
            return qid, qtype, False
        diff = "".join(difflib.unified_diff(
            edit["old_text"].splitlines(keepends=True),
            edit["new_text"].splitlines(keepends=True),
            fromfile=f"a/{edit['path']}", tofile=f"b/{edit['path']}", n=3,
        ))
        all_diffs.append(diff)

    # Run test
    result = _run_toy_test(qid, solver_type, temp_dir)
    status = "PASS" if result["passed"] else "FAIL"
    print(f"Q{qid:2d} ({qtype:16s}): {status}")
    if not result["passed"]:
        for line in result.get("output", "").split("\n"):
            if any(kw in line for kw in ["assert", "FAILED", "Error"]):
                print(f"    {line.strip()}")
    sys.stdout.flush()

    # Collect reasoning from edits
    all_reasoning = []
    for edit in edits:
        r = edit.get("reasoning")
        if r:
            all_reasoning.append(r)

    # Save result with diff and reasoning
    combined_diff = "\n".join(all_diffs)
    with open(os.path.join(query_dir, "result.json"), "w") as f:
        json.dump({
            "query_id": qid,
            "type": qtype,
            "question": q["question"],
            "entities": entities,
            "success": result["passed"],
            "diff": combined_diff,
            "reasoning": all_reasoning,
        }, f, indent=2)
    with open(os.path.join(query_dir, "edit.diff"), "w") as f:
        f.write(combined_diff)

    return qid, qtype, result["passed"]


def main():
    import argparse
    from collections import defaultdict
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="insertion", choices=["insertion", "hexaly"])
    parser.add_argument("--queries", default="all")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Use expanded queries if available, otherwise original
    expanded = "src/configs/queries_expanded.json"
    original = "src/configs/queries.json"
    qfile = expanded if os.path.exists(expanded) else original
    qs = json.load(open(qfile))["queries"]
    print(f"Using: {qfile} ({len(qs)} queries)")
    if args.queries != "all":
        query_ids = set(int(x) for x in args.queries.split(","))
        qs = [q for q in qs if q["query_id"] in query_ids]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("context_runs", f"{timestamp}_claude_baseline")
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Claude Baseline: {args.solver} ({len(qs)} queries)")
    print(f"{'='*60}\n")

    start = time.time()
    results = []
    for q in qs:
        qid, qtype, success = run_query(q, args.solver, output_dir)
        results.append((qid, qtype, success))

    elapsed = time.time() - start
    passed = sum(1 for _, _, s in results if s)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} ({100*passed/total:.0f}%) in {elapsed:.0f}s")
    print(f"{'='*60}")

    by_type = defaultdict(lambda: [0, 0])
    for qid, qtype, success in sorted(results):
        by_type[qtype][1] += 1
        if success:
            by_type[qtype][0] += 1
    for t, (p, tot) in sorted(by_type.items()):
        print(f"  {t:16s}: {p}/{tot}")

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump({
            "pipeline": "claude_baseline",
            "total_time": round(elapsed, 1),
            "results": [{"query_id": qid, "type": qt, "success": s} for qid, qt, s in results],
        }, f, indent=2)


if __name__ == "__main__":
    main()
