"""Run pipeline on queries with results saved to context_runs/.

Modes:
  Standard  — one solver at a time (any --llm backend)
  Paired    — insertion + hexaly run in parallel per query, then a combined
              hexaly-with-insertion-warmstart scenario is executed.
              Activated automatically when --llm claude-cli.
"""

import asyncio
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime

# Bootstrap: add project root to path and load .env before importing src packages
_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _project_root)

_env_file = os.path.join(_project_root, ".env")
if os.path.isfile(_env_file):
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

from src.configs import config_parser as cfg  # noqa: E402
from src.core.pipeline import run_pipeline  # noqa: E402
from src.core.validation import validate_constraint  # noqa: E402
from src.mcp.serena import SerenaClient  # noqa: E402
from src.utils.code_tools import apply_edit_to_temp_dir, copy_policy_to_dir  # noqa: E402
from src.utils.test_runner import run_toy_test  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_schedule(query_dir: str, q: dict) -> tuple:
    """Load bus_schedule_after.json and run constraint validation. Returns (satisfied, detail)."""
    schedule_path = os.path.join(query_dir, "output", "bus_schedule_after.json")
    if not os.path.isfile(schedule_path):
        return None, "no schedule file"
    try:
        with open(schedule_path) as f:
            schedule = json.load(f)
        vresult = validate_constraint(schedule, q)
        return vresult["satisfied"], vresult["detail"]
    except Exception as e:
        return None, f"validation error: {e}"


def _copy_toy_output(temp_dir: str, query_dir: str, qid: int):
    toy_output = os.path.join(temp_dir, "output", f"query_{qid}_modified")
    dest_output = os.path.join(query_dir, "output")
    if os.path.isdir(toy_output):
        shutil.copytree(toy_output, dest_output, dirs_exist_ok=True)


# ---------------------------------------------------------------------------
# Standard single-solver run
# ---------------------------------------------------------------------------

async def run_one(q, solver_type, output_dir, shared_serena=None):
    """Run a single query for one solver and save results."""
    qid = q["query_id"]
    qtype = q["type"]

    query_dir = os.path.join(output_dir, f"query_{qid}", solver_type)
    os.makedirs(query_dir, exist_ok=True)

    temp_dir = os.path.join(query_dir, "temp_code")
    os.makedirs(temp_dir, exist_ok=True)
    logs_dir = os.path.join(query_dir, "logs")

    t0 = time.time()
    try:
        result = await run_pipeline(
            constraint=q["question"],
            query_id=qid,
            solver_type=solver_type,
            temp_code_dir=temp_dir,
            logs_dir=logs_dir,
            shared_serena=shared_serena,
        )
        elapsed = time.time() - t0
        success = result.get("success", False)
        diff_summary = result.get("diff", "")[:80]
        injection_count = len(result.get("injection_points", []))
    except Exception as e:
        elapsed = time.time() - t0
        success = False
        diff_summary = f"ERROR: {e}"
        injection_count = 0
        result = {"success": False, "diff": str(e), "edit": None}

    _copy_toy_output(temp_dir, query_dir, qid)

    constraint_satisfied, constraint_detail = _validate_schedule(query_dir, q)

    # Save edit for combined scenario (claude-cli returns edit dict)
    edit = result.get("edit")
    if edit:
        with open(os.path.join(query_dir, "edit.json"), "w") as f:
            json.dump(edit, f, indent=2)

    status = "PASS" if success else "FAIL"
    cv = "" if constraint_satisfied is None else (" ✓constraint" if constraint_satisfied else " ✗constraint")
    print(f"Q{qid:2d} [{solver_type:9s}] ({qtype:16s}): {status}{cv} ({elapsed:.0f}s)")
    if constraint_satisfied is False:
        print(f"       constraint detail: {constraint_detail}")
    sys.stdout.flush()

    with open(os.path.join(query_dir, "result.json"), "w") as f:
        json.dump({
            "query_id": qid,
            "type": qtype,
            "question": q["question"],
            "solver_type": solver_type,
            "success": success,
            "constraint_satisfied": constraint_satisfied,
            "constraint_detail": constraint_detail,
            "diff": result.get("diff", "")[:500],
            "elapsed_seconds": round(elapsed, 1),
            "injection_points_found": injection_count,
        }, f, indent=2)

    return qid, qtype, success, constraint_satisfied, diff_summary, result


# ---------------------------------------------------------------------------
# Combined scenario: apply both insertion + hexaly edits, run hexaly toy test
# ---------------------------------------------------------------------------

def _run_combined_scenario(q, output_dir, ins_result, hex_result):
    """Apply insertion + hexaly edits to a fresh temp dir and run the hexaly toy test.

    Hexaly internally uses PrunedInsertionPolicy as warm-start, so both modified
    policies exercise together.

    Returns dict with keys: success, constraint_satisfied, constraint_detail, note.
    """
    qid = q["query_id"]
    qtype = q["type"]

    combined_dir = os.path.join(output_dir, f"query_{qid}", "combined")
    os.makedirs(combined_dir, exist_ok=True)
    combined_temp = os.path.join(combined_dir, "temp_code")
    os.makedirs(combined_temp, exist_ok=True)
    logs_dir = os.path.join(combined_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    env = os.environ.copy()

    ins_temp = ins_result.get("temp_code_dir") if ins_result else None
    hex_temp = hex_result.get("temp_code_dir") if hex_result else None
    ins_edit = ins_result.get("edit") if ins_result else None
    hex_edit = hex_result.get("edit") if hex_result else None

    has_ins = bool(ins_temp and os.path.isdir(os.path.join(ins_temp, "policy")))
    has_hex = bool(hex_temp and os.path.isdir(os.path.join(hex_temp, "policy")))
    has_edit = bool(ins_edit or hex_edit)

    if not has_ins and not has_hex and not has_edit:
        note = "no edits available from either solver"
        _write_combined_result(combined_dir, q, False, None, note, note)
        return {"success": False, "constraint_satisfied": None,
                "constraint_detail": note, "note": note}

    # Build combined policy dir by merging final modified files from each solver.
    # Prefer copying final temp dirs (which include fix-round patches) over re-applying edits.
    import policy as _pol
    _pol_file = _pol.__file__
    assert _pol_file is not None
    policy_temp = os.path.join(combined_temp, "policy")
    copy_policy_to_dir(os.path.dirname(_pol_file), policy_temp)

    applied = []

    if has_ins:
        # Copy only materialized (non-symlink) files — these are the edited ones
        ins_policy = os.path.join(ins_temp, "policy")
        for fname in os.listdir(ins_policy):
            src = os.path.join(ins_policy, fname)
            dst = os.path.join(policy_temp, fname)
            if os.path.isfile(src) and not os.path.islink(src):  # only real files (edited)
                if os.path.islink(dst):  # replace symlink with real file
                    os.unlink(dst)
                shutil.copy2(src, dst)
        applied.append("insertion")
    elif ins_edit:
        ok = apply_edit_to_temp_dir(ins_edit, combined_temp, env, logs_dir)
        if ok:
            applied.append("insertion")
        else:
            with open(os.path.join(logs_dir, "combined.log"), "a") as f:
                f.write("WARNING: insertion edit failed to apply\n")

    if has_hex:
        # Copy only materialized (non-symlink) hexaly files
        hex_policy = os.path.join(hex_temp, "policy")
        for fname in os.listdir(hex_policy):
            src = os.path.join(hex_policy, fname)
            dst = os.path.join(policy_temp, fname)
            if os.path.isfile(src) and not os.path.islink(src):
                if os.path.islink(dst):
                    os.unlink(dst)
                shutil.copy2(src, dst)
        applied.append("hexaly")
    elif hex_edit:
        ok = apply_edit_to_temp_dir(hex_edit, combined_temp, env, logs_dir)
        if ok:
            applied.append("hexaly")
        else:
            with open(os.path.join(logs_dir, "combined.log"), "a") as f:
                f.write("WARNING: hexaly edit failed to apply\n")

    if not applied:
        note = "both edits failed to apply"
        _write_combined_result(combined_dir, q, False, None, note, note)
        return {"success": False, "constraint_satisfied": None,
                "constraint_detail": note, "note": note}

    # Run hexaly toy test (uses insertion as warm-start internally)
    test_result = run_toy_test(qid, "hexaly", combined_temp)
    success = test_result["passed"]

    # Copy toy output for inspection
    _copy_toy_output(combined_temp, combined_dir, qid)

    constraint_satisfied, constraint_detail = _validate_schedule(combined_dir, q)

    note = f"applied: {', '.join(applied)}"
    _write_combined_result(combined_dir, q, success, constraint_satisfied,
                           constraint_detail, note)

    status = "PASS" if success else "FAIL"
    cv = "" if constraint_satisfied is None else (" ✓constraint" if constraint_satisfied else " ✗constraint")
    print(f"Q{qid:2d} [combined  ] ({qtype:16s}): {status}{cv}  [{note}]")
    sys.stdout.flush()

    return {
        "success": success,
        "constraint_satisfied": constraint_satisfied,
        "constraint_detail": constraint_detail,
        "note": note,
    }


def _write_combined_result(combined_dir, q, success, cv, detail, note):
    with open(os.path.join(combined_dir, "result_combined.json"), "w") as f:
        json.dump({
            "query_id": q["query_id"],
            "type": q["type"],
            "question": q["question"],
            "solver_type": "combined_hexaly_warmstart",
            "success": success,
            "constraint_satisfied": cv,
            "constraint_detail": detail,
            "note": note,
        }, f, indent=2)


# ---------------------------------------------------------------------------
# Paired run: insertion + hexaly in parallel, then combined scenario
# ---------------------------------------------------------------------------

async def run_paired(q, output_dir, shared_serena=None):
    """Run insertion and hexaly pipelines in parallel, then run combined scenario."""
    qid = q["query_id"]

    import concurrent.futures
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _run_sync(solver_type):
        """Run run_one in its own event loop in a thread."""
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(
                run_one(q, solver_type, output_dir, shared_serena=shared_serena)
            )
        finally:
            new_loop.close()

    ins_ret, hex_ret = await asyncio.gather(
        loop.run_in_executor(executor, _run_sync, "insertion"),
        loop.run_in_executor(executor, _run_sync, "hexaly"),
        return_exceptions=True,
    )
    executor.shutdown(wait=False)

    # Unpack results (handle exceptions from either solver)
    if isinstance(ins_ret, Exception):
        print(f"Q{qid:2d} [insertion ] ERROR: {ins_ret}")
        ins_result = None
        ins_row = (qid, q["type"], False, None, str(ins_ret))
    else:
        *ins_row_vals, ins_result = ins_ret
        ins_row = tuple(ins_row_vals)

    if isinstance(hex_ret, Exception):
        print(f"Q{qid:2d} [hexaly    ] ERROR: {hex_ret}")
        hex_result = None
        hex_row = (qid, q["type"], False, None, str(hex_ret))
    else:
        *hex_row_vals, hex_result = hex_ret
        hex_row = tuple(hex_row_vals)

    # Combined scenario (sync — runs in the event loop thread, ~fast)
    combined = _run_combined_scenario(q, output_dir, ins_result, hex_result)

    return ins_row, hex_row, combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="both",
                        choices=["insertion", "hexaly", "both"],
                        help="Solver(s) to run. Ignored for --llm claude-cli (always runs both + combined).")
    parser.add_argument("--queries", default="all",
                        help="'all' or comma-separated IDs like '1,2,3'")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--llm", default="gemini",
                        choices=["gemini", "groq", "amplify", "claude-cli"],
                        help="LLM backend: gemini/groq (native ADK tools), amplify (prompt-based), claude-cli (Claude CLI paired mode)")
    args = parser.parse_args()

    if args.llm == "amplify":
        os.environ["USE_PROMPT_TOOLS"] = "1"
    elif args.llm == "groq":
        os.environ["USE_GROQ"] = "1"
    elif args.llm == "claude-cli":
        os.environ["USE_CLAUDE_CLI"] = "1"

    paired_mode = args.solver == "both"

    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))

    expanded = "src/configs/queries_expanded.json"
    original = "src/configs/queries.json"
    qfile = expanded if os.path.exists(expanded) else original
    qs = json.load(open(qfile))["queries"]
    print(f"Using: {qfile} ({len(qs)} queries)")

    if args.queries != "all":
        query_ids = set()
        for part in args.queries.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                query_ids.update(range(int(lo), int(hi) + 1))
            else:
                query_ids.add(int(part))
        qs = [q for q in qs if q["query_id"] in query_ids]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    llm_tag = args.llm
    output_dir = os.path.join("context_runs", f"{timestamp}_{llm_tag}")
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()

    # ── Paired mode (claude-cli): insertion + hexaly in parallel per query ──
    if paired_mode:
        print(f"\n{'='*60}")
        print(f"Mode: PAIRED (insertion + hexaly parallel + combined scenario)")
        print(f"{'='*60}")

        all_ins, all_hex, all_combined = [], [], []

        policy_root = str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT else "."
        shared_serena = SerenaClient(policy_root)
        serena_ok = shared_serena.start()
        if serena_ok:
            print(f"  Serena started (shared)")
        else:
            print(f"  WARNING: Serena unavailable")
            shared_serena = None

        for i in range(0, len(qs), args.batch_size):
            batch = qs[i: i + args.batch_size]
            print(f"\n--- Batch {i // args.batch_size + 1}: "
                  f"Q{batch[0]['query_id']}-Q{batch[-1]['query_id']} ---")
            sys.stdout.flush()

            batch_results = await asyncio.gather(
                *[run_paired(q, output_dir, shared_serena=shared_serena) for q in batch]
            )

            for ins_row, hex_row, combined in batch_results:
                all_ins.append(ins_row)
                all_hex.append(hex_row)
                all_combined.append((ins_row[0], ins_row[1], combined))

        if shared_serena:
            shared_serena.stop()

        total_time = time.time() - start

        print(f"\n\n{'='*60}")
        print(f"SUMMARY (paired) — {total_time:.0f}s total")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        for label, rows in [("insertion", all_ins), ("hexaly", all_hex)]:
            passed = sum(1 for r in rows if r[2])
            cv_pass = sum(1 for r in rows if r[2] and r[3] is True)
            cv_fail = sum(1 for r in rows if r[2] and r[3] is False)
            print(f"\n{label}: {passed}/{len(rows)} toy pass "
                  f"| constraint: {cv_pass} pass, {cv_fail} fail")

        comb_pass = sum(1 for _, _, c in all_combined if c.get("success"))
        comb_cv_pass = sum(1 for _, _, c in all_combined if c.get("success") and c.get("constraint_satisfied") is True)
        comb_cv_fail = sum(1 for _, _, c in all_combined if c.get("success") and c.get("constraint_satisfied") is False)
        print(f"\ncombined (hexaly+warmstart): {comb_pass}/{len(all_combined)} toy pass "
              f"| constraint: {comb_cv_pass} pass, {comb_cv_fail} fail")

        summary = {
            "timestamp": timestamp,
            "pipeline": "claude-cli-paired",
            "total_time_seconds": round(total_time, 1),
            "results": {
                "insertion": [
                    {"query_id": r[0], "type": r[1], "success": r[2], "constraint_satisfied": r[3]}
                    for r in all_ins
                ],
                "hexaly": [
                    {"query_id": r[0], "type": r[1], "success": r[2], "constraint_satisfied": r[3]}
                    for r in all_hex
                ],
                "combined": [
                    {"query_id": qid, "type": qtype,
                     "success": c.get("success"), "constraint_satisfied": c.get("constraint_satisfied"),
                     "note": c.get("note", "")}
                    for qid, qtype, c in all_combined
                ],
            },
        }

    # ── Standard mode: one solver at a time ─────────────────────────────────
    else:
        solver_types = ["insertion", "hexaly"] if args.solver == "both" else [args.solver]
        all_results = []

        for solver_type in solver_types:
            print(f"\n{'='*60}")
            print(f"Solver: {solver_type}")
            print(f"{'='*60}")

            policy_root = str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT else "."
            shared_serena = SerenaClient(policy_root)
            serena_ok = shared_serena.start()
            if serena_ok:
                print(f"  Serena started (shared)")
            else:
                print(f"  WARNING: Serena unavailable")
                shared_serena = None

            for i in range(0, len(qs), args.batch_size):
                batch = qs[i: i + args.batch_size]
                print(f"\n--- Batch {i // args.batch_size + 1}: "
                      f"Q{batch[0]['query_id']}-Q{batch[-1]['query_id']} ---")
                sys.stdout.flush()
                results = await asyncio.gather(
                    *[run_one(q, solver_type, output_dir, shared_serena=shared_serena) for q in batch]
                )
                all_results.extend((solver_type, *r[:5]) for r in results)
                if i + args.batch_size < len(qs):
                    await asyncio.sleep(30)  # 30s between batches (RPM)

            if shared_serena:
                shared_serena.stop()

        total_time = time.time() - start

        print(f"\n\n{'='*60}")
        print(f"SUMMARY — {total_time:.0f}s total")
        print(f"Output: {output_dir}")
        print(f"{'='*60}")

        for solver_type in solver_types:
            solver_results = [(qid, qt, s, cv, d)
                              for st, qid, qt, s, cv, d in all_results if st == solver_type]
            passed = sum(1 for _, _, s, _, _ in solver_results if s)
            cv_passed = sum(1 for _, _, s, cv, _ in solver_results if s and cv is True)
            cv_failed = sum(1 for _, _, s, cv, _ in solver_results if s and cv is False)
            total = len(solver_results)
            if total == 0:
                continue
            print(f"\n{solver_type}: {passed}/{total} toy test pass "
                  f"| constraint verified: {cv_passed} pass, {cv_failed} fail")

            by_type = defaultdict(lambda: [0, 0, 0])
            for qid, qtype, success, cv, _ in sorted(solver_results):
                by_type[qtype][2] += 1
                cv_str = "" if cv is None else (" ✓" if cv else " ✗cv")
                status = "PASS" if success else "FAIL"
                print(f"  Q{qid:2d} ({qtype:16s}): {status}{cv_str}")
                if success:
                    by_type[qtype][0] += 1
                if cv is True:
                    by_type[qtype][1] += 1

            print(f"\n  By type (toy/constraint/total):")
            for t, (tp, cp, tot) in sorted(by_type.items()):
                print(f"    {t:16s}: {tp}/{tot} toy, {cp}/{tot} constraint")

        summary = {
            "timestamp": timestamp,
            "pipeline": args.llm,
            "total_time_seconds": round(total_time, 1),
            "results": [
                {"solver": st, "query_id": qid, "type": qt, "success": s, "constraint_satisfied": cv}
                for st, qid, qt, s, cv, _ in all_results
            ],
        }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {output_dir}/summary.json")


if __name__ == "__main__":
    asyncio.run(main())
