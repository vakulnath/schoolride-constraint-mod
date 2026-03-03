import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

from src.configs.config_parser import (
    SBR_VENV_PY, TOY_TEST_PATH, SOLVERS_ROOT, SOLVERS_SRC, SIM_ROOT, SIM_SRC, SOLVERS_POLICY_ROOT,
    CONTEXT_RUNS_ROOT, VANILLA_BASELINE_DIR, TEMP_CODE_DIR, OUTPUT_DIR, POLICY_DIR, LOGS_DIR,
    MAIN_AGENT_LOG, AGENT_LOG, LLM_EDIT_LOG, SCHEDULE_DIFF_LOG, QUERY_SUMMARY_LOG, PANEL_JUDGMENT_LOG,
    INSERTION_SOLVER_DIR, HEXALY_SOLVER_DIR, INSERTION_TEMP_CODE_DIR, HEXALY_TEMP_CODE_DIR
)
from src.core.claude_agent import run_claude_agent
from src.core.api_agent import run_api_agent
from src.core.main_agent import run_main_agent
from src.core.schedule_diff import compare_schedules
from src.utils.utils import load_env_file, load_queries, pick_query


def _write_query_summary_log(outdir: str, query_id: int, question: str, summary: Dict[str, Any]) -> None:
    """Write a comprehensive summary log for the query."""
    insertion_ok = summary.get("insertion_ok", False)
    hexaly_ok = summary.get("hexaly_ok", False)
    combined_ok = summary.get("combined_ok")
    diff_result = summary.get("schedule_diff")
    timings = summary.get("timings", [])

    # Calculate total time
    total_time = sum(t.get("duration", 0) for t in timings)

    # Build summary text
    insertion_status = "PASS" if insertion_ok else ("FAIL" if summary.get("insertion_ok") is False else "NO EDIT")
    hexaly_status = "PASS" if hexaly_ok else ("FAIL" if summary.get("hexaly_ok") is False else "NO EDIT")
    combined_status = "PASS" if combined_ok else ("FAIL" if combined_ok is False else "SKIPPED")

    lines = [
        f"Query {query_id}: {question}",
        "=" * 80,
        "",
        "SOLVER RESULTS:",
        f"  Insertion:     {insertion_status}",
        f"  Hexaly:        {hexaly_status}",
        "",
        "COMBINED TEST:",
        f"  Status:        {combined_status}",
        "",
    ]

    if combined_ok is None or combined_ok is False:
        if not insertion_ok or not hexaly_ok:
            lines.append("  Reason:        One or more solvers failed")
        elif combined_ok is False:
            lines.append("  Reason:        Combined test failed despite individual solver passes")

    # Agent trial counts
    agent_trials = summary.get("agent_trials", {})
    if agent_trials:
        lines.extend([
            "",
            "AGENT TRIALS:",
        ])
        if "insertion_judgment_agent" in agent_trials or "hexaly_judgment_agent" in agent_trials:
            # Dual solver mode
            ins_ja = agent_trials.get("insertion_judgment_agent", 0)
            hex_ja = agent_trials.get("hexaly_judgment_agent", 0)
            ins_er = agent_trials.get("insertion_error_recovery", 0)
            hex_er = agent_trials.get("hexaly_error_recovery", 0)
            lines.append(f"  Insertion Judgment Agent:  {ins_ja} round(s)")
            lines.append(f"  Hexaly Judgment Agent:     {hex_ja} round(s)")
            if ins_er > 0 or hex_er > 0:
                lines.append(f"  Insertion Error Recovery:  {ins_er} round(s)")
                lines.append(f"  Hexaly Error Recovery:     {hex_er} round(s)")
        else:
            # Single solver or CLI mode
            if "judgment_agent" in agent_trials:
                lines.append(f"  Judgment Agent:        {agent_trials['judgment_agent']} round(s)")
            if "error_recovery" in agent_trials:
                lines.append(f"  Error Recovery:        {agent_trials['error_recovery']} round(s)")
        if "constraint_retry_rounds" in agent_trials:
            lines.append(f"  Constraint Retry:      {agent_trials['constraint_retry_rounds']} round(s)")

    lines.extend([
        "",
        "CONSTRAINT VERIFICATION:",
    ])

    if diff_result:
        constraint_satisfied = diff_result.get("constraint_satisfied")
        if constraint_satisfied is True:
            lines.append("  Status:        PASS")
            lines.append(f"  Detail:        {diff_result.get('constraint_detail', 'N/A')}")
        elif constraint_satisfied is False:
            lines.append("  Status:        FAIL")
            lines.append(f"  Detail:        {diff_result.get('constraint_detail', 'N/A')}")
        else:
            lines.append("  Status:        N/A (no modified schedule available)")

        regression_free = diff_result.get("regression_free")
        if regression_free is not None:
            regression_status = "PASS (no stops lost)" if regression_free else "FAIL (stops lost)"
            lines.append(f"  Regression:    {regression_status}")
    else:
        lines.append("  Status:        N/A (no schedule diff available)")

    lines.extend([
        "",
        "TIMING:",
    ])
    for timing in timings:
        step = timing.get("step", "UNKNOWN")
        duration = timing.get("duration", 0)
        lines.append(f"  {step:<20} {duration:>8.1f}s")

    lines.extend([
        f"  {'TOTAL':<20} {total_time:>8.1f}s",
        "",
    ])

    # Write to file
    log_path = os.path.join(outdir, "QUERY_SUMMARY.txt")
    with open(log_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None, help="Repository root path (defaults to current directory)")
    parser.add_argument("--queries", default="src/configs/queries.json")
    parser.add_argument("--query-id", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--toy-test", default=None)
    parser.add_argument("--sbr-venv-python", default=None)
    parser.add_argument("--toy-data-root", default=None)
    parser.add_argument("--runtime-retries", type=int, default=1)
    parser.add_argument(
        "--agent-type",
        choices=["cli", "api"],
        default="cli",
        help=(
            "Agent mode: "
            "'cli' = Claude Code CLI, "
            "'api' = API agent with native tool-calling over MCP tools"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=["insertion", "both"],
        default="insertion",
        help=(
            "Solver scope for API mode: "
            "'insertion' = 4-agent flow for insertion heuristic only, "
            "'both' = parallel 4-agent flows for insertion + hexaly"
        ),
    )
    parser.add_argument(
        "--nversion",
        action="store_true",
        default=False,
        help="Use N-version Main Agent orchestration (3×GA + 3×JA panel) instead of single GA",
    )
    args = parser.parse_args()

    # Load environment from project root
    os.environ.update(load_env_file(".env"))

    # Context root: solvers source directory (for schema loading and original file diffs)
    context_root = str(SOLVERS_SRC) if SOLVERS_SRC else str(SOLVERS_ROOT)
    if not os.path.isdir(context_root):
        print(f"\nERROR: Solvers source not found at {context_root}")
        print("Check SOLVERS_ROOT / SOLVERS_SRC in environment or src/configs/config.ini")
        sys.exit(1)
    print(f"Using solvers source: {context_root}")

    # Load queries (relative to project root)
    queries_path = args.queries
    if not os.path.isabs(queries_path) and not os.path.exists(queries_path):
        # Try relative path from project root
        pass  # queries_path is already relative, just use it as-is

    queries = load_queries(queries_path)

    env = os.environ.copy()
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    # Select agent function based on --agent-type.
    if args.agent_type == "cli":
        run_agent = run_claude_agent
        agent_env = env.copy()
    else:
        run_agent = run_api_agent
        agent_env = env.copy()
        agent_env["LLM_PROVIDER"] = "gemini"

    print(f"Starting pipeline with agent={args.agent_type}, query_id={args.query_id}, queries found: {len(queries)}")

    # ---------------------------------------------------------------

    def _run_toy_test(
        temp_code_dir: str,
        output_dir: str,
        query_id: int,
        label: str = "modified",
        policy: str = "insertion",
    ) -> bool:
        """Run the toy test and write results.

        Args:
            temp_code_dir: Directory whose ``policy/`` sub-package is on PYTHONPATH.
            output_dir: Where to write stdout/stderr/status files.
            query_id: Used to set TOY_DATA_DIRNAME so the test writes to a
                      unique sub-folder.
            label: ``"vanilla"`` or ``"modified"`` — prefixed to output filenames
                   so both runs can coexist in the same output_dir.
            policy: ``"insertion"`` or ``"hexaly"`` — sets TOY_POLICY env var
                    to override the config file default.
        """
        toy_test = args.toy_test or (str(TOY_TEST_PATH) if TOY_TEST_PATH else None)
        if not toy_test or not os.path.exists(toy_test):
            raise RuntimeError("Toy test path not configured. Set TOY_TEST_PATH in config.")

        sbr_venv_python = args.sbr_venv_python or (str(SBR_VENV_PY) if SBR_VENV_PY else None)
        if not sbr_venv_python or not os.path.exists(sbr_venv_python):
            sbr_venv_python = sys.executable

        policy_dir = os.path.join(temp_code_dir, POLICY_DIR)
        if not os.path.isdir(policy_dir):
            with open(os.path.join(output_dir, f"{label}_toy_run.status"), "w") as f:
                f.write("failed_no_policy_dir\n")
            return False

        # Ensure __init__.py exists
        init_py = os.path.join(policy_dir, "__init__.py")
        if not os.path.exists(init_py):
            original_init = os.path.join(str(SOLVERS_POLICY_ROOT), "__init__.py")
            if os.path.exists(original_init):
                shutil.copy2(original_init, init_py)
            else:
                with open(init_py, "w") as f:
                    f.write("")

        test_env = env.copy()
        sim_src = str(SIM_SRC) if SIM_SRC else ""
        pythonpath_parts = [temp_code_dir]
        if SOLVERS_SRC and os.path.isdir(str(SOLVERS_SRC)):
            solvers_src_str = str(SOLVERS_SRC)
            if "site-packages" not in solvers_src_str:
                pythonpath_parts.append(solvers_src_str)
        if sim_src and os.path.isdir(sim_src):
            pythonpath_parts.append(sim_src)
        test_env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

        # Vanilla and modified runs write to separate sub-directories so
        # their bus_schedule_after.json files don't overwrite each other.
        data_dirname = f"query_{query_id}_{label}"
        test_env["TOY_DATA_DIRNAME"] = data_dirname
        # Use absolute path so pytest can find it regardless of rootdir
        abs_output_dir = os.path.abspath(output_dir)
        test_env["TOY_DATA_ROOT"] = args.toy_data_root if args.toy_data_root else abs_output_dir
        test_env["TOY_POLICY"] = policy
        # Set SIM_ROOT so config path resolution works in installed mode
        if SIM_ROOT:
            test_env["SIM_ROOT"] = str(SIM_ROOT)

        # For modified runs, set the constraint query ID so toy test knows which assertion to run
        if label == "modified" and query_id > 0:
            test_env["CONSTRAINT_QUERY_ID"] = str(query_id)

        proc = subprocess.Popen(
            [sbr_venv_python, "-m", "pytest", "-s", toy_test],
            env=test_env,
            cwd=str(SIM_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_text, stderr_text = proc.communicate()

        # Print to terminal so hexaly model init / solver output is visible
        if stdout_text:
            print(stdout_text)
        if stderr_text:
            print(stderr_text, file=sys.stderr)

        with open(os.path.join(output_dir, f"{label}_toy_run.out"), "w") as f:
            f.write(stdout_text)
        with open(os.path.join(output_dir, f"{label}_toy_run.err"), "w") as f:
            f.write(stderr_text)

        has_traceback = "Traceback" in stdout_text or "Traceback" in stderr_text
        ok = proc.returncode == 0 and not has_traceback
        with open(os.path.join(output_dir, f"{label}_toy_run.status"), "w") as f:
            f.write("ok\n" if ok else "failed\n")
        return ok

    # ---------------------------------------------------------------
    # 4-agent AFL-style flow (reusable for both insertion and hexaly)
    # ---------------------------------------------------------------

    def _run_4agent_flow(
        solver_type: str,
        question: str,
        context_root: str,
        agent_env: Dict[str, str],
        flow_logs_dir: str,
        flow_temp_code_dir: str,
        flow_output_dir: str,
        base_dir: str,
        query_id: int,
    ) -> Dict[str, Any]:
        """Run 4-agent AFL-style orchestration for one solver type.

        Returns::

            {
                "success": bool,
                "edit": dict or None,
                "modified_ok": bool,
                "cost_usd": float,
                "input_tokens": int,
                "output_tokens": int,
            }
        """
        from src.agents.generation_agent import run_generation_agent
        from src.agents.judgment_agent import run_judgment_agent
        from src.agents.error_analysis_agent import run_error_analysis_agent
        from src.agents.revision_agent import run_revision_agent

        local_code_dir = os.path.join(flow_temp_code_dir, POLICY_DIR)
        tag = solver_type.upper()[:3]  # INS or HEX

        MAX_JA_ROUNDS = 2
        MAX_FIX_ROUNDS = 3

        total_cost = 0.0
        total_input = 0
        total_output = 0
        timings: List[Dict[str, Any]] = []

        # Track agent trial counts for summary reporting
        ja_rounds_completed = 0
        fix_rounds_completed = 0

        def _track(result: Dict):
            nonlocal total_cost, total_input, total_output
            total_cost += result.get("cost_usd", 0)
            total_input += result.get("input_tokens", 0)
            total_output += result.get("output_tokens", 0)

        def _restore():
            """Reset temp policy dir to pristine copy before a revision."""
            if SOLVERS_POLICY_ROOT and os.path.isdir(str(SOLVERS_POLICY_ROOT)):
                if os.path.exists(local_code_dir):
                    shutil.rmtree(local_code_dir)
                shutil.copytree(str(SOLVERS_POLICY_ROOT), local_code_dir)

        def _get_diff(edit: Dict[str, str]) -> str:
            return edit.get("diff", "")

        def _read_edited_source(edit: Dict[str, str]) -> str:
            """Read the original (pre-edit) function for scope verification."""
            fname = edit.get("path", "")
            diff = edit.get("diff", "")
            if not fname or not diff:
                return ""
            orig_dir = str(SOLVERS_POLICY_ROOT) if SOLVERS_POLICY_ROOT else ""
            if not orig_dir:
                return ""
            fpath = os.path.join(orig_dir, os.path.basename(fname))
            if not os.path.exists(fpath):
                return ""

            with open(fpath, "r") as f:
                lines = f.readlines()

            changed_line = None
            for dline in diff.splitlines():
                m = re.match(r"^@@ -(\d+)", dline)
                if m:
                    changed_line = int(m.group(1)) - 1
                    break
            if changed_line is None:
                return ""

            func_start = None
            for i in range(min(changed_line, len(lines) - 1), -1, -1):
                if lines[i].lstrip().startswith("def "):
                    func_start = i
                    break
            if func_start is None:
                return ""

            func_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
            func_end = len(lines)
            for i in range(func_start + 1, len(lines)):
                stripped = lines[i].lstrip()
                if not stripped or stripped.startswith("#"):
                    continue
                line_indent = len(lines[i]) - len(stripped)
                if line_indent <= func_indent and (
                    stripped.startswith("def ") or stripped.startswith("class ")
                ):
                    func_end = i
                    break

            return "".join(lines[func_start:func_end])

        def _fail_result():
            return {
                "success": False, "edit": None, "modified_ok": False,
                "cost_usd": total_cost, "input_tokens": total_input,
                "output_tokens": total_output, "timings": timings,
            }

        # ---- Phase 1: Generation Agent (GA) ----
        print(f"[{tag}-GA] Generating edit...")
        t0 = time.time()
        ga_result = run_generation_agent(
            constraint=question,
            context_root=context_root,
            env=agent_env,
            logs_dir=flow_logs_dir,
            temp_code_dir=flow_temp_code_dir,
            solver_type=solver_type,
        )
        _track(ga_result)
        timings.append({"step": f"{tag}-GA", "duration": time.time() - t0})

        current_edit = ga_result.get("edit")

        if not (ga_result.get("success") and current_edit):
            print(f"[{tag}-GA] Failed: {ga_result.get('error', 'no edit produced')}")
            return _fail_result()

        print(f"[{tag}-GA] Edit: {current_edit.get('path')}")
        diff_text = _get_diff(current_edit)

        # ---- Phase 2: Judgment Agent (JA) pre-flight ----
        ja_approved = False
        t0 = time.time()
        for ja_round in range(MAX_JA_ROUNDS):
            ja_rounds_completed += 1
            print(f"[{tag}-JA] Reviewing edit (round {ja_round + 1}/{MAX_JA_ROUNDS})...")
            ja_result = run_judgment_agent(
                constraint=question,
                diff=diff_text,
                env=agent_env,
                logs_dir=flow_logs_dir,
                base_dir=base_dir,
                source_context=_read_edited_source(current_edit),
                solver_type=solver_type,
            )
            _track(ja_result)

            if ja_result["approved"]:
                print(f"[{tag}-JA] APPROVED")
                ja_approved = True
                break

            print(f"[{tag}-JA] REJECTED — {ja_result['feedback'][:200]}")

            if ja_round < MAX_JA_ROUNDS - 1:
                print(f"[{tag}-RA] Revising based on JA feedback...")
                _restore()
                ra_result = run_revision_agent(
                    constraint=question,
                    feedback=ja_result["feedback"],
                    feedback_source="judgment",
                    previous_diff=diff_text,
                    context_root=context_root,
                    env=agent_env,
                    logs_dir=flow_logs_dir,
                    temp_code_dir=flow_temp_code_dir,
                    solver_type=solver_type,
                )
                _track(ra_result)

                if ra_result.get("success") and ra_result.get("edit"):
                    current_edit = ra_result["edit"]
                    diff_text = _get_diff(current_edit)
                    print(f"[{tag}-RA] Revised edit: {current_edit.get('path')}")
                else:
                    print(f"[{tag}-RA] Failed: {ra_result.get('error', 'no edit')}")
                    break

        timings.append({"step": f"{tag}-JA", "duration": time.time() - t0})

        if not ja_approved:
            print(f"[{tag}] Edit not approved after max JA rounds — skipping toy test")
            return _fail_result()

        applied_edits = [current_edit]
        with open(os.path.join(flow_logs_dir, LLM_EDIT_LOG), "w") as f:
            f.write(json.dumps({
                "edit": current_edit,
                "thoughts": ga_result.get("thoughts", []),
            }, indent=2))

        # ---- Phase 3: Toy test ----
        print(f"[{tag}-TOY] Running modified toy test...")
        t0 = time.time()
        modified_ok = _run_toy_test(
            flow_temp_code_dir, flow_output_dir, query_id, label="modified",
            policy=solver_type,
        )
        timings.append({"step": f"{tag}-TOY", "duration": time.time() - t0})

        # ---- Phase 4: Error recovery (EAA → RA → JA → re-test) ----
        prev_diff_text = None
        t0 = time.time()
        for fix_round in range(MAX_FIX_ROUNDS):
            if modified_ok:
                break

            fix_rounds_completed += 1
            traceback_text = ""
            for fname in ("modified_toy_run.err", "modified_toy_run.out"):
                fpath = os.path.join(flow_output_dir, fname)
                if os.path.exists(fpath):
                    with open(fpath, "r") as f:
                        traceback_text += f.read()
            if not any(kw in traceback_text for kw in ("Traceback", "Error", "ERRORS")):
                break

            print(f"\n[{tag}-EAA] Diagnosing failure (round {fix_round + 1}/{MAX_FIX_ROUNDS})...")
            eaa_result = run_error_analysis_agent(
                constraint=question,
                diff=diff_text,
                traceback_text=traceback_text,
                env=agent_env,
                logs_dir=flow_logs_dir,
            )
            _track(eaa_result)
            print(f"[{tag}-EAA] Diagnosis: {eaa_result['diagnosis'][:200]}")

            print(f"[{tag}-RA] Revising based on EAA diagnosis...")
            _restore()
            ra_result = run_revision_agent(
                constraint=question,
                feedback=eaa_result["diagnosis"],
                feedback_source="error_analysis",
                previous_diff=diff_text,
                context_root=context_root,
                env=agent_env,
                logs_dir=flow_logs_dir,
                temp_code_dir=flow_temp_code_dir,
                solver_type=solver_type,
            )
            _track(ra_result)

            if not (ra_result.get("success") and ra_result.get("edit")):
                print(f"[{tag}-RA] Failed: {ra_result.get('error', 'no edit')}")
                break

            current_edit = ra_result["edit"]
            diff_text = _get_diff(current_edit)

            if diff_text == prev_diff_text:
                print(f"[{tag}] RA produced identical edit — stopping fix loop")
                break
            prev_diff_text = diff_text

            print(f"[{tag}-RA] Revised edit: {current_edit.get('path')}")

            print(f"[{tag}-JA] Reviewing RA fix (fix round {fix_round + 1})...")
            ja_fix_result = run_judgment_agent(
                constraint=question,
                diff=diff_text,
                env=agent_env,
                logs_dir=flow_logs_dir,
                base_dir=base_dir,
                source_context=_read_edited_source(current_edit),
                solver_type=solver_type,
            )
            _track(ja_fix_result)

            if not ja_fix_result["approved"]:
                print(f"[{tag}-JA] REJECTED RA fix — {ja_fix_result['feedback'][:200]}")
                continue

            print(f"[{tag}-JA] APPROVED RA fix")
            applied_edits = [current_edit]
            modified_ok = _run_toy_test(
                flow_temp_code_dir, flow_output_dir, query_id, label="modified",
                policy=solver_type,
            )
            if modified_ok:
                print(f"[{tag}-TOY] PASSED after fix round {fix_round + 1}")

        eaa_duration = time.time() - t0
        if eaa_duration > 0.1:  # only log if recovery loop actually ran
            timings.append({"step": f"{tag}-EAA-RETRY", "duration": eaa_duration})

        return {
            "success": len(applied_edits) > 0,
            "edit": applied_edits[0] if applied_edits else None,
            "modified_ok": modified_ok,
            "cost_usd": total_cost,
            "input_tokens": total_input,
            "output_tokens": total_output,
            "timings": timings,
            "agent_trials": {
                "judgment_agent": ja_rounds_completed,
                "error_recovery": fix_rounds_completed,
            },
        }

    # ---------------------------------------------------------------

    def run_single_query(
        q: Dict[str, Any],
        query_id: int,
        base_outdir: str,
        vanilla_ok: bool,
        vanilla_schedule: str,
    ) -> Dict[str, Any]:
        query_start = time.time()
        query_timings: List[Dict[str, Any]] = []
        print(f"\nRunning query {query_id}: {q.get('question', '')}")
        query_suffix = f"query_{int(query_id)}"
        outdir = base_outdir if os.path.basename(base_outdir) == query_suffix else os.path.join(base_outdir, query_suffix)
        os.makedirs(outdir, exist_ok=True)

        # Set up directory structure
        logs_dir = os.path.join(outdir, LOGS_DIR)
        temp_code_dir = os.path.join(outdir, TEMP_CODE_DIR)
        output_dir = os.path.join(outdir, OUTPUT_DIR)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(temp_code_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Copy policy package to temp_code/ so agent edits it there
        local_code_dir = os.path.join(temp_code_dir, POLICY_DIR)
        if SOLVERS_POLICY_ROOT and os.path.isdir(str(SOLVERS_POLICY_ROOT)):
            if os.path.exists(local_code_dir):
                shutil.rmtree(local_code_dir)
            shutil.copytree(str(SOLVERS_POLICY_ROOT), local_code_dir)
            print(f"Copied policy package to {local_code_dir}")
        os.makedirs(local_code_dir, exist_ok=True)

        # --- 2. Run agent + test ---
        question = q.get('question', '')
        print(f"Constraint: {question}")

        applied_edits: List[Dict[str, str]] = []
        modified_ok = False
        insertion_result: Dict[str, Any] = {}
        hexaly_result: Dict[str, Any] = {}
        constraint_retry_rounds = 0

        # N-VERSION MAIN AGENT ORCHESTRATION (3 GA + 3 JA panel)
        if args.agent_type == "api" and args.nversion and args.solver == "insertion":
            # ============================================================
            # INSERTION-ONLY N-VERSION MAIN AGENT ORCHESTRATION
            #   Main Agent orchestrates: 3×GA (Flash/Pro/2.5Pro) in parallel
            #                           3×JA panel judges in parallel
            #                           → select winner via consensus
            #   Then: Toy Test → EAA → RA → JA (single) if failed
            # ============================================================
            insertion_result = run_main_agent(
                constraint=question,
                context_root=context_root,
                solver_type="insertion",
                env=agent_env,
                logs_dir=logs_dir,
                temp_code_dir_base=temp_code_dir,
            )
            query_timings.extend(insertion_result.get("timings", []))

            total_cost = insertion_result.get("cost_usd", 0)
            total_input = insertion_result.get("input_tokens", 0)
            total_output = insertion_result.get("output_tokens", 0)

            if insertion_result.get("winner_edit"):
                applied_edits = [insertion_result["winner_edit"]]

            agent_result = {
                "success": len(applied_edits) > 0,
                "edit": applied_edits[0] if applied_edits else None,
                "iterations": None,
                "cost_usd": total_cost,
                "input_tokens": total_input,
                "output_tokens": total_output,
            }
            agent_success = agent_result["success"]
            agent_edit = agent_result["edit"]

        elif args.agent_type == "api" and args.solver == "insertion":
            # ============================================================
            # INSERTION-ONLY 4-AGENT AFL-STYLE ORCHESTRATION
            #   Single flow: GA → JA → (RA if rejected) → toy test
            #                → (EAA → RA → JA if failed) → re-test
            # ============================================================
            insertion_result = _run_4agent_flow(
                "insertion", question, context_root, agent_env,
                logs_dir, temp_code_dir, output_dir,
                base_outdir, query_id,
            )
            query_timings.extend(insertion_result.get("timings", []))

            total_cost = insertion_result.get("cost_usd", 0)
            total_input = insertion_result.get("input_tokens", 0)
            total_output = insertion_result.get("output_tokens", 0)
            modified_ok = insertion_result.get("modified_ok", False)

            if insertion_result.get("edit"):
                applied_edits = [insertion_result["edit"]]

            agent_result = {
                "success": len(applied_edits) > 0,
                "edit": applied_edits[0] if applied_edits else None,
                "iterations": None,
                "cost_usd": total_cost,
                "input_tokens": total_input,
                "output_tokens": total_output,
            }
            agent_success = agent_result["success"]
            agent_edit = agent_result["edit"]

        elif args.agent_type == "api" and args.solver == "both":
            # ============================================================
            # PARALLEL 4-AGENT AFL-STYLE ORCHESTRATION
            #   Two flows run in parallel:
            #   - Insertion: edits insert_heuristic.py / insertion_policy.py
            #   - Hexaly:    edits hexaly_planner_cleanup.py / hexaly_helper.py
            #   Each flow: GA → JA → (RA if rejected) → toy test
            #              → (EAA → RA → JA if failed) → re-test
            # ============================================================

            # Set up separate dirs for each solver type
            insertion_temp = os.path.join(outdir, INSERTION_TEMP_CODE_DIR)
            hexaly_temp = os.path.join(outdir, HEXALY_TEMP_CODE_DIR)
            insertion_logs = os.path.join(logs_dir, INSERTION_SOLVER_DIR)
            hexaly_logs = os.path.join(logs_dir, HEXALY_SOLVER_DIR)
            insertion_output = os.path.join(output_dir, INSERTION_SOLVER_DIR)
            hexaly_output = os.path.join(output_dir, HEXALY_SOLVER_DIR)

            for d in (insertion_temp, hexaly_temp, insertion_logs, hexaly_logs,
                       insertion_output, hexaly_output):
                os.makedirs(d, exist_ok=True)

            # Copy pristine policy to both temp dirs
            for td in (insertion_temp, hexaly_temp):
                pd = os.path.join(td, POLICY_DIR)
                if SOLVERS_POLICY_ROOT and os.path.isdir(str(SOLVERS_POLICY_ROOT)):
                    if os.path.exists(pd):
                        shutil.rmtree(pd)
                    shutil.copytree(str(SOLVERS_POLICY_ROOT), pd)

            # Run both flows in parallel
            print(f"\n[PARALLEL] Launching insertion + hexaly flows for Q{query_id}...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                ins_future = executor.submit(
                    _run_4agent_flow,
                    "insertion", question, context_root, agent_env,
                    insertion_logs, insertion_temp, insertion_output,
                    base_dir, query_id,
                )
                hex_future = executor.submit(
                    _run_4agent_flow,
                    "hexaly", question, context_root, agent_env,
                    hexaly_logs, hexaly_temp, hexaly_output,
                    base_dir, query_id,
                )

                insertion_result = ins_future.result()
                hexaly_result = hex_future.result()

            query_timings.extend(insertion_result.get("timings", []))
            query_timings.extend(hexaly_result.get("timings", []))

            # Report individual results
            total_cost = 0.0
            total_input = 0
            total_output = 0

            for result, label in [(insertion_result, "insertion"), (hexaly_result, "hexaly")]:
                total_cost += result.get("cost_usd", 0)
                total_input += result.get("input_tokens", 0)
                total_output += result.get("output_tokens", 0)

                edit = result.get("edit")
                if edit:
                    applied_edits.append(edit)
                    ok_str = "PASS" if result.get("modified_ok") else "FAIL"
                    print(f"[{label.upper()}] {ok_str} — edit: {edit.get('path')}")
                else:
                    print(f"[{label.upper()}] No edit produced")

            # Merge both edits into the combined temp_code_dir for final test
            # Only run combined test if BOTH solvers passed
            both_solvers_passed = (
                insertion_result.get("modified_ok", False) and
                hexaly_result.get("modified_ok", False)
            )

            if applied_edits:
                if os.path.exists(local_code_dir):
                    shutil.rmtree(local_code_dir)
                shutil.copytree(str(SOLVERS_POLICY_ROOT), local_code_dir)

                for edit, flow_temp in [
                    (insertion_result.get("edit"), insertion_temp),
                    (hexaly_result.get("edit"), hexaly_temp),
                ]:
                    if not edit:
                        continue
                    basename = os.path.basename(edit.get("path", ""))
                    if not basename:
                        continue
                    src = os.path.join(flow_temp, POLICY_DIR, basename)
                    dst = os.path.join(local_code_dir, basename)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        print(f"[COMBINED] Merged {basename}")

                # Run combined toy test ONLY if both solvers passed
                if both_solvers_passed:
                    print("[COMBINED-TOY] Running combined toy test...")
                    modified_ok = _run_toy_test(
                        temp_code_dir, output_dir, query_id, label="modified",
                        policy="hexaly",
                    )
                    if modified_ok:
                        print("[COMBINED-TOY] PASSED")
                    else:
                        print("[COMBINED-TOY] FAILED")
                else:
                    print("[COMBINED-TOY] SKIPPED — one or more solvers failed")

            # Save both edits to logs
            with open(os.path.join(logs_dir, LLM_EDIT_LOG), "w") as f:
                f.write(json.dumps({
                    "insertion_edit": insertion_result.get("edit"),
                    "hexaly_edit": hexaly_result.get("edit"),
                    "insertion_ok": insertion_result.get("modified_ok", False),
                    "hexaly_ok": hexaly_result.get("modified_ok", False),
                }, indent=2))

            # Build unified agent_result for downstream summary
            agent_result = {
                "success": len(applied_edits) > 0,
                "edit": applied_edits[0] if applied_edits else None,
                "iterations": None,
                "cost_usd": total_cost,
                "input_tokens": total_input,
                "output_tokens": total_output,
            }
            agent_success = agent_result["success"]
            agent_edit = agent_result["edit"]

        else:
            # ============================================================
            # CLI MODE: SINGLE-AGENT FLOW (original behavior)
            # ============================================================
            agent_result = run_agent(
                constraint=question,
                context_root=context_root,
                env=agent_env,
                logs_dir=logs_dir,
                temp_code_dir=temp_code_dir,
            )

            agent_success = agent_result.get('success', False)
            agent_edit = agent_result.get('edit')

            if agent_success and agent_edit:
                print(f"Agent edited: {agent_edit.get('path')}")
                with open(os.path.join(logs_dir, LLM_EDIT_LOG), "w") as f:
                    f.write(json.dumps({"edit": agent_edit, "thoughts": agent_result.get('thoughts', [])}, indent=2))
                applied_edits = [agent_edit]
            else:
                print(f"Agent failed: {agent_result.get('error', 'no edit produced')}")

            # --- Modified toy test (after agent edit) ---
            if applied_edits:
                print("Running modified toy test...")
                modified_ok = _run_toy_test(temp_code_dir, output_dir, query_id, label="modified")

                max_toy_attempts = 3
                for fix_attempt in range(1, max_toy_attempts):
                    if modified_ok:
                        break

                    traceback_text = ""
                    for fname in ("modified_toy_run.err", "modified_toy_run.out"):
                        fpath = os.path.join(output_dir, fname)
                        if os.path.exists(fpath):
                            with open(fpath, "r") as f:
                                traceback_text += f.read()
                    if not any(kw in traceback_text for kw in ("Traceback", "Error", "ERRORS")):
                        break

                    print(f"\nToy test failed — retrying with error feedback (attempt {fix_attempt + 1}/{max_toy_attempts})")

                    if SOLVERS_POLICY_ROOT and applied_edits:
                        filename = os.path.basename(applied_edits[0].get("path", ""))
                        if filename:
                            original_file = os.path.join(str(SOLVERS_POLICY_ROOT), filename)
                            dest_file = os.path.join(local_code_dir, filename)
                            if os.path.exists(original_file):
                                shutil.copy2(original_file, dest_file)
                                print(f"Restored original {filename}")

                    prev_edit = applied_edits[0] if applied_edits else {}
                    error_constraint = (
                        f"{question}\n\n"
                        f"IMPORTANT: A previous edit attempt caused this runtime error:\n"
                        f"```\n{traceback_text[-1000:]}\n```\n"
                        f"The previous edit was in file: {prev_edit.get('path', '?')}\n"
                        f"Try a DIFFERENT function or a DIFFERENT approach to avoid this error."
                    )

                    retry_result = run_agent(
                        constraint=error_constraint,
                        context_root=context_root,
                        env=agent_env,
                        logs_dir=logs_dir,
                        temp_code_dir=temp_code_dir,
                    )

                    if retry_result.get('success') and retry_result.get('edit'):
                        applied_edits = [retry_result['edit']]
                        print(f"Retry produced new edit: {retry_result['edit'].get('path')}")
                        modified_ok = _run_toy_test(temp_code_dir, output_dir, query_id, label="modified")
                        if modified_ok:
                            print(f"Toy test PASSED after retry {fix_attempt}")
                    else:
                        print(f"Retry failed: {retry_result.get('error', 'unknown')}")
                        break

        # --- 4. Schedule diff + constraint verification (unified) ---
        t0 = time.time()
        modified_schedule = os.path.join(
            output_dir, f"query_{query_id}_modified", "bus_schedule_after.json"
        )

        diff_result = None
        if os.path.exists(vanilla_schedule) and os.path.exists(modified_schedule):
            diff_result = compare_schedules(vanilla_schedule, modified_schedule, query=q)
            print(f"Schedule diff:\n{diff_result.get('summary', '')}")

            if diff_result.get("regression_free"):
                print("Regression check: PASS (no stops lost)")
            elif diff_result.get("regression_free") is False:
                print(f"Regression check: FAIL — lost stops: {diff_result.get('stops_lost')}")

            satisfied = diff_result.get("constraint_satisfied")
            if satisfied is True:
                print(f"Constraint check: PASS — {diff_result['constraint_detail']}")
            elif satisfied is False:
                print(f"Constraint check: FAIL — {diff_result['constraint_detail']}")
            else:
                print(f"Constraint check: N/A — {diff_result.get('constraint_detail', '')}")

            with open(os.path.join(logs_dir, SCHEDULE_DIFF_LOG), "w") as f:
                json.dump(diff_result, f, indent=2)
        else:
            missing = []
            if not os.path.exists(vanilla_schedule):
                missing.append("vanilla")
            if not os.path.exists(modified_schedule):
                missing.append("modified")
            print(f"Schedule diff: SKIPPED (missing {', '.join(missing)} schedule)")
        query_timings.append({"step": "SCHEDULE-DIFF", "duration": time.time() - t0})

        # --- 4b. Schedule diff → EAA constraint retry (API agent only) ---
        # When the toy test passed but the constraint is NOT satisfied in the
        # output schedule, feed the violation through EAA → RA → JA → re-test.
        MAX_CONSTRAINT_RETRIES = 2
        if (
            args.agent_type == "api"
            and diff_result
            and diff_result.get("constraint_satisfied") is False
            and modified_ok
            and applied_edits
        ):
            t0 = time.time()
            from src.agents.error_analysis_agent import run_error_analysis_agent
            from src.agents.revision_agent import run_revision_agent
            from src.agents.judgment_agent import run_judgment_agent

            retry_diff_text = applied_edits[0].get("diff", "")
            retry_solver = args.solver if args.solver != "both" else "insertion"
            constraint_retry_rounds = 0

            for cr_round in range(MAX_CONSTRAINT_RETRIES):
                constraint_retry_rounds += 1
                print(
                    f"\n[CONSTRAINT-RETRY] Round {cr_round + 1}/{MAX_CONSTRAINT_RETRIES}"
                    f" — constraint not satisfied: {diff_result.get('constraint_detail', '')}"
                )

                # Format constraint violation details for EAA
                violation_text = (
                    f"CONSTRAINT VERIFICATION FAILED (schedule diff check)\n"
                    f"Constraint: {question}\n"
                    f"Verification result: {diff_result.get('constraint_detail', '')}\n"
                    f"Schedule summary:\n{diff_result.get('summary', '')}\n"
                )

                print(f"[CONSTRAINT-EAA] Diagnosing constraint violation...")
                eaa_result = run_error_analysis_agent(
                    constraint=question,
                    diff=retry_diff_text,
                    traceback_text=violation_text,
                    env=agent_env,
                    logs_dir=logs_dir,
                )
                agent_result["cost_usd"] = agent_result.get("cost_usd", 0) + eaa_result.get("cost_usd", 0)
                agent_result["input_tokens"] = agent_result.get("input_tokens", 0) + eaa_result.get("input_tokens", 0)
                agent_result["output_tokens"] = agent_result.get("output_tokens", 0) + eaa_result.get("output_tokens", 0)
                print(f"[CONSTRAINT-EAA] Diagnosis: {eaa_result['diagnosis'][:200]}")

                # Restore pristine policy before revision
                _retry_policy = os.path.join(temp_code_dir, POLICY_DIR)
                if SOLVERS_POLICY_ROOT and os.path.isdir(str(SOLVERS_POLICY_ROOT)):
                    if os.path.exists(_retry_policy):
                        shutil.rmtree(_retry_policy)
                    shutil.copytree(str(SOLVERS_POLICY_ROOT), _retry_policy)

                print(f"[CONSTRAINT-RA] Revising based on constraint violation...")
                ra_result = run_revision_agent(
                    constraint=question,
                    feedback=eaa_result["diagnosis"],
                    feedback_source="constraint_violation",
                    previous_diff=retry_diff_text,
                    context_root=context_root,
                    env=agent_env,
                    logs_dir=logs_dir,
                    temp_code_dir=temp_code_dir,
                    solver_type=retry_solver,
                )
                agent_result["cost_usd"] += ra_result.get("cost_usd", 0)
                agent_result["input_tokens"] += ra_result.get("input_tokens", 0)
                agent_result["output_tokens"] += ra_result.get("output_tokens", 0)

                if not (ra_result.get("success") and ra_result.get("edit")):
                    print(f"[CONSTRAINT-RA] Failed: {ra_result.get('error', 'no edit')}")
                    break

                new_edit = ra_result["edit"]
                new_diff = new_edit.get("diff", "")
                print(f"[CONSTRAINT-RA] Revised edit: {new_edit.get('path')}")

                if new_diff == retry_diff_text:
                    print(f"[CONSTRAINT-RETRY] RA produced identical edit — stopping")
                    break
                retry_diff_text = new_diff

                # JA review
                print(f"[CONSTRAINT-JA] Reviewing revised edit...")
                ja_result = run_judgment_agent(
                    constraint=question,
                    diff=new_diff,
                    env=agent_env,
                    logs_dir=logs_dir,
                    base_dir=base_dir,
                    solver_type=retry_solver,
                )
                agent_result["cost_usd"] += ja_result.get("cost_usd", 0)
                agent_result["input_tokens"] += ja_result.get("input_tokens", 0)
                agent_result["output_tokens"] += ja_result.get("output_tokens", 0)

                if not ja_result["approved"]:
                    print(f"[CONSTRAINT-JA] REJECTED — {ja_result['feedback'][:200]}")
                    continue

                print(f"[CONSTRAINT-JA] APPROVED")
                applied_edits = [new_edit]

                # Re-run toy test
                print(f"[CONSTRAINT-TOY] Re-running toy test...")
                modified_ok = _run_toy_test(
                    temp_code_dir, output_dir, query_id, label="modified",
                    policy=retry_solver,
                )
                if not modified_ok:
                    print(f"[CONSTRAINT-TOY] FAILED — runtime error after constraint fix")
                    break

                print(f"[CONSTRAINT-TOY] PASSED")

                # Re-run schedule diff
                modified_schedule = os.path.join(
                    output_dir, f"query_{query_id}_modified", "bus_schedule_after.json"
                )
                if os.path.exists(vanilla_schedule) and os.path.exists(modified_schedule):
                    diff_result = compare_schedules(vanilla_schedule, modified_schedule, query=q)
                    print(f"[CONSTRAINT-DIFF] {diff_result.get('summary', '')}")

                    with open(os.path.join(logs_dir, SCHEDULE_DIFF_LOG), "w") as f:
                        json.dump(diff_result, f, indent=2)

                    if diff_result.get("constraint_satisfied") is True:
                        print(f"[CONSTRAINT-RETRY] PASS — constraint now satisfied!")
                        break
                    else:
                        print(f"[CONSTRAINT-RETRY] Still not satisfied: "
                              f"{diff_result.get('constraint_detail', '')}")
                else:
                    print(f"[CONSTRAINT-RETRY] Schedule file missing — cannot verify")
                    break
            query_timings.append({"step": "CONSTRAINT-RETRY", "duration": time.time() - t0})

        # --- Timing summary ---
        query_total = time.time() - query_start
        query_timings.append({"step": "TOTAL", "duration": query_total})
        print(f"\n  Q{query_id} timing:")
        for t in query_timings:
            print(f"    {t['step']:20s} {t['duration']:6.1f}s")

        # Collect agent trial information
        agent_trials_dict: Dict[str, Any] = {}

        # For API mode, extract trials from results
        if args.agent_type == "api":
            if args.solver == "insertion" and insertion_result:
                if "agent_trials" in insertion_result:
                    agent_trials_dict = insertion_result["agent_trials"].copy()
            elif args.solver == "both" and (insertion_result or hexaly_result):
                # Combine trials from both solvers
                ins_trials = insertion_result.get("agent_trials", {})
                hex_trials = hexaly_result.get("agent_trials", {})
                if ins_trials or hex_trials:
                    agent_trials_dict["insertion_judgment_agent"] = ins_trials.get("judgment_agent", 0)
                    agent_trials_dict["hexaly_judgment_agent"] = hex_trials.get("judgment_agent", 0)
                    agent_trials_dict["insertion_error_recovery"] = ins_trials.get("error_recovery", 0)
                    agent_trials_dict["hexaly_error_recovery"] = hex_trials.get("error_recovery", 0)

        # Add constraint retry tracking
        if constraint_retry_rounds > 0:
            agent_trials_dict["constraint_retry_rounds"] = constraint_retry_rounds

        summary: Dict[str, Any] = {
            "query_id": query_id,
            "question": q.get("question", ""),
            "entities": q.get("entities", {}),
            "agent": {
                "success": agent_success,
                "iterations": agent_result.get("iterations"),
                "cost_usd": agent_result.get("cost_usd"),
                "input_tokens": agent_result.get("input_tokens"),
                "output_tokens": agent_result.get("output_tokens"),
                "edit_path": agent_edit.get("path") if agent_edit else None,
            },
            "edit_applied": len(applied_edits) > 0,
            "vanilla_test": {"passed": vanilla_ok},
            "agent_trials": agent_trials_dict,
        }

        # Toy test status (modified run)
        modified_status_path = os.path.join(output_dir, "modified_toy_run.status")
        if os.path.exists(modified_status_path):
            with open(modified_status_path) as f:
                toy_status = f.read().strip()
            summary["toy_test"] = {"passed": toy_status == "ok"}
        else:
            summary["toy_test"] = None

        if diff_result:
            summary["schedule_diff"] = {
                "regression_free": diff_result.get("regression_free"),
                "constraint_satisfied": diff_result.get("constraint_satisfied"),
                "constraint_detail": diff_result.get("constraint_detail"),
                "stops_lost": diff_result.get("stops_lost", []),
                "reassignments": diff_result.get("reassignments", []),
                "buses_changed": diff_result.get("buses_changed", []),
            }
        else:
            summary["schedule_diff"] = None

        summary["timings"] = query_timings

        # Write query summary log for clarity
        _write_query_summary_log(outdir, query_id, question, summary)

        print(f"Wrote outputs to {outdir}")
        return summary

    # ---------------------------------------------------------------
    # Run queries
    # ---------------------------------------------------------------

    all_summaries: List[Dict[str, Any]] = []
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_outdir = args.outdir or os.path.join("context_runs", run_name)
    os.makedirs(base_outdir, exist_ok=True)

    # ---------------------------------------------------------------
    # Run vanilla baseline ONCE (same unmodified policy for all queries)
    # ---------------------------------------------------------------
    vanilla_dir = os.path.join(base_outdir, VANILLA_BASELINE_DIR)
    vanilla_code_dir = os.path.join(vanilla_dir, TEMP_CODE_DIR)
    vanilla_output_dir = os.path.join(vanilla_dir, OUTPUT_DIR)
    os.makedirs(vanilla_code_dir, exist_ok=True)
    os.makedirs(vanilla_output_dir, exist_ok=True)

    vanilla_policy_dir = os.path.join(vanilla_code_dir, POLICY_DIR)
    if SOLVERS_POLICY_ROOT and os.path.isdir(str(SOLVERS_POLICY_ROOT)):
        if os.path.exists(vanilla_policy_dir):
            shutil.rmtree(vanilla_policy_dir)
        shutil.copytree(str(SOLVERS_POLICY_ROOT), vanilla_policy_dir)

    print("Running vanilla baseline (once for all queries)...")
    vanilla_ok = _run_toy_test(vanilla_code_dir, vanilla_output_dir, 0, label="vanilla")
    vanilla_schedule = os.path.join(
        vanilla_output_dir, "query_0_vanilla", "bus_schedule_after.json"
    )
    if vanilla_ok:
        print(f"Vanilla baseline: PASSED (schedule at {vanilla_schedule})")
    else:
        print("Vanilla baseline: FAILED (baseline itself has errors)")

    # ---------------------------------------------------------------

    if args.query_id is None:
        for q in queries:
            qid = q.get("query_id")
            if qid is None:
                continue
            try:
                qid_int = int(qid)
            except Exception:
                continue
            summary = run_single_query(q, qid_int, base_outdir, vanilla_ok, vanilla_schedule)
            all_summaries.append(summary)
    else:
        q = pick_query(queries, args.query_id)
        print(f"Found query: {q}")
        print(f"Output directory: {base_outdir}")
        summary = run_single_query(q, int(args.query_id), base_outdir, vanilla_ok, vanilla_schedule)
        all_summaries.append(summary)

    # Write run summary
    run_summary = {
        "run_dir": base_outdir,
        "queries": all_summaries,
    }
    run_summary_path = os.path.join(base_outdir, "run_summary.json")
    with open(run_summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
    print(f"\nRun summary saved to {run_summary_path}")

    # --- Verification summary ---
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    passed = 0
    failed = 0
    for s in all_summaries:
        qid = s.get("query_id", "?")
        toy = s.get("toy_test") or {}
        sd = s.get("schedule_diff") or {}
        toy_ok = toy.get("passed", False)
        constraint_ok = sd.get("constraint_satisfied")
        regression_free = sd.get("regression_free")
        detail = sd.get("constraint_detail", "")

        if toy_ok and constraint_ok is True and regression_free is not False:
            status = "PASS"
            passed += 1
        elif not toy_ok:
            status = "FAIL (toy test)"
            detail = "runtime error or test failure"
            failed += 1
        elif constraint_ok is False:
            status = "FAIL (constraint)"
            failed += 1
        elif regression_free is False:
            status = "FAIL (regression)"
            detail = f"lost stops: {sd.get('stops_lost', [])}"
            failed += 1
        else:
            status = "SKIP (no verification)"
            detail = "no schedule produced"
            failed += 1

        reassign_count = len(sd.get("reassignments", []))
        buses_changed = len(sd.get("buses_changed", []))
        diff_info = f" [{reassign_count} reassignments, {buses_changed} buses changed]" if reassign_count else ""

        # Per-query total time from timings
        q_timings = s.get("timings") or []
        total_entry = next((t for t in q_timings if t["step"] == "TOTAL"), None)
        time_str = f" [{total_entry['duration']:.1f}s]" if total_entry else ""
        print(f"  Q{qid}: {status} — {detail}{diff_info}{time_str}")

    total = passed + failed
    total_cost = sum(
        (s.get("agent") or {}).get("cost_usd") or 0 for s in all_summaries
    )
    total_input = sum(
        (s.get("agent") or {}).get("input_tokens") or 0 for s in all_summaries
    )
    total_output = sum(
        (s.get("agent") or {}).get("output_tokens") or 0 for s in all_summaries
    )
    total_time = sum(
        next((t["duration"] for t in (s.get("timings") or []) if t["step"] == "TOTAL"), 0)
        for s in all_summaries
    )
    print("-" * 60)
    print(f"  {passed}/{total} passed")
    print(f"  Tokens: {total_input:,} input, {total_output:,} output")
    print(f"  Cost: ${total_cost:.4f}")
    print(f"  Total time: {total_time:.1f}s")
    if failed == 0:
        print("  ALL QUERIES VERIFIED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
