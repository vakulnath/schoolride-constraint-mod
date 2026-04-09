"""
Claude CLI backend — runs `claude -p` (non-interactive) to generate a code edit.

This backend does NOT use the multi-agent pipeline. Instead, it invokes the
Claude CLI with the constraint description and lets Claude explore the policy
codebase using its built-in tools (Read, Glob, Grep, Bash) and produce a
single edit in XML format.

Public entry point:
    run_claude_cli_pipeline(constraint, query_id, solver_type, ...) -> dict
"""

from __future__ import annotations

import asyncio
import difflib
import json as _json
import os
import re
import shutil
import subprocess
import traceback
from typing import Any, Dict, List, Optional

from src.configs import config_parser as cfg
from src.utils.code_tools import (
    copy_policy_to_dir,
    load_constraints as _load_constraints,
    load_schema as _load_schema,
    parse_edit_suggestion as _parse_edit_suggestion,
    write_file as _write_file,
    apply_edit_to_temp_dir as _apply_edit_to_temp_dir,
)
from src.utils.test_runner import run_toy_test as _run_toy_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MAX_ATTEMPTS = 5
_CLAUDE_MODEL = os.environ.get("CLAUDE_CLI_MODEL", "claude-sonnet-4-6")


def _make_diff(e: dict) -> str:
    d = e.get("diff", "")
    if d:
        return d
    old, new = e.get("old_text", ""), e.get("new_text", "")
    if old and new:
        return "".join(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile=f"a/{e.get('path', '?')}", tofile=f"b/{e.get('path', '?')}", n=3,
        ))
    return ""


_SOLVER_FILES = {
    "insertion": ["insert_heuristic.py", "insertion_policy.py", "pruned_insertion.py", "pruning_candidates.py"],
    "hexaly":    ["hexaly_planner_modular.py", "hexaly_helper.py"],
}


def _build_prompt(
    constraint: str,
    schema_content: str,
    constraint_content: str,
    solver_type: str = "insertion",
    error_feedback: Optional[str] = None,
) -> str:
    """Build the prompt sent to `claude -p`."""
    solver_files = _SOLVER_FILES.get(solver_type, _SOLVER_FILES["insertion"])
    files_str = ", ".join(f"`{f}`" for f in solver_files)
    lines = [
        f"You are a code editing assistant working on a Python school-bus routing solver ({solver_type} variant).",
        "",
        "## Your task",
        "Enforce the following routing constraint by editing the policy source files:",
        "",
        "<constraint>",
        constraint,
        "</constraint>",
        "",
        f"## Solver: {solver_type}",
        f"You MUST only edit files relevant to the {solver_type} solver: {files_str}.",
        "Do NOT edit insertion files when working on hexaly, and vice versa.",
        "",
        "## Solver API schema",
        "<schema>",
        schema_content,
        "</schema>",
        "",
        "## Existing constraints (for context)",
        "<existing_constraints>",
        constraint_content,
        "</existing_constraints>",
        "",
        "## Instructions",
        f"1. Use Read, Glob, and Grep to explore the {solver_type} policy files listed above.",
        "2. Find the function(s) where this constraint should be enforced.",
        "3. Make the MINIMAL edit required — do not refactor unrelated code.",
        "4. Output your edit using the XML format below.",
        "5. Use EXACT text from the file for <old_text> so the patch applies cleanly.",
        "",
        "## Output format",
        "Output ONE edit block (or multiple if needed for different functions):",
        "",
        "<edit>",
        "  <relative_path>policy/filename.py</relative_path>",
        "  <old_text>exact lines to replace (copy verbatim from file)</old_text>",
        "  <new_text>replacement lines with constraint enforced</new_text>",
        "  <explanation>one sentence describing the change</explanation>",
        "</edit>",
        "",
        "Important: Do NOT include anything outside the <edit> tags in your final answer.",
        "If you need multiple edits (e.g. for two functions), output multiple <edit> blocks.",
    ]

    if error_feedback:
        lines += [
            "",
            "## Previous attempt failed with this error",
            "<error>",
            error_feedback,
            "</error>",
            "",
            "Please diagnose the error and produce a corrected edit.",
        ]

    return "\n".join(lines)


async def _run_claude_cli(
    prompt: str,
    policy_dir: str,
    timeout: int = 300,
    logs_dir: Optional[str] = None,
    attempt: int = 1,
) -> str:
    """Run `claude -p <prompt>` with stream-json output, log the decision trace, and return the result text."""
    model = os.environ.get("CLAUDE_CLI_MODEL", _CLAUDE_MODEL)
    proc = await asyncio.create_subprocess_exec(
        "claude", "-p", prompt,
        "--allowedTools", "Read,Glob,Grep",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
        "--model", model,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=policy_dir,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        raise RuntimeError(f"claude CLI timed out after {timeout}s")

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        raise RuntimeError(
            f"claude CLI exited with code {proc.returncode}\n"
            f"stderr: {stderr_text[:500]}"
        )

    # Parse JSONL stream and extract result + decision trace
    result_text = ""
    trace_lines: List[str] = []
    step = 0

    for raw_line in stdout_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = _json.loads(raw_line)
        except _json.JSONDecodeError:
            continue

        etype = event.get("type")

        if etype == "assistant":
            for block in event.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "text":
                    text = block.get("text", "").strip()
                    if text:
                        step += 1
                        # Keep first 400 chars of each reasoning block
                        trace_lines.append(f"[{step}] thinking: {text[:400]}")
                elif btype == "tool_use":
                    step += 1
                    tool_name = block.get("name", "?")
                    tool_input = block.get("input", {})
                    trace_lines.append(f"[{step}] {tool_name}: {_json.dumps(tool_input)}")

        elif etype == "tool":
            content = str(event.get("content", ""))
            # Truncate large file reads in the trace
            preview = content[:600].replace("\n", "↵") if content else ""
            trace_lines.append(f"    → {preview}")

        elif etype == "result":
            result_text = event.get("result", "")
            cost = event.get("cost_usd")
            subtype = event.get("subtype", "")
            trace_lines.append(f"\n=== result ({subtype}, ${cost}) ===\n{result_text[:300]}")

    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        # Raw JSONL stream
        with open(os.path.join(logs_dir, f"claude_stream_{attempt}.jsonl"), "w") as f:
            f.write(stdout_text)
        # Human-readable decision trace
        with open(os.path.join(logs_dir, f"claude_trace_{attempt}.txt"), "w") as f:
            f.write("\n".join(trace_lines))

    return result_text or stdout_text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_claude_cli_pipeline(
    constraint: str,
    query_id: int,
    solver_type: str = "insertion",
    temp_code_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    shared_serena=None,  # unused — Claude CLI has its own tools
) -> Dict[str, Any]:
    """Run a single-agent pipeline via the `claude -p` CLI.

    Claude explores the policy files autonomously using Read/Glob/Grep and
    outputs an edit in XML format. On test failure the error is fed back and
    Claude retries (up to MAX_ATTEMPTS times).
    """
    logs_dir = logs_dir or f"/tmp/claude_cli_logs_{query_id}"
    os.makedirs(logs_dir, exist_ok=True)
    temp_code_dir = temp_code_dir or f"/tmp/claude_cli_temp_{query_id}"
    os.makedirs(temp_code_dir, exist_ok=True)

    def _log(msg: str):
        print(msg)
        with open(os.path.join(logs_dir, "pipeline.log"), "a") as f:
            f.write(msg + "\n")

    _log(f"=== Claude CLI Pipeline: {constraint[:80]} ({solver_type}) ===")

    env = os.environ.copy()

    # Load shared context
    constraint_content = _load_constraints()
    schema_content = _load_schema(".", solver_type)

    # Resolve policy directory
    policy_dir = (
        str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT
        else os.path.join(".", "policy")
    )

    # Set up temp policy copy
    policy_temp = os.path.join(temp_code_dir, "policy")
    if not os.path.isdir(policy_temp):
        import policy as _pol
        _pol_file = _pol.__file__
        assert _pol_file is not None, "Cannot locate policy package"
        copy_policy_to_dir(os.path.dirname(_pol_file), policy_temp)
        _log(f"  Copied policy to {policy_temp}")

    winner_diff = ""
    edit = None
    error_feedback: Optional[str] = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        _log(f"\n--- Attempt {attempt}/{MAX_ATTEMPTS} ---")

        prompt = _build_prompt(
            constraint=constraint,
            schema_content=schema_content,
            constraint_content=constraint_content,
            solver_type=solver_type,
            error_feedback=error_feedback,
        )

        # Save prompt for debugging
        _write_file(logs_dir, f"prompt_{attempt}.txt", prompt)

        try:
            output_text = await _run_claude_cli(
                prompt=prompt,
                policy_dir=policy_temp,
                timeout=300,
                logs_dir=logs_dir,
                attempt=attempt,
            )
        except Exception as e:
            _log(f"  Claude CLI error: {type(e).__name__}: {e}")
            _log(traceback.format_exc())
            error_feedback = f"{type(e).__name__}: {e}"
            continue

        _write_file(logs_dir, f"claude_output_{attempt}.txt", output_text)
        _log(f"  Claude output: {output_text[:200]}")

        # Parse the XML edit from the output
        candidate = _parse_edit_suggestion(output_text, logs_dir)
        if not candidate:
            _log(f"  No valid edit found in output — retrying")
            error_feedback = (
                "Your previous response did not contain a valid <edit> block. "
                "Please output the edit using the exact XML format specified."
            )
            continue

        winner_diff = _make_diff(candidate)
        edit = candidate
        _log(f"  Parsed edit for {edit.get('path', '?')}")

        # Apply edit to temp dir
        applied = _apply_edit_to_temp_dir(edit, temp_code_dir, env, logs_dir)
        if not applied:
            _log("  Apply FAILED — old_text may not match; retrying")
            error_feedback = (
                "The edit could not be applied because <old_text> did not match the file. "
                "Please use Read to get the EXACT current content of the file and copy it verbatim."
            )
            # Reset temp dir to clean copy for next attempt
            shutil.rmtree(policy_temp, ignore_errors=True)
            import policy as _pol
            _pol_file = _pol.__file__
            assert _pol_file is not None
            copy_policy_to_dir(os.path.dirname(_pol_file), policy_temp)
            continue

        _log(f"  Applied edit to {edit.get('path', '?')}")

        # Run toy test
        result = _run_toy_test(query_id, solver_type, temp_code_dir)

        if result["passed"]:
            _log(f"  Attempt {attempt}: PASS")
            return {
                "success": True,
                "diff": winner_diff,
                "edit": edit,
                "injection_points": [],
            }

        test_output = result.get("output", "")
        _write_file(logs_dir, f"test_output_{attempt}.txt", test_output)

        # Extract a concise error summary for the retry prompt
        error_lines = [l.strip() for l in test_output.split("\n")
                       if any(kw in l for kw in ["FAILED", "assert", "AssertionError", "Error"])]
        error_summary = "\n".join(error_lines[:5]) if error_lines else test_output[-2000:][:500]

        _log(f"  Attempt {attempt}: FAIL")
        for line in error_lines[:3]:
            _log(f"    {line}")

        error_feedback = (
            f"The toy test failed after your edit was applied.\n"
            f"Test output (excerpt):\n{error_summary}\n\n"
            f"Previous edit applied to {edit.get('path', '?')}:\n"
            f"```diff\n{winner_diff[:1000]}\n```\n"
        )

        # Reset temp dir for next attempt so we start from a clean state
        shutil.rmtree(policy_temp, ignore_errors=True)
        import policy as _pol
        _pol_file = _pol.__file__
        assert _pol_file is not None
        copy_policy_to_dir(os.path.dirname(_pol_file), policy_temp)

    _log(f"  All {MAX_ATTEMPTS} attempts exhausted")
    return {
        "success": False,
        "diff": winner_diff,
        "edit": edit,
        "injection_points": [],
    }
