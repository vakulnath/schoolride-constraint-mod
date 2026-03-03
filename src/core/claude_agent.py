"""
Claude Code CLI agent for constraint implementation.

Calls the `claude` CLI in non-interactive mode (-p) with native tools
(Read, Edit, Grep, Glob). No custom XML parsing or tool routing —
Claude Code handles everything natively.

Same return interface as run_contract_agent() for easy pipeline swapping.
"""

import difflib
import hashlib
import importlib.util
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from src.configs.config_parser import (
    CONSTRAINTS_FILE, POLICY_DIR, TEMP_CODE_DIR, OUTPUT_DIR, LOGS_DIR, LLM_EDIT_LOG
)


def _load_constraints() -> str:
    """Load the problem constraints from configuration."""
    if CONSTRAINTS_FILE.exists():
        with open(CONSTRAINTS_FILE, "r") as f:
            return f.read()
    return "(Constraints not available)"


def _load_schema(context_root: str) -> str:
    """Load the domain schema (insertion.py) for context."""
    # Check context root directly
    schema_path = os.path.join(context_root, "insertion.py")
    if not os.path.exists(schema_path):
        # Try schemas subdirectory
        for root, dirs, files in os.walk(context_root):
            if "insertion.py" in files and "schemas" in root:
                schema_path = os.path.join(root, "insertion.py")
                break

    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            return f.read()

    # Try importlib as fallback
    try:
        spec = importlib.util.find_spec("school_bus_routing.schemas.insertion")
        if spec and spec.origin:
            with open(spec.origin, "r") as f:
                return f.read()
    except Exception:
        pass

    return "(Schema not available)"


def _snapshot_files(directory: str) -> Dict[str, str]:
    """Snapshot all .py files in a directory with their MD5 hashes."""
    snapshots = {}
    if not os.path.isdir(directory):
        return snapshots
    for fname in os.listdir(directory):
        if fname.endswith(".py"):
            fpath = os.path.join(directory, fname)
            with open(fpath, "rb") as f:
                snapshots[fname] = hashlib.md5(f.read()).hexdigest()
    return snapshots


def _detect_edits(
    policy_dir: str,
    original_dir: str,
    before_snapshots: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """
    Compare current files against pre-run snapshots to detect edits.

    Returns edit info dict or None if no changes detected.
    """
    if not os.path.isdir(policy_dir):
        return None

    for fname, old_hash in before_snapshots.items():
        fpath = os.path.join(policy_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            new_hash = hashlib.md5(f.read()).hexdigest()
        if new_hash != old_hash:
            # File was modified — generate diff
            orig_path = os.path.join(original_dir, fname)
            if os.path.exists(orig_path):
                with open(orig_path, "r") as f:
                    orig_lines = f.readlines()
                with open(fpath, "r") as f:
                    new_lines = f.readlines()
                diff = "".join(difflib.unified_diff(
                    orig_lines, new_lines,
                    fromfile=f"a/{fname}", tofile=f"b/{fname}",
                ))
            else:
                diff = "(original not available for diff)"

            return {
                "path": fname,
                "success": True,
                "diff": diff,
            }

    return None


def _log(logs_dir: Optional[str], filename: str, content: str):
    """Append to a log file."""
    if logs_dir:
        with open(os.path.join(logs_dir, filename), "a") as f:
            f.write(content + "\n")


def run_claude_agent(
    constraint: str,
    context_root: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    temp_code_dir: Optional[str] = None,
    max_turns: int = 10,
) -> Dict[str, Any]:
    """
    Run Claude Code CLI to implement a constraint.

    Same interface as run_contract_agent() for easy swapping in pipeline.

    Args:
        constraint: The constraint to implement (natural language)
        context_root: Path to solvers source (for schema loading and diffs)
        env: Environment variables (not used directly — CLI uses its own auth)
        logs_dir: Directory for log files
        temp_code_dir: Directory with policy/ subdirectory for editing
        max_turns: Max agentic turns for Claude Code

    Returns:
        Dict with: success, edit, thoughts, iterations, cost_usd
    """
    constraint_content = _load_constraints()
    schema_content = _load_schema(context_root)

    # Working directory: temp_code_dir contains policy/ with files to edit
    work_dir = temp_code_dir or context_root
    policy_dir = os.path.join(work_dir, POLICY_DIR)

    # Snapshot files before running so we can detect changes
    before = _snapshot_files(policy_dir)

    # Build the system prompt (appended to Claude's default system prompt)
    append_prompt = f"""## PROBLEM CONSTRAINTS (must not be violated by your edit)

{constraint_content}

## DOMAIN SCHEMA

{schema_content}

## RULES

- Search the policy/ directory for code relevant to the constraint
- Read the function to understand what variables are in scope
- Make a MINIMAL edit — a simple if/continue or if/return None
- Before editing, verify that variables your constraint needs are in scope
- If a variable is not in scope, find a different function where it IS in scope
- USE ONLY simple Python control flow: continue, return False, return None
- Your new constraint must NOT violate existing constraints listed above
- Keep edits small (3-10 lines). Do NOT duplicate existing logic."""

    prompt = f"Implement this constraint by editing the code in the policy/ directory: {constraint}"

    # Build CLI command
    cmd = [
        "claude",
        "-p", prompt,
        "--append-system-prompt", append_prompt,
        "--output-format", "stream-json",
        "--tools", "Read,Edit,Grep,Glob",
        "--permission-mode", "acceptEdits",
        "--model", "sonnet",
    ]

    _log(logs_dir, AGENT_LOG, f"=== Claude Code CLI agent ===")
    _log(logs_dir, AGENT_LOG, f"Constraint: {constraint}")
    _log(logs_dir, AGENT_LOG, f"Working dir: {work_dir}")
    _log(logs_dir, AGENT_LOG, f"Policy dir: {policy_dir}")
    _log(logs_dir, AGENT_LOG, f"Files: {list(before.keys())}")
    _log(logs_dir, AGENT_LOG, f"Command: {json.dumps(cmd)}")

    # Run Claude Code CLI
    try:
        result = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes
        )
    except subprocess.TimeoutExpired:
        _log(logs_dir, AGENT_LOG, "TIMEOUT: Claude Code took >5 minutes")
        return {
            "success": False,
            "error": "Claude Code timed out (5min)",
            "edit": None,
            "thoughts": [],
            "iterations": 0,
        }
    except FileNotFoundError:
        _log(logs_dir, AGENT_LOG, "ERROR: claude CLI not found")
        return {
            "success": False,
            "error": "Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code",
            "edit": None,
            "thoughts": [],
            "iterations": 0,
        }

    # Parse stream-json output (one JSON object per line)
    raw_stdout = result.stdout or ""
    raw_stderr = result.stderr or ""

    messages: List[Dict[str, Any]] = []
    final_result: Optional[Dict[str, Any]] = None
    for line in raw_stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            messages.append(msg)
            if msg.get("type") == "result":
                final_result = msg
        except json.JSONDecodeError:
            continue

    # Log full transcript and raw output
    if logs_dir:
        with open(os.path.join(logs_dir, "claude_stream.jsonl"), "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
        with open(os.path.join(logs_dir, "claude_stderr.txt"), "w") as f:
            f.write(raw_stderr)
        # Write human-readable thinking log
        with open(os.path.join(logs_dir, "claude_thinking.log"), "w") as f:
            for msg in messages:
                mtype = msg.get("type", "")
                if mtype == "assistant":
                    content = msg.get("message", {}).get("content", [])
                    for block in content:
                        if block.get("type") == "text":
                            f.write(f"[THINKING] {block['text']}\n\n")
                        elif block.get("type") == "tool_use":
                            f.write(f"[TOOL_CALL] {block.get('name', '?')}({json.dumps(block.get('input', {}), indent=2)[:500]})\n\n")
                elif mtype == "tool_result":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        preview = content[:300]
                    elif isinstance(content, list):
                        preview = str(content)[:300]
                    else:
                        preview = str(content)[:300]
                    f.write(f"[TOOL_RESULT] {preview}\n\n")
                elif mtype == "result":
                    f.write(f"[RESULT] {msg.get('result', '')[:500]}\n")

    if not final_result:
        _log(logs_dir, AGENT_LOG, f"No result message in stream output ({len(messages)} messages parsed)")
        return {
            "success": False,
            "error": f"No result in CLI stream output (exit code {result.returncode})",
            "edit": None,
            "thoughts": [raw_stderr[:500]],
            "iterations": 0,
        }

    response_text = final_result.get("result", "")
    cost = final_result.get("total_cost_usd", 0)
    num_turns = final_result.get("num_turns", 0)
    is_error = final_result.get("is_error", False)

    _log(logs_dir, AGENT_LOG, f"Claude Code completed: turns={num_turns}, cost=${cost:.4f}, error={is_error}")
    _log(logs_dir, AGENT_LOG, f"Response:\n{response_text[:1000]}")

    if is_error:
        _log(logs_dir, AGENT_LOG, f"Claude Code reported error: {response_text[:500]}")
        return {
            "success": False,
            "error": response_text[:500],
            "edit": None,
            "thoughts": [response_text],
            "iterations": num_turns,
            "cost_usd": cost,
        }

    # Detect edits by comparing file hashes
    original_policy_dir = os.path.join(context_root, POLICY_DIR)
    edit_info = _detect_edits(policy_dir, original_policy_dir, before)

    if edit_info:
        _log(logs_dir, AGENT_LOG, f"Edit detected: {edit_info['path']}")
        if logs_dir and edit_info.get("diff"):
            with open(os.path.join(logs_dir, "edit.diff"), "w") as f:
                f.write(edit_info["diff"])
        return {
            "success": True,
            "edit": edit_info,
            "thoughts": [response_text],
            "iterations": num_turns,
            "cost_usd": cost,
        }

    _log(logs_dir, AGENT_LOG, "No file changes detected after Claude Code run")
    return {
        "success": False,
        "error": "No edits detected in policy/ files",
        "edit": None,
        "thoughts": [response_text],
        "iterations": num_turns,
        "cost_usd": cost,
    }
