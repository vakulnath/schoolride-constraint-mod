"""
Pipeline dispatcher — routes to the appropriate backend.

Architecture:
    Main Agent  (src/agents/main_agent.py)          — explores code, finds injection points
    GA          (src/agents/generation_agent.py)    — generates code edit from focused context
    JA          (src/agents/judgment_agent.py)      — validates the edit pre-execution
    EAA         (src/agents/error_analysis_agent.py)— diagnoses test failures, tool-using
    RA          (src/agents/revision_agent.py)      — corrects edits based on feedback

Backend is selected by environment variable:
    USE_CLAUDE_CLI=1   → Claude CLI backend  (src/backends/claude_cli.py)
    USE_PROMPT_TOOLS=1 → Amplify prompt-tools backend (src/backends/prompt_tools.py)
    USE_AMPLIFY=1      → Legacy alias for USE_PROMPT_TOOLS
    USE_GROQ=1         → Groq via ADK LiteLLM  (src/backends/native.py)
    default            → Gemini via ADK  (src/backends/native.py)
"""

from __future__ import annotations

import difflib
import json
import os
import re
from typing import Any, Dict, List, Optional

from src.mcp.serena import SerenaClient


# ---------------------------------------------------------------------------
# Shared helpers (used by multiple backends)
# ---------------------------------------------------------------------------

def _parse_ja_verdict(text: str):
    """Parse JA verdict in either format:
        right: True/False + jud: explanation  (AFL format)
        choice: N/None + jud: explanation     (multi-candidate format)

    Returns (right: bool, jud: str).
    Falls back to scanning for APPROVED: TRUE if no structured format found.
    """
    right = None
    choice = None
    jud = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("right:"):
            val = line.split(":", 1)[1].strip().lower()
            right = val == "true"
        elif line.lower().startswith("choice:"):
            val = line.split(":", 1)[1].strip().lower()
            if val == "none":
                choice = None
            else:
                try:
                    choice = int(val)  # 1-based
                except ValueError:
                    choice = None
        elif line.lower().startswith("jud:"):
            jud = line.split(":", 1)[1].strip()
    # Resolve right from choice if not set directly
    if right is None and choice is not None:
        right = True  # a valid choice means approved
    elif right is None and choice is None and jud:
        right = False  # choice: None with feedback means rejected
    if right is None:
        # Legacy fallback
        upper = text.upper()
        right = "APPROVED: TRUE" in upper or "APPROVED:TRUE" in upper
        if not jud:
            jud = text.strip()
    return right, jud


def _parse_injection_points(text: str) -> List[Dict[str, Any]]:
    """Parse injection_points JSON from Main Agent output."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{[^{}]*\"injection_points\"[^{}]*\[.*?\]\s*\})", text, re.DOTALL)
    if not m:
        m = re.search(r"\"injection_points\"\s*:\s*(\[.*?\])", text, re.DOTALL)
        if m:
            try:
                arr = json.loads(m.group(1))
                return arr if isinstance(arr, list) else []
            except json.JSONDecodeError:
                pass
        return []
    try:
        data = json.loads(m.group(1))
        return data.get("injection_points", []) if isinstance(data, dict) else []
    except json.JSONDecodeError:
        try:
            arr_m = re.search(r'"injection_points"\s*:\s*(\[.*\])', m.group(1), re.DOTALL)
            if arr_m:
                return json.loads(arr_m.group(1))
        except Exception:
            pass
        return []


def _make_diff(e: dict) -> str:
    """Build a unified diff string from an edit dict (includes extra_edits blocks)."""
    def _one(edit: dict) -> str:
        d = edit.get("diff", "")
        if d:
            return d
        old, new = edit.get("old_text", ""), edit.get("new_text", "")
        if old and new:
            return "".join(difflib.unified_diff(
                old.splitlines(keepends=True), new.splitlines(keepends=True),
                fromfile=f"a/{edit.get('path', '?')}", tofile=f"b/{edit.get('path', '?')}", n=3,
            ))
        return ""

    parts = [_one(e)] + [_one(x) for x in e.get("extra_edits", [])]
    return "\n".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

async def run_pipeline(
    constraint: str,
    query_id: int,
    solver_type: str = "insertion",
    temp_code_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    shared_serena: Optional[SerenaClient] = None,
) -> Dict[str, Any]:
    """Run the full pipeline for one constraint query.

    Backend is selected by environment variable:
      USE_CLAUDE_CLI=1   → Claude CLI (claude -p)
      USE_PROMPT_TOOLS=1 → Amplify prompt-tools backend
      USE_AMPLIFY=1      → Legacy alias for USE_PROMPT_TOOLS
      USE_GROQ=1         → Groq via ADK native tools
      default            → Gemini via ADK native tools
    """
    if os.environ.get("USE_CLAUDE_CLI") == "1":
        from src.backends.claude_cli import run_claude_cli_pipeline
        return await run_claude_cli_pipeline(
            constraint=constraint,
            query_id=query_id,
            solver_type=solver_type,
            temp_code_dir=temp_code_dir,
            logs_dir=logs_dir,
            shared_serena=shared_serena,
        )

    if os.environ.get("USE_PROMPT_TOOLS") == "1" or os.environ.get("USE_AMPLIFY") == "1":
        from src.backends.prompt_tools import run_prompt_tools_pipeline
        return await run_prompt_tools_pipeline(
            constraint=constraint,
            query_id=query_id,
            solver_type=solver_type,
            temp_code_dir=temp_code_dir,
            logs_dir=logs_dir,
            shared_serena=shared_serena,
        )

    # DSL pipeline: compile constraint to verified snippet, then locate + inject
    if os.environ.get("USE_DSL") == "1":
        from src.backends.native import run_dsl_pipeline
        return await run_dsl_pipeline(
            constraint=constraint,
            query_id=query_id,
            solver_type=solver_type,
            temp_code_dir=temp_code_dir,
            logs_dir=logs_dir,
            shared_serena=shared_serena,
        )

    # Default: native ADK (Gemini or Groq)
    from src.backends.native import run_native_pipeline
    return await run_native_pipeline(
        constraint=constraint,
        query_id=query_id,
        solver_type=solver_type,
        temp_code_dir=temp_code_dir,
        logs_dir=logs_dir,
        shared_serena=shared_serena,
    )
