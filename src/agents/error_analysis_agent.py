"""
Error Analysis Agent (EAA) — diagnoses runtime errors after toy-test failure.

Tool-free, single-turn API call.  Takes the constraint, the diff that was
applied, and the traceback, and returns a structured diagnosis that the
Revision Agent can act on.
"""

from typing import Any, Dict, Optional

from core.api_agent import api_single_turn


_EAA_SYSTEM_PROMPT = """\
You are a Python debugging expert specialising in school-bus routing
policy code.

Given:
  • The routing constraint that was being implemented.
  • The code edit (unified diff) that was applied.
  • The runtime error / traceback produced by the test harness.

Your task:

1) ROOT CAUSE   — Identify the exact cause of the failure (wrong variable
                  name, wrong scope, missing import, off-by-one, violated
                  invariant, etc.).
2) EXPLANATION  — In 2-3 sentences, explain *why* the edit caused this
                  error.
3) FIX STRATEGY — Give a specific, actionable suggestion.  Should the fix
                  be in the same function?  A different function?  A
                  different approach entirely?

Be precise: reference exact variable names, function names, and — where
possible — line numbers from the traceback.  Focus on the MOST LIKELY
single root cause; do not enumerate every theoretical possibility.
"""


def run_error_analysis_agent(
    constraint: str,
    diff: str,
    traceback_text: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Diagnose a toy-test failure.

    Returns::

        {
            "diagnosis": str,         # full EAA response text
            "cost_usd": float,
            "input_tokens": int,
            "output_tokens": int,
        }
    """
    # Keep traceback to a reasonable size to avoid blowing token limits.
    traceback_trimmed = traceback_text[-3000:] if len(traceback_text) > 3000 else traceback_text

    user_prompt = (
        "## Constraint being implemented\n"
        f"{constraint}\n\n"
        "## Edit that was applied\n"
        f"```diff\n{diff}\n```\n\n"
        "## Runtime error / traceback\n"
        f"```\n{traceback_trimmed}\n```\n\n"
        "Diagnose the root cause and suggest a concrete fix."
    )

    result = api_single_turn(
        user_prompt=user_prompt,
        system_prompt=_EAA_SYSTEM_PROMPT,
        env=env,
        logs_dir=logs_dir,
        log_prefix="error_analysis_agent",
        max_tokens=2048,
    )

    return {
        "diagnosis": (result.get("text") or "").strip(),
        "cost_usd": result.get("cost_usd", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
    }
