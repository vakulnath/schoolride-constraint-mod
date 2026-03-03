"""
Judgment Agent (JA) — validates a proposed edit before execution.

Tool-free, single-turn API call.  Reviews the diff against the constraint
and the function source (with docstring contracts) to catch obvious mistakes
*before* burning 30 s on a toy-test run.
"""

from typing import Any, Dict, Optional

from core.api_agent import api_single_turn, _log


_JA_SYSTEM_PROMPT = """\
You are a strict code reviewer for school-bus routing policy edits.

A Generation Agent has proposed a minimal Python code edit (shown as a
unified diff) to implement a routing constraint.  Your job is to judge
whether the edit is ready for execution.

You will be given the original function source code.  The function's
docstring contains Preconditions, Postconditions, and Invariants that
describe the hard constraints the code must satisfy.  Use these to
evaluate safety.

Evaluate the edit on FIVE criteria:

1) CORRECTNESS  — Does the diff implement the stated constraint?
2) SAFETY       — Does it preserve all existing logic?  Would it violate
                  any of the preconditions, postconditions, or invariants
                  documented in the function docstring?
3) SYNTAX       — Is the Python syntactically correct?  Is indentation
                  consistent with the surrounding code?
4) SCOPE        — Are all referenced variables actually in scope at the
                  edit location?  Use the original function source
                  provided to verify — do NOT guess about scope.
5) MINIMALITY   — Is the change minimal (guard clause / condition), not a
                  large rewrite?

IMPORTANT: Your response MUST start with exactly these three lines:

APPROVED: TRUE   (or FALSE)
ISSUES: <one-line summary of problems, or "None">
SUGGESTIONS: <concrete fix instructions if FALSE, or "None">

Do NOT include any other text before the verdict lines above.
"""


def run_judgment_agent(
    constraint: str,
    diff: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    base_dir: Optional[str] = None,
    source_context: Optional[str] = None,
    solver_type: str = "insertion",
) -> Dict[str, Any]:
    """Review *diff* against *constraint* and the function source.

    Args:
        source_context: Full source of the edited function (includes
            docstring with Preconditions/Postconditions/Invariants).
            JA uses this to verify variable scope and safety.

    Returns::

        {
            "approved": bool,
            "feedback": str,          # full JA response text
            "cost_usd": float,
            "input_tokens": int,
            "output_tokens": int,
        }
    """
    source_section = ""
    if source_context:
        source_section = (
            "## Original function source (pre-edit — includes contract in docstring)\n"
            f"```python\n{source_context}\n```\n\n"
        )

    user_prompt = (
        "## Constraint to implement\n"
        f"{constraint}\n\n"
        f"{source_section}"
        "## Proposed edit (unified diff)\n"
        f"```diff\n{diff}\n```\n\n"
        "Judge this edit.  Is it correct, safe, and ready for execution?"
    )

    result = api_single_turn(
        user_prompt=user_prompt,
        system_prompt=_JA_SYSTEM_PROMPT,
        env=env,
        logs_dir=logs_dir,
        log_prefix="judgment_agent",
        max_tokens=2048,
    )

    text = (result.get("text") or "").strip()

    # Retry once if Gemini returned an empty response (transient API issue)
    if not text:
        _log(logs_dir, "judgment_agent.log",
             f"Empty response (finishReason={result.get('finish_reason', '?')}) — retrying once")
        result = api_single_turn(
            user_prompt=user_prompt,
            system_prompt=_JA_SYSTEM_PROMPT,
            env=env,
            logs_dir=logs_dir,
            log_prefix="judgment_agent",
            max_tokens=2048,
            temperature=0.2,
        )
        text = (result.get("text") or "").strip()

    # If still empty after retry, default to APPROVED — let the toy test
    # be the real validator rather than silently rejecting a good edit.
    if not text:
        _log(logs_dir, "judgment_agent.log",
             "Still empty after retry — defaulting to APPROVED (toy test will validate)")
        approved = True
    else:
        upper = text.upper()
        approved = "APPROVED: TRUE" in upper or "APPROVED:TRUE" in upper

    _log(logs_dir, "judgment_agent.log", f"Approved: {approved}")

    return {
        "approved": approved,
        "feedback": text,
        "cost_usd": result.get("cost_usd", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
    }


_PANEL_JA_SYSTEM_PROMPT = """\
You are a code review judge on a panel of three judges evaluating code solutions.

You will receive THREE proposed code edits (diffs) for the same routing constraint.
Each edit was produced by a different LLM model. Your job is to review ALL THREE
independently and vote for the BEST one.

Evaluate each diff on: CORRECTNESS, SAFETY, SYNTAX, SCOPE, MINIMALITY.
Assign each a score from 0-10.

VOTE for the highest-scoring diff that is APPROVED. If no diffs are approvable
(all have critical issues), VOTE: none.

Respond with EXACTLY this format:

VOTE: <ga_flash | ga_pro | ga_25pro | none>
SCORES:
  ga_flash:  <score>/10 — <one-line reason>
  ga_pro:    <score>/10 — <one-line reason>
  ga_25pro:  <score>/10 — <one-line reason>
WINNER_REASONING: <explain why winner is best, or why all were rejected>

Your response MUST start with these exact lines. No preamble.
"""


def run_panel_judgment_agent(
    constraint: str,
    diffs: Dict[str, str],
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    judge_id: int = 1,
    solver_type: str = "insertion",
) -> Dict[str, Any]:
    """Panel judge: review ALL 3 GA diffs, vote for best.

    Args:
        constraint: The constraint statement
        diffs: Dict with keys "ga_flash", "ga_pro", "ga_25pro", values are unified diffs
        judge_id: 1, 2, or 3 (for logging)

    Returns:
        {
            "vote": str,           # "ga_flash" | "ga_pro" | "ga_25pro" | "none"
            "scores": dict,        # {"ga_flash": score, ...}
            "winner_reasoning": str,
            "raw_feedback": str,   # full JA response
            "cost_usd": float,
            "input_tokens": int,
            "output_tokens": int,
        }
    """
    # Build diffs section (all 3 side-by-side)
    diffs_section = "## Three proposed edits (unified diffs)\n\n"
    for model_name, diff_text in diffs.items():
        diffs_section += f"### {model_name}\n```diff\n{diff_text}\n```\n\n"

    user_prompt = (
        "## Constraint to implement\n"
        f"{constraint}\n\n"
        f"{diffs_section}"
        "Review all three diffs independently. Vote for the best one."
    )

    result = api_single_turn(
        user_prompt=user_prompt,
        system_prompt=_PANEL_JA_SYSTEM_PROMPT,
        env=env,
        logs_dir=logs_dir,
        log_prefix=f"panel_judge_{judge_id}",
        max_tokens=2048,
    )

    text = (result.get("text") or "").strip()

    # Retry once if empty
    if not text:
        result = api_single_turn(
            user_prompt=user_prompt,
            system_prompt=_PANEL_JA_SYSTEM_PROMPT,
            env=env,
            logs_dir=logs_dir,
            log_prefix=f"panel_judge_{judge_id}",
            max_tokens=2048,
            temperature=0.2,
        )
        text = (result.get("text") or "").strip()

    # Parse response: extract vote and scores
    vote = "none"
    scores = {}
    winner_reasoning = ""

    lines = text.split("\n")
    for i, line in enumerate(lines):
        line_upper = line.upper()
        if line_upper.startswith("VOTE:"):
            vote_part = line.split(":", 1)[1].strip().lower()
            if vote_part in ["ga_flash", "ga_pro", "ga_25pro", "none"]:
                vote = vote_part
        elif line_upper.startswith("WINNER_REASONING:"):
            winner_reasoning = line.split(":", 1)[1].strip()
        elif any(f"{m}:" in line for m in ["ga_flash:", "ga_pro:", "ga_25pro:"]):
            for model in ["ga_flash", "ga_pro", "ga_25pro"]:
                if f"{model}:" in line:
                    try:
                        score_str = line.split(":")[1].split("/")[0].strip()
                        scores[model] = int(score_str)
                    except (ValueError, IndexError):
                        scores[model] = 0

    _log(logs_dir, f"panel_judge_{judge_id}.log", f"Vote: {vote}")

    return {
        "vote": vote,
        "scores": scores,
        "winner_reasoning": winner_reasoning,
        "raw_feedback": text,
        "cost_usd": result.get("cost_usd", 0),
        "input_tokens": result.get("input_tokens", 0),
        "output_tokens": result.get("output_tokens", 0),
    }
