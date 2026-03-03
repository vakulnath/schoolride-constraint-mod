"""
Revision Agent (RA) — corrects edits based on JA or EAA feedback.

Has the same tools as the Generation Agent (search_code, read_function,
apply_edit).  The key difference is the enriched prompt that includes the
previous diff and structured feedback so the model doesn't repeat the same
mistake.
"""

from typing import Any, Dict, Optional

from core.api_agent import run_api_agent


def run_revision_agent(
    constraint: str,
    feedback: str,
    feedback_source: str,
    previous_diff: str,
    context_root: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    temp_code_dir: Optional[str] = None,
    max_turns: int = 10,
    solver_type: str = "insertion",
) -> Dict[str, Any]:
    """Run the Revision Agent with enriched context.

    Parameters
    ----------
    feedback_source : str
        ``"judgment"`` when the JA rejected the edit, or
        ``"error_analysis"`` when the EAA diagnosed a runtime failure.
    previous_diff : str
        Unified diff of the edit that was rejected / failed.
    feedback : str
        Full text from the JA or EAA.

    Returns the same shape as ``run_api_agent``.
    """
    if feedback_source == "judgment":
        enriched = (
            f"{constraint}\n\n"
            f"IMPORTANT — A previous edit was REJECTED by the review agent.\n\n"
            f"Previous edit diff:\n```diff\n{previous_diff}\n```\n\n"
            f"Review feedback:\n{feedback}\n\n"
            f"Apply a CORRECTED edit that addresses every issue listed above.\n"
            f"Do NOT repeat the same mistake."
        )
    elif feedback_source == "error_analysis":
        enriched = (
            f"{constraint}\n\n"
            f"IMPORTANT — A previous edit caused a RUNTIME ERROR.\n\n"
            f"Previous edit diff:\n```diff\n{previous_diff}\n```\n\n"
            f"Error diagnosis:\n{feedback}\n\n"
            f"Apply a CORRECTED edit that avoids this error.\n"
            f"If the diagnosis suggests a different function or approach, follow it."
        )
    elif feedback_source == "constraint_violation":
        enriched = (
            f"{constraint}\n\n"
            f"IMPORTANT — A previous edit PASSED the runtime test but FAILED "
            f"constraint verification in the output schedule.\n\n"
            f"Previous edit diff:\n```diff\n{previous_diff}\n```\n\n"
            f"Constraint violation analysis:\n{feedback}\n\n"
            f"Apply a CORRECTED edit that ensures the constraint is properly enforced.\n"
            f"The code ran without errors, but the resulting schedule did not satisfy "
            f"the constraint. The edit may need a stronger enforcement mechanism."
        )
    else:
        enriched = constraint

    return run_api_agent(
        constraint=enriched,
        context_root=context_root,
        env=env,
        logs_dir=logs_dir,
        temp_code_dir=temp_code_dir,
        max_turns=max_turns,
        solver_type=solver_type,
    )
