"""
Generation Agent (GA) — produces code edits via native tool-calling.

Wraps the existing run_api_agent, which uses search_code / read_function /
apply_edit tools to implement a constraint as a minimal policy-file edit.
"""

from typing import Any, Dict, List, Optional

from core.api_agent import run_api_agent


def run_generation_agent(
    constraint: str,
    context_root: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    temp_code_dir: Optional[str] = None,
    max_turns: int = 12,
    solver_type: str = "insertion",
    model_override: Optional[str] = None,
    prefetched_context: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the Generation Agent.

    Returns the same shape as ``run_api_agent``:
        success, edit, thoughts, iterations, cost_usd,
        input_tokens, output_tokens
    """
    return run_api_agent(
        constraint=constraint,
        context_root=context_root,
        env=env,
        logs_dir=logs_dir,
        temp_code_dir=temp_code_dir,
        max_turns=max_turns,
        solver_type=solver_type,
        model_override=model_override,
        prefetched_context=prefetched_context,
    )
