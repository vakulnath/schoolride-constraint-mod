"""
Main Agent (MA) — LLM orchestrator for N-version GA + JA ensemble.

Coordinates 3 GA subagents (different models) and 3 JA subagents (panel judges)
to generate, evaluate, and select the best code edit for a constraint.

Enforces that all 6 subagents are called before finalizing winner selection.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agents.generation_agent import run_generation_agent
from agents.judgment_agent import run_panel_judgment_agent
from core.api_agent import _log, _snapshot_files, _api_generate, _resolve_model


# Three models for GA subagents (fast to expensive)
NVERSION_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-2.5-pro",
]

MAIN_AGENT_SYSTEM_PROMPT = """\
You are a code verification orchestrator for an N-version programming ensemble.

Your job is to:
1. Coordinate three Generation Agents (GA) with different models
2. Collect all 3 GA outputs (diffs)
3. Coordinate three Judgment Agents (JA panel judges) to review all 3 diffs
4. Select the winner based on consensus

CRITICAL: You MUST follow the strict protocol below. Do NOT deviate.

STEP 1 (single response): Call spawn_ga_subagent for ALL THREE models simultaneously:
  - gemini-3-flash-preview
  - gemini-3-pro-preview
  - gemini-2.5-pro
Do NOT proceed to Step 2 until you have made all 3 spawn_ga_subagent calls in one response.

STEP 2 (after receiving all 3 GA outputs): Call spawn_ja_subagent for EACH judge.
  You MUST call spawn_ja_subagent three times — once for each judge (judge_id 1, 2, 3).
  Each JA will review all 3 diffs and vote for the best.
Do NOT proceed to Step 3 until you have made all 3 spawn_ja_subagent calls.

STEP 3 (after receiving all 3 JA panel verdicts): Call finalize() with:
  - winner_model: the model ("ga_flash" | "ga_pro" | "ga_25pro") whose diff you select
  - reasoning: explain why this diff wins

Selection criteria (after reviewing JA panel verdicts):
  1. If all 3 JA agree on same winner → that is your choice
  2. If 2 JA agree on same winner → that is your choice
  3. If 1-1-1 tie or no JA voted (all "none") → use your own judgment:
     a) Read all 3 diffs and JA feedback
     b) Pick the diff with best CORRECTNESS, SAFETY, SCOPE, MINIMALITY
     c) Or, if all were rejected by JAs, indicate that reinvocation is needed

You MUST NOT call finalize before completing Steps 1 and 2.
"""

MAIN_AGENT_TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "spawn_ga_subagent",
                "description": "Spawn a Generation Agent with a specific model to produce a code edit. "
                               "Call this for ALL THREE models in a single response (batch them).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "model": {
                            "type": "STRING",
                            "enum": [
                                "gemini-3-flash-preview",
                                "gemini-3-pro-preview",
                                "gemini-2.5-pro",
                            ],
                            "description": "One of the three models to use for this GA",
                        }
                    },
                    "required": ["model"],
                },
            },
            {
                "name": "spawn_ja_subagent",
                "description": "Spawn a Judgment Agent judge to review ALL 3 GA diffs and vote for best. "
                               "Call this for all 3 judges in one response (batch them).",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "judge_id": {
                            "type": "STRING",
                            "enum": ["1", "2", "3"],
                            "description": "Judge panel ID (1, 2, or 3)",
                        }
                    },
                    "required": ["judge_id"],
                },
            },
            {
                "name": "finalize",
                "description": "Select the winner diff after reviewing all GA and JA results. "
                               "You MUST call spawn_ga_subagent 3 times and spawn_ja_subagent 3 times first.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "winner_model": {
                            "type": "STRING",
                            "description": "The winning GA model: ga_flash, ga_pro, or ga_25pro",
                        },
                        "reasoning": {
                            "type": "STRING",
                            "description": "Why this diff was selected: consensus scores, minimality, etc.",
                        },
                    },
                    "required": ["winner_model", "reasoning"],
                },
            },
        ]
    }
]


def run_main_agent(
    constraint: str,
    context_root: str,
    solver_type: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    temp_code_dir_base: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main Agent: orchestrates N-version ensemble (3 GA + 3 JA panel).

    Returns:
        {
            "success": bool,
            "winner_model": str,              # e.g. "gemini-3-pro-preview"
            "winner_edit": dict,              # {path, diff} from winning GA
            "panel_report": dict,             # all 3 JA votes + reasoning
            "reasoning": str,                 # MA's final selection reasoning
            "ga_results": List[dict],         # all 3 GA outputs
            "ja_results": List[dict],         # all 3 JA panel verdicts
            "retry_count": int,
            "cost_usd": float,
            "input_tokens": int,
            "output_tokens": int,
        }
    """
    os.makedirs(logs_dir or ".", exist_ok=True)
    _log(logs_dir, "main_agent.log", "=== Main Agent Orchestrator ===")
    _log(logs_dir, "main_agent.log", f"Constraint: {constraint}")

    api_key = env.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"success": False, "error": "GEMINI_API_KEY not set"}

    model = _resolve_model(env)
    _log(logs_dir, "main_agent.log", f"Model: {model}")

    # Main Agent's user prompt
    user_prompt = (
        f"Implement this constraint:\n{constraint}\n\n"
        "Use the protocol below: spawn 3 GA subagents, then 3 JA subagents, then finalize."
    )

    contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": user_prompt}]}]
    system_instruction = MAIN_AGENT_SYSTEM_PROMPT
    tools = MAIN_AGENT_TOOLS

    # State tracking
    ga_results: Dict[str, Dict[str, Any]] = {}  # model -> GA result
    ja_results: List[Dict[str, Any]] = []  # JA verdicts
    tool_call_history: List[Dict[str, Any]] = []
    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    retry_count = 0
    max_ga_retries = 2

    # --- Main LLM Turn Loop ---
    for turn in range(1, 15):  # max 15 turns
        _log(logs_dir, "main_agent.log", f"\n--- Turn {turn} --- (contents: {len(contents)} parts)")
        _log(logs_dir, "main_agent.log", f"GA results so far: {list(ga_results.keys())}")

        payload = {
            "contents": contents,
            "generationConfig": {"temperature": 0.0},
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "tools": tools,
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        }

        response = _api_generate(
            api_key=api_key,
            model=model,
            payload=payload,
        )

        _log(logs_dir, "main_agent.log", f"=== Turn {turn} ===")
        _log(logs_dir, "main_agent.log", f"Model: {model}")

        if response.get("error"):
            _log(logs_dir, "main_agent.log", f"API error: {response['error']}")
            return {"success": False, "error": response["error"]}

        # Parse Gemini response structure
        candidates = response.get("candidates", [])
        if not candidates:
            _log(logs_dir, "main_agent.log", "No candidates in response")
            break

        candidate = candidates[0]
        candidate_content = candidate.get("content", {})
        parts = candidate_content.get("parts", [])

        # Extract function calls from parts
        function_calls = [p.get("functionCall") for p in parts if isinstance(p, dict) and p.get("functionCall")]

        if not function_calls:
            # No more function calls → LLM is done
            finish_reason = candidate.get("finishReason", "")
            _log(logs_dir, "main_agent.log", f"Finish reason: {finish_reason}")
            break

        # Append model response to contents with required thought_signature
        model_parts = []
        for i, fc in enumerate(function_calls):
            # Ensure thought_signature exists (required by Gemini API for parallel tool calls)
            if "thought_signature" not in fc:
                fc["thought_signature"] = f"tool_call_{turn}_{i}"
            model_parts.append({"functionCall": fc})
        contents.append({"role": "model", "parts": model_parts})

        # --- Dispatch tool calls ---
        tool_responses: List[Dict[str, Any]] = []

        # Execute batched function calls in parallel if multiple
        if len(function_calls) > 1:
            with ThreadPoolExecutor(max_workers=len(function_calls)) as executor:
                futures = [
                    executor.submit(
                        _dispatch_main_tool,
                        fc,
                        constraint,
                        context_root,
                        solver_type,
                        env,
                        logs_dir,
                        temp_code_dir_base,
                        ga_results,
                        tool_call_history,
                    )
                    for fc in function_calls
                ]
                tool_responses = []
                for i, f in enumerate(futures):
                    try:
                        result = f.result()
                        tool_responses.append(result)
                    except Exception as exc:
                        _log(logs_dir, "main_agent.log", f"Tool {i} error: {exc}")
                        tool_responses.append({"error": str(exc)})
        else:
            # Single tool call
            tool_responses = [
                _dispatch_main_tool(
                    function_calls[0],
                    constraint,
                    context_root,
                    solver_type,
                    env,
                    logs_dir,
                    temp_code_dir_base,
                    ga_results,
                    tool_call_history,
                )
            ]

        # Append tool responses to contents
        _log(logs_dir, "main_agent.log", f"Tool responses: {len(tool_responses)} responses for {len(function_calls)} calls")
        for i, fc in enumerate(function_calls):
            if i < len(tool_responses):
                resp = tool_responses[i]
                _log(logs_dir, "main_agent.log", f"  Tool {i} ({fc['name']}): {'success' if not resp.get('error') else 'error'}")
                contents.append({
                    "role": "tool",
                    "parts": [{
                        "functionResponse": {
                            "name": fc["name"],
                            "response": {"result": resp}
                        }
                    }]
                })

        # Check if finalize was called with all subagents invoked
        finalize_calls = [h for h in tool_call_history if h["name"] == "finalize"]
        if finalize_calls:
            # Main Agent is done — extract winner
            finalize_args = finalize_calls[-1]["args"]
            winner_model = finalize_args.get("winner_model")
            reasoning = finalize_args.get("reasoning")

            # Map winner_model name back to original model string
            model_map = {
                "ga_flash": "gemini-3-flash-preview",
                "ga_pro": "gemini-3-pro-preview",
                "ga_25pro": "gemini-2.5-pro",
            }
            actual_winner_model = model_map.get(winner_model, winner_model)

            if actual_winner_model in ga_results:
                winner_edit = ga_results[actual_winner_model].get("edit")
                panel_report = {
                    "votes": [r.get("vote") for r in ja_results],
                    "verdicts": ja_results,
                }

                _log(logs_dir, "main_agent.log",
                     f"\n=== WINNER SELECTED ===\nModel: {actual_winner_model}\nReasoning: {reasoning}")

                return {
                    "success": True,
                    "winner_model": actual_winner_model,
                    "winner_edit": winner_edit,
                    "panel_report": panel_report,
                    "reasoning": reasoning,
                    "ga_results": list(ga_results.values()),
                    "ja_results": ja_results,
                    "retry_count": retry_count,
                    "cost_usd": total_cost,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                }
            else:
                return {
                    "success": False,
                    "error": f"Winner model '{actual_winner_model}' not found in GA results"
                }

    # If we exited loop without finalize
    _log(logs_dir, "main_agent.log", f"Loop exited after {turn} turns without finalize")
    _log(logs_dir, "main_agent.log", f"Final GA results: {list(ga_results.keys())}")
    _log(logs_dir, "main_agent.log", f"Final JA results count: {len(ja_results)}")
    return {
        "success": False,
        "error": "Main Agent loop exited without finalizing",
        "ga_results": list(ga_results.keys()),
        "ja_results_count": len(ja_results),
        "cost_usd": total_cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }


def _dispatch_main_tool(
    function_call: Dict[str, Any],
    constraint: str,
    context_root: str,
    solver_type: str,
    env: Dict[str, str],
    logs_dir: Optional[str],
    temp_code_dir_base: Optional[str],
    ga_results: Dict[str, Dict[str, Any]],
    tool_call_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Dispatch a Main Agent tool call and return the result."""

    name = function_call.get("name")
    args = function_call.get("args", {})

    tool_call_history.append({"name": name, "args": args})
    _log(logs_dir, "main_agent.log", f"  Tool: {name} {args}")

    if name == "spawn_ga_subagent":
        return _spawn_ga_tool(
            args,
            constraint,
            context_root,
            solver_type,
            env,
            logs_dir,
            temp_code_dir_base,
            ga_results,
        )

    elif name == "spawn_ja_subagent":
        return _spawn_ja_tool(
            args,
            constraint,
            env,
            logs_dir,
            ga_results,
        )

    elif name == "finalize":
        return _finalize_tool(args, tool_call_history)

    else:
        return {"error": f"Unknown tool: {name}"}


def _spawn_ga_tool(
    args: Dict[str, str],
    constraint: str,
    context_root: str,
    solver_type: str,
    env: Dict[str, str],
    logs_dir: Optional[str],
    temp_code_dir_base: Optional[str],
    ga_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Spawn one GA subagent with specified model."""

    model = args.get("model")
    if not model:
        return {"error": "model argument required"}

    # Map model name to shorthand for directory
    model_shorthand = model.split("-")[1]  # e.g. "flash" from "gemini-3-flash-preview"

    ga_logs = os.path.join(logs_dir or ".", f"ga_{model_shorthand}")
    os.makedirs(ga_logs, exist_ok=True)

    # Run GA with this model override and shared context_root
    # (context_root already has policy files copied by the pipeline)
    result = run_generation_agent(
        constraint=constraint,
        context_root=context_root,
        solver_type=solver_type,
        env=env,
        logs_dir=ga_logs,
        temp_code_dir=temp_code_dir_base,
        model_override=model,
        # TODO: pass prefetched_context here (Main Agent's gathered context)
    )

    ga_results[model] = result

    return {
        "model": model,
        "success": result.get("success", False),
        "edit": result.get("edit"),
        "cost_usd": result.get("cost_usd", 0),
    }


def _spawn_ja_tool(
    args: Dict[str, Any],
    constraint: str,
    env: Dict[str, str],
    logs_dir: Optional[str],
    ga_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Spawn one JA panel judge to review all 3 GA diffs."""

    judge_id_str = args.get("judge_id")
    if not judge_id_str:
        return {"error": "judge_id argument required"}
    judge_id = int(judge_id_str)

    # Collect all 3 GA diffs
    diffs = {}
    for model, ga_result in ga_results.items():
        model_shorthand = model.split("-")[1]
        edit = ga_result.get("edit")
        if edit:
            diffs[f"ga_{model_shorthand}"] = edit.get("diff", "")
        else:
            diffs[f"ga_{model_shorthand}"] = "(GA produced no edit)"

    ja_logs = os.path.join(logs_dir or ".", f"ja_{judge_id}")
    os.makedirs(ja_logs, exist_ok=True)

    result = run_panel_judgment_agent(
        constraint=constraint,
        diffs=diffs,
        env=env,
        logs_dir=ja_logs,
        judge_id=judge_id,
    )

    return {
        "judge_id": judge_id,
        "vote": result.get("vote"),
        "scores": result.get("scores", {}),
        "winner_reasoning": result.get("winner_reasoning"),
        "cost_usd": result.get("cost_usd", 0),
    }


def _finalize_tool(
    args: Dict[str, str],
    tool_call_history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Validate that all 6 subagents were called, then return OK."""

    ga_calls = [h for h in tool_call_history if h["name"] == "spawn_ga_subagent"]
    ja_calls = [h for h in tool_call_history if h["name"] == "spawn_ja_subagent"]

    if len(ga_calls) < 3:
        return {
            "error": f"spawn_ga_subagent called {len(ga_calls)} times. "
                     f"You must call it 3 times (one per model). Call remaining models now."
        }

    if len(ja_calls) < 3:
        return {
            "error": f"spawn_ja_subagent called {len(ja_calls)} times. "
                     f"You must call it 3 times (one per judge). Call remaining judges now."
        }

    winner_model = args.get("winner_model")
    if not winner_model:
        return {"error": "winner_model argument required"}

    if winner_model not in ["ga_flash", "ga_pro", "ga_25pro"]:
        return {
            "error": f"winner_model '{winner_model}' invalid. "
                     f"Must be one of: ga_flash, ga_pro, ga_25pro"
        }

    return {
        "ok": True,
        "winner_model": winner_model,
        "reasoning": args.get("reasoning", ""),
    }
