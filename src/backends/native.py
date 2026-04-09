"""
Native backend — Gemini/Groq via Google ADK (native tool calling).

Public entry point:
    run_native_pipeline(constraint, query_id, solver_type, ...) -> dict
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import traceback
from typing import Any, Dict, List, Optional

from google.adk import Agent
from google.adk.apps import App
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part, GenerateContentConfig

from src.configs import config_parser as cfg
from src.agents.main_agent import create_main_agent
from src.agents.generation_agent import create_generation_agent
from src.agents.judgment_agent import create_judgment_agent
from src.agents.error_analysis_agent import create_error_analysis_agent
from src.utils.code_tools import (
    copy_policy_to_dir,
    load_constraints as _load_constraints,
    load_schema as _load_schema,
    search_tool as _search_tool,
    read_function_tool as _read_function_tool,
    parse_edit_suggestion as _parse_edit_suggestion,
    write_file as _write_file,
    apply_edit_to_temp_dir as _apply_edit_to_temp_dir,
    apply_edit_tool as _apply_edit_tool,
)
from src.mcp.serena import SerenaClient
from src.utils.test_runner import run_toy_test as _run_toy_test
from src.core.pipeline import _parse_ja_verdict, _parse_injection_points, _make_diff


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------

_LLM_CONFIG = GenerateContentConfig(temperature=0)


def _get_model(role: str = "default"):
    """Return the model for a given agent role.

    Backends (selected by env var):
      USE_GROQ=1   — Llama via Groq (native tool calling)
      default      — Gemini via Google AI (native tool calling)
    """
    if os.environ.get("USE_GROQ") == "1":
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key:
            from google.adk.models.lite_llm import LiteLlm
            models = {
                "explore": "groq/llama-3.3-70b-versatile",
                "ga": "groq/llama-3.3-70b-versatile",
                "eaa": "groq/llama-3.3-70b-versatile",
                "default": "groq/llama-3.3-70b-versatile",
            }
            return LiteLlm(model=models.get(role, models["default"]), api_key=groq_key)
    return cfg.DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Tool functions for tool-using agents (Main Agent, EAA, RA)
# ---------------------------------------------------------------------------

def _make_tools(solver_type: str, env: Dict[str, str], serena: Optional[SerenaClient] = None) -> List:
    """Create tool functions scoped to the given solver type."""

    index_workspace = (
        str(cfg.HEXALY_INDEX_WORKSPACE) if solver_type == "hexaly"
        else str(cfg.INDEX_WORKSPACE)
    )
    work_dir = (
        str(cfg.SOLVERS_POLICY_ROOT).rsplit("/policy", 1)[0]
        if cfg.SOLVERS_POLICY_ROOT else "."
    )
    policy_dir = (
        str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT
        else os.path.join(work_dir, "policy")
    )

    def search_code(query: str, limit: int = 5) -> dict:
        """Semantic search in the indexed policy codebase.
        Returns code snippets with function names and docstrings.
        Use descriptive queries about OPERATIONS, not function names."""
        return _search_tool(
            index_workspace=index_workspace,
            query=query,
            limit=max(1, min(limit, 20)),
            env=env,
        )

    def read_function(relative_path: str, function_name: str) -> dict:
        """Read the full source of a specific Python function from the policy codebase.
        Only read the ONE function you need — be selective."""
        return _read_function_tool(
            work_dir=work_dir,
            policy_dir=policy_dir,
            relative_path=relative_path,
            function_name=function_name,
            env=env,
        )

    def _serena_path(p: str) -> str:
        """Serena project root = policy dir, so strip leading 'policy/' prefix if present."""
        if p.startswith("policy/"):
            return p[len("policy/"):]
        return p

    def get_symbols_overview(relative_path: str) -> dict:
        """List ALL real function and class names in a file. Pass the filename or
        policy/filename.py — returns Function/Class lists you can pass to read_function."""
        if serena:
            return serena.call_tool("get_symbols_overview", {"relative_path": _serena_path(relative_path)})
        return {"error": "Serena not available"}

    def find_symbol(name_path_pattern: str, substring_matching: bool = True) -> dict:
        """Find a symbol by partial name across the codebase. Returns fqn, file, line."""
        if serena:
            return serena.call_tool("find_symbol", {
                "name_path_pattern": name_path_pattern,
                "substring_matching": substring_matching,
                "include_info": True,
            })
        return {"error": "Serena not available"}

    def find_referencing_symbols(name_path: str, relative_path: str) -> dict:
        """Find all symbols that reference (call) a given symbol. LSP-based callers list."""
        if serena:
            return serena.call_tool("find_referencing_symbols", {
                "name_path": name_path,
                "relative_path": _serena_path(relative_path),
            })
        return {"error": "Serena not available"}

    return [search_code, read_function, get_symbols_overview, find_symbol, find_referencing_symbols]


# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

async def _run_agent(agent: Agent, message: str, timeout: int = 180,
                     max_retries: int = 5, log_turns: bool = False,
                     max_turns: int = 0, use_context_cache: bool = False) -> str:
    """Run an ADK agent and return its final text output. Retries on transient errors.

    Args:
        max_turns: If > 0, stop collecting after this many ADK turns (prevents
                   infinite tool-call loops). 0 means unlimited.
        use_context_cache: If True, wrap agent in App with ContextCacheConfig so ADK
                           automatically manages caching for multi-turn sessions.
    """
    for retry in range(max_retries):
        try:
            svc = InMemorySessionService()
            if use_context_cache:
                app = App(
                    name="v3",
                    root_agent=agent,
                    context_cache_config=ContextCacheConfig(
                        min_tokens=1024,
                        ttl_seconds=3600,
                        cache_intervals=50,
                    ),
                )
                runner = Runner(app=app, session_service=svc)
            else:
                runner = Runner(app_name="v3", agent=agent, session_service=svc)
            session = await svc.create_session(app_name="v3", user_id="pipeline")
            msg = Content(parts=[Part(text=message)], role="user")

            final_text = ""
            all_text_parts = []
            turn_num = 0

            async def _collect():
                nonlocal final_text, turn_num
                async for event in runner.run_async(
                    session_id=session.id, user_id="pipeline", new_message=msg
                ):
                    # Log cache usage when tokens are being served from cache
                    if event.usage_metadata:
                        cached_toks = getattr(event.usage_metadata, "cached_content_token_count", 0) or 0
                        if cached_toks > 0:
                            prompt_toks = getattr(event.usage_metadata, "prompt_token_count", 0) or 0
                            print(f"    [cache hit] cached={cached_toks} / prompt={prompt_toks} tokens")
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            if log_turns:
                                if hasattr(part, "function_call") and part.function_call:
                                    fc = part.function_call
                                    args_str = str(dict(fc.args))[:200] if fc.args else "{}"
                                    print(f"    [turn {turn_num}] TOOL: {fc.name}({args_str})")
                                if hasattr(part, "text") and part.text:
                                    print(f"    [turn {turn_num}] TEXT: {part.text[:300]}")
                            if hasattr(part, "text") and part.text:
                                final_text = part.text
                                all_text_parts.append(part.text)
                    if event.content:
                        turn_num += 1
                        if max_turns > 0 and turn_num >= max_turns:
                            break
                if "injection_points" in final_text:
                    return final_text
                combined = "\n".join(all_text_parts)
                if "injection_points" in combined:
                    return combined
                return final_text

            return await asyncio.wait_for(_collect(), timeout=timeout)

        except Exception as e:
            err_str = str(e)
            is_timeout = isinstance(e, TimeoutError)
            is_transient = is_timeout or any(code in err_str for code in [
                "429", "503", "500", "502", "504", "UNAVAILABLE", "overloaded",
                "high demand", "RESOURCE_EXHAUSTED", "quota",
            ])
            # Timeout errors: retry at most once (large context is unlikely to improve on retry)
            max_r = 2 if is_timeout else max_retries
            if is_transient and retry < max_r - 1:
                wait = 30 if is_timeout else 15 * (2 ** retry)
                print(f"    [retry {retry+1}/{max_r}] {type(e).__name__}: {err_str[:120]}... waiting {wait}s")
                await asyncio.sleep(wait)
                continue
            print(f"    [FATAL after {retry+1} attempts] {type(e).__name__}: {err_str[:200]}")
            print(traceback.format_exc())
            raise
    return ""  # unreachable — loop always returns or raises


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def run_native_pipeline(
    constraint: str,
    query_id: int,
    solver_type: str = "insertion",
    temp_code_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    shared_serena: Optional[SerenaClient] = None,
) -> Dict[str, Any]:
    """Run the full five-agent pipeline using native ADK tool calling (Gemini or Groq)."""

    logs_dir = logs_dir or f"/tmp/adk_logs_{query_id}"
    os.makedirs(logs_dir, exist_ok=True)
    temp_code_dir = temp_code_dir or f"/tmp/adk_temp_{query_id}"
    os.makedirs(temp_code_dir, exist_ok=True)

    def _log_msg(msg: str):
        print(msg)
        with open(os.path.join(logs_dir, "pipeline.log"), "a") as f:
            f.write(msg + "\n")

    _log_msg(f"=== Native ADK Pipeline: {constraint[:80]} ({solver_type}) ===")

    env = os.environ.copy()
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(env_file):
        with open(env_file) as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    env.setdefault(_k.strip(), _v.strip())

    # Models
    explore_model = _get_model("explore")
    eaa_model = _get_model("eaa")

    # Shared context
    constraint_content = _load_constraints()
    schema_content = _load_schema(".", solver_type)

    # Tools — wrap with trace logging for explore phase
    raw_tools = _make_tools(solver_type, env, shared_serena)

    trace_file = os.path.join(logs_dir, "explore_trace.txt") if logs_dir else None

    def _make_traced(base_tools, trace_path):
        import functools
        traced = []
        for fn in base_tools:
            def _wrap(f=fn):
                @functools.wraps(f)
                def _traced(**kwargs):
                    result = f(**kwargs)
                    if trace_path:
                        with open(trace_path, "a") as tf:
                            tf.write(f"\n>>> {f.__name__}({json.dumps(kwargs, indent=2)})\n")
                            tf.write(f"<<< {json.dumps(result, indent=2)}\n")
                    return result
                return _traced
            traced.append(_wrap())
        return traced

    tools = _make_traced(raw_tools, trace_file)

    pre_discovery = ""

    # ===================================================================
    # Phase 1: Main Agent explores code
    # ===================================================================
    _log_msg("\n--- Phase 1: Main Agent explores code ---")

    main_agent = create_main_agent(explore_model, tools, constraint_content, schema_content)

    user_msg = (
        f"<constraint>\n{constraint}\n</constraint>\n"
        + pre_discovery
        + "\nFind the function(s) where this constraint should be enforced. "
        "If the constraint has MULTIPLE requirements (e.g., same-group assignment AND ordering), "
        "you may need multiple injection points — list the most critical one first. "
        "Read each function, extract api_examples, output the injection_points JSON."
    )

    try:
        # max_turns=25: up to ~12 tool call+response pairs + final JSON output turn
        explore_text = await _run_agent(main_agent, user_msg, timeout=600,
                                        log_turns=True, max_turns=25)
    except Exception as e:
        _log_msg(f"  Main Agent error: {type(e).__name__}: {e}")
        _log_msg(traceback.format_exc())
        return {"success": False, "diff": f"Explore failed: {type(e).__name__}: {e}"}

    if logs_dir:
        _write_file(logs_dir, "explore_output.txt", explore_text)

    injection_points = _parse_injection_points(explore_text)
    _log_msg(f"  Found {len(injection_points)} injection point(s)")

    if not injection_points:
        _log_msg("  No injection points found — cannot proceed")
        return {"success": False, "diff": "No injection points found"}

    # Read full source for each injection point
    work_dir = (
        str(cfg.SOLVERS_POLICY_ROOT).rsplit("/policy", 1)[0]
        if cfg.SOLVERS_POLICY_ROOT else "."
    )
    policy_dir = (
        str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT
        else os.path.join(work_dir, "policy")
    )

    for ip in injection_points:
        fname = ip.get("file", "")
        func = ip.get("function", "")
        if fname and func and not ip.get("source"):
            result = _read_function_tool(
                work_dir=work_dir, policy_dir=policy_dir,
                relative_path=fname, function_name=func, env=env,
            )
            if result.get("ok"):
                ip["source"] = result["functionSource"]
                _log_msg(f"  Attached source for {func} ({len(ip['source'])} chars)")

    for i, ip in enumerate(injection_points):
        _log_msg(f"  IP{i+1}: {ip.get('function')} in {ip.get('file')} [{ip.get('role')}]")

    # ===================================================================
    # Phase 2: Generator → Verifier loop
    # ===================================================================
    _log_msg("\n--- Phase 2: Generator -> Verifier loop ---")

    # Build context package from ALL injection points
    def _build_context_package() -> str:
        parts = [f"## Constraint\n{constraint}\n"]
        for i, ip in enumerate(injection_points):
            header = f"## Function {i+1}: {ip.get('file')}::{ip.get('function')} [{ip.get('role', '')}]"
            parts.append(header)
            if ip.get("injection_description"):
                parts.append(f"Injection description: {ip['injection_description']}")
            if ip.get("anchor_line"):
                parts.append(f"Anchor line: {ip['anchor_line']}")
            if ip.get("scope_vars"):
                parts.append(f"Variables in scope: {ip['scope_vars']}")
            api_ex = ip.get("api_examples", [])
            if api_ex:
                ex_str = "\n".join(f"  {x}" for x in api_ex) if isinstance(api_ex, list) else str(api_ex)
                parts.append(f"API examples:\n```python\n{ex_str}\n```")
            if ip.get("source"):
                parts.append(f"Full source:\n```python\n{ip['source']}\n```")
            parts.append("")
        return "\n".join(parts)

    context_package = _build_context_package()

    gen_model = _get_model("default")
    ja_model  = _get_model("default")
    N_CANDIDATES = 1  # 1 GA + 1 JA per round (reduced from 3 to lower API load)
    # Phase 2 agents: no static_instruction — context_package included in message body
    gen_agent = create_generation_agent(gen_model, raw_tools, constraint_content, schema_content)
    ja_agent  = create_judgment_agent(ja_model)

    def _apply_edit_to_source(source: str, edit_text: str) -> str:
        """Return source with old_text replaced by new_text, or original on failure."""
        import re as _re
        olds = _re.findall(r"<old_text>\n?(.*?)\n?</old_text>", edit_text, _re.DOTALL)
        news = _re.findall(r"<new_text>\n?(.*?)\n?</new_text>", edit_text, _re.DOTALL)
        result = source
        for old, new in zip(olds, news):
            if old in result:
                result = result.replace(old, new, 1)
        return result

    def _build_ja_prompt(candidates: list) -> str:
        parts = [f"## Constraint to implement\n{constraint}\n"]
        source_summary = "\n\n".join(
            f"### {ip.get('function')} ({ip.get('file')})\n```python\n{ip.get('source', '')}\n```"
            for ip in injection_points if ip.get("source")
        )
        if source_summary:
            parts.append(f"## Function source(s)\n{source_summary}\n")
        for i, et in enumerate(candidates):
            applied_sources = []
            for ip in injection_points:
                src = ip.get("source", "")
                if src:
                    modified = _apply_edit_to_source(src, et)
                    applied_sources.append(
                        f"### {ip.get('function')} ({ip.get('file')}) — after edit\n```python\n{modified}\n```"
                    )
            parts.append(f"## Proposed edit {i + 1}")
            if applied_sources:
                parts.append("\n".join(applied_sources))
            parts.append(f"Raw edit:\n{et}\n")
        parts.append("Review all proposed edits and choose the best correct one.")
        return "\n".join(parts)

    edit = None
    winner_diff = ""
    prev_edit_text = None
    jud_feedback = None
    MAX_GEN_ROUNDS = 10

    for gen_round in range(1, MAX_GEN_ROUNDS + 1):
        # First pass: N parallel GAs; subsequent rounds: single revision
        if jud_feedback is None:
            prompt = context_package + "\nGenerate the edit(s). Use tools if you need to look up any API or variable."
            role = "GEN"
            n_ga = N_CANDIDATES
        else:
            prompt = (
                context_package
                + f"\n## Previous edit attempt (for reference only — do NOT use its new_text as old_text)\n{prev_edit_text}\n\n"
                + f"## Verifier feedback\n{jud_feedback}\n\n"
                + "Generate a FRESH edit against the ORIGINAL source shown in the context package above. "
                "Your old_text MUST be copied verbatim from the Original Source in 'Full source' above — NOT from the previous edit's new_text. "
                "If the verifier provided exact corrected code, use it as the new_text VERBATIM."
            )
            role = "REV"
            n_ga = 1  # single revision per round

        # Run n_ga GA agents in parallel (using raw_tools — no explore trace/limit)
        ga_agents = [
            create_generation_agent(gen_model, raw_tools, constraint_content, schema_content)
            for _ in range(n_ga)
        ]
        ga_tasks = [_run_agent(a, prompt, timeout=600) for a in ga_agents]
        ga_results = await asyncio.gather(*ga_tasks, return_exceptions=True)

        # Save and filter valid candidates
        valid_texts = []
        for i, res in enumerate(ga_results):
            suffix = f"_gen{i+1}" if n_ga > 1 else ""
            et = res if isinstance(res, str) else ""
            if logs_dir:
                _write_file(logs_dir, f"edit_round_{gen_round}_{role.lower()}{suffix}.txt", et)
            if et and _parse_edit_suggestion(et, logs_dir):
                valid_texts.append(et)
            elif not isinstance(res, str):
                _log_msg(f"  {role} round {gen_round} GA{i+1} error: {res}")
            else:
                _log_msg(f"  {role} round {gen_round} GA{i+1}: no valid XML")

        if not valid_texts:
            _log_msg(f"  Round {gen_round}: all GAs produced no valid edit — retrying")
            jud_feedback = "All candidates produced no valid XML. Output ONLY the XML edit format with <relative_path>, <old_text>, <new_text>, <explanation> tags."
            continue

        prev_edit_text = valid_texts[0]

        # Run N_CANDIDATES JAs in parallel, each reviewing ALL valid candidates
        ja_prompt = _build_ja_prompt(valid_texts)
        ja_agents = [create_judgment_agent(ja_model) for _ in range(N_CANDIDATES)]
        ja_tasks = [_run_agent(a, ja_prompt, timeout=240) for a in ja_agents]
        ja_results = await asyncio.gather(*ja_tasks, return_exceptions=True)

        # Collect and log JA verdicts
        votes = []
        for i, res in enumerate(ja_results):
            ja_text = res if isinstance(res, str) else ""
            if logs_dir:
                _write_file(logs_dir, f"ja_round_{gen_round}_ja{i+1}.txt", ja_text)
            if ja_text:
                right, jud = _parse_ja_verdict(ja_text)
                votes.append((right, jud))

        # Tally: need 2+ votes for a candidate (choice 1 = valid_texts[0])
        # right=True means some candidate approved; pick the approved ones
        approved_votes = [(r, j) for r, j in votes if r]
        _log_msg(f"  Round {gen_round} [{role}]: {len(approved_votes)}/{len(votes)} JAs approved")

        min_approval = max(1, N_CANDIDATES - 1)  # need majority (1 of 1, or 2 of 3)
        if len(approved_votes) >= min_approval:
            # Majority approved — use first valid candidate
            edit = _parse_edit_suggestion(valid_texts[0], logs_dir)
            winner_diff = _make_diff(edit)
            _log_msg(f"  Majority winner: candidate 1 ({len(approved_votes)}/{len(votes)} votes)")
            break

        # No majority — collect feedback for next round
        jud_feedback = next(
            (j for _, j in votes if j and j.lower() != "none"),
            "No majority — all edits had issues"
        )

    if not edit:
        _log_msg("  Generator loop produced no valid edit")
        return {"success": False, "diff": "No edit produced"}

    # ===================================================================
    # Phase 3: Apply edit + run toy test (retry on old_text mismatch)
    # ===================================================================
    _log_msg("\n--- Phase 3: Apply + Test ---")

    MAX_APPLY_RETRIES = 3
    applied = False
    for apply_attempt in range(MAX_APPLY_RETRIES):
        applied = _apply_edit_to_temp_dir(edit, temp_code_dir, env, logs_dir)
        if applied:
            break
        # Apply failed — read actual file to show GA the correct content
        rel_path = edit.get("path", "")
        old_text_failed = edit.get("old_text", "")
        policy_dir = os.path.join(temp_code_dir, "policy")
        src_file = os.path.join(policy_dir, os.path.basename(rel_path))
        # Fall back to original policy root if symlink not set up yet
        if not os.path.isfile(src_file) and cfg.SOLVERS_POLICY_ROOT:
            src_file = os.path.join(str(cfg.SOLVERS_POLICY_ROOT), os.path.basename(rel_path))
        file_snippet = ""
        if os.path.isfile(src_file):
            with open(src_file) as _f:
                file_lines = _f.readlines()
            first_line = next((l.strip() for l in old_text_failed.splitlines() if l.strip()), "")
            for _i, _line in enumerate(file_lines):
                if first_line[:50] in _line:
                    _start = max(0, _i - 2)
                    _end = min(len(file_lines), _i + 20)
                    file_snippet = "".join(file_lines[_start:_end])
                    break
        _log_msg(f"  Apply FAILED (attempt {apply_attempt+1}/{MAX_APPLY_RETRIES})")
        if apply_attempt >= MAX_APPLY_RETRIES - 1:
            break
        # Retry GA: show it the actual file content so it can correct old_text
        apply_retry_prompt = (
            context_package
            + f"\n## Previous edit attempt (FAILED — old_text did not match file)\n"
            + f"old_text used:\n```\n{old_text_failed}\n```\n\n"
            + (f"## Actual file content near expected location\n```python\n{file_snippet}\n```\n\n" if file_snippet else "")
            + "The old_text above was WRONG — it did not appear verbatim in the file.\n"
            + "Generate a CORRECTED edit. Copy old_text EXACTLY from the Actual file content shown above.\n"
            + "Pay special attention to function call argument lists — include EVERY argument in the exact order shown."
        )
        try:
            retry_gen_text = await _run_agent(gen_agent, apply_retry_prompt, timeout=240)
            if logs_dir:
                _write_file(logs_dir, f"apply_retry_{apply_attempt+1}_gen.txt", retry_gen_text)
            retry_edit = _parse_edit_suggestion(retry_gen_text, logs_dir)
            if retry_edit:
                edit = retry_edit
                winner_diff = _make_diff(edit)
        except Exception as _e:
            _log_msg(f"  Apply retry GA error: {type(_e).__name__}: {_e}")
            break

    if not applied:
        _log_msg("  Apply FAILED after all retries")
        return {"success": False, "diff": "Edit application failed"}
    n_blocks = 1 + len(edit.get("extra_edits", []))
    _log_msg(f"  Applied {n_blocks} edit block(s) to {edit.get('path', '?')}")

    # ===================================================================
    # Phase 4: AFL correction loop — run until test passes
    # Each iteration: EAA diagnoses → inner AFL loop (RA → JA → RA → ...) → apply → retest
    # ===================================================================
    _log_msg("\n--- Phase 4: AFL correction loop (run until test passes) ---")

    MAX_TEST_ATTEMPTS = 10   # safety cap on total test runs
    MAX_FIX_ROUNDS = 5       # safety cap on RA→JA rounds per test failure
    test_attempt = 0

    # Phase 4 agents use static_instruction for the stable context (constraint + function
    # sources). This keeps the prefix identical across all correction rounds so providers
    # can cache it (Gemini via ContextCacheConfig in App, Anthropic/OpenAI automatically).
    # Turn messages contain only the small dynamic delta (error + prev edit + EAA output).
    source_summary = "\n\n".join(
        f"### {ip.get('function')} ({ip.get('file')})\n```python\n{ip.get('source', '')}\n```"
        for ip in injection_points if ip.get("source")
    )
    gen_agent_p4 = create_generation_agent(
        gen_model, raw_tools, constraint_content, schema_content,
        static_instruction=context_package,
    )
    ja_agent_p4 = create_judgment_agent(
        ja_model,
        static_instruction=(
            f"## Constraint\n{constraint}\n\n## Function source(s)\n{source_summary}"
            if source_summary else f"## Constraint\n{constraint}"
        ),
    )
    eaa_agent = create_error_analysis_agent(
        eaa_model, tools, constraint_content, schema_content,
        static_instruction=context_package,
    )
    _log_msg(f"  Phase 4 agents: static_instruction set ({len(context_package):,} chars cached)")

    while test_attempt < MAX_TEST_ATTEMPTS:
        test_attempt += 1
        result = _run_toy_test(query_id, solver_type, temp_code_dir)

        if result["passed"]:
            _log_msg(f"  Test attempt {test_attempt}: PASS")
            return {
                "success": True,
                "diff": winner_diff,
                "injection_points": injection_points,
            }

        test_output = result.get("output", "")
        for line in test_output.split("\n"):
            if any(kw in line for kw in ["FAILED", "assert", "Error", "PASSED", "collected"]):
                _log_msg(f"    {line.strip()}")
        _log_msg(f"  Test attempt {test_attempt}: FAIL")
        if logs_dir:
            _write_file(logs_dir, f"test_output_{test_attempt}.txt", test_output)

        error_summary = test_output

        # EAA turn message: just the dynamic delta. Constraint + function sources
        # are in static_instruction and cached by the provider.
        eaa_prompt = (
            f"## Test error\n```\n{error_summary}\n```\n\n"
            f"## Previous edit\n{prev_edit_text}\n\n"
            "Diagnose the failure mode. Search for the function that CONTROLS it.\n"
            "Use find_referencing_symbols to trace up from evaluation functions to the assignment loop.\n"
            "Read the controlling function. Output the corrected edit."
        )
        try:
            eaa_text = await _run_agent(eaa_agent, eaa_prompt, timeout=240, log_turns=True,
                                        use_context_cache=True)
            if logs_dir:
                _write_file(logs_dir, f"eaa_output_{test_attempt}.txt", eaa_text)
            _log_msg(f"  EAA: {eaa_text[:200]}")
        except Exception as e:
            _log_msg(f"  EAA error: {type(e).__name__}: {e}")
            _log_msg(traceback.format_exc())
            continue

        fix_jud_feedback = None
        fix_edit = None
        fix_diff = ""
        fix_prev_text = None

        for fix_round in range(MAX_FIX_ROUNDS):
            fix_round_num = fix_round + 1

            # Turn messages contain only the dynamic delta — context_package is in
            # static_instruction and cached by the provider across all rounds.
            if fix_jud_feedback is None:
                fix_prompt = (
                    f"## Previous edit (caused test failure)\n{prev_edit_text}\n\n"
                    f"## Error analysis\n{eaa_text}\n\n"
                    "Generate the corrected edit. If the diagnosis points to a different "
                    "function, use your tools to read it first."
                )
            else:
                fix_prompt = (
                    f"## Previous fix attempt\n{fix_prev_text}\n\n"
                    f"## Verifier feedback\n{fix_jud_feedback}\n\n"
                    f"## Error context\n{eaa_text}\n\n"
                    "Revise the fix to address all issues."
                )

            try:
                fix_text = await _run_agent(gen_agent_p4, fix_prompt, timeout=240,
                                            use_context_cache=True)
                if logs_dir:
                    _write_file(logs_dir, f"fix_{test_attempt}_{fix_round_num}.txt", fix_text)
                fix_prev_text = fix_text
                candidate_fix = _parse_edit_suggestion(fix_text, logs_dir)
            except Exception as e:
                _log_msg(f"  Fix round {fix_round_num} error: {type(e).__name__}: {e}")
                break

            if not candidate_fix:
                _log_msg(f"  Fix round {fix_round_num}: no valid edit — retrying")
                continue

            fix_diff = _make_diff(candidate_fix)
            fix_edit = candidate_fix

            # JA turn message: just the proposed fix. Constraint + sources are in static_instruction.
            ja_fix_prompt = f"## Proposed fix\n{fix_text}\n\nJudge this fix."
            try:
                ja_fix_text = await _run_agent(ja_agent_p4, ja_fix_prompt, timeout=240,
                                               use_context_cache=True)
                if logs_dir:
                    _write_file(logs_dir, f"ja_fix_{test_attempt}_{fix_round_num}.txt", ja_fix_text)
                right, jud = _parse_ja_verdict(ja_fix_text)
                _log_msg(f"  Fix round {fix_round_num} [GEN->JA]: {'approved' if right else f'rejected: {jud}'}")
            except Exception as e:
                _log_msg(f"  JA fix error (proceeding): {type(e).__name__}: {e}")
                right, jud = True, ""

            if right:
                break
            fix_jud_feedback = jud

        if fix_edit and _apply_edit_to_temp_dir(fix_edit, temp_code_dir, env, logs_dir):
            n_blocks = 1 + len(fix_edit.get("extra_edits", []))
            _log_msg(f"  Fix applied ({n_blocks} block(s)) to {fix_edit.get('path', '?')}")
            winner_diff = fix_diff
            prev_edit_text = fix_prev_text or fix_diff
        else:
            _log_msg("  No valid fix or apply failed — continuing")

    return {
        "success": False,
        "diff": winner_diff,
        "injection_points": injection_points,
    }


# ---------------------------------------------------------------------------
# DSL pipeline — NL → IR → verified snippet → locate → inject
# ---------------------------------------------------------------------------

async def run_dsl_pipeline(
    constraint: str,
    query_id: int,
    solver_type: str = "insertion",
    temp_code_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    shared_serena: Optional[SerenaClient] = None,
) -> Dict[str, Any]:
    """DSL pipeline: compile constraint to a verified snippet, then locate + inject.

    Differences from run_native_pipeline:
      - Phase 0 (new): NL → IR → snippet via DSL_OR compiler (no LLM code generation)
      - Phase 1: Main Agent still finds injection point (same as normal)
      - Phase 2: GA only locates old_text and wraps the pre-compiled snippet — it
                 must NOT modify the constraint logic
      - Phase 3+4: identical to normal pipeline
    """
    from src.dsl.adapter import compile_for_solver, build_dsl_ga_prompt, DSLCompileError

    logs_dir = logs_dir or f"/tmp/dsl_logs_{query_id}"
    os.makedirs(logs_dir, exist_ok=True)
    temp_code_dir = temp_code_dir or f"/tmp/dsl_temp_{query_id}"
    os.makedirs(temp_code_dir, exist_ok=True)

    def _log_msg(msg: str):
        print(msg)
        with open(os.path.join(logs_dir, "pipeline.log"), "a") as f:
            f.write(msg + "\n")

    _log_msg(f"=== DSL Pipeline: {constraint[:80]} ({solver_type}) ===")

    env = os.environ.copy()
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(env_file):
        with open(env_file) as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    env.setdefault(_k.strip(), _v.strip())

    # ===================================================================
    # Phase 0: Compile NL → IR → verified snippet
    # ===================================================================
    _log_msg("\n--- Phase 0: DSL compile ---")
    try:
        dsl_result = compile_for_solver(constraint, solver_type)
        snippet = dsl_result["snippet"]
        ir_type = dsl_result["ir_type"]
        _log_msg(f"  IR: {ir_type}  fields: {dsl_result['ir_dict']}")
        _log_msg(f"  Snippet ({len(snippet)} chars):\n{snippet}")
        if logs_dir:
            _write_file(logs_dir, "dsl_ir.json", json.dumps(dsl_result["ir_dict"], indent=2))
            _write_file(logs_dir, "dsl_snippet.py", snippet)
    except DSLCompileError as e:
        _log_msg(f"  DSL compile failed: {e}")
        _log_msg("  Falling back to normal LLM pipeline")
        return await run_native_pipeline(
            constraint=constraint,
            query_id=query_id,
            solver_type=solver_type,
            temp_code_dir=temp_code_dir,
            logs_dir=logs_dir,
            shared_serena=shared_serena,
        )

    # ===================================================================
    # Phase 1: Main Agent finds injection point (same as normal pipeline)
    # ===================================================================
    _log_msg("\n--- Phase 1: Main Agent finds injection point ---")

    explore_model = _get_model("explore")
    eaa_model = _get_model("eaa")
    gen_model = _get_model("default")
    ja_model  = _get_model("default")

    constraint_content = _load_constraints()
    schema_content = _load_schema(".", solver_type)

    raw_tools = _make_tools(solver_type, env, shared_serena)
    trace_file = os.path.join(logs_dir, "explore_trace.txt")

    import functools
    def _make_traced_tools(base_tools, trace_path):
        traced = []
        for fn in base_tools:
            def _wrap(f=fn):
                @functools.wraps(f)
                def _traced(**kwargs):
                    result = f(**kwargs)
                    with open(trace_path, "a") as tf:
                        tf.write(f"\n>>> {f.__name__}({json.dumps(kwargs, indent=2)})\n")
                        tf.write(f"<<< {json.dumps(result, indent=2)}\n")
                    return result
                return _traced
            traced.append(_wrap())
        return traced

    tools = _make_traced_tools(raw_tools, trace_file)

    main_agent = create_main_agent(explore_model, tools, constraint_content, schema_content)
    user_msg = (
        f"<constraint>\n{constraint}\n</constraint>\n"
        "\nFind the function(s) where this constraint should be enforced. "
        "Read each function, extract api_examples, output the injection_points JSON."
    )

    try:
        explore_text = await _run_agent(main_agent, user_msg, timeout=600,
                                        log_turns=True, max_turns=25)
    except Exception as e:
        _log_msg(f"  Main Agent error: {type(e).__name__}: {e}")
        return {"success": False, "diff": f"Explore failed: {e}"}

    if logs_dir:
        _write_file(logs_dir, "explore_output.txt", explore_text)

    injection_points = _parse_injection_points(explore_text)
    _log_msg(f"  Found {len(injection_points)} injection point(s)")

    if not injection_points:
        return {"success": False, "diff": "No injection points found"}

    work_dir = (
        str(cfg.SOLVERS_POLICY_ROOT).rsplit("/policy", 1)[0]
        if cfg.SOLVERS_POLICY_ROOT else "."
    )
    policy_dir = (
        str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT
        else os.path.join(work_dir, "policy")
    )

    for ip in injection_points:
        fname = ip.get("file", "")
        func = ip.get("function", "")
        if fname and func and not ip.get("source"):
            result = _read_function_tool(
                work_dir=work_dir, policy_dir=policy_dir,
                relative_path=fname, function_name=func, env=env,
            )
            if result.get("ok"):
                ip["source"] = result["functionSource"]

    source_summary = "\n\n".join(
        f"### {ip.get('function')} ({ip.get('file')})\n```python\n{ip.get('source', '')}\n```"
        for ip in injection_points if ip.get("source")
    )

    # ===================================================================
    # Phase 2: GA locates injection point and wraps pre-compiled snippet
    # ===================================================================
    _log_msg("\n--- Phase 2: GA locates injection point + wraps snippet ---")

    ga_prompt = build_dsl_ga_prompt(dsl_result, source_summary)

    gen_agent = create_generation_agent(gen_model, raw_tools, constraint_content, schema_content)
    ja_agent  = create_judgment_agent(ja_model)

    edit = None
    winner_diff = ""
    prev_edit_text = None
    jud_feedback = None
    MAX_GEN_ROUNDS = 5  # fewer rounds needed — logic is pre-compiled

    for gen_round in range(1, MAX_GEN_ROUNDS + 1):
        if jud_feedback is not None:
            ga_prompt = (
                ga_prompt
                + f"\n\n## Previous attempt\n{prev_edit_text}\n\n"
                + f"## Reviewer feedback\n{jud_feedback}\n\n"
                + "Fix the old_text — copy it EXACTLY from the function source above. "
                "The snippet logic must remain unchanged."
            )

        try:
            ga_text = await _run_agent(gen_agent, ga_prompt, timeout=600)
        except Exception as e:
            _log_msg(f"  GA round {gen_round} error: {type(e).__name__}: {e}")
            break

        if logs_dir:
            _write_file(logs_dir, f"dsl_ga_round_{gen_round}.txt", ga_text)

        candidate = _parse_edit_suggestion(ga_text, logs_dir)
        if not candidate:
            _log_msg(f"  Round {gen_round}: no valid XML")
            jud_feedback = "No valid XML produced. Output ONLY <relative_path>/<old_text>/<new_text>/<explanation> tags."
            prev_edit_text = ga_text
            continue

        prev_edit_text = ga_text

        # Build JA prompt with the pre-compiled snippet for context
        ja_prompt = (
            f"## Constraint\n{constraint}\n\n"
            f"## Pre-compiled snippet (must appear in new_text verbatim)\n"
            f"```python\n{snippet}\n```\n\n"
            f"## Function source(s)\n{source_summary}\n\n"
            f"## Proposed edit\n{ga_text}\n\n"
            "Judge this edit. Verify the snippet appears in new_text and old_text matches the source."
        )
        try:
            ja_text = await _run_agent(ja_agent, ja_prompt, timeout=240)
            if logs_dir:
                _write_file(logs_dir, f"dsl_ja_round_{gen_round}.txt", ja_text)
            right, jud = _parse_ja_verdict(ja_text)
            _log_msg(f"  Round {gen_round}: {'approved' if right else f'rejected: {jud}'}")
        except Exception as e:
            _log_msg(f"  JA error (proceeding): {type(e).__name__}: {e}")
            right, jud = True, ""

        if right:
            edit = candidate
            winner_diff = _make_diff(edit)
            break

        jud_feedback = jud

    if not edit:
        _log_msg("  DSL GA loop produced no valid edit")
        return {"success": False, "diff": "No edit produced", "dsl_ir": dsl_result["ir_dict"]}

    # ===================================================================
    # Phase 3+4: Apply + test + correction loop (same as normal pipeline)
    # ===================================================================
    _log_msg("\n--- Phase 3: Apply + Test ---")

    MAX_APPLY_RETRIES = 3
    applied = False
    for apply_attempt in range(MAX_APPLY_RETRIES):
        applied = _apply_edit_to_temp_dir(edit, temp_code_dir, env, logs_dir)
        if applied:
            break
        rel_path = edit.get("path", "")
        old_text_failed = edit.get("old_text", "")
        src_file = os.path.join(temp_code_dir, "policy", os.path.basename(rel_path))
        if not os.path.isfile(src_file) and cfg.SOLVERS_POLICY_ROOT:
            src_file = os.path.join(str(cfg.SOLVERS_POLICY_ROOT), os.path.basename(rel_path))
        file_snippet = ""
        if os.path.isfile(src_file):
            with open(src_file) as _f:
                file_lines = _f.readlines()
            first_line = next((l.strip() for l in old_text_failed.splitlines() if l.strip()), "")
            for _i, _line in enumerate(file_lines):
                if first_line[:50] in _line:
                    _start = max(0, _i - 2)
                    _end = min(len(file_lines), _i + 20)
                    file_snippet = "".join(file_lines[_start:_end])
                    break
        _log_msg(f"  Apply FAILED (attempt {apply_attempt+1}/{MAX_APPLY_RETRIES})")
        if apply_attempt >= MAX_APPLY_RETRIES - 1:
            break
        retry_prompt = (
            f"## Pre-compiled snippet (must appear in new_text)\n```python\n{snippet}\n```\n\n"
            f"## Previous edit (FAILED — old_text did not match)\n{old_text_failed}\n\n"
            + (f"## Actual file content near expected location\n```python\n{file_snippet}\n```\n\n" if file_snippet else "")
            + "Copy old_text EXACTLY from the actual file content. Do NOT modify the snippet."
        )
        try:
            retry_text = await _run_agent(gen_agent, retry_prompt, timeout=240)
            retry_edit = _parse_edit_suggestion(retry_text, logs_dir)
            if retry_edit:
                edit = retry_edit
                winner_diff = _make_diff(edit)
        except Exception:
            break

    if not applied:
        return {"success": False, "diff": "Edit application failed", "dsl_ir": dsl_result["ir_dict"]}

    _log_msg("\n--- Phase 4: Correction loop ---")

    context_package = (
        f"## Constraint\n{constraint}\n\n"
        f"## DSL IR\nType: {ir_type}\nFields: {dsl_result['ir_dict']}\n\n"
        f"## Pre-compiled snippet\n```python\n{snippet}\n```\n\n"
        f"## Function source(s)\n{source_summary}"
    )

    gen_agent_p4 = create_generation_agent(
        gen_model, raw_tools, constraint_content, schema_content,
        static_instruction=context_package,
    )
    ja_agent_p4 = create_judgment_agent(
        ja_model,
        static_instruction=f"## Constraint\n{constraint}\n\n## Function source(s)\n{source_summary}",
    )
    eaa_agent = create_error_analysis_agent(
        eaa_model, tools, constraint_content, schema_content,
        static_instruction=context_package,
    )

    MAX_TEST_ATTEMPTS = 10
    MAX_FIX_ROUNDS = 5
    test_attempt = 0

    while test_attempt < MAX_TEST_ATTEMPTS:
        test_attempt += 1
        test_result = _run_toy_test(query_id, solver_type, temp_code_dir)

        if test_result["passed"]:
            _log_msg(f"  Test attempt {test_attempt}: PASS")
            return {
                "success": True,
                "diff": winner_diff,
                "injection_points": injection_points,
                "dsl_ir": dsl_result["ir_dict"],
                "dsl_snippet": snippet,
            }

        test_output = test_result.get("output", "")
        _log_msg(f"  Test attempt {test_attempt}: FAIL")
        if logs_dir:
            _write_file(logs_dir, f"test_output_{test_attempt}.txt", test_output)

        eaa_prompt = (
            f"## Test error\n```\n{test_output}\n```\n\n"
            f"## Previous edit\n{prev_edit_text}\n\n"
            "Diagnose the failure. The pre-compiled snippet is in static_instruction — "
            "if the snippet logic is correct, the issue is in WHERE or HOW it was injected.\n"
            "Use find_referencing_symbols to trace to the assignment loop if needed.\n"
            "Output the corrected edit."
        )
        try:
            eaa_text = await _run_agent(eaa_agent, eaa_prompt, timeout=240, log_turns=True,
                                        use_context_cache=True)
            if logs_dir:
                _write_file(logs_dir, f"eaa_output_{test_attempt}.txt", eaa_text)
        except Exception as e:
            _log_msg(f"  EAA error: {type(e).__name__}: {e}")
            continue

        fix_jud_feedback = None
        fix_edit = None
        fix_diff = ""
        fix_prev_text = None

        for fix_round in range(MAX_FIX_ROUNDS):
            fix_round_num = fix_round + 1
            if fix_jud_feedback is None:
                fix_prompt = (
                    f"## Previous edit (caused test failure)\n{prev_edit_text}\n\n"
                    f"## Error analysis\n{eaa_text}\n\n"
                    "Generate the corrected edit. The pre-compiled snippet logic is trusted — "
                    "fix the injection location or variable names if needed."
                )
            else:
                fix_prompt = (
                    f"## Previous fix attempt\n{fix_prev_text}\n\n"
                    f"## Verifier feedback\n{fix_jud_feedback}\n\n"
                    f"## Error context\n{eaa_text}\n\nRevise the fix."
                )

            try:
                fix_text = await _run_agent(gen_agent_p4, fix_prompt, timeout=240,
                                            use_context_cache=True)
                if logs_dir:
                    _write_file(logs_dir, f"fix_{test_attempt}_{fix_round_num}.txt", fix_text)
                fix_prev_text = fix_text
                candidate_fix = _parse_edit_suggestion(fix_text, logs_dir)
            except Exception as e:
                _log_msg(f"  Fix round {fix_round_num} error: {type(e).__name__}: {e}")
                break

            if not candidate_fix:
                continue

            fix_diff = _make_diff(candidate_fix)
            fix_edit = candidate_fix

            ja_fix_prompt = f"## Proposed fix\n{fix_text}\n\nJudge this fix."
            try:
                ja_fix_text = await _run_agent(ja_agent_p4, ja_fix_prompt, timeout=240,
                                               use_context_cache=True)
                right, jud = _parse_ja_verdict(ja_fix_text)
                _log_msg(f"  Fix round {fix_round_num}: {'approved' if right else f'rejected: {jud}'}")
            except Exception:
                right, jud = True, ""

            if right:
                break
            fix_jud_feedback = jud

        if fix_edit and _apply_edit_to_temp_dir(fix_edit, temp_code_dir, env, logs_dir):
            winner_diff = fix_diff
            prev_edit_text = fix_prev_text or fix_diff

    return {
        "success": False,
        "diff": winner_diff,
        "injection_points": injection_points,
        "dsl_ir": dsl_result["ir_dict"],
    }
