"""
Prompt-tools backend — API without native tool calling — prompt-based tool loop.

Endpoint: {AMPLIFY_BASE_URL}/chat  (POST)
Auth:     Authorization: Bearer <AMPLIFY_API_KEY>

Since Amplify does not support OpenAI-style tool_calls, this backend implements
prompt-based tool calling: tools are described in the system prompt, the model
emits <tool_call> XML, we execute the tool, and loop.

Public entry point:
    run_prompt_tools_pipeline(constraint, query_id, solver_type, ...) -> dict
"""

from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import traceback
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from src.configs import config_parser as cfg
from src.utils.code_tools import (
    copy_policy_to_dir,
    load_constraints as _load_constraints,
    load_schema as _load_schema,
    search_tool,
    read_function_tool,
    parse_edit_suggestion,
    write_file,
    apply_edit_to_temp_dir,
)
from src.utils.test_runner import run_toy_test
from src.agents.main_agent import _INSTRUCTION as _MAIN_INSTRUCTION
from src.agents.generation_agent import _INSTRUCTION as _GA_INSTRUCTION
from src.agents.judgment_agent import _INSTRUCTION as _JA_INSTRUCTION
from src.agents.error_analysis_agent import _INSTRUCTION as _EAA_INSTRUCTION
from src.agents.revision_agent import _INSTRUCTION as _RA_INSTRUCTION


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

def _amplify_chat(
    messages: List[Dict[str, str]],
    *,
    model_id: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """Single call to Amplify /chat. Returns the response text.

    Retries up to 3 times on transient server errors (5xx HTTP codes or
    Amplify returning an error string as the response body).
    """
    import time as _time

    _TRANSIENT_MARKERS = (
        "can only concatenate str",
        "internal server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
    )

    payload = json.dumps({"data": {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "dataSources": [],
        "messages": messages,
        "options": {
            "skipRag": True,
            "model": {"id": model_id},
        },
    }}).encode()

    for attempt in range(4):  # up to 4 attempts (3 retries)
        req = urllib.request.Request(
            f"{base_url}/chat",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (500, 502, 503, 504) and attempt < 3:
                _time.sleep(2 ** attempt)  # 1s, 2s, 4s back-off
                continue
            raise

        if not body.get("success"):
            raise RuntimeError(f"Amplify error: {body.get('message', body)}")

        data = body.get("data", "")

        # Detect transient server-side error strings returned as data
        if isinstance(data, str) and any(m in data.lower() for m in _TRANSIENT_MARKERS):
            if attempt < 3:
                _time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Amplify transient error (gave up): {data[:200]}")

        if not isinstance(data, str):
            data = json.dumps(data)  # unexpected dict — convert so callers always get str

        return data

    raise RuntimeError("Amplify: max retries exceeded")


# ---------------------------------------------------------------------------
# Prompt-based tool calling
# ---------------------------------------------------------------------------

_TOOL_SYSTEM_PREFIX = """\
You have access to tools. To call a tool, emit:

<tool_call name="TOOL_NAME">
{"arg1": value1, "arg2": value2}
</tool_call>

The result will be given back to you. Call tools as needed, then provide your final answer.

Available tools:
"""

_TOOL_CALL_RE = re.compile(
    r'<tool_call\s+name=["\']?(\w+)["\']?>\s*(.*?)\s*</tool_call>',
    re.DOTALL,
)


def _format_tool_docs(tools: List[Callable]) -> str:
    """Build tool descriptions from function docstrings + signatures."""
    import inspect
    lines = []
    for fn in tools:
        sig = inspect.signature(fn)
        def _ann(p):
            if p.annotation == inspect.Parameter.empty:
                return "any"
            ann = p.annotation
            return getattr(ann, "__name__", str(ann))
        params = ", ".join(
            f"{name}: {_ann(p)}"
            + (f" = {p.default!r}" if p.default != inspect.Parameter.empty else "")
            for name, p in sig.parameters.items()
        )
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        lines.append(f"- {fn.__name__}({params}): {doc}")
    return "\n".join(lines)


def _run_tool_loop(
    system: str,
    user_message: str,
    tools: List[Callable],
    *,
    model_id: str,
    api_key: str,
    base_url: str,
    max_turns: int = 20,
    max_tokens: int = 4096,
    log_turns: bool = False,
) -> str:
    """Multi-turn tool loop for Amplify.

    Sends system + user message, parses <tool_call> tags from response,
    executes tools, and feeds results back until no more tool calls or
    max_turns is reached.
    """
    tool_map = {fn.__name__: fn for fn in tools}
    tool_docs = _format_tool_docs(tools)
    full_system = _TOOL_SYSTEM_PREFIX + tool_docs + "\n\n" + system

    # Amplify uses role=system in the messages array
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": user_message},
    ]

    last_text = ""
    for turn in range(max_turns):
        text = _amplify_chat(
            messages,
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
        last_text = text

        calls = _TOOL_CALL_RE.findall(text)
        if not calls:
            break  # No more tool calls — final answer

        if log_turns:
            for name, args_str in calls:
                try:
                    args = json.loads(args_str) if args_str.strip() else {}
                    first_val = next(iter(args.values()), "") if args else ""
                    hint = f'"{str(first_val)[:60]}"' if first_val else ""
                except Exception:
                    hint = ""
                print(f"    [turn {turn}] TOOL: {name}({hint})")

        # Append assistant turn + tool results
        messages.append({"role": "assistant", "content": text})

        result_parts = []
        for tool_name, args_str in calls:
            fn = tool_map.get(tool_name)
            if fn is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    args = json.loads(args_str) if args_str.strip() else {}
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}
            result_parts.append(
                f'<tool_result name="{tool_name}">\n{json.dumps(result, indent=2)}\n</tool_result>'
            )

        messages.append({"role": "user", "content": "\n\n".join(result_parts)})

    return last_text


def _run_simple(
    system: str,
    user_message: str,
    *,
    model_id: str,
    api_key: str,
    base_url: str,
    max_tokens: int = 4096,
) -> str:
    """Single-turn call (no tools) for GA and JA."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]
    return _amplify_chat(
        messages,
        model_id=model_id,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Verdict parser (shared with ADK pipeline)
# ---------------------------------------------------------------------------

def _parse_ja_verdict(text: str):
    """Parse JA verdict: choice: N|None + jud: explanation.

    Returns (choice_index_or_none, jud) where choice_index is 0-based
    (i.e. choice:1 → 0, choice:2 → 1, etc.) or None if no valid edit.
    """
    choice = None
    jud = ""
    for line in text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("choice:"):
            val = line.split(":", 1)[1].strip().lower()
            if val == "none":
                choice = None
            else:
                try:
                    choice = int(val) - 1  # convert 1-based to 0-based
                except ValueError:
                    choice = None
        elif line.lower().startswith("jud:"):
            jud = line.split(":", 1)[1].strip()
    return choice, jud


def _make_diff(e: dict) -> str:
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
# Pipeline
# ---------------------------------------------------------------------------

async def run_prompt_tools_pipeline(
    constraint: str,
    query_id: int,
    solver_type: str = "insertion",
    temp_code_dir: Optional[str] = None,
    logs_dir: Optional[str] = None,
    shared_serena=None,
) -> Dict[str, Any]:
    """Run the full five-agent pipeline using the Amplify prompt-tools backend."""

    logs_dir = logs_dir or f"/tmp/amplify_logs_{query_id}"
    os.makedirs(logs_dir, exist_ok=True)
    temp_code_dir = temp_code_dir or f"/tmp/amplify_temp_{query_id}"
    os.makedirs(temp_code_dir, exist_ok=True)

    def _log(msg: str):
        print(msg)
        with open(os.path.join(logs_dir, "pipeline.log"), "a") as f:
            f.write(msg + "\n")

    _log(f"=== Prompt-Tools Pipeline: {constraint[:80]} ({solver_type}) ===")

    # Credentials
    api_key = os.environ.get("AMPLIFY_API_KEY", "")
    base_url = os.environ.get("AMPLIFY_BASE_URL", "https://prod-api.vanderbilt.ai")
    model_id = os.environ.get("AMPLIFY_MODEL", "us.anthropic.claude-sonnet-4-20250514-v1:0")

    def _call(system, user, tools=None, max_turns=20, log_turns=False, max_tokens=4096):
        if tools:
            return _run_tool_loop(
                system, user, tools,
                model_id=model_id, api_key=api_key, base_url=base_url,
                max_turns=max_turns, max_tokens=max_tokens, log_turns=log_turns,
            )
        return _run_simple(system, user, model_id=model_id, api_key=api_key,
                           base_url=base_url, max_tokens=max_tokens)

    # Shared context
    env = os.environ.copy()
    constraint_content = _load_constraints()
    schema_content = _load_schema(".", solver_type)

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

    # Tool functions
    def search_code(query: str, limit: int = 5) -> dict:
        """Semantic search in the indexed policy codebase."""
        return search_tool(index_workspace=index_workspace, query=query,
                           limit=max(1, min(limit, 10)), env=env)

    def read_function(relative_path: str, function_name: str) -> dict:
        """Read the full source of a specific Python function."""
        return read_function_tool(work_dir=work_dir, policy_dir=policy_dir,
                                  relative_path=relative_path,
                                  function_name=function_name, env=env)

    def _serena_path(p: str) -> str:
        """Serena project root = policy dir, so strip leading 'policy/' prefix if present."""
        if p.startswith("policy/"):
            return p[len("policy/"):]
        return p

    def get_symbols_overview(relative_path: str) -> dict:
        """List ALL real function and class names in a file. Pass the filename or
        policy/filename.py — returns Function/Class lists you can pass to read_function."""
        if shared_serena:
            return shared_serena.call_tool("get_symbols_overview", {"relative_path": _serena_path(relative_path)})
        return {"error": "Serena not available"}

    def find_symbol(name_path_pattern: str, substring_matching: bool = True) -> dict:
        """Find a symbol by partial name across the codebase. Returns fqn, file, line."""
        if shared_serena:
            return shared_serena.call_tool("find_symbol", {
                "name_path_pattern": name_path_pattern,
                "substring_matching": substring_matching,
                "include_info": True,
            })
        return {"error": "Serena not available"}

    def find_referencing_symbols(name_path: str, relative_path: str) -> dict:
        """Find all symbols that reference (call) a given symbol. LSP-based callers list."""
        if shared_serena:
            return shared_serena.call_tool("find_referencing_symbols", {
                "name_path": name_path,
                "relative_path": _serena_path(relative_path),
            })
        return {"error": "Serena not available"}

    tools = [search_code, read_function, get_symbols_overview, find_symbol, find_referencing_symbols]

    _fmt = dict(
        constraints=f"<constraints>\n{constraint_content}\n</constraints>",
        schema=f"<schema>\n{schema_content}\n</schema>",
    )
    main_system = _MAIN_INSTRUCTION.format(**_fmt)
    ga_system   = _GA_INSTRUCTION.format(**_fmt)
    ja_system   = _JA_INSTRUCTION
    eaa_system  = _EAA_INSTRUCTION.format(**_fmt)
    ra_system   = _RA_INSTRUCTION.format(**_fmt)

    # ===================================================================
    # Phase 1: Main Agent explores code
    # ===================================================================
    _log("\n--- Phase 1: Main Agent explores code ---")

    # List only the policy files that actually exist — prevents the model from
    # trying invented paths like hexaly_solver.py, routing/hexaly_replan.py, etc.
    _policy_files = sorted(
        f for f in os.listdir(policy_dir) if f.endswith(".py") and f != "__init__.py"
    ) if os.path.isdir(policy_dir) else []
    _files_line = (
        f"The ONLY Python files in this codebase are: {', '.join(_policy_files)}. "
        "Do NOT attempt to read any other file path.\n\n"
        if _policy_files else ""
    )

    user_msg = (
        f"<constraint>\n{constraint}\n</constraint>\n\n"
        + _files_line
        + "Find the function(s) where this constraint should be enforced. "
        "Read each function, extract api_examples, output the injection_points JSON."
    )

    from src.core.pipeline import _parse_injection_points

    def _make_traced_tools(base_tools, trace_path, successful_reads):
        """Wrap tools to log call args + result to trace_path.
        Tracks files successfully read via read_function into successful_reads set.
        Corrects the LLM when it calls read_function with an invented path."""
        traced = []
        for fn in base_tools:
            def _wrap(f=fn):
                import functools
                @functools.wraps(f)
                def _traced(**kwargs):
                    result = f(**kwargs)
                    if f.__name__ == "read_function":
                        if result.get("ok"):
                            successful_reads.add(os.path.basename(kwargs.get("relative_path", "")))
                        elif "File not found" in result.get("error", ""):
                            result = {
                                "ok": False,
                                "error": (
                                    f"File not found: {kwargs.get('relative_path')}. "
                                    "IMPORTANT: This file does not exist in the codebase. "
                                    "You must only call read_function with paths returned by "
                                    "search_code or find_symbol. Do NOT guess or invent file paths. "
                                    "Use search_code to find the correct file."
                                ),
                            }
                        elif "not found in" in result.get("error", ""):
                            fname = kwargs.get("function_name", "")
                            fpath = kwargs.get("relative_path", "")
                            mod = os.path.splitext(os.path.basename(fpath))[0]
                            result = {
                                "ok": False,
                                "error": (
                                    f"Function '{fname}' does not exist in {fpath}. "
                                    "IMPORTANT: Do NOT guess function names. "
                                    f"Call find_symbol(module='{mod}') to list ALL real "
                                    f"function names in {fpath}, then use the exact name returned."
                                ),
                            }
                    if trace_path:
                        with open(trace_path, "a") as tf:
                            tf.write(f"\n>>> {f.__name__}({json.dumps(kwargs, indent=2)})\n")
                            tf.write(f"<<< {json.dumps(result, indent=2)}\n")
                    return result
                return _traced
            traced.append(_wrap())
        return traced

    injection_points = []
    for explore_attempt in range(1, 4):  # up to 3 attempts if no IPs found
        trace_file = os.path.join(logs_dir, f"explore_trace{'_' + str(explore_attempt) if explore_attempt > 1 else ''}.txt") if logs_dir else None
        successful_reads: set = set()
        traced_tools = _make_traced_tools(tools, trace_file, successful_reads)
        try:
            explore_text = _call(main_system, user_msg, tools=traced_tools,
                                 max_turns=10, log_turns=True)
        except Exception as e:
            _log(f"  Main Agent attempt {explore_attempt} error: {e}")
            if explore_attempt == 3:
                _log(traceback.format_exc())
                return {"success": False, "diff": f"Explore failed: {e}"}
            continue

        write_file(logs_dir, f"explore_output{'_' + str(explore_attempt) if explore_attempt > 1 else ''}.txt",
                   explore_text)
        injection_points = _parse_injection_points(explore_text)

        # Drop any IP whose file was never successfully read — prevents hallucinated paths
        if successful_reads:
            injection_points = [
                ip for ip in injection_points
                if os.path.basename(ip.get("file", "")) in successful_reads
            ]

        _log(f"  Explore attempt {explore_attempt}: found {len(injection_points)} injection point(s)")
        if injection_points:
            break
        _log("  No injection points parsed — retrying explore")

    if not injection_points:
        return {"success": False, "diff": "No injection points found"}

    # Attach full source to each injection point
    for ip in injection_points:
        fname, func = ip.get("file", ""), ip.get("function", "")
        if fname and func and not ip.get("source"):
            result = read_function(fname, func)
            if result.get("ok"):
                ip["source"] = result["functionSource"]

    for i, ip in enumerate(injection_points):
        _log(f"  IP{i+1}: {ip.get('function')} in {ip.get('file')} [{ip.get('role')}]")

    # ===================================================================
    # Phase 2: Generator → Verifier loop
    # ===================================================================
    _log("\n--- Phase 2: Generator -> Verifier loop ---")

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

    edit = None
    winner_diff = ""
    prev_edit_text = None
    jud_feedback = None
    MAX_GEN_ROUNDS = 10
    N_CANDIDATES = 3  # parallel GAs for majority voting on first pass

    import concurrent.futures as _cf

    def _apply_edit_to_source(source: str, edit_text: str) -> str:
        """Return source with old_text replaced by new_text from the edit, or original on failure."""
        import re as _re
        olds = _re.findall(r"<old_text>\n?(.*?)\n?</(?:new|old)_text>", edit_text, _re.DOTALL)
        news = _re.findall(r"<new_text>\n?(.*?)\n?</(?:new|old)_text>", edit_text, _re.DOTALL)
        result = source
        for old, new in zip(olds, news):
            if old in result:
                result = result.replace(old, new, 1)
        return result

    def _build_ja_prompt(candidates: list) -> str:
        """Build a JA prompt presenting all GA candidates with their edits shown in context."""
        parts = [f"## Constraint to implement\n{constraint}\n"]
        for i, et in enumerate(candidates):
            # Show the edit applied to the full function source so placement is visible
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

    def _run_ja_vote(ja_prompt: str) -> tuple:
        """Run one JA that reviews all candidates. Returns (choice_0based, jud)."""
        try:
            ja_text = _call(ja_system, ja_prompt)
        except Exception:
            return None, ""  # JA error — abstain
        return _parse_ja_verdict(ja_text)

    for gen_round in range(1, MAX_GEN_ROUNDS + 1):
        # 3 parallel GAs
        feedback_block = (
            f"\n## Previous attempt feedback\n{jud_feedback}\n\n"
            "Fix the issues described above."
            if jud_feedback else ""
        )
        prompt = context_package + feedback_block + "\nGenerate the edit(s)."

        def _run_ga(_):
            return _call(ga_system, prompt, tools=None)

        with _cf.ThreadPoolExecutor(max_workers=N_CANDIDATES) as exc:
            ga_futures = [exc.submit(_run_ga, i) for i in range(N_CANDIDATES)]
            ga_texts = []
            for i, f in enumerate(ga_futures):
                try:
                    ga_texts.append(f.result())
                except Exception as e:
                    _log(f"  GEN candidate {i+1} error: {e}")
                    ga_texts.append(None)

        # Log and save each GA output
        valid_texts = []
        valid_idx = []
        for i, et in enumerate(ga_texts):
            write_file(logs_dir, f"edit_round_{gen_round}_gen{i+1}.txt", et or "")
            cand = parse_edit_suggestion(et, logs_dir) if et else None
            if cand:
                valid_texts.append(et)
                valid_idx.append(i)
            else:
                _log(f"  GEN candidate {i+1}: no valid XML")

        if not valid_texts:
            _log(f"  Round {gen_round}: all GAs produced no valid XML — retrying")
            jud_feedback = "All candidates failed to produce valid XML edit blocks. Output ONLY the XML edit format with <relative_path>, <old_text>, <new_text>, <explanation> tags."
            continue

        # 3 parallel JAs each reviewing ALL valid candidates
        ja_prompt = _build_ja_prompt(valid_texts)
        with _cf.ThreadPoolExecutor(max_workers=N_CANDIDATES) as exc:
            vote_futures = [exc.submit(_run_ja_vote, ja_prompt) for _ in range(N_CANDIDATES)]
            votes = [f.result() for f in vote_futures]

        # Tally votes (choices are 0-based indices into valid_texts)
        from collections import Counter
        vote_counts = Counter(c for c, _ in votes if c is not None and 0 <= c < len(valid_texts))
        _log(f"  JA votes: {dict(vote_counts)} (out of {len(valid_texts)} valid candidates)")

        # Pick candidate with majority (2+) votes
        winner_idx = next((idx for idx, cnt in vote_counts.most_common() if cnt >= 2), None)

        if winner_idx is not None:
            winning_text = valid_texts[winner_idx]
            edit = parse_edit_suggestion(winning_text, logs_dir)
            winner_diff = _make_diff(edit)
            _log(f"  Majority winner: candidate {valid_idx[winner_idx] + 1} ({vote_counts[winner_idx]}/3 votes)")
            break

        # No majority — collect feedback and retry all 3 GAs
        jud_feedback = next(
            (jud for _, jud in votes if jud and jud.lower() != "none"),
            "No majority — all edits had issues"
        )
        _log(f"  No majority in round {gen_round}, retrying with feedback")

    if not edit:
        return {"success": False, "diff": "No edit produced"}

    # ===================================================================
    # Phase 3: Apply edit
    # ===================================================================
    _log("\n--- Phase 3: Apply + Test ---")

    if not apply_edit_to_temp_dir(edit, temp_code_dir, env, logs_dir):
        _log("  Apply FAILED")
        return {"success": False, "diff": "Edit application failed"}
    n_blocks = 1 + len(edit.get("extra_edits", []))
    _log(f"  Applied {n_blocks} edit block(s) to {edit.get('path', '?')}")

    # ===================================================================
    # Phase 4: Correction loop (EAA → Generator → Verifier)
    # ===================================================================
    _log("\n--- Phase 4: Correction loop ---")

    MAX_TEST_ATTEMPTS = 10
    MAX_FIX_ROUNDS = 5

    # Phase 4 uses tools that read from the TEMP directory so EAA/GEN see the
    # already-modified files (not the originals), giving correct old_text.
    temp_policy_dir = os.path.join(temp_code_dir, "policy")

    def read_function_temp(relative_path: str, function_name: str) -> dict:
        """Read the full source of a function from the modified temp policy directory."""
        return read_function_tool(work_dir=temp_code_dir, policy_dir=temp_policy_dir,
                                  relative_path=relative_path,
                                  function_name=function_name, env=env)

    fix_tools = [search_code, read_function_temp, get_symbols_overview, find_symbol, find_referencing_symbols]

    for test_attempt in range(1, MAX_TEST_ATTEMPTS + 1):
        result = run_toy_test(query_id, solver_type, temp_code_dir)

        if result["passed"]:
            _log(f"  Test attempt {test_attempt}: PASS")
            return {"success": True, "diff": winner_diff, "injection_points": injection_points,
                    "edit": edit, "temp_code_dir": temp_code_dir}

        test_output = result.get("output", "")
        _log(f"  Test attempt {test_attempt}: FAIL")
        write_file(logs_dir, f"test_output_{test_attempt}.txt", test_output)

        error_summary = test_output

        eaa_prompt = (
            f"## Constraint\n{constraint}\n\n"
            f"## Test error\n```\n{error_summary}\n```\n\n"
            f"## Previous edit\n{prev_edit_text}\n\n"
            "IMPORTANT: The policy files have already been modified. Use read_function_temp to read "
            "the CURRENT state of any function before writing old_text — it may differ from the original. "
            "Diagnose the failure mode, read the controlling function, output the corrected edit."
        )
        try:
            eaa_text = _call(eaa_system, eaa_prompt, tools=fix_tools, log_turns=True)
            write_file(logs_dir, f"eaa_output_{test_attempt}.txt", eaa_text)
            _log(f"  EAA: {eaa_text[:200]}")
        except Exception as e:
            _log(f"  EAA error: {e}")
            continue

        fix_jud_feedback = None
        fix_edit = None
        fix_diff = ""
        fix_prev_text = None

        for fix_round in range(1, MAX_FIX_ROUNDS + 1):
            if fix_jud_feedback is None:
                fix_prompt = (
                    context_package
                    + f"\n## Previous edit (caused test failure)\n{prev_edit_text}\n\n"
                    + f"## Error analysis\n{eaa_text}\n\n"
                    + "IMPORTANT: Files are already modified — use read_function_temp to read the "
                    "CURRENT state before writing old_text. Generate the corrected edit."
                )
            else:
                fix_prompt = (
                    context_package
                    + f"\n## Previous fix attempt\n{fix_prev_text}\n\n"
                    + f"## Verifier feedback\n{fix_jud_feedback}\n\n"
                    + f"## Error context\n{eaa_text}\n\n"
                    + "IMPORTANT: Use read_function_temp to read current file state before writing old_text. "
                    "Revise the fix to address all issues."
                )
            try:
                fix_text = _call(ra_system, fix_prompt, tools=fix_tools)
                write_file(logs_dir, f"fix_{test_attempt}_{fix_round}.txt", fix_text)
                fix_prev_text = fix_text
                candidate_fix = parse_edit_suggestion(fix_text, logs_dir)
            except Exception as e:
                _log(f"  Fix round {fix_round} error: {e}")
                break

            if not candidate_fix:
                _log(f"  Fix round {fix_round}: no valid edit — retrying")
                continue
            fix_diff = _make_diff(candidate_fix)
            fix_edit = candidate_fix

            source_summary = "\n\n".join(
                f"### {ip.get('function')} ({ip.get('file')})\n```python\n{ip.get('source', '')}\n```"
                for ip in injection_points if ip.get("source")
            )
            ja_fix_prompt = (
                f"## Constraint\n{constraint}\n\n"
                + (f"## Function source(s)\n{source_summary}\n\n" if source_summary else "")
                + f"## Proposed edit 1\n{fix_text}\n\nReview all proposed edits and choose the best correct one."
            )
            try:
                ja_fix_text = _call(ja_system, ja_fix_prompt)
                write_file(logs_dir, f"ja_fix_{test_attempt}_{fix_round}.txt", ja_fix_text)
                choice, jud = _parse_ja_verdict(ja_fix_text)
                right = choice is not None
                _log(f"  Fix round {fix_round} [RA->JA]: {'approved' if right else f'rejected: {jud}'}")
            except Exception as e:
                _log(f"  JA fix error (proceeding): {e}")
                right, jud = True, ""

            if right:
                break
            fix_jud_feedback = jud

        if fix_edit and apply_edit_to_temp_dir(fix_edit, temp_code_dir, env, logs_dir):
            n_blocks = 1 + len(fix_edit.get("extra_edits", []))
            _log(f"  Fix applied ({n_blocks} block(s)) to {fix_edit.get('path', '?')}")
            winner_diff = fix_diff
            prev_edit_text = fix_prev_text or fix_diff
        else:
            _log("  No valid fix or apply failed — continuing")

    return {"success": False, "diff": winner_diff, "injection_points": injection_points}
