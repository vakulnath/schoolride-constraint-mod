"""
API agent for constraint implementation with native tool-calling.

Uses LLM function-calling and executes local tools that bridge to:
- Claude Context MCP (semantic search)
- Filesystem MCP (read/edit policy files)

Return shape matches run_claude_agent() for pipeline compatibility.
"""

import difflib
import hashlib
import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.clients.fs_mcp import fs_edit_file, fs_read_file
from src.clients.mcp_client import extract_text, mcp_search
from src.configs.config_parser import (
    INDEX_WORKSPACE, HEXALY_INDEX_WORKSPACE, CONSTRAINTS_FILE,
    POLICY_DIR, TEMP_CODE_DIR, OUTPUT_DIR, LOGS_DIR, AGENT_LOG, LLM_EDIT_LOG, SCHEDULE_DIFF_LOG
)
from src.utils.utils import dedupe_results


def _log(logs_dir: Optional[str], filename: str, content: str):
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, filename), "a") as f:
            f.write(content + "\n")


def _write_file(logs_dir: Optional[str], filename: str, content: str):
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, filename), "w") as f:
            f.write(content)


def _truncate(text: Any, limit: int = 300) -> str:
    if text is None:
        return ""
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"... <truncated {len(s) - limit} chars>"


def _summarize_tool_args(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    if name == "apply_edit":
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        return {
            "relative_path": args.get("relative_path"),
            "explanation": _truncate(args.get("explanation", ""), 220),
            "old_text_chars": len(old_text),
            "old_text_preview": _truncate(old_text, 220),
            "new_text_chars": len(new_text),
            "new_text_preview": _truncate(new_text, 220),
        }
    return args


def _summarize_tool_result(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    if name == "search_code":
        top: List[Dict[str, Any]] = []
        for s in (result.get("results") or [])[:5]:
            top.append(
                {
                    "relativePath": s.get("relativePath"),
                    "functionName": s.get("functionName"),
                    "lines": f"{s.get('startLine')}-{s.get('endLine')}",
                    "nodeType": s.get("nodeType"),
                    "className": s.get("className"),
                    "callsTo": s.get("callsTo"),
                    "calledBy": s.get("calledBy"),
                    "docstring": s.get("docstring"),
                    "content_preview": _truncate(s.get("content", ""), 220),
                }
            )
        return {"count": result.get("count", 0), "top": top}

    if name == "read_function":
        source = str(result.get("functionSource", ""))
        return {
            "ok": result.get("ok"),
            "path": result.get("path"),
            "fileName": result.get("fileName"),
            "functionName": result.get("functionName"),
            "function_chars": len(source),
            "function_preview": _truncate(source, 300),
            "error": result.get("error"),
        }

    if name == "apply_edit":
        return {
            "ok": result.get("ok"),
            "path": result.get("path"),
            "fileName": result.get("fileName"),
            "explanation": _truncate(result.get("explanation", ""), 220),
            "error": result.get("error"),
        }

    return result


def _load_constraints() -> str:
    if CONSTRAINTS_FILE.exists():
        with open(CONSTRAINTS_FILE, "r") as f:
            return f.read()
    return "(Constraints not available)"


def _load_schema(context_root: str, solver_type: str = "insertion") -> str:
    schema_name = "hexaly.py" if solver_type == "hexaly" else "insertion.py"
    schema_path = os.path.join(context_root, schema_name)
    if not os.path.exists(schema_path):
        for root, _dirs, files in os.walk(context_root):
            if schema_name in files and "schemas" in root:
                schema_path = os.path.join(root, schema_name)
                break

    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            return f.read()

    try:
        spec_name = "school_bus_routing.schemas.hexaly" if solver_type == "hexaly" else "school_bus_routing.schemas.insertion"
        spec = importlib.util.find_spec(spec_name)
        if spec and spec.origin:
            with open(spec.origin, "r") as f:
                return f.read()
    except Exception:
        pass
    return "(Schema not available)"


def _snapshot_files(directory: str) -> Dict[str, str]:
    snapshots: Dict[str, str] = {}
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
    if not os.path.isdir(policy_dir):
        return None

    for fname, old_hash in before_snapshots.items():
        fpath = os.path.join(policy_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, "rb") as f:
            new_hash = hashlib.md5(f.read()).hexdigest()
        if new_hash != old_hash:
            orig_path = os.path.join(original_dir, fname)
            if os.path.exists(orig_path):
                with open(orig_path, "r") as f:
                    orig_lines = f.readlines()
                with open(fpath, "r") as f:
                    new_lines = f.readlines()
                diff = "".join(
                    difflib.unified_diff(
                        orig_lines,
                        new_lines,
                        fromfile=f"a/{fname}",
                        tofile=f"b/{fname}",
                    )
                )
            else:
                diff = "(original not available for diff)"

            return {
                "path": fname,
                "success": True,
                "diff": diff,
            }
    return None


def _resolve_target_file(work_dir: str, policy_dir: str, relative_path: str) -> str:
    target = os.path.join(policy_dir, os.path.basename(relative_path))
    if os.path.isfile(target):
        return target
    return os.path.join(work_dir, relative_path)


def _extract_function(source: str, function_name: str) -> Optional[str]:
    lines = source.splitlines(keepends=True)
    start_idx = None
    base_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {function_name}("):
            start_idx = i
            base_indent = len(line) - len(stripped)
            break

    if start_idx is None:
        return None

    # First, skip past the function signature (may span multiple lines until we
    # find the colon that ends it).  We look for a line whose stripped form ends
    # with ":".
    body_start = start_idx + 1
    for i in range(start_idx, len(lines)):
        if lines[i].rstrip().endswith(":"):
            body_start = i + 1
            break

    # Now collect the body: every line that is blank or indented more than
    # base_indent.
    end_idx = body_start
    for i in range(body_start, len(lines)):
        line = lines[i]
        if not line.strip():
            end_idx = i + 1
            continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent <= base_indent:
            break
        end_idx = i + 1

    return "".join(lines[start_idx:end_idx])


def _fix_indentation(file_path: str, old_text: str, new_text: str) -> Tuple[str, str]:
    with open(file_path, "r") as f:
        file_content = f.read()

    anchor = old_text.split("\n")[0].strip()
    if not anchor:
        return old_text, new_text

    file_indent = ""
    for line in file_content.split("\n"):
        if line.strip() == anchor:
            file_indent = line[: len(line) - len(line.lstrip())]
            break

    if not file_indent:
        return old_text, new_text

    first_line = old_text.split("\n")[0]
    if len(first_line) - len(first_line.lstrip()) == len(file_indent):
        return old_text, new_text

    def add_indent(text: str) -> str:
        result = []
        for line in text.split("\n"):
            result.append(file_indent + line if line.strip() else line)
        return "\n".join(result)

    return add_indent(old_text), add_indent(new_text)


# Pricing per 1M tokens (USD). Update when models change.
_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-3-pro-preview":   {"input": 1.25, OUTPUT_DIR: 10.00},
    "gemini-3-flash-preview": {"input": 0.15, OUTPUT_DIR: 3.50},
    "gemini-2.5-flash":       {"input": 0.15, OUTPUT_DIR: 3.50},
    "gemini-2.5-pro":         {"input": 1.25, OUTPUT_DIR: 10.00},
    "gemini-flash-latest":    {"input": 0.15, OUTPUT_DIR: 3.50},
    "gemini-pro-latest":      {"input": 1.25, OUTPUT_DIR: 10.00},
}
_DEFAULT_PRICING = {"input": 1.25, OUTPUT_DIR: 10.00}

# Fallback model chain: on 403 (quota/rate-limit), walk down to the next model.
# Order: best → worst.  gemini-3-pro → 3-flash → 2.5-pro → 2.5-flash.
_FALLBACK_CHAIN: Dict[str, str] = {
    "gemini-3-flash-preview": "gemini-3-pro-preview",  # Flash is now primary, pro is fallback
    "gemini-3-pro-preview": "gemini-2.5-pro",
    "gemini-2.5-pro": "gemini-2.5-flash",
    "gemini-pro-latest": "gemini-flash-latest",
    "gemini-flash-latest": "gemini-2.5-pro",
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICING.get(model, _DEFAULT_PRICING)
    return (input_tokens * prices["input"] + output_tokens * prices[OUTPUT_DIR]) / 1_000_000


def _resolve_model(env: Dict[str, str]) -> str:
    return (
        env.get("GEMINI_TOOLS_MODEL")
        or env.get("GEMINI_MODEL_EDIT")
        or env.get("GEMINI_MODEL")
        or "gemini-3-pro-preview"
    )


# ── Gemini Context Caching ─────────────────────────────────────────────
# Caches systemInstruction + tools so they aren't re-sent every turn.
# 90% discount on cached input tokens.  Persists across queries within
# the same process via _CACHE_REGISTRY.

_CACHE_REGISTRY: Dict[str, str] = {}  # (model:solver_type) -> cache name


def _create_cached_content(
    api_key: str,
    model: str,
    system_prompt: str,
    tools: List[Dict[str, Any]],
    ttl: str = "600s",
) -> Optional[str]:
    """Create a Gemini cached content for the system prompt and tools.

    Returns the cache resource name (e.g. 'cachedContents/xxx') or None.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/cachedContents"
    payload = {
        "model": f"models/{model}",
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": tools,
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        "ttl": ttl,
    }
    try:
        resp = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("name")
        return None
    except Exception:
        return None


def _get_or_create_cache(
    api_key: str,
    model: str,
    solver_type: str,
    system_prompt: str,
    tools: List[Dict[str, Any]],
    logs_dir: Optional[str] = None,
) -> Optional[str]:
    """Get existing cache or create a new one.  Keyed by model + solver_type."""
    cache_key = f"{model}:{solver_type}"
    if cache_key in _CACHE_REGISTRY:
        return _CACHE_REGISTRY[cache_key]

    cache_name = _create_cached_content(api_key, model, system_prompt, tools)
    if cache_name:
        _CACHE_REGISTRY[cache_key] = cache_name
        _log(logs_dir, AGENT_LOG, f"Created context cache: {cache_name}")
    else:
        _log(logs_dir, AGENT_LOG,
             "Context caching unavailable (content may be below minimum) — using inline prompt")
    return cache_name


def api_single_turn(
    user_prompt: str,
    system_prompt: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
    log_prefix: str = "agent",
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Single-turn API call with no tools.  Used by JA and EAA agents."""
    api_key = env.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return {
            "text": "",
            "error": "GEMINI_API_KEY not set",
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

    model = _resolve_model(env)

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": int(max_tokens),
        },
    }

    _log(logs_dir, f"{log_prefix}.log", f"=== {log_prefix} (single-turn) ===")
    _log(logs_dir, f"{log_prefix}.log", f"Model: {model}")

    if logs_dir:
        _write_file(logs_dir, f"{log_prefix}_prompt.txt", f"## System\n{system_prompt}\n\n## User\n{user_prompt}")

    # Try primary model, then walk the fallback chain on 403.
    response = None
    last_error: Optional[Exception] = None
    current_model = model
    while True:
        try:
            response = _api_generate(api_key=api_key, model=current_model, payload=payload)
            model = current_model  # persist whichever model succeeded
            break
        except Exception as exc:
            last_error = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            is_rate_limit = (
                isinstance(exc, requests.exceptions.HTTPError)
                and status_code in (403, 429)  # 403 Forbidden, 429 Too Many Requests
            )
            fallback = _FALLBACK_CHAIN.get(current_model) if is_rate_limit else None
            if fallback:
                _log(logs_dir, f"{log_prefix}.log",
                     f"Rate limit ({status_code}) on {current_model} — falling back to {fallback}")
                current_model = fallback
                continue
            _log(logs_dir, f"{log_prefix}.log", f"API error: {exc}")
            return {
                "text": "",
                "error": str(exc),
                "cost_usd": 0,
                "input_tokens": 0,
                "output_tokens": 0,
            }

    usage = response.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    cost = _estimate_cost(model, input_tokens, output_tokens)

    candidates = response.get("candidates") or []
    text = ""
    finish_reason = ""
    if candidates:
        text = _extract_text_from_candidate(candidates[0])
        finish_reason = candidates[0].get("finishReason", "")
        # Log safety / blocking metadata for debugging empty responses
        safety_ratings = candidates[0].get("safetyRatings", [])
        if not text or finish_reason not in ("STOP", "MAX_TOKENS", ""):
            _log(logs_dir, f"{log_prefix}.log",
                 f"WARNING: empty/blocked response — finishReason={finish_reason}, "
                 f"safetyRatings={safety_ratings}")
    else:
        # No candidates at all — check promptFeedback for blocking
        prompt_feedback = response.get("promptFeedback", {})
        if prompt_feedback:
            _log(logs_dir, f"{log_prefix}.log",
                 f"WARNING: no candidates — promptFeedback={prompt_feedback}")

    _log(logs_dir, f"{log_prefix}.log", f"Response ({len(text)} chars): {text[:500]}")
    _log(logs_dir, f"{log_prefix}.log", f"Tokens: in={input_tokens} out={output_tokens} cost=${cost:.4f}")

    return {
        "text": text,
        "finish_reason": finish_reason,
        "cost_usd": cost,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _extract_text_from_candidate(candidate: Dict[str, Any]) -> str:
    content = (candidate.get("content") or {})
    parts = content.get("parts") or []
    chunks: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text", "")
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def _build_api_payload(
    contents: List[Dict[str, Any]],
    system_prompt: str,
    tools: List[Dict[str, Any]],
    temperature: float = 0.0,
    cached_content: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
        },
    }
    if cached_content:
        # System prompt + tools are in the cache; don't re-send them.
        payload["cachedContent"] = cached_content
    else:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        payload["tools"] = tools
        payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
    return payload


def _api_generate(
    api_key: str,
    model: str,
    payload: Dict[str, Any],
    max_retries: int = 3,
) -> Dict[str, Any]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url,
                headers={
                    "x-goog-api-key": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=300,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            continue
        if resp.status_code >= 400:
            error_detail = resp.text if resp.text else resp.json().get("error", {}).get("message", "Unknown error")
            error_msg = f"HTTP {resp.status_code}: {error_detail}"
            raise requests.exceptions.HTTPError(error_msg, response=resp)
        return resp.json()
    if last_exc:
        raise last_exc
    resp.raise_for_status()  # type: ignore[possibly-undefined]
    return resp.json()  # type: ignore[possibly-undefined]


def _search_tool(
    index_workspace: str,
    query: str,
    limit: int,
    env: Dict[str, str],
) -> Dict[str, Any]:
    raw = mcp_search(index_workspace, query=query, limit=limit, env=env)
    parsed = dedupe_results(extract_text(raw))
    compact: List[Dict[str, Any]] = []
    for s in parsed[:limit]:
        compact.append(
            {
                "relativePath": s.get("relativePath"),
                "functionName": s.get("functionName"),
                "startLine": s.get("startLine"),
                "endLine": s.get("endLine"),
                "nodeType": s.get("nodeType"),
                "className": s.get("className"),
                "callsTo": s.get("callsTo"),
                "calledBy": s.get("calledBy"),
                "docstring": s.get("docstring"),
                "content": (s.get("content") or "")[:1400],
            }
        )
    return {"count": len(compact), "results": compact}


def _read_function_tool(
    work_dir: str,
    policy_dir: str,
    relative_path: str,
    function_name: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    target_file = _resolve_target_file(work_dir, policy_dir, relative_path)
    if not os.path.isfile(target_file):
        return {"ok": False, "error": f"File not found: {target_file}"}
    full_source = fs_read_file(target_file, env=env)
    func_source = _extract_function(full_source, function_name)
    if not func_source:
        return {
            "ok": False,
            "error": f"Function '{function_name}' not found in {os.path.basename(target_file)}",
            "path": target_file,
        }
    return {
        "ok": True,
        "path": target_file,
        "fileName": os.path.basename(target_file),
        "functionName": function_name,
        "functionSource": func_source,
        "functionSourceTruncated": False,
    }



def _apply_edit_tool(
    work_dir: str,
    policy_dir: str,
    relative_path: str,
    old_text: str,
    new_text: str,
    explanation: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    target_file = _resolve_target_file(work_dir, policy_dir, relative_path)
    if not os.path.isfile(target_file):
        return {"ok": False, "error": f"File not found: {target_file}"}

    full_source = fs_read_file(target_file, env=env)
    if old_text not in full_source:
        fixed_old, fixed_new = _fix_indentation(target_file, old_text, new_text)
        if fixed_old in full_source:
            old_text, new_text = fixed_old, fixed_new
        else:
            return {
                "ok": False,
                "error": "oldText not found in target file (even after indentation fix)",
                "path": target_file,
            }

    result = fs_edit_file(
        path=target_file,
        edits=[{"oldText": old_text, "newText": new_text}],
        dry_run=False,
        env=env,
    )
    error = result.get("error") or result.get("isError")
    if error:
        return {"ok": False, "error": str(error), "path": target_file}

    return {
        "ok": True,
        "path": target_file,
        "fileName": os.path.basename(target_file),
        "explanation": explanation or "",
    }


def _tool_declarations() -> List[Dict[str, Any]]:
    return [
        {
            "functionDeclarations": [
                {
                    "name": "search_code",
                    "description": (
                        "Semantic search in indexed policy code via Claude Context MCP. "
                        "Use a generic query (no concrete stop IDs/bus IDs)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "read_function",
                    "description": (
                        "Read a file and extract a named Python function using Filesystem MCP."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {"type": "string"},
                            "function_name": {"type": "string"},
                        },
                        "required": ["relative_path", "function_name"],
                    },
                },
                {
                    "name": "apply_edit",
                    "description": (
                        "Apply a minimal oldText/newText edit to a policy file via Filesystem MCP."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "relative_path": {"type": "string"},
                            "old_text": {"type": "string"},
                            "new_text": {"type": "string"},
                            "explanation": {"type": "string"},
                        },
                        "required": ["relative_path", "old_text", "new_text"],
                    },
                },
            ]
        }
    ]


def run_api_agent(
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
    """
    Run API agent with native function-calling and local MCP-backed tools.

    Args:
        solver_type: "insertion" (default) or "hexaly". Controls which
            index workspace, contract, and system prompt the agent uses.
    """
    api_key = env.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        return {
            "success": False,
            "error": "GEMINI_API_KEY not set",
            "edit": None,
            "thoughts": [],
            "iterations": 0,
            "cost_usd": 0,
        }

    model = model_override if model_override else _resolve_model(env)

    if solver_type == "hexaly":
        index_workspace = str(HEXALY_INDEX_WORKSPACE)
    else:
        index_workspace = str(INDEX_WORKSPACE)
    constraint_content = _load_constraints()
    schema_content = _load_schema(context_root, solver_type)

    work_dir = temp_code_dir or context_root
    policy_dir = os.path.join(work_dir, POLICY_DIR)
    if not os.path.isdir(policy_dir):
        return {
            "success": False,
            "error": f"No policy directory found at {policy_dir}",
            "edit": None,
            "thoughts": [],
            "iterations": 0,
            "cost_usd": 0,
        }

    before = _snapshot_files(policy_dir)
    thoughts: List[str] = []

    system_prompt = f"""\
<role>
You are a precise Python coding agent that implements routing constraints
by making minimal edits to policy source files.
You are analytical, methodical, and persistent.
</role>

<instructions>
Before taking any action (tool call or response), reason through these steps:

1) Logical dependencies and constraints:
   - Analyze the constraint against the contract rules below.
   - Determine order of operations: which function to search, read, then edit.
   - Identify which variables must be in scope at the edit location.
   - Ensure the edit does not violate any existing hard constraint (C1-C6)
     listed in the <constraints> section below.

2) Risk assessment:
   - search_code and read_function are LOW-RISK reads. Call them freely
     with available information rather than deliberating.
   - apply_edit is HIGH-RISK (state-changing). Before calling it:
     a) Confirm you have read the target function source.
     b) Verify old_text matches the file exactly (indentation matters).
     c) Verify new_text preserves all existing logic and only adds the
        new constraint guard.

3) Persistence and recovery:
   - If a tool returns an error, change your strategy — do NOT repeat the
     same call with the same arguments.
   - If the function you expected is not found, search for alternative
     functions or locations where the needed variables are in scope.
   - Do not give up until you have exhausted all reasonable approaches.

4) Precision:
   - Make MINIMAL edits: typically a guard clause, continue, or early return.
   - Do NOT rewrite entire functions or refactor unrelated code.
   - Use exact variable names and indentation from the source you read.
   - Quote the smallest unique old_text snippet needed to anchor the edit.
</instructions>

<tools>
- search_code(query, limit): Semantic search over indexed policy code.
  Craft a query that describes the OPERATION the constraint requires —
  do NOT include concrete stop/bus IDs (the index is code, not data).
  Examples:
    - "assign specific stop to a specific bus" (for stop→bus assignment)
    - "limit number of stops per bus" (for max-stops constraint)
    - "enforce ordering between two stops on a route" (for precedence)
    - "bus load capacity check" (for capacity constraint)
    - "place two stops on the same bus" (for co-location)
  LOW-RISK — call freely.
- read_function(relative_path, function_name): Read exact function source
  from a policy file. LOW-RISK.
- apply_edit(relative_path, old_text, new_text, explanation): Apply a
  minimal old_text/new_text replacement to a policy file. HIGH-RISK.
</tools>

<workflow>
1. Search: call search_code with a query tailored to the constraint.
   Describe the operation/behavior the constraint requires, NOT a fixed
   phrase.  Do NOT use concrete IDs in the query.
2. Read: pick the most relevant function where the needed variables are in
   scope (check the docstring's Scope section). Call read_function to get
   the full source.
3. Plan: identify the exact insertion point. Determine old_text (existing
   lines) and new_text (existing lines + new guard).
4. Edit: call apply_edit with the minimal change.
5. Done: provide a short summary of what changed and why it satisfies the
   constraint without violating C1-C6.
</workflow>

<constraints>
{constraint_content}
</constraints>

<schema>
{schema_content}
</schema>"""

    user_prompt = (
        "<task>\n"
        "Implement this constraint by editing policy code:\n\n"
        f"{constraint}\n"
        "</task>\n\n"
        "<final_instruction>\n"
        "Use tools to search, inspect code, and apply exactly one minimal edit.\n"
        "Think step-by-step: search -> read -> plan -> edit -> summarize.\n"
        "\n"
        "AFTER successfully editing the policy code, also GENERATE A PYTEST FUNCTION that validates\n"
        "the constraint is satisfied in the output schedule. The pytest should:\n"
        "  1. Load the modified bus_schedule_after.json from the toy test output\n"
        "  2. Verify the constraint is satisfied (e.g., check if stop is on correct bus, capacity respected, etc.)\n"
        "  3. Assert with a clear error message\n"
        "  4. Be enclosed in triple backticks as ```python ... ```\n"
        "\n"
        "Format the pytest like this:\n"
        "```python\n"
        "def test_constraint_query_ID(modified_schedule):\n"
        "    '''Validate constraint is satisfied in output schedule'''\n"
        "    # Load or receive schedule data\n"
        "    # Verify constraint\n"
        "    # Assert\n"
        "```\n"
        "</final_instruction>"
    )

    tools = _tool_declarations()
    contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": user_prompt}]}]
    # Inject pre-fetched context (e.g., from Main Agent's MCP search phase)
    if prefetched_context:
        contents = prefetched_context + contents

    # Try to cache system prompt + tools (reused across turns and queries)
    cache_name = _get_or_create_cache(
        api_key=api_key,
        model=model,
        solver_type=solver_type,
        system_prompt=system_prompt,
        tools=tools,
        logs_dir=logs_dir,
    )

    _log(logs_dir, AGENT_LOG, "=== API Agent ===")
    _log(logs_dir, AGENT_LOG, f"Constraint: {constraint}")
    _log(logs_dir, AGENT_LOG, f"Model: {model}")
    _log(logs_dir, AGENT_LOG, f"Index workspace: {index_workspace}")
    _log(logs_dir, AGENT_LOG, f"Policy dir: {policy_dir}")
    _log(logs_dir, AGENT_LOG, f"Cache: {cache_name or 'disabled'}")
    _log(logs_dir, AGENT_LOG, "=" * 80)

    # Persist exact initial prompts for easy inspection.
    _write_file(logs_dir, "api_system_prompt.txt", system_prompt)
    _write_file(logs_dir, "api_user_prompt.txt", user_prompt)
    _write_file(
        logs_dir,
        "api_prompt.txt",
        f"## System Prompt\n\n{system_prompt}\n\n## User Prompt\n\n{user_prompt}\n",
    )

    final_text = ""
    tool_call_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0

    for turn in range(max_turns):
        _log(logs_dir, AGENT_LOG, f"\n=== Turn {turn + 1}/{max_turns} ===")

        payload = _build_api_payload(
            contents=contents,
            system_prompt=system_prompt,
            tools=tools,
            cached_content=cache_name,
        )
        if logs_dir:
            req_path = os.path.join(logs_dir, f"api_turn_{turn + 1:02d}_request.json")
            with open(req_path, "w") as f:
                json.dump(payload, f, indent=2)

        # Try primary model, walk fallback chain on 403.
        current_model = model
        while True:
            try:
                response = _api_generate(
                    api_key=api_key, model=current_model, payload=payload,
                )
                if current_model != model:
                    model = current_model  # persist for remaining turns
                break
            except Exception as exc:
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                should_fallback = (
                    isinstance(exc, requests.exceptions.HTTPError)
                    and status_code in (403, 429, 500, 502, 503, 504)  # Rate limits + server errors
                )
                fallback = _FALLBACK_CHAIN.get(current_model) if should_fallback else None
                if fallback:
                    error_type = "rate limit" if status_code in (403, 429) else "server error"
                    _log(logs_dir, AGENT_LOG,
                         f"{error_type.capitalize()} ({status_code}) on {current_model} — falling back to {fallback}")
                    current_model = fallback
                    cache_name = _get_or_create_cache(
                        api_key, current_model, solver_type,
                        system_prompt, tools, logs_dir,
                    )
                    payload = _build_api_payload(
                        contents=contents,
                        system_prompt=system_prompt,
                        tools=tools,
                        cached_content=cache_name,
                    )
                    continue
                _log(logs_dir, AGENT_LOG, f"API error: {exc}")
                cost = _estimate_cost(model, total_input_tokens, total_output_tokens)
                return {
                    "success": False,
                    "error": f"API error: {exc}",
                    "edit": None,
                    "thoughts": thoughts,
                    "iterations": turn + 1,
                    "cost_usd": cost,
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                }

        if logs_dir:
            with open(os.path.join(logs_dir, "api_tools_stream.jsonl"), "a") as f:
                f.write(json.dumps(response) + "\n")

        # Track token usage
        usage = response.get("usageMetadata") or {}
        turn_input = usage.get("promptTokenCount", 0)
        turn_output = usage.get("candidatesTokenCount", 0)
        turn_cached = usage.get("cachedContentTokenCount", 0)
        total_input_tokens += turn_input
        total_output_tokens += turn_output
        total_cached_tokens += turn_cached
        _log(logs_dir, AGENT_LOG,
             f"Tokens: input={turn_input} (cached={turn_cached}), output={turn_output}")

        candidates = response.get("candidates") or []
        if not candidates:
            _log(logs_dir, AGENT_LOG, "No API candidates returned")
            cost = _estimate_cost(model, total_input_tokens, total_output_tokens)
            return {
                "success": False,
                "error": "No candidates from API",
                "edit": None,
                "thoughts": thoughts,
                "iterations": turn + 1,
                "cost_usd": cost,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }

        candidate_content = candidates[0].get("content") or {}
        parts = candidate_content.get("parts") or []
        contents.append(candidate_content)

        finish_reason = candidates[0].get("finishReason", "")
        function_calls = [p.get("functionCall") for p in parts if isinstance(p, dict) and p.get("functionCall")]
        text = _extract_text_from_candidate(candidates[0])
        if text:
            final_text = text
            thoughts.append(f"[assistant] {text[:300]}")
            _log(logs_dir, AGENT_LOG, "Assistant response:")
            _log(logs_dir, AGENT_LOG, _truncate(text, 800))

        if not function_calls:
            if finish_reason == "MAX_TOKENS":
                _log(logs_dir, AGENT_LOG, "Hit MAX_TOKENS — continuing so model can finish")
                continue
            break

        for fc in function_calls:
            tool_call_count += 1
            name = fc.get("name", "")
            args = fc.get("args", {}) or {}
            if not isinstance(args, dict):
                args = {}

            _log(logs_dir, AGENT_LOG, f"Tool call: {name}")
            _log(
                logs_dir,
                AGENT_LOG,
                json.dumps(_summarize_tool_args(name, args), indent=2),
            )
            thoughts.append(f"[tool_call] {name}")

            if name == "search_code":
                query = str(args.get("query", "")).strip()
                limit = int(args.get("limit", 5) or 5)
                tool_result = _search_tool(index_workspace=index_workspace, query=query, limit=max(1, min(limit, 10)), env=env)
            elif name == "read_function":
                rel = str(args.get("relative_path", "")).strip()
                fn = str(args.get("function_name", "")).strip()
                tool_result = _read_function_tool(
                    work_dir=work_dir,
                    policy_dir=policy_dir,
                    relative_path=rel,
                    function_name=fn,
                    env=env,
                )
            elif name == "apply_edit":
                rel = str(args.get("relative_path", "")).strip()
                old_text = str(args.get("old_text", ""))
                new_text = str(args.get("new_text", ""))
                explanation = str(args.get("explanation", ""))
                tool_result = _apply_edit_tool(
                    work_dir=work_dir,
                    policy_dir=policy_dir,
                    relative_path=rel,
                    old_text=old_text,
                    new_text=new_text,
                    explanation=explanation,
                    env=env,
                )
            else:
                tool_result = {"ok": False, "error": f"Unknown tool: {name}"}

            _log(logs_dir, AGENT_LOG, f"Tool result: {name}")
            _log(
                logs_dir,
                AGENT_LOG,
                json.dumps(_summarize_tool_result(name, tool_result), indent=2),
            )

            contents.append(
                {
                    "role": "tool",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": {"result": tool_result},
                            }
                        }
                    ],
                }
            )

    cost = _estimate_cost(model, total_input_tokens, total_output_tokens)
    _log(logs_dir, AGENT_LOG,
         f"Total tokens: input={total_input_tokens} (cached={total_cached_tokens}), "
         f"output={total_output_tokens}, cost=${cost:.4f}")

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
            "thoughts": thoughts + ([final_text] if final_text else []),
            "iterations": min(max_turns, max(1, tool_call_count)),
            "cost_usd": cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        }

    _log(logs_dir, AGENT_LOG, "No file changes detected after API agent run")
    return {
        "success": False,
        "error": "No edits detected in policy/ files",
        "edit": None,
        "thoughts": thoughts + ([final_text] if final_text else []),
        "iterations": min(max_turns, max(1, tool_call_count)),
        "cost_usd": cost,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }
