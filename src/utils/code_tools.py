"""
Shared code tools: search, read, edit, and parse utilities used by all pipeline backends.
"""

from __future__ import annotations

import difflib
import importlib.util
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from src.mcp.context import extract_text, mcp_search
from src.configs import config_parser as cfg
from src.utils.utils import dedupe_results


# ---------------------------------------------------------------------------
# Policy copy helper
# ---------------------------------------------------------------------------

def copy_policy_to_dir(src_dir: str, dst_dir: str) -> None:
    """Symlink all policy .py files from src_dir into dst_dir.

    Every file is a symlink to the original package — zero disk cost.
    When apply_edit_to_temp_dir later edits a file it materialises that
    specific symlink into a real copy first, so each query's modified file
    is preserved independently without duplicating the rest.
    """
    os.makedirs(dst_dir, exist_ok=True)
    abs_src = os.path.abspath(src_dir)
    for fname in os.listdir(abs_src):
        if not fname.endswith(".py"):
            continue
        src = os.path.join(abs_src, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.symlink(src, dst)


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def log_to_file(logs_dir: Optional[str], filename: str, content: str):
    """Append content to a log file."""
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, filename), "a") as f:
            f.write(content + "\n")


def write_file(logs_dir: Optional[str], filename: str, content: str):
    """Write content to a file in logs_dir."""
    if logs_dir:
        os.makedirs(logs_dir, exist_ok=True)
        with open(os.path.join(logs_dir, filename), "w") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_constraints() -> str:
    """Load contents of constraints.txt."""
    if cfg.CONSTRAINTS_FILE.exists():
        with open(cfg.CONSTRAINTS_FILE, "r") as f:
            return f.read()
    return "(Constraints not available)"


def load_schema(context_root: str, solver_type: str = "insertion") -> str:
    """Load the solver API schema file."""
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
        spec_name = (
            "school_bus_routing.schemas.hexaly"
            if solver_type == "hexaly"
            else "school_bus_routing.schemas.insertion"
        )
        spec = importlib.util.find_spec(spec_name)
        if spec and spec.origin:
            with open(spec.origin, "r") as f:
                return f.read()
    except Exception:
        pass
    return "(Schema not available)"


# ---------------------------------------------------------------------------
# Edit parser
# ---------------------------------------------------------------------------

def parse_edit_suggestion(
    text: str,
    logs_dir: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Parse XML-tagged edit(s) from agent output.

    Supports multiple edit blocks in a single response (e.g. one for the
    function signature and one for the body).  Returns the first edit as the
    primary dict; any additional edits are stored in the ``extra_edits`` list.

    Expected format (repeat as needed):
        <relative_path>filename.py</relative_path>
        <old_text>exact lines to replace</old_text>
        <new_text>replacement lines</new_text>
        <explanation>why</explanation>
    """
    if not isinstance(text, str):
        return None

    paths = re.findall(r"<relative_path>(.*?)</relative_path>", text, re.DOTALL)
    olds  = re.findall(r"<old_text>\n?(.*?)\n?</old_text>", text, re.DOTALL)
    # Accept </new_text> or mis-tagged </old_text> as closing tag for new_text
    news  = re.findall(r"<new_text>\n?(.*?)\n?</(?:new|old)_text>", text, re.DOTALL)
    expls = re.findall(r"<explanation>(.*?)</explanation>", text, re.DOTALL)

    # Fallback: some models generate <filename.py> instead of <relative_path>filename.py</relative_path>
    if not paths and olds and news:
        fallback_paths = re.findall(r"<([\w/._-]+\.py)>", text)
        if fallback_paths:
            paths = fallback_paths

    if not (paths and olds and news):
        return None

    edits = []
    for i in range(min(len(paths), len(olds), len(news))):
        rel = paths[i].strip()
        old_text = olds[i]
        new_text = news[i]
        explanation = expls[i].strip() if i < len(expls) else ""
        diff_text = "".join(difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{os.path.basename(rel)}",
            tofile=f"b/{os.path.basename(rel)}",
        ))
        edits.append({
            "path": os.path.basename(rel),
            "success": True,
            "diff": diff_text,
            "old_text": old_text,
            "new_text": new_text,
            "explanation": explanation,
        })

    primary = edits[0]
    if len(edits) > 1:
        primary["extra_edits"] = edits[1:]
    return primary


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def _resolve_target_file(work_dir: str, policy_dir: str, relative_path: str) -> str:
    target = os.path.join(policy_dir, os.path.basename(relative_path))
    if os.path.isfile(target):
        return target
    return os.path.join(work_dir, relative_path)


def _extract_function(source: str, function_name: str) -> Optional[str]:
    """Extract a function definition by name from Python source."""
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

    body_start = start_idx + 1
    for i in range(start_idx, len(lines)):
        if lines[i].rstrip().endswith(":"):
            body_start = i + 1
            break

    end_idx = body_start
    for i in range(body_start, len(lines)):
        line = lines[i]
        if not line.strip():
            end_idx = i + 1
            continue
        if len(line) - len(line.lstrip()) <= base_indent:
            break
        end_idx = i + 1

    return "".join(lines[start_idx:end_idx])


def _fix_indentation(file_path: str, old_text: str, new_text: str) -> Tuple[str, str]:
    """Try to match old_text indentation to what's actually in the file."""
    with open(file_path, "r") as f:
        file_content = f.read()

    old_lines = old_text.splitlines()
    if not old_lines:
        return old_text, new_text

    first_non_empty = next((l for l in old_lines if l.strip()), None)
    if not first_non_empty:
        return old_text, new_text

    old_indent = len(first_non_empty) - len(first_non_empty.lstrip())

    # Find the actual indentation in the file
    for line in file_content.splitlines():
        if first_non_empty.strip() in line:
            actual_indent = len(line) - len(line.lstrip())
            if actual_indent != old_indent:
                delta = actual_indent - old_indent
                def _reindent(text: str, d: int) -> str:
                    result = []
                    for l in text.splitlines(keepends=True):
                        if l.strip():
                            spaces = len(l) - len(l.lstrip())
                            new_spaces = max(0, spaces + d)
                            result.append(" " * new_spaces + l.lstrip())
                        else:
                            result.append(l)
                    return "".join(result)
                return _reindent(old_text, delta), _reindent(new_text, delta)
            break

    return old_text, new_text


# ---------------------------------------------------------------------------
# MCP tool wrappers
# ---------------------------------------------------------------------------

def search_tool(
    index_workspace: str,
    query: str,
    limit: int,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Semantic search in an indexed codebase via MCP."""
    raw = mcp_search(index_workspace, query=query, limit=limit, env=env)
    parsed = dedupe_results(extract_text(raw))
    return {
        "count": len(parsed[:limit]),
        "results": [
            {
                "relativePath": s.get("relativePath"),
                "functionName": s.get("functionName"),
                "startLine": s.get("startLine"),
                "endLine": s.get("endLine"),
                "nodeType": s.get("nodeType"),
                "className": s.get("className"),
                # callsTo/calledBy omitted: names without file paths cause hallucinated read_function calls
                # Use find_referencing_symbols(name_path, relative_path) Serena tool instead
                "docstring": s.get("docstring"),
                "content": s.get("content") or "",
            }
            for s in parsed[:limit]
        ],
    }


def read_function_tool(
    work_dir: str,
    policy_dir: str,
    relative_path: str,
    function_name: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Read the full source of a function from the policy codebase."""
    from src.mcp.filesystem import fs_read_file

    target_file = _resolve_target_file(work_dir, policy_dir, relative_path)
    if not os.path.isfile(target_file) and cfg.SOLVERS_POLICY_ROOT:
        orig = os.path.join(str(cfg.SOLVERS_POLICY_ROOT), os.path.basename(relative_path))
        if os.path.isfile(orig):
            target_file = orig
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
    }


def apply_edit_tool(
    work_dir: str,
    policy_dir: str,
    relative_path: str,
    old_text: str,
    new_text: str,
    explanation: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Apply an old_text → new_text edit to a file."""
    from src.mcp.filesystem import fs_edit_file, fs_read_file

    target_file = _resolve_target_file(work_dir, policy_dir, relative_path)
    if not os.path.isfile(target_file) and cfg.SOLVERS_POLICY_ROOT:
        basename = os.path.basename(relative_path)
        orig = os.path.join(str(cfg.SOLVERS_POLICY_ROOT), basename)
        if os.path.isfile(orig):
            os.makedirs(policy_dir, exist_ok=True)
            dest = os.path.join(policy_dir, basename)
            shutil.copy2(orig, dest)
            target_file = dest
    if not os.path.isfile(target_file):
        return {"ok": False, "error": f"File not found: {target_file}"}

    full_source = fs_read_file(target_file, env=env)
    if old_text not in full_source:
        fixed_old, fixed_new = _fix_indentation(target_file, old_text, new_text)
        if fixed_old in full_source:
            old_text, new_text = fixed_old, fixed_new
        else:
            lines = full_source.splitlines()
            first_line = next((l.strip() for l in old_text.splitlines() if l.strip()), "")
            snippet = ""
            if first_line:
                for i, line in enumerate(lines):
                    if first_line[:40] in line:
                        start = max(0, i - 3)
                        end = min(len(lines), i + 10)
                        snippet = "\n".join(lines[start:end])
                        break
            return {
                "ok": False,
                "error": (
                    "old_text not found in file. "
                    f"Current file near expected location:\n```\n{snippet}\n```\n"
                    "Adjust old_text to match the CURRENT file content."
                ),
                "path": target_file,
            }

    result = fs_edit_file(
        path=target_file,
        edits=[{"oldText": old_text, "newText": new_text}],
        dry_run=False,
        env=env,
    )
    if result.get("error") or result.get("isError"):
        return {"ok": False, "error": str(result.get("error")), "path": target_file}

    return {"ok": True, "path": target_file, "fileName": os.path.basename(target_file)}


def apply_edit_to_temp_dir(
    edit: Dict[str, Any],
    temp_code_dir: str,
    env: Dict[str, str],
    logs_dir: Optional[str] = None,
) -> bool:
    """Apply an edit dict to a temp copy of the policy directory."""
    policy_dir = os.path.join(temp_code_dir, "policy")
    if not os.path.isdir(policy_dir):
        import policy as _pol
        _pol_file = _pol.__file__
        assert _pol_file is not None, "Cannot locate policy package"
        copy_policy_to_dir(os.path.dirname(_pol_file), policy_dir)

    def _apply_one(e: Dict[str, Any]) -> bool:
        """Materialise symlink if needed, then apply one edit block."""
        tgt = os.path.join(policy_dir, os.path.basename(e["path"]))
        if os.path.islink(tgt):
            real_src = os.path.realpath(tgt)
            os.unlink(tgt)
            shutil.copy2(real_src, tgt)
        r = apply_edit_tool(
            work_dir=temp_code_dir,
            policy_dir=policy_dir,
            relative_path=e["path"],
            old_text=e["old_text"],
            new_text=e["new_text"],
            explanation=e.get("explanation", ""),
            env=env,
        )
        return r.get("ok", False)

    if not _apply_one(edit):
        return False

    # Apply any additional edit blocks (e.g. signature + body in separate blocks)
    for extra in edit.get("extra_edits", []):
        if not _apply_one(extra):
            return False

    if logs_dir:
        all_diffs = [edit.get("diff", "")] + [
            e.get("diff", "") for e in edit.get("extra_edits", [])
        ]
        with open(os.path.join(logs_dir, "applied_edit.diff"), "w") as f:
            f.write("\n".join(d for d in all_diffs if d))

    return True
