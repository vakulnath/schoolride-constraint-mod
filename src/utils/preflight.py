#!/usr/bin/env python3
"""
Preflight checks for MCP + agent runtime readiness.

Run:
    python -m src.core.preflight
"""

from __future__ import annotations

import configparser
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from src.mcp.context import (
    MCPClient,
    _CONTEXT_MCP_DIST,
    _FS_MCP_DIST,
    get_context_client,
    get_filesystem_client,
)
from src.utils.utils import load_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_INI = PROJECT_ROOT / "src" / "configs" / "config.ini"


def _ok(msg: str):
    print(f"[OK]   {msg}")


def _fail(msg: str):
    print(f"[FAIL] {msg}")


def _warn(msg: str):
    print(f"[WARN] {msg}")


def _load_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(load_env_file(str(PROJECT_ROOT / ".env")))
    return env


def _workspace_paths() -> Tuple[Path, Path]:
    cfg = configparser.ConfigParser()
    cfg.read(CONFIG_INI)
    insertion = Path(cfg.get("paths", "index_workspace"))
    hexaly = Path(cfg.get("paths", "hexaly_index_workspace"))
    if not insertion.is_absolute():
        insertion = PROJECT_ROOT / insertion
    if not hexaly.is_absolute():
        hexaly = PROJECT_ROOT / hexaly
    return insertion, hexaly


def _list_tools(client: MCPClient, timeout_s: float = 30.0) -> List[str]:
    resp = client._send_raw(
        {"jsonrpc": "2.0", "id": 99991, "method": "tools/list", "params": {}},
        timeout_s=timeout_s,
    )
    return [t.get("name") for t in resp.get("result", {}).get("tools", [])]


def _is_error_result(result: Dict) -> bool:
    if result.get("error"):
        return True
    if result.get("isError"):
        return True
    text_blocks = result.get("content", [])
    if isinstance(text_blocks, list):
        for b in text_blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                text = str(b.get("text", ""))
                if text.startswith("Error:"):
                    return True
    return False


def main() -> int:
    env = _load_env()
    failures = 0

    print("== MCP/Agent Preflight ==")
    print(f"Project root: {PROJECT_ROOT}")

    # File existence checks
    if Path(_CONTEXT_MCP_DIST).is_file():
        _ok(f"Claude Context MCP dist found: {_CONTEXT_MCP_DIST}")
    else:
        failures += 1
        _fail(f"Missing Claude Context MCP dist: {_CONTEXT_MCP_DIST}")

    if Path(_FS_MCP_DIST).is_file():
        _ok(f"Filesystem MCP dist found: {_FS_MCP_DIST}")
    else:
        failures += 1
        _fail(f"Missing Filesystem MCP dist: {_FS_MCP_DIST}")

    uvx = shutil.which("uvx") or str(Path.home() / ".local" / "bin" / "uvx")
    if Path(uvx).is_file():
        _ok(f"uvx found: {uvx} (required for Serena MCP)")
    else:
        failures += 1
        _fail(f"Missing uvx binary: {uvx} (required for Serena MCP — install via: curl -LsSf https://astral.sh/uv/install.sh | sh)")

    if env.get("GEMINI_API_KEY"):
        _ok("GEMINI_API_KEY is set")
    else:
        failures += 1
        _fail("GEMINI_API_KEY is missing")

    insertion_ws, hexaly_ws = _workspace_paths()
    for label, ws in [("insertion", insertion_ws), ("hexaly", hexaly_ws)]:
        if ws.is_dir():
            _ok(f"{label} workspace exists: {ws}")
        else:
            failures += 1
            _fail(f"{label} workspace missing: {ws}")

    # Claude Context MCP checks
    try:
        client = get_context_client(env=env)
        try:
            tools = _list_tools(client)
            expected = {"search_code", "index_codebase", "clear_index", "get_indexing_status"}
            missing = sorted(expected - set(tools))
            if missing:
                failures += 1
                _fail(f"Context MCP missing tools: {missing}")
            else:
                _ok(f"Context MCP tools available: {tools}")

            for label, ws in [("insertion", insertion_ws), ("hexaly", hexaly_ws)]:
                status_result = client.call("get_indexing_status", {"path": str(ws)}, timeout_s=30.0)
                if _is_error_result(status_result):
                    failures += 1
                    _fail(f"Context MCP get_indexing_status failed for {label}: {status_result}")
                else:
                    _ok(f"Context MCP indexing status reachable for {label}")

                search_result = client.call(
                    "search_code",
                    {"path": str(ws), "query": "constraint", "limit": 1},
                    timeout_s=60.0,
                )
                if _is_error_result(search_result):
                    failures += 1
                    _fail(f"Context MCP search_code failed for {label}: {search_result}")
                else:
                    _ok(f"Context MCP search_code works for {label}")
        finally:
            client.close()
    except Exception as e:
        failures += 1
        _fail(f"Context MCP startup failed: {e}")

    # Filesystem MCP checks
    try:
        fs_client = get_filesystem_client([str(PROJECT_ROOT)], env=env)
        try:
            fs_tools = _list_tools(fs_client)
            expected_fs = {"read_text_file", "read_multiple_files", "edit_file"}
            missing_fs = sorted(expected_fs - set(fs_tools))
            if missing_fs:
                failures += 1
                _fail(f"Filesystem MCP missing tools: {missing_fs}")
            else:
                _ok(f"Filesystem MCP tools available: {fs_tools}")

            read_result = fs_client.call(
                "read_text_file",
                {"path": str(PROJECT_ROOT / "src" / "mcp" / "context.py"), "head": 3},
                timeout_s=30.0,
            )
            if _is_error_result(read_result):
                failures += 1
                _fail(f"Filesystem MCP read_text_file failed: {read_result}")
            else:
                _ok("Filesystem MCP read_text_file works")
        finally:
            fs_client.close()
    except Exception as e:
        failures += 1
        _fail(f"Filesystem MCP startup failed: {e}")

    # Pathfinder MCP checks
    try:
        pf_client = MCPClient([pf, "serve", "--project", str(PROJECT_ROOT)], env=env)
        try:
            pf_tools = _list_tools(pf_client, timeout_s=120.0)
            expected_pf = {"find_symbol", "get_callers", "get_callees"}
            missing_pf = sorted(expected_pf - set(pf_tools))
            if missing_pf:
                failures += 1
                _fail(f"Pathfinder MCP missing expected tools: {missing_pf}")
            else:
                _ok(f"Pathfinder MCP tools available: {pf_tools}")
        finally:
            pf_client.close()
    except Exception as e:
        failures += 1
        _fail(f"Pathfinder MCP startup failed: {e}")

    if failures:
        print(f"\nPreflight FAILED with {failures} issue(s).")
        return 1

    print("\nPreflight PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
