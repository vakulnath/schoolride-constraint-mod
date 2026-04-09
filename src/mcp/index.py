#!/usr/bin/env python3
"""
Index client for the Claude Context MCP server.

Copies policy files into workspace directories and indexes them via
the Claude Context MCP.  Supports both insertion and Hexaly workspaces.

Usage:
    python index_client.py                     # Index insertion (default)
    python index_client.py --workspace hexaly  # Index Hexaly
    python index_client.py --workspace all     # Index both
    python index_client.py --clear             # Clear and re-index
    python index_client.py --status            # Check indexing status
"""

import argparse
import os
import shutil
import time
from typing import Dict, List, Optional

from src.mcp.context import get_context_client, extract_text
from src.configs import config_parser as cfg
from src.utils.utils import load_env_file

# ── Paths ────────────────────────────────────────────────────────────────


# ── Helpers ──────────────────────────────────────────────────────────────

def _copy_files(file_list: List[str], src_dir, dst_dir) -> List[str]:
    """Copy files from src_dir into dst_dir. Returns list of copied paths."""
    os.makedirs(dst_dir, exist_ok=True)

    if src_dir is None or not os.path.isdir(str(src_dir)):
        raise RuntimeError(
            f"Source directory not found: {src_dir}\n"
            "Set SOLVERS_ROOT or SOLVERS_POLICY_ROOT in .env or environment."
        )

    copied = []
    for fname in file_list:
        src = os.path.join(str(src_dir), fname)
        dst = os.path.join(str(dst_dir), fname)
        if not os.path.isfile(src):
            print(f"  Warning: Source file not found, skipping: {src}")
            continue
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"  Copied {fname}")

    if not copied:
        raise RuntimeError(f"No files copied from {src_dir}")
    return copied


def _load_env() -> Dict[str, str]:
    env_path = ".env"
    return {**os.environ, **load_env_file(env_path)}


def _index_workspace(
    workspace_path,
    file_list: List[str],
    label: str,
    clear: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Copy files and index a single workspace."""
    if env is None:
        env = _load_env()

    print(f"\nCopying {label} files to {workspace_path}")
    if clear:
        # Remove stale .py files not in the current file_list
        dst = str(workspace_path)
        if os.path.isdir(dst):
            current = set(file_list)
            for f in os.listdir(dst):
                if f.endswith(".py") and f not in current:
                    os.remove(os.path.join(dst, f))
                    print(f"  Removed stale file: {f}")
    _copy_files(file_list, cfg.SOLVERS_POLICY_ROOT, workspace_path)

    print("\nConnecting to Claude Context MCP server...")
    client = get_context_client(env)

    try:
        if clear:
            print(f"\nClearing existing {label} index...")
            result = client.call(
                "clear_index",
                {"path": str(workspace_path)},
                timeout_s=60.0,
            )
            msg = extract_text(result)
            print(f"  {msg}")

        print(f"\nStarting indexing ({label}): {workspace_path}")
        result = client.call(
            "index_codebase",
            {"path": str(workspace_path), "force": clear, "splitter": "ast"},
            timeout_s=300.0,
        )
        msg = extract_text(result)
        print(f"  {msg}")

        if "already indexed" in msg.lower():
            print(f"\nAlready indexed ({label}). Use --clear to force re-index.")
            return msg

        print(f"\nWaiting for {label} indexing to complete...")
        for i in range(200):
            time.sleep(3)
            result = client.call(
                "get_indexing_status",
                {"path": str(workspace_path)},
                timeout_s=30.0,
            )
            status = extract_text(result)

            if "%" in status:
                pct_line = [s for s in status.split("\n") if "%" in s]
                print(f"  [{i*3:>3}s] {pct_line[0].strip() if pct_line else status[:100]}")
            else:
                print(f"  [{i*3:>3}s] {status[:100]}")

            if "fully indexed" in status.lower():
                print(f"\nIndexing complete ({label})!")
                return status

            if "failed" in status.lower() or "error" in status.lower():
                print(f"\nIndexing failed ({label}): {status}")
                return status

        print(f"\nTimed out waiting for {label} indexing (10 min).")
        return "TIMEOUT"

    finally:
        client.close()


# ── Public API ───────────────────────────────────────────────────────────

def index_codebase(
    clear: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Index the insertion workspace (backward-compatible)."""
    return _index_workspace(cfg.INDEX_WORKSPACE, cfg.INDEX_FILES, "insertion", clear, env)


def index_hexaly(
    clear: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Index the Hexaly workspace."""
    return _index_workspace(cfg.HEXALY_INDEX_WORKSPACE, cfg.HEXALY_INDEX_FILES, "hexaly", clear, env)


def get_indexing_status(
    workspace=None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """Check the indexing status of a workspace."""
    if env is None:
        env = _load_env()
    ws = str(workspace or cfg.INDEX_WORKSPACE)
    client = get_context_client(env)
    try:
        result = client.call(
            "get_indexing_status",
            {"path": ws},
            timeout_s=30.0,
        )
        return extract_text(result)
    finally:
        client.close()


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Index policy files for semantic code search."
    )
    parser.add_argument(
        "--workspace",
        choices=["insertion", "hexaly", "all"],
        default="insertion",
        help="Which workspace to index (default: insertion).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index and re-index from scratch.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Just print current indexing status and exit.",
    )
    args = parser.parse_args()

    if args.status:
        if args.workspace in ("insertion", "all"):
            print("=== Insertion workspace ===")
            print(get_indexing_status(cfg.INDEX_WORKSPACE))
        if args.workspace in ("hexaly", "all"):
            print("=== Hexaly workspace ===")
            print(get_indexing_status(cfg.HEXALY_INDEX_WORKSPACE))
        return

    if args.workspace in ("insertion", "all"):
        result = index_codebase(clear=args.clear)
        print(f"\nInsertion: {result[:200]}")

    if args.workspace in ("hexaly", "all"):
        result = index_hexaly(clear=args.clear)
        print(f"\nHexaly: {result[:200]}")


if __name__ == "__main__":
    main()
