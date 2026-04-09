"""
Filesystem MCP client for secure file read/write/edit operations.

Wraps the @modelcontextprotocol/server-filesystem MCP server
(built from local repo at servers/src/filesystem).
"""

import os
from typing import Any, Dict, List, Optional

from src.mcp.context import MCPClient, get_filesystem_client, extract_text


def _get_fs_client(allowed_dirs: List[str], env: Optional[Dict[str, str]] = None) -> MCPClient:
    """Get a Filesystem MCP client with given allowed directories."""
    return get_filesystem_client(allowed_dirs, env=env)


def fs_read_files(
    paths: List[str],
    env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Read multiple files via the Filesystem MCP server.

    Args:
        paths: List of absolute file paths to read.
        env: Optional environment overrides.

    Returns:
        Concatenated file contents as a string.
    """
    if not paths:
        return ""

    # Determine allowed directories from the file paths
    allowed_dirs = list({os.path.dirname(p) for p in paths})
    client = _get_fs_client(allowed_dirs, env=env)
    try:
        result = client.call(
            "read_multiple_files",
            {"paths": paths},
            timeout_s=30.0,
        )
        return extract_text(result)
    finally:
        client.close()


def fs_read_file(
    path: str,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    """
    Read a single file via the Filesystem MCP server.

    Args:
        path: Absolute file path.
        head: Only return first N lines.
        tail: Only return last N lines.
        env: Optional environment overrides.

    Returns:
        File content as string.
    """
    allowed_dirs = [os.path.dirname(path)]
    client = _get_fs_client(allowed_dirs, env=env)
    try:
        args: Dict[str, Any] = {"path": path}
        if head is not None:
            args["head"] = head
        if tail is not None:
            args["tail"] = tail
        result = client.call("read_text_file", args, timeout_s=30.0)
        return extract_text(result)
    finally:
        client.close()


def fs_edit_file(
    path: str,
    edits: List[Dict[str, str]],
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Apply line-based edits to a file via the Filesystem MCP server.

    Args:
        path: Absolute file path to edit.
        edits: List of ``{"oldText": "...", "newText": "..."}`` dicts.
        dry_run: If True, preview changes without writing.
        env: Optional environment overrides.

    Returns:
        MCP result dict.
    """
    allowed_dirs = [os.path.dirname(path)]
    client = _get_fs_client(allowed_dirs, env=env)
    try:
        result = client.call(
            "edit_file",
            {"path": path, "edits": edits, "dryRun": dry_run},
            timeout_s=30.0,
        )
        return result
    finally:
        client.close()


def apply_line_edits_in_root(
    repo_root: str,
    edits: List[Dict[str, str]],
    env: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Apply a list of edits within a repository root.

    Each edit dict should have: ``path`` (relative or absolute), ``old``, ``new``.
    The filesystem MCP's ``edit_file`` is used with ``oldText``/``newText``.

    Args:
        repo_root: Root directory of the repository.
        edits: List of edit dicts with ``path``, ``old``, ``new`` keys.
        env: Optional environment overrides.

    Returns:
        True if all edits succeeded, False otherwise.
    """
    client = _get_fs_client([repo_root], env=env)
    try:
        for edit in edits:
            file_path = edit.get("path", "")
            if not os.path.isabs(file_path):
                file_path = os.path.join(repo_root, file_path)
            old_text = edit.get("old", "")
            new_text = edit.get("new", "")
            if not old_text or not new_text:
                continue
            result = client.call(
                "edit_file",
                {
                    "path": file_path,
                    "edits": [{"oldText": old_text, "newText": new_text}],
                    "dryRun": False,
                },
                timeout_s=30.0,
            )
            if result.get("error") or result.get("isError"):
                return False
        return True
    finally:
        client.close()
