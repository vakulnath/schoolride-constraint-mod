"""
Generic MCP (Model Context Protocol) client for stdio-based MCP servers.

Provides:
- MCPClient: Low-level JSON-RPC client that spawns an MCP server subprocess
- get_context_client: Factory for Claude Context MCP (semantic code search)
- mcp_search: High-level search helper
- extract_text: Extract text content from MCP response dicts
"""

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class MCPClient:
    """
    JSON-RPC 2.0 client that communicates with an MCP server over stdio.

    Usage::

        client = MCPClient(["node", "/path/to/mcp/dist/index.js", ...args], env)
        result = client.call("search_code", {"path": "/repo", "query": "foo"}, timeout_s=60)
        client.close()
    """

    def __init__(self, cmd: List[str], env: Optional[Dict[str, str]] = None):
        self._cmd = cmd
        self._env = env or os.environ.copy()
        self._proc: Optional[subprocess.Popen] = None
        self._id = 0
        self._lock = threading.Lock()
        self._start()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def _start(self):
        """Spawn the MCP server subprocess."""
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self._env,
            bufsize=0,
        )
        # MCP servers using StdioServerTransport expect an initialize handshake
        self._initialize()

    def _initialize(self):
        """Send the MCP initialize handshake."""
        resp = self._send_raw({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "constraint-modification-client",
                    "version": "1.0.0",
                },
            },
        })
        # Send initialized notification (no id, no response expected)
        self._write_message({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })
        return resp

    def close(self):
        """Terminate the MCP server subprocess."""
        if self._proc and self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------ #
    # JSON-RPC transport
    # ------------------------------------------------------------------ #

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _write_message(self, msg: dict):
        """Write a JSON-RPC message to the server's stdin."""
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(msg)
        # MCP uses newline-delimited JSON (no Content-Length header)
        data = body + "\n"
        self._proc.stdin.write(data.encode("utf-8"))
        self._proc.stdin.flush()

    def _read_message(self, timeout_s: float = 120.0) -> dict:
        """
        Read a JSON-RPC response from the server's stdout.

        Handles the case where the server might write log lines to stderr
        or non-JSON lines to stdout (which we skip).
        """
        assert self._proc is not None and self._proc.stdout is not None
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._proc.poll() is not None:
                stderr_out = ""
                try:
                    if self._proc.stderr is not None:
                        stderr_out = self._proc.stderr.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                raise RuntimeError(
                    f"MCP server process exited with code {self._proc.returncode}. "
                    f"stderr: {stderr_out[:20000]}"
                )

            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                return json.loads(line_str)
            except json.JSONDecodeError:
                # Server might emit non-JSON log lines — skip them
                continue

        raise TimeoutError(f"MCP server did not respond within {timeout_s}s")

    def _send_raw(self, msg: dict, timeout_s: float = 120.0) -> dict:
        """Send a message and wait for the matching response by id."""
        msg_id = msg.get("id")
        self._write_message(msg)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = self._read_message(timeout_s=deadline - time.time())
            # Skip notifications (no id) or responses for other ids
            if resp.get("id") == msg_id:
                return resp
        raise TimeoutError(f"No response for message id={msg_id} within {timeout_s}s")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_s: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Call an MCP tool and return the parsed result.

        Returns a dict with either:
        - ``{"content": [...], ...}`` on success
        - ``{"error": "message"}`` on error
        """
        msg = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        resp = self._send_raw(msg, timeout_s=timeout_s)

        if "error" in resp:
            return {"error": resp["error"].get("message", str(resp["error"]))}

        result = resp.get("result", {})
        return result


# ====================================================================== #
# Claude Context MCP helpers
# ====================================================================== #

# Project root: constraint_modification/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Absolute path to Claude Context MCP server from node_modules
_CONTEXT_MCP_DIST = str(
    _PROJECT_ROOT / "node_modules" / "claude-context" / "packages" / "mcp" / "dist" / "index.js"
)

# Absolute path to Filesystem MCP server from node_modules
_FS_MCP_DIST = str(
    _PROJECT_ROOT / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js"
)


def _build_context_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build environment variables for the Claude Context MCP server."""
    base = os.environ.copy()
    if env:
        base.update(env)
    # Ensure critical vars are set
    required_keys = {
        "EMBEDDING_PROVIDER": "Ollama",
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": "nomic-embed-text",
        "MILVUS_ADDRESS": "localhost:19530",
    }
    for k, default in required_keys.items():
        if k not in base or not base[k]:
            base[k] = default
    # Disable cloud sync by default for local dev
    base.setdefault("DISABLE_CLOUD_SYNC", "1")
    return base


def get_context_client(env: Optional[Dict[str, str]] = None) -> MCPClient:
    """
    Create an MCPClient connected to the local Claude Context MCP server.

    The server is spawned from:
      <project_root>/node_modules/claude-context/packages/mcp/dist/index.js
    """
    if not os.path.isfile(_CONTEXT_MCP_DIST):
        raise FileNotFoundError(
            f"Claude Context MCP not built. Expected: {_CONTEXT_MCP_DIST}\n"
            "Run: cd node_modules/claude-context && pnpm install && pnpm run --filter @zilliz/claude-context-mcp build"
        )
    mcp_env = _build_context_env(env)
    return MCPClient(["node", _CONTEXT_MCP_DIST], env=mcp_env)


def get_filesystem_client(
    allowed_dirs: List[str],
    env: Optional[Dict[str, str]] = None,
) -> MCPClient:
    """
    Create an MCPClient connected to the Filesystem MCP server.

    The server is spawned from:
      <project_root>/node_modules/@modelcontextprotocol/server-filesystem/dist/index.js
    """
    if not os.path.isfile(_FS_MCP_DIST):
        raise FileNotFoundError(
            f"Filesystem MCP not built. Expected: {_FS_MCP_DIST}\n"
            "Run: npm install in constraint_modification directory"
        )
    cmd = ["node", _FS_MCP_DIST] + allowed_dirs
    return MCPClient(cmd, env=env or os.environ.copy())


def extract_text(result: Dict[str, Any]) -> str:
    """
    Extract the concatenated text content from an MCP tool-call result.

    MCP results have the shape ``{"content": [{"type": "text", "text": "..."}]}``.
    """
    if "error" in result:
        return f"ERROR: {result['error']}"

    content = result.get("content", [])
    if isinstance(content, str):
        return content

    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block["text"])
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(parts)


def mcp_search(
    repo_path: str,
    query: str,
    limit: int = 10,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Search indexed code via the Claude Context MCP server.

    Spawns a fresh MCP server, executes ``search_code``, and shuts down.

    Args:
        repo_path: Absolute path to the indexed codebase directory.
        query: Natural language search query.
        limit: Maximum number of results.
        env: Environment variable overrides.

    Returns:
        Raw MCP result dict (pass to ``extract_text`` for string output).
    """
    client = get_context_client(env)
    try:
        result = client.call(
            "search_code",
            {"path": repo_path, "query": query, "limit": limit},
            timeout_s=120.0,
        )
        return result
    finally:
        client.close()
