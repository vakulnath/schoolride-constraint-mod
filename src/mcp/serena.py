"""
SerenaClient — manages a Serena MCP server subprocess for LSP-backed code navigation.

Provides: get_symbols_overview, find_symbol, find_referencing_symbols
Uses Pyright (LSP) for accurate cross-file symbol resolution and reference finding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

_UVX = os.path.expanduser("~/.local/bin/uvx")
if not os.path.isfile(_UVX):
    _UVX = "uvx"

_SERENA_FROM = "git+https://github.com/oraios/serena"


class SerenaClient:
    """Manages a Serena MCP server subprocess for LSP-backed code navigation."""

    def __init__(self, project_path: str):
        # Serena requires absolute paths
        self.project_path = os.path.abspath(project_path)
        self.proc: Optional[subprocess.Popen] = None
        self._responses: Dict[int, Any] = {}
        self._lock: Optional[threading.Lock] = None
        self._reader_thread: Optional[threading.Thread] = None

    def start(self) -> bool:
        """Start Serena MCP server and wait for it to be ready."""
        self._lock = threading.Lock()
        try:
            cmd = [
                _UVX, "-p", "3.13", "--from", _SERENA_FROM,
                "serena", "start-mcp-server",
                "--project", self.project_path,
                "--transport", "stdio",
                "--enable-web-dashboard", "False",
                "--open-web-dashboard", "False",
            ]
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            def _reader():
                assert self.proc and self.proc.stdout
                while self.proc.poll() is None:
                    try:
                        line = self.proc.stdout.readline()
                        if not line.strip():
                            continue
                        d = json.loads(line.strip())
                        msg_id = d.get("id")
                        if msg_id is not None:
                            assert self._lock
                            with self._lock:
                                self._responses[msg_id] = d
                    except (json.JSONDecodeError, ValueError):
                        continue
                    except Exception:
                        break

            self._reader_thread = threading.Thread(target=_reader, daemon=True)
            self._reader_thread.start()

            # MCP initialize handshake
            init_msg = json.dumps({
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "constraint_solver", "version": "1.0"},
                },
                "id": 0,
            })
            assert self.proc.stdin
            self.proc.stdin.write(init_msg + "\n")
            self.proc.stdin.flush()

            # Wait for Serena to start LSP and index the project (~20s)
            time.sleep(25)
            if self.proc.poll() is not None:
                return False

            # Activate the project — try by absolute path first, then by basename
            for project_ref in [self.project_path, os.path.basename(self.project_path)]:
                result = self.call_tool("activate_project", {"project": project_ref})
                result_str = str(result)
                if "is activated" in result_str or "activated" in result_str.lower():
                    break
            return self.proc.poll() is None

        except FileNotFoundError:
            return False
        except Exception:
            return False

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Call a Serena MCP tool and return the parsed result."""
        if not self.proc or self.proc.poll() is not None:
            return {"error": "Serena not running"}

        msg_id = int(time.time() * 1000) % 100000
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
            "id": msg_id,
        })
        assert self.proc.stdin
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()

        # Serena LSP calls can take a few seconds
        for _ in range(150):  # up to 15 seconds
            time.sleep(0.1)
            assert self._lock
            with self._lock:
                if msg_id in self._responses:
                    resp = self._responses.pop(msg_id)
                    # Extract text content from MCP response
                    content = resp.get("result", {}).get("content", [])
                    if content and isinstance(content, list):
                        text = content[0].get("text", "")
                        try:
                            return json.loads(text)
                        except json.JSONDecodeError:
                            return {"result": text}
                    error = resp.get("error")
                    if error:
                        return {"error": str(error)}
                    return {"result": resp.get("result")}

        return {"error": f"Timeout waiting for Serena response to {tool_name}"}

    def stop(self):
        """Stop the Serena server."""
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.communicate(timeout=5)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def __del__(self):
        self.stop()
