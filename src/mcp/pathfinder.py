"""
PathfinderClient — manages a pathfinder MCP server for structural code analysis.

Provides: find_symbol, get_callers, get_call_details (call graph queries).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

# Prefer venv-local pathfinder binary so it works without system PATH setup
_VENV_BIN = os.path.join(os.path.dirname(sys.executable))
_PATHFINDER_BIN = os.path.join(_VENV_BIN, "pathfinder")
if not os.path.isfile(_PATHFINDER_BIN):
    _PATHFINDER_BIN = "pathfinder"  # fall back to PATH


class PathfinderClient:
    """Manages a pathfinder MCP server subprocess for structural code queries."""

    def __init__(self, project_path: str):
        # Pathfinder requires absolute paths — relative paths cause 0 functions indexed
        self.project_path = os.path.abspath(project_path)
        self.proc: Optional[subprocess.Popen] = None
        self._responses: Dict[int, str] = {}
        self._reader_thread = None
        self._lock = None

    def start(self) -> bool:
        """Start pathfinder server and initialize MCP protocol."""
        import threading
        self._lock = threading.Lock()
        try:
            self.proc = subprocess.Popen(
                [_PATHFINDER_BIN, "serve", "--project", self.project_path, "--no-banner"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            def _reader():
                while self.proc and self.proc.poll() is None:
                    try:
                        assert self.proc.stdout is not None
                        line = self.proc.stdout.readline()
                        if not line:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        d = json.loads(line)
                        msg_id = d.get("id")
                        if msg_id is not None:
                            content = d.get("result", {}).get("content", [])
                            text_parts = [c.get("text", "") for c in content]
                            result_text = "".join(text_parts)
                            if not result_text:
                                result_text = json.dumps(d.get("result", {}))
                            assert self._lock is not None
                            with self._lock:
                                self._responses[msg_id] = result_text
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
            assert self.proc.stdin is not None
            self.proc.stdin.write(init_msg + "\n")
            self.proc.stdin.flush()
            time.sleep(5)  # Wait for index to build
            return True
        except FileNotFoundError:
            return False
        except Exception:
            return False

    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a pathfinder MCP tool and return the result."""
        if not self.proc or self.proc.poll() is not None:
            return {"error": "Pathfinder not running"}
        msg_id = int(time.time() * 1000) % 100000
        msg = json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args},
            "id": msg_id,
        })
        assert self.proc.stdin is not None
        self.proc.stdin.write(msg + "\n")
        self.proc.stdin.flush()

        for _ in range(40):  # Up to 4 seconds
            time.sleep(0.1)
            assert self._lock is not None
            with self._lock:
                if msg_id in self._responses:
                    result_text = self._responses.pop(msg_id)
                    try:
                        return json.loads(result_text)
                    except json.JSONDecodeError:
                        return {"result": result_text}

        return {"error": f"Timeout waiting for pathfinder response to {tool_name}"}

    def stop(self):
        """Stop the pathfinder server."""
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
