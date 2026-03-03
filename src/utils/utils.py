import json
import os
import re
from typing import Any, Dict, List


def load_queries(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "queries" in data and isinstance(data["queries"], list):
            return data["queries"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return [data]
    raise ValueError(f"Unsupported queries format in {path}")


def pick_query(queries: List[Dict[str, Any]], query_id: int) -> Dict[str, Any]:
    for q in queries:
        if int(q.get("query_id", -1)) == int(query_id):
            return q
    raise ValueError(f"query_id {query_id} not found in {queries}")


def load_env_file(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not os.path.exists(path):
        return env
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'").strip('"')
    return env


def dedupe_results(raw_text: str) -> List[Dict[str, Any]]:
    """
    Parse the formatted text output from Claude Context MCP ``search_code``
    into a list of structured result dicts.

    Each result dict contains:
        - relativePath (str)
        - startLine (int)
        - endLine (int)
        - functionName (str or None)
        - nodeType (str or None)
        - className (str or None)
        - callsTo (list[str] or None)
        - calledBy (list[str] or None)
        - docstring (str or None)
        - content (str)
        - rank (int)
        - language (str)

    The MCP output format looks like::

        1. Code snippet (python) [codebaseInfo]
           Location: path/file.py:10-50
           Function: funcName (method_definition)
           Docstring: some text...
           Calls: func1, func2
           Called by: func3
           Rank: 1
           Context:
        ```python
        code here
        ```
    """
    if not raw_text or raw_text.startswith("ERROR:"):
        return []

    results: List[Dict[str, Any]] = []

    # Split on result headers: "N. Code snippet (...) [...]"
    # Pattern matches lines like "1. Code snippet (python) [index_workspace]"
    header_pattern = re.compile(
        r'^(\d+)\.\s+Code snippet\s+\((\w+)\)\s+\[([^\]]*)\]',
        re.MULTILINE,
    )

    matches = list(header_pattern.finditer(raw_text))
    if not matches:
        return []

    for i, match in enumerate(matches):
        # Extract the block for this result (from this header to the next header)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        block = raw_text[start:end]

        language = match.group(2)

        result: Dict[str, Any] = {
            "relativePath": None,
            "startLine": None,
            "endLine": None,
            "functionName": None,
            "nodeType": None,
            "className": None,
            "callsTo": None,
            "calledBy": None,
            "docstring": None,
            "content": "",
            "rank": int(match.group(1)),
            "language": language,
        }

        # Parse Location: path:startLine-endLine
        loc_match = re.search(r'Location:\s*(\S+):(\d+)-(\d+)', block)
        if loc_match:
            result["relativePath"] = loc_match.group(1)
            result["startLine"] = int(loc_match.group(2))
            result["endLine"] = int(loc_match.group(3))

        # Parse Function: name (nodeType)  or  Function: ClassName.name (nodeType)
        func_match = re.search(r'Function:\s*(?:(\w+)\.)?(\S+?)(?:\s+\((\w+)\))?\s*$', block, re.MULTILINE)
        if func_match:
            result["className"] = func_match.group(1)  # may be None
            result["functionName"] = func_match.group(2)
            result["nodeType"] = func_match.group(3)  # may be None
        else:
            # Simpler format: "Function: name"
            simple_func = re.search(r'Function:\s*(\S+)', block)
            if simple_func:
                result["functionName"] = simple_func.group(1)

        # Parse Docstring: ...
        doc_match = re.search(r'Docstring:\s*(.+?)(?=\n\s{3}\w|\n```|\Z)', block, re.DOTALL)
        if doc_match:
            result["docstring"] = doc_match.group(1).strip()

        # Parse Calls: func1, func2, ...
        calls_match = re.search(r'Calls:\s*(.+)', block)
        if calls_match:
            calls_str = calls_match.group(1).strip()
            result["callsTo"] = [c.strip() for c in calls_str.split(",") if c.strip()]

        # Parse Called by: func1, func2, ...
        called_by_match = re.search(r'Called by:\s*(.+)', block)
        if called_by_match:
            cb_str = called_by_match.group(1).strip()
            result["calledBy"] = [c.strip() for c in cb_str.split(",") if c.strip()]

        # Parse Rank
        rank_match = re.search(r'Rank:\s*(\d+)', block)
        if rank_match:
            result["rank"] = int(rank_match.group(1))

        # Extract code content from the fenced code block
        code_match = re.search(r'```\w*\n(.*?)```', block, re.DOTALL)
        if code_match:
            result["content"] = code_match.group(1)

        # Dedupe: skip if we already have a result with the same path + lines
        key = (result["relativePath"], result["startLine"], result["endLine"])
        if not any(
            (r["relativePath"], r["startLine"], r["endLine"]) == key
            for r in results
        ):
            results.append(result)

    return results
