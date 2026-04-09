#!/usr/bin/env python3
"""
Test script to verify callsTo metadata and docstrings from Claude Context MCP.

This script:
1. Searches for code using Claude Context MCP
2. Checks if callsTo metadata is returned
3. Checks if docstrings appear in the content
4. Prints the raw output to see what's actually coming back
"""

import json
import os
import sys

# Add the src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from mcp.context import mcp_search, extract_text, get_context_client
from utils.utils import dedupe_results, load_env_file


def test_callsto_metadata():
    """Test if callsTo metadata is being returned from Claude Context MCP."""

    # Load environment from parent directory
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    env = {**os.environ, **load_env_file(env_path)}

    # Indexed insertion workspace path (created by index_client.py)
    repo_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "context_runs",
        "index_workspace",
        "insertion",
    )

    # Test queries - these should find functions that CALL other functions
    test_queries = [
        "replan_bus_routes",  # Should show calls to evaluate_bus_candidate
        "evaluate_bus_candidate",  # Should show calls to simulate_arrivals_detailed
        "insertion_constraints",  # Should show calls to simulate_arrivals_detailed
    ]

    print("=" * 80)
    print("TESTING CALLSTO METADATA FROM CLAUDE CONTEXT MCP")
    print("=" * 80)
    print(f"Repo path: {repo_path}")
    print()

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print("=" * 60)

        # Execute search
        result = mcp_search(repo_path, query, limit=5, env=env)
        raw_text = extract_text(result)

        # Print raw output (first 3000 chars)
        print("\n--- RAW MCP OUTPUT (first 3000 chars) ---")
        print(raw_text[:3000])
        if len(raw_text) > 3000:
            print(f"... [{len(raw_text) - 3000} more chars]")

        # Parse results
        print("\n--- PARSED RESULTS ---")
        parsed = dedupe_results(raw_text)

        if not parsed:
            print("No results parsed!")
            continue

        for i, r in enumerate(parsed):
            print(f"\n[{i+1}] {r.get('relativePath')}:{r.get('startLine')}-{r.get('endLine')}")
            print(f"    functionName: {r.get('functionName', '(not found)')}")
            print(f"    nodeType: {r.get('nodeType', '(not found)')}")
            print(f"    className: {r.get('className', '(not found)')}")
            print(f"    callsTo: {r.get('callsTo', '(not found)')}")
            print(f"    calledBy: {r.get('calledBy', '(not found)')}")

            # Check if callsTo exists
            if r.get('callsTo'):
                print(f"    ✓ callsTo metadata FOUND: {r['callsTo']}")
            else:
                print(f"    ✗ callsTo metadata NOT FOUND")

            # Check if docstring exists in content
            content = r.get('content', '')
            has_docstring = '"""' in content or "'''" in content
            if has_docstring:
                # Extract first docstring
                if '"""' in content:
                    start = content.find('"""')
                    end = content.find('"""', start + 3)
                    if end > start:
                        docstring_preview = content[start:end+3][:200]
                        print(f"    ✓ DOCSTRING FOUND: {docstring_preview}...")
                else:
                    print("    ✓ DOCSTRING FOUND (single quotes)")
            else:
                print("    ✗ NO DOCSTRING in content")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


def test_raw_mcp_response():
    """Print the raw JSON response from MCP to see exact structure."""

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    env = {**os.environ, **load_env_file(env_path)}

    # Indexed insertion workspace path (created by index_client.py)
    repo_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "context_runs",
        "index_workspace",
        "insertion",
    )

    print("\n" + "=" * 80)
    print("RAW MCP CLIENT RESPONSE (before extract_text)")
    print("=" * 80)

    client = get_context_client(env)
    result = client.call(
        "search_code",
        {"path": repo_path, "query": "replan_bus_routes", "limit": 3},
        timeout_s=120.0,
    )

    print("\n--- Full MCP Response ---")
    print(json.dumps(result, indent=2, default=str)[:5000])


if __name__ == "__main__":
    test_callsto_metadata()
    print("\n\n")
    test_raw_mcp_response()
