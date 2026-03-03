#!/usr/bin/env python3
"""
Test the constraint agent.
"""

import os
import sys

# Add path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.constraint_agent import run_constraint_agent
from src.utils.utils import load_env_file


def main():
    # Load environment - start with system env, then overlay .env file
    env = dict(os.environ)
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        env_vars = load_env_file(env_path)
        env.update(env_vars)
        # Also set in os.environ so llm_client can find them
        os.environ.update(env_vars)

    # Context root is the indexed workspace
    context_root = os.path.join(os.path.dirname(__file__), "..", "context_runs", "index_workspace")

    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(__file__), "test_constraint_agent_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Test constraint
    constraint = "What if stop 20006 must be served by bus 0?"

    print(f"Running constraint agent for: {constraint}")
    print(f"Logs will be written to: {logs_dir}")
    print()

    result = run_constraint_agent(
        constraint=constraint,
        context_root=context_root,
        env=env,
        logs_dir=logs_dir,
        max_iterations=10,
    )

    print("\n=== Result ===")
    print(f"Success: {result.get('success')}")
    print(f"Iterations: {result.get('iterations')}")

    if result.get("edit"):
        print(f"\nEdit:")
        print(f"  Path: {result['edit'].get('path')}")
        print(f"  Success: {result['edit'].get('success')}")
        if result['edit'].get('diff'):
            print(f"  Diff:\n{result['edit']['diff']}")

    if result.get("error"):
        print(f"\nError: {result['error']}")

    if result.get("thoughts"):
        print(f"\nThoughts ({len(result['thoughts'])}):")
        for i, thought in enumerate(result["thoughts"], 1):
            print(f"  {i}. {thought[:100]}...")


if __name__ == "__main__":
    main()
