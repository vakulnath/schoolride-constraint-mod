"""Test ONLY the Main Agent explore phase for all 35 queries.

Reports which function the agent found for each query.
No GA, no test, no EAA — just explore.
"""

import asyncio
import json
import os
import sys

_project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _project_root)

with open(os.path.join(os.path.dirname(__file__), "..", "..", ".env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

from google.adk import Agent
from google.genai.types import GenerateContentConfig
from src.core.pipeline import (
    _make_tools, _run_agent, _parse_injection_points,
    _LLM_CONFIG,
)
from src.core.api_agent_v2 import PathfinderClient
from src.configs import config_parser as cfg
from src.core._archive.api_agent import _load_constraints, _load_schema


async def test_explore(solver_type: str, queries: list):
    """Run explore for each query and report results."""

    env = os.environ.copy()
    constraint_content = _load_constraints()
    schema_content = _load_schema(".", solver_type)
    model_name = cfg.DEFAULT_MODEL

    if solver_type == "hexaly":
        _index_files = set(os.path.basename(f) for f in cfg.HEXALY_INDEX_FILES)
    else:
        _index_files = set(os.path.basename(f) for f in cfg.INDEX_FILES)

    # Shared pathfinder
    policy_root = str(cfg.SOLVERS_POLICY_ROOT) if cfg.SOLVERS_POLICY_ROOT else "."
    pf = PathfinderClient(policy_root)
    pf.start()

    # Pre-discover
    solver_keywords = ["solver", "replan", "evaluate", "optimize", "planner", "constraint"]
    discovered = []
    for kw in solver_keywords:
        result = pf.call_tool("find_symbol", {"name": kw, "type": "function_definition"})
        for m in result.get("matches", []):
            fname = os.path.basename(m.get("file", ""))
            if fname in _index_files:
                discovered.append(f"  - {m.get('fqn', '')} ({fname}:{m.get('line', 0)})")
    pre_discovery = ""
    if discovered:
        pre_discovery = "\n<discovered_functions>\n" + "\n".join(discovered[:15]) + "\n</discovered_functions>\n"

    tools = _make_tools(solver_type, env, pf)

    main_agent = Agent(
        name="main_explorer",
        model=model_name,
        description="Explores solver code to find where to inject a constraint.",
        instruction=(
            "You explore a routing solver codebase to find where a constraint should be enforced.\n"
            "You have tools: search_code, read_function, find_symbol, get_callers, get_call_details.\n\n"
            "WORKFLOW:\n"
            "1. Look at the <discovered_functions> list in the user message. These were found\n"
            "   via call graph analysis and are the most likely injection points.\n"
            "2. Use read_function on the most promising discovered function — read its docstring\n"
            "   to check if it has the right variables in scope (bus_id, stop, etc.).\n"
            "3. The injection point must be where the solver DECIDES eligibility — where it\n"
            "   accepts or rejects a candidate. Validation functions that check AFTER insertion\n"
            "   are NOT the right place. Read the docstring to tell the difference.\n"
            "4. Once you find the right function, output the injection_points JSON.\n\n"
            "BE SELECTIVE: Read 1 function, check its docstring, output JSON. Do not read more.\n\n"
            "Output format:\n"
            "```json\n"
            '{"injection_points": [{"file": "...", "function": "...", "role": "...", '
            '"anchor_line": "exact code line", "injection_description": "what to add", '
            '"scope_vars": "vars in scope", "api_examples": ["existing constraint pattern"]}]}\n'
            "```\n\n"
            f"<constraints>\n{constraint_content}\n</constraints>\n"
            f"<schema>\n{schema_content}\n</schema>\n"
        ),
        tools=tools,
        generate_content_config=_LLM_CONFIG,
    )

    results = []
    for q in queries:
        qid = q["query_id"]
        constraint = q["question"]
        qtype = q["type"]

        user_msg = (
            f"<constraint>\n{constraint}\n</constraint>\n"
            + pre_discovery
            + "\nFind the ONE function where this constraint should be enforced. "
            "Read it, extract api_examples, output the injection_points JSON."
        )

        try:
            text = await _run_agent(main_agent, user_msg, timeout=180)
            ips = _parse_injection_points(text)
            if ips:
                funcs = ", ".join(f"{ip['function']} ({ip['file']})" for ip in ips)
                print(f"Q{qid:2d} ({qtype:16s}): {funcs}")
                results.append((qid, qtype, True, funcs))
            else:
                print(f"Q{qid:2d} ({qtype:16s}): NO INJECTION POINTS")
                results.append((qid, qtype, False, "none"))
        except Exception as e:
            print(f"Q{qid:2d} ({qtype:16s}): ERROR — {str(e)[:80]}")
            results.append((qid, qtype, False, str(e)[:80]))

        sys.stdout.flush()
        await asyncio.sleep(2)  # gentle rate limiting

    pf.stop()

    # Summary
    print(f"\n{'='*60}")
    found = sum(1 for _, _, ok, _ in results if ok)
    print(f"Found injection points: {found}/{len(results)}")
    print(f"\nFailures:")
    for qid, qtype, ok, detail in results:
        if not ok:
            print(f"  Q{qid:2d} ({qtype}): {detail}")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="insertion", choices=["insertion", "hexaly"])
    parser.add_argument("--queries", default="all")
    args = parser.parse_args()

    os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
    qs = json.load(open("src/configs/queries.json"))["queries"]

    if args.queries != "all":
        query_ids = set(int(x) for x in args.queries.split(","))
        qs = [q for q in qs if q["query_id"] in query_ids]

    await test_explore(args.solver, qs)


if __name__ == "__main__":
    asyncio.run(main())
