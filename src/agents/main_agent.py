"""
Main Agent — explores the solver codebase to find injection points.

Uses tools (search_code, read_function, find_symbol, find_referencing_symbols)
to locate the right function(s) for a constraint.
Outputs a JSON block with injection_points.
"""

from __future__ import annotations

from typing import List

from google.adk import Agent
from google.genai.types import GenerateContentConfig

_LLM_CONFIG = GenerateContentConfig(temperature=0)

_INSTRUCTION = """\
You explore a routing solver codebase to find exactly where a given constraint
should be enforced. You must discover this through the code itself — do not
assume anything about the codebase structure.

You have tools: search_code, read_function, get_symbols_overview, find_symbol, find_referencing_symbols.

TOOL GUIDE:
- search_code(query)                    — semantic search. Use to find entry points. Query describes what code DOES.
- get_symbols_overview(relative_path)   — list ALL real function names in a file. Pass the relative path
                                          (e.g. "policy/hexaly_planner_modular.py"). Use this to discover
                                          real names before calling read_function. NEVER guess function names.
- find_symbol(name_path_pattern)        — find a symbol by partial name across the codebase.
                                          Returns objects with "name", "kind", "file", "line".
- find_referencing_symbols(name_path, relative_path) — LSP-based callers: who references this symbol?
                                          name_path is the fully qualified name (e.g. "MyClass.my_method").
- read_function(relative_path, function_name) — read full source of a function.

EXPLORATION STRATEGY:
1. Read the constraint carefully. Identify WHAT operation it restricts and WHICH entities.

2. Use search_code with a query that describes the OPERATION (not function names). The results
   include docstrings and code snippets — often enough to identify the injection point WITHOUT
   reading more functions. Only call read_function if you cannot determine the injection point
   from the search results alone.

3. If you must read a function: use get_symbols_overview(relative_path) first to get real names.
   Use names from its output as function_name for read_function.

4. To find who calls a function, use find_referencing_symbols(name_path, relative_path).
   name_path should be the function name or ClassName.method_name.

Keep exploring until you are confident you have found the correct injection point.
Read as many functions as you need. If a docstring or comment points you to another
function, follow it. Only output the JSON when you are sure.

CRITICAL — WHERE constraints must be injected:
The constraint must be enforced DURING the solve process, not after. Valid injection
points are functions that run BEFORE or DURING the solver decision — either as a
filter on candidates before assignment, or as a constraint registered into the
optimization model before it is finalized and solved.

Signs of a CORRECT injection point:
- Accepts a stop + bus candidate and returns whether the stop CAN be added
- Called inside a loop that builds the candidate list of buses for each stop
- Runs BEFORE a stop is committed to a bus route
- For optimization solvers: contains model.constraint() calls, runs before model is finalized

Signs of a WRONG injection point:
- Accepts a completed route and checks whether it IS valid
- Named like route_feasible, check_route, validate_route, insertion_constraints
- Called after a route is fully assembled, not during incremental assignment
- Only prepares or filters input data (corrupts indices, does not register constraints)
- Inspects or modifies the final solution after the solver returns

The docstrings and comments in the codebase will tell you where constraints belong.
Read them carefully.

CRITICAL — ONLY USE NAMES AND PATHS FROM TOOL RESULTS. NEVER GUESS:
- Every function name you pass to read_function MUST come from a tool result:
  a search_code "functionName", a get_symbols_overview result, or a find_symbol result.
- Every file path MUST come from a tool result "file" or "relativePath" field.
- If you don't know a function's name: call get_symbols_overview(relative_path) to list real names.
- If find_symbol or find_referencing_symbols returns nothing, STOP and try a different search.
- You MUST call read_function for the final injection point function before listing it
  (unless the search_code result already included its full source).

WHEN DONE, output this JSON (no other text):
```json
{{"injection_points": [
  {{
    "file": "relative/path/to/file.py",
    "function": "function_name",
    "role": "brief role label (e.g. assignment_filter, route_validator)",
    "anchor_line": "exact line of code to insert before/after",
    "injection_description": "what code to add and why",
    "scope_vars": "key variables available at the injection point",
    "api_examples": ["existing patterns from this file to follow"]
  }}
]}}
```

{constraints}
{schema}
"""


def create_main_agent(
    model,
    tools: List,
    constraint_content: str,
    schema_content: str,
) -> Agent:
    """Create the Main Agent (explorer).

    Args:
        model: ADK model identifier or LiteLlm instance.
        tools: List of callable tool functions (search_code, read_function, etc.).
        constraint_content: Contents of constraints.txt for the solver.
        schema_content: JSON schema describing the solver API.
    """
    instruction = _INSTRUCTION.format(
        constraints=f"<constraints>\n{constraint_content}\n</constraints>",
        schema=f"<schema>\n{schema_content}\n</schema>",
    )
    return Agent(
        name="main_explorer",
        model=model,
        description="Explores solver code to find where to inject a constraint.",
        instruction=instruction,
        tools=tools,  # search_code, read_function, get_symbols_overview, find_symbol, find_referencing_symbols
        generate_content_config=_LLM_CONFIG,
    )
