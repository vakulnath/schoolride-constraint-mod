# Constraint Modification Pipeline

LLM-powered system for modifying production routing solvers to satisfy natural
language constraints expressed by a dispatcher during a live school-bus disruption.

---

## Architecture Overview

Two parallel pipelines, selected by environment variable:

```
Natural Language constraint
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│  LLM Pipeline (default)         DSL Pipeline (USE_DSL=1)│
│                                                         │
│  Main Agent (finds injection)   nl_to_routing_optl()    │
│       ↓                              ↓                  │
│  GA → JA (generate + verify)    OPTL source (IR)        │
│       ↓                              ↓                  │
│  Apply edit                     compile_optl()          │
│       ↓                              ↓                  │
│  Toy test                       snippet + GA locates     │
│       ↓                         injection point         │
│  EAA → RA (fix if failed)            ↓                  │
│       ↓                         Apply + test + fix      │
└─────────────────────────────────────────────────────────┘
           │
           ▼
    schedule_diff.py (constraint verification)
```

### LLM Pipeline (5 agents)
1. **Main Agent** — explores codebase with tools, finds injection point, outputs JSON
2. **Generation Agent (GA)** — generates `old_text`/`new_text` XML edit
3. **Judgment Agent (JA)** — validates edit pre-execution
4. **Error Analysis Agent (EAA)** — diagnoses toy test failures
5. **Revision Agent (RA)** — corrects failed edits

### DSL Pipeline (USE_DSL=1)
- **Phase 0**: NL → OPTL (one LLM call) → compile to verified snippet (deterministic)
- **Phase 1**: Main Agent finds injection point (same as LLM pipeline)
- **Phase 2**: GA locates `old_text`, wraps pre-compiled snippet
- **Phase 3+4**: same apply/test/fix loop
- Falls back to LLM pipeline automatically if DSL compile fails

---

## OPTL as Common Interface

OPTL (`DSL_OR/`) is the **common interface** between both solvers.
The same constraint is defined once and compiled to both backends:

```
PROBLEM must_assign
PARAM stop_id = 20006
PARAM bus_id = 2
SUBJECT TO
  must_assign: stop_id == bus_id
SOLVE WITH INSERTION        ← swap for HEXALY_ROUTING
```

Two routing backends added to OPTL:
- `INSERTION` → guard clause for `evaluate_bus_candidate()` in `insert_heuristic.py`
- `HEXALY_ROUTING` → `model.constraint()` calls for `hexaly_planner_modular.py`

See `DSL_OR/README.md` for full OPTL documentation.

---

## Running

```bash
# LLM pipeline (default — Gemini via ADK)
python -m src.core.run --solver insertion --queries 1,2,3

# DSL pipeline
USE_DSL=1 python -m src.core.run --solver insertion --queries 1,2,3

# Groq backend
USE_GROQ=1 python -m src.core.run --solver insertion

# Both solvers
python -m src.core.run --solver both --batch-size 7
```

---

## Environment Variables

| Variable          | Effect                                      |
|-------------------|---------------------------------------------|
| `USE_DSL=1`       | DSL pipeline (NL → OPTL → snippet)         |
| `USE_GROQ=1`      | Groq via ADK LiteLLM                        |
| `USE_CLAUDE_CLI=1`| Claude CLI backend                          |
| `USE_PROMPT_TOOLS=1` | Amplify prompt-tools backend             |
| default           | Gemini via Google ADK                       |

---

## Setup on a New Machine

```bash
git clone https://github.com/vakulnath/schoolride-constraint-mod constraint_modification
cd constraint_modification

# DSL_OR has its own .git — clone separately
# (TODO: make submodule or fork to vakulnath/DSL_OR)
git clone https://github.com/jptalusan/DSL_OR DSL_OR
# Then copy optl/routing/ and updated optl/backends/__init__.py from another machine
# OR rsync from Mac: rsync -avz --exclude='.git' -e "ssh -p 222" \
#   /path/to/DSL_OR/ vakul@ds3:~/constraint-solver/constraint_modification/DSL_OR/

# Install dependencies
pip install hexaly -i https://pip.hexaly.com
pip install -e .

# Copy .env with API keys and paths
cp /path/to/.env .env

# Set Hexaly license
export HX_LICENSE_PATH=/opt/hexaly_14_5/license.dat
```

### Required `.env` keys
```
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...        # for DSL NL parsing
GOOGLE_API_KEY=...           # alternative for DSL NL parsing
GEMINI_MODEL=gemini-2.0-flash
SOLVERS_SRC=./schoolride-solvers/src
SOLVERS_POLICY_ROOT=./schoolride-solvers/src/policy
SIM_ROOT=./schoolride-simulator
SIM_SRC=./schoolride-simulator/src
```

---

## Project Structure

```
constraint_modification/
├── src/
│   ├── agents/
│   │   ├── main_agent.py           — explores codebase, finds injection points
│   │   ├── generation_agent.py     — generates code edits
│   │   ├── judgment_agent.py       — validates edits pre-execution
│   │   ├── error_analysis_agent.py — diagnoses test failures
│   │   └── revision_agent.py       — corrects failed edits
│   ├── backends/
│   │   ├── native.py               — LLM pipeline + DSL pipeline (run_native_pipeline, run_dsl_pipeline)
│   │   ├── claude_cli.py           — Claude CLI backend
│   │   └── prompt_tools.py         — Amplify backend
│   ├── core/
│   │   ├── pipeline.py             — dispatcher (USE_DSL check here)
│   │   ├── run.py                  — main entry point
│   │   └── validation.py           — per-type constraint validators
│   ├── dsl/
│   │   └── adapter.py              — compile_for_solver(), build_dsl_ga_prompt()
│   ├── configs/
│   │   └── config_parser.py        — loads .env, resolves paths
│   ├── mcp/
│   │   └── serena.py               — LSP-based code navigation
│   └── utils/
│       ├── schedule_diff.py        — post-execution constraint verification (7 types)
│       ├── code_tools.py           — search/read/apply edit tools
│       └── test_runner.py          — toy test runner
├── DSL_OR/                         — OPTL compiler (see DSL_OR/README.md)
│   └── optl/routing/               — NEW: routing constraint DSL module
├── schoolride-solvers/             — solver source (local editable install)
│   └── src/policy/
│       ├── insert_heuristic.py     — insertion solver (evaluate_bus_candidate = injection point)
│       └── hexaly_planner_modular.py — hexaly solver (modular, clean injection hooks)
├── schoolride-simulator/           — simulator source (local editable install)
├── pyproject.toml                  — deps: optl @ DSL_OR, school-solvers, school-simulator
└── .env                            — API keys + paths (NOT in git)
```

---

## Constraint Types (7 total)

| Type             | Entities                    | Verified by schedule_diff.py |
|------------------|-----------------------------|------------------------------|
| `must_assign`    | stop_id, bus_id             | ✓                            |
| `must_not_assign`| stop_id, bus_id             | ✓                            |
| `same_bus`       | stop_id_1, stop_id_2        | ✓                            |
| `not_same_bus`   | stop_id_1, stop_id_2        | ✓                            |
| `ordering`       | stop_id_1, stop_id_2        | ✓                            |
| `capacity`       | bus_id, capacity            | ✓                            |
| `max_stops`      | bus_id, max_stops           | ✓                            |

---

## Key Design Decisions

**Why two pipelines?**
LLM pipeline handles complex/ambiguous constraints. DSL pipeline handles standard
constraint types deterministically with verified snippets. DSL is the baseline
for comparison in the paper.

**Why not one function per solver?**
`evaluate_bus_candidate` handles bus-level eligibility but lacks `inserted_nodes`
(cross-bus assignment state). Same_bus/ordering sometimes need `replan_bus_routes`.
hexaly_planner_modular.py already has clean separated hooks: `_add_per_vehicle_constraints()`
and `_build_global_objective()`.

**Why not use existing HEXALY/HEURISTIC backends?**
They generate complete standalone solver programs. We need snippets that inject
into an already-running solver with existing state. The routing backends
(INSERTION, HEXALY_ROUTING) generate only the constraint logic — no imports,
no optimizer setup, no solve call.

**Is it truly solver-agnostic?**
At the OPTL level yes — same constraint definition compiles to both backends.
At the backend level no — backends are coupled to specific variable names in
our solvers. True solver-agnosticism requires standardized hook functions in
each solver (small refactor, not done yet).

---

## Paper Positioning

| System    | Problem                        | Approach                |
|-----------|-------------------------------|-------------------------|
| AFL       | New solver for VRP benchmarks  | Generate complete solver|
| OptiChat  | Add constraints to Pyomo/MILP  | Re-solve with Gurobi    |
| **Ours**  | Modify production solver during live disruption | LLM modifies existing code |

Key differentiators:
- Heterogeneous legacy solvers (insertion heuristic + Hexaly PDPTW)
- Live operational state cannot be discarded (warm start, current locations)
- Human-guided: dispatcher NL query during active disruption
- OPTL as common interface across solver architectures
