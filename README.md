# Constraint Modification (OPTL-Only)

This repository now uses a single constraint path:

`Natural language -> LLM -> OPTL -> backend codegen -> injected temp policy -> toy tests`

Legacy edit-diff pipeline code is no longer the active implementation.

## Core Entry Point

Use `run_optl.py` for all runs.

The OPTL compiler now lives in the local [`optl/`](/Users/vakulnath/constraint-solver/constraint_modification/optl) package and is imported as the installed Python package `optl`.

```bash
cd constraint_modification
.venv/bin/python3 src/core/run_optl.py --solver both --batch-size 7
```

Useful variants:

```bash
# Insertion only
.venv/bin/python3 src/core/run_optl.py --solver insertion

# Both (insertion + hexaly, where hexaly uses insertion warm start)
.venv/bin/python3 src/core/run_optl.py --solver both

# Specific queries
.venv/bin/python3 src/core/run_optl.py --solver both --queries 19,20,30,31
```

## What OPTL Does

1. Parses generated constraints into an AST.
2. Compiles AST to backend-specific code:
   - insertion: `_optl_constraints.py`
   - hexaly: `_optl_hexaly_constraints.py`
3. Injects hooks into temporary solver copies in `context_runs/.../temp_code/policy`.
4. Runs toy tests against the temporary code.

No base solver files are modified in-place during evaluation runs.

## Output Layout

Each run writes to:

`context_runs/{timestamp}_optl/`

Per query+solver, key files are:

- `result.json`
- `logs/pipeline.log`
- `logs/artifacts/round_XX_constraint.optl`
- `logs/artifacts/round_XX_generated_backend.py`
- `logs/artifacts/round_XX_*diff`
- `temp_code/policy/...` (temporary injected policy files)
- `output/...` (toy test outputs such as `bus_schedule_after.json` when produced)

## Compatibility Wrappers

These modules now delegate to the OPTL flow for backward compatibility:

- `src/core/api_agent.py`
- `src/core/run_all.py`
- `src/core/run_10x.py`

Use `run_optl.py` directly for authoritative behavior.
