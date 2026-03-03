# Constraint Modification Pipeline

4-agent architecture for modifying school bus routing solver constraints via LLM-generated code edits.

## Agents

| Agent | Role | File |
|-------|------|------|
| **GA** (Generation Agent) | Generates constraint edit + unit tests | `src/core/contract_agent.py` |
| **JA** (Judgement Agent) | Verifies edit correctness (pre-checks + LLM review) | `src/core/judgement_agent.py` |
| **RA** (Revision Agent) | Fixes failed edits based on JA/EAA/unit test feedback | `src/core/revision_agent.py` |
| **EAA** (Error Analysis Agent) | Analyzes runtime errors structurally | `src/core/error_analysis_agent.py` |

## Pipeline Flow

```
GA → {edit, test_cases} → Pre-checks → JA → [RA loop] → Unit tests → [RA loop] → Toy test → [EAA → RA loop]
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 20+
- Docker (for Milvus)
- Ollama (for embeddings)

### 1. Install Python packages

```bash
cd constraint_modification
python3 -m venv venv
source venv/bin/activate
pip install -e ../school-solvers
pip install -e ../school-simulator
pip install -r requirements.txt
```

### 2. Start Milvus (vector database)

```bash
# Start all Milvus containers
docker restart milvus-etcd milvus-minio milvus-standalone

# Verify all 3 are healthy
docker ps --format "{{.Names}}\t{{.Status}}" | grep milvus
```

All 3 must be running: `milvus-etcd`, `milvus-minio`, `milvus-standalone`.

### 3. Start Ollama (embedding model)

```bash
ollama serve &
ollama pull nomic-embed-text
```

### 4. Build Claude Context MCP

```bash
cd /path/to/claude-context
pnpm install && pnpm build:mcp
```

### 5. Index the codebase

```bash
cd constraint_modification
source venv/bin/activate
python3 -m src.clients.index_client --config configs/config.py --force --clear
```

This indexes the 3 policy files (`insert_heuristic.py`, `policy_helpers.py`, `insertion_policy.py`) using the AST splitter. Each function becomes a searchable chunk with its docstring as metadata.

To re-index after code changes, run the same command.

## Running the Pipeline

```bash
cd constraint_modification
source venv/bin/activate
python3 -m src.core.pipeline --query-id 1 --agent-type contract
```

### Query IDs

Defined in `configs/queries.json`:

| ID | Query |
|----|-------|
| 1 | Stop 20006 must be served by bus 0 |
| 2 | Stop 20002 must not be served by bus 2 |
| 7 | Stop 20006 must be served before stop 20002 |
| 3 | Bus 2 capacity reduced to 12 |

### Output

Results are saved to `context_runs/{timestamp}/query_{id}/`:

```
logs/           # agent.log, prompts, responses, token usage
  edit.diff     # the applied diff
  constraint_test.py  # generated unit tests
temp_code/      # modified policy files
output/         # toy test results
```

## Configuration

- `configs/config.py` — paths, index files, search limit
- `configs/constraints.txt` — existing hard constraints (C1-C6) and soft objectives (O1-O4)
- `configs/queries.json` — test queries with entities

## Architecture Notes

- **Context retrieval**: Agents use targeted semantic search (`mcp_search`) to retrieve relevant function snippets instead of dumping all docstrings upfront. The constraints file (`constraints.txt`) is always included as a concise guide.
- **Unit tests**: GA generates tests that call the modified function directly with minimal mock data. No solver run needed. Tests verify REJECT/ACCEPT/NO SIDE EFFECT.
- **File isolation**: Edits go to `temp_code/policy/` — the index workspace stays clean.
