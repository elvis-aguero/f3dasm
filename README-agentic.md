# agentic-f3dasm

---

## Summary

`agentic-f3dasm` adds an LLM-orchestrated loop on top of f3dasm:

```bash
uv run python -m f3dasm.agentic <study-dir>
```

Drop a `PROBLEM_STATEMENT.md` into a study directory and the runtime will orchestrate a graph of LLM agents that read, plan, write and execute Python code, and return a structured result. The only required input is a natural-language problem description.

The default topology is two nodes:

- a **Strategizer** that reads the study tree, forms hypotheses, delegates concrete tasks, and signals `Done` when results are satisfactory;
- an **Implementer** that receives each task, writes and executes Python inside `workspace/`, and returns a structured `## Report`.

---

## Architecture

### LangGraph under the hood

The runtime compiles an `Agent`/`Graph` spec into a LangGraph `StateGraph`. Each agent definition becomes a LangGraph node; routing is handled by `Command(goto=...)` driven by the closure tools the agent calls. `AgenticState` (a `MessagesState` subclass) carries messages, budget counters, and routing metadata across turns.

Users never interact with LangGraph directly — the `Agent`/`Graph`/`Edge` API is the only surface.

### Agent definitions

Every node in the topology is a Python class that subclasses `Agent`. You configure it with class-level attributes:

```python
from f3dasm.agentic.agents import StrategizerAgent, ImplementerAgent
from f3dasm.agentic import Edge, Graph, AgenticRun

class MyOrchestrator(StrategizerAgent):
    system_prompt = "You coordinate the search."
    tools = frozenset({"Done", "Ask", "WriteMarkdown", "ReadNote"})
    reset_on_checkpoint = False

class MyWorker(ImplementerAgent):
    model = "claude-haiku-4-5-20251001"
    tools = frozenset({"Bash", "Read", "Write", "Edit", "Glob", "Grep", "ReportEvals"})

graph = Graph(
    nodes={"orch": MyOrchestrator(), "worker": MyWorker()},
    edges=(Edge("orch", "worker"),),
    entry="orch",
)

AgenticRun(study_dir="studies/my_problem", graph=graph).execute()
```

`StrategizerAgent` and `ImplementerAgent` are the library's canonical defaults (in `f3dasm.agentic.agents`). Users subclass them — or subclass `Agent` directly with `role = "strategizer"` or `role = "implementer"` — wherever is convenient: a run script, a notebook, or a dedicated file.

### Agent class attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `system_prompt` | `str` | `""` | System prompt for this node's LLM session |
| `tools` | `frozenset[str]` | `frozenset()` | Declared tool names (opt-in, conservative) |
| `role` | `str` | `"implementer"` | `"strategizer"` or `"implementer"` — controls which routing closures are injected |
| `backend` | `str \| None` | `None` | Per-node backend override (`"claude"` or `"ollama"`); `None` inherits run-level config |
| `model` | `str \| None` | `None` | Model identifier; `None` inherits run-level config |
| `reset_on_checkpoint` | `bool` | `True` | Whether to clear conversation history between delegations |
| `description` | `str \| None` | `None` | Human-readable description (documentation only) |

### Graph and Edge

```python
Graph(
    nodes: dict[str, Agent],   # name → Agent instance
    edges: tuple[Edge, ...],   # directed delegation edges
    entry: str,                # node that receives the initial briefing (required)
)

Edge(
    source: str,               # delegating node name
    target: str,               # receiving node name
    preamble: str = "",        # text prepended to the task message on this edge
)
```

`Graph.entry` has no default — it must be declared explicitly.

**Multi-target delegation:** a strategizer with multiple outgoing edges calls `Delegate(target="name", intent="...", expected_report="...")`. The `target` is validated against declared outgoing edges; an error string is returned if the name is unknown. `Edge.preamble` is injected into the task message automatically when that edge is traversed.

**Routing is state-carried:** when the strategizer delegates, it writes its own name as `return_to` into `AgenticState`. The implementer reads `state["return_to"]` for its `goto` — no node name is hardcoded anywhere.

### Tool system

`Agent.tools` is a `frozenset[str]`. **Default is `frozenset()` — no tools.** There are three categories:

#### 1. Native backend tools — declare in `Agent.tools`

| Name | Claude | Ollama |
|---|---|---|
| `"Bash"` | SDK `Bash` | subprocess wrapper |
| `"Read"` | SDK `Read` | `StructuredTool` wrapper |
| `"Write"` | SDK `Write` | `StructuredTool` wrapper |
| `"Edit"` | SDK `Edit` | `StructuredTool` wrapper |
| `"Glob"` | SDK `Glob` | `StructuredTool` wrapper |
| `"Grep"` | SDK `Grep` | `StructuredTool` wrapper |

#### 2. Protocol closure tools — declare in `Agent.tools`

Python callables built by the runtime and injected into the session:

| Name | Role | What it does |
|---|---|---|
| `"Done"` | strategizer | End the run with a summary |
| `"Ask"` | strategizer (entry) | Ask the human operator a question |
| `"WriteMarkdown"` | strategizer | Write a `.md` note to `runs/<ts>/strategizer_notes/` |
| `"ReadNote"` | strategizer | Read a file from the study tree |
| `"ReportEvals"` | implementer | Report function evaluation count for this task |

#### 3. Topology-injected tools — **never declare in `Agent.tools`**

Injected automatically by the runtime based on the graph; declared in `Agent.tools` has no effect:

| Tool | Injected when |
|---|---|
| `"Delegate"` | Node has `role = "strategizer"` |

---

## Getting started

### Install

```bash
uv venv
uv pip install -e ".[agentic]"
```

The `agentic` extra installs `claude-agent-sdk`, `langchain-*`, and `langgraph`. For Claude, you also need the `claude` CLI binary on your `PATH` with an active session (`claude` once interactively to authenticate).

### Run an existing study

```bash
uv run python -m f3dasm.agentic studies/agentic_modular_resonance
```

### Make your own study

```bash
mkdir studies/my_problem
cat > studies/my_problem/PROBLEM_STATEMENT.md << 'EOF'
Find (x, y) in [0,1]² that maximises f(x,y) implemented in sim.py.
Do not run more than 200 evaluations.
EOF
cp my_simulator.py studies/my_problem/sim.py
uv run python -m f3dasm.agentic studies/my_problem
```

### config.yaml reference

```yaml
model: claude-haiku-4-5-20251001   # LLM model
backend: claude                     # "claude" (default) or "ollama"
budget: "01:00:00"                  # HH:MM:SS wall-clock budget (soft warning when exceeded)
eval_budget: 5000                   # max evaluations (soft warning when exceeded)
required_deliverables:              # files that must exist before Done is accepted
  - workspace/replicate.py
  - workspace/solution.md
```

**Budgets are soft constraints.** When `budget` or `eval_budget` is exceeded, a warning is appended to the strategizer's next message context; the run continues. The strategizer is expected to wrap up.

**`required_deliverables` is a hard gate.** When the strategizer calls `Done` (or produces a response with no routing tool), the runtime checks whether all listed paths exist under `study_dir/`. If any are missing, the strategizer receives an error message listing the missing files and must delegate their creation before `Done` is accepted.

---

## Python API

```python
from f3dasm.agentic import AgenticRun, Graph, Edge
from f3dasm.agentic.agents import StrategizerAgent, ImplementerAgent

# Default graph (no custom graph needed for simple cases)
run = AgenticRun(study_dir="studies/my_problem")
report = run.execute()   # returns final report text

# Custom graph
class FastWorker(ImplementerAgent):
    model = "claude-haiku-4-5-20251001"
    backend = "ollama"        # per-node backend override

graph = Graph(
    nodes={"s": StrategizerAgent(), "w": FastWorker()},
    edges=(Edge("s", "w", preamble="Work in workspace/. Use f3dasm."),),
    entry="s",
)
AgenticRun(study_dir="studies/my_problem", graph=graph).execute()
```

### Key public symbols

| Symbol | Description |
|---|---|
| `Agent` | Base class for all agent nodes |
| `StrategizerAgent` | Default orchestrator (role="strategizer") |
| `ImplementerAgent` | Default worker (role="implementer") |
| `Graph` | Directed agent topology |
| `Edge` | Directed delegation edge with optional `preamble` |
| `AgenticRun` | Run entry point |
| `AgenticState` | LangGraph state schema |
| `ClaudeAdapter` | Backend adapter using claude-agent-sdk |
| `OllamaAdapter` | Backend adapter using LangGraph + Ollama |
| `LookupDataGenerator` | Nearest-neighbour pool evaluator |
| `AgenticOptimizer` | f3dasm `Optimizer` interface wrapper |

---

## Backends

### Claude (default)

Uses the claude-agent-sdk (`claude-agent-sdk` package). `ClaudeAdapter` wraps a `query()` call with an MCP server for closure tools. Native tools are passed as SDK tool names.

### Ollama

Uses a locally-running Ollama server via `ChatOpenAI(base_url="http://localhost:11434/v1")` and LangGraph's `create_react_agent`. All native tools are custom `StructuredTool` wrappers (no external deps beyond `langchain-core`). Closure tools work identically to Claude.

```yaml
backend: ollama
model: qwen2.5:7b   # any Ollama model with tool-calling support
```

Test with the smallest confirmed tool-calling model: `ollama pull qwen2.5:0.5b`.

Per-node backend override: set `backend = "ollama"` on an individual `Agent` subclass to use Ollama for that node while the rest use Claude.

---

## Safeguards

- **Strategizer prompts** address anchoring bias, confirmation bias, sycophancy, and premature convergence.
- **Implementer prompts** enforce a three-stage protocol (Restate / Inventory / Plan) before code execution and a structured `## Report` block on output.
- **Corrective retry**: if the implementer's response fails format checks, one retry fires with a `REFLECT:` diagnosis.
- **Soft budget warnings**: when wall-clock or eval budgets are exceeded, the strategizer is warned in-context; no hard stop.
- **Required deliverables gate**: configurable list of files that must exist before `Done` is accepted.
- **`ReportEvals`**: implementer closure that writes eval counts back into `AgenticState.evals_used`; surfaced in `solution.md` metadata.

---

## Repository layout

```text
src/f3dasm/agentic/
    __init__.py            # public API
    __main__.py            # CLI entry point

src/f3dasm/_src/agentic/
    agents.py              # StrategizerAgent, ImplementerAgent, _default_graph
    agent_runtime.py       # AgenticRun (config loading, execute(), _make_adapter)
    agent_prompts.py       # system prompts and template constants
    graph_builder.py       # build_graph (compiles Agent/Graph spec → LangGraph)
    graph_state.py         # AgenticState
    nodes.py               # StrategizerNode, ImplementerNode (LangGraph nodes)
    backends/
        base.py            # Agent, Edge, Graph
        claude.py          # ClaudeAdapter
        ollama.py          # OllamaAdapter
    lookup.py              # LookupDataGenerator
    optimizer.py           # AgenticOptimizer

tests/agentic/
    test_nodes.py
    test_graph_builder.py
    test_graph_state.py
    test_agentic_run.py
    test_agent_prompts.py
    test_claude_adapter.py
    test_ollama_adapter.py
    test_lookup.py

studies/<study>/
    PROBLEM_STATEMENT.md   # the only required file
    config.yaml            # optional: model, backend, budget, required_deliverables, …
    workspace/             # implementer scratch space (persists across runs)
    runs/<ts>/
        solution.md        # final report + run metadata
        run.log
        strategizer_notes/ # planner .md lab notebook
```

---

## Studies

| Study | Description |
|---|---|
| `agentic_modular_resonance` | Two-parameter integer optimisation: maximise `ord(k,m)/ln(m)` over `k∈[2,50]`, `m∈[1000,100000]`. Uses f3dasm `Domain`+`DataGenerator`. Confirmed optimum: `k=6, m=99991, resonance≈8685`. |
| `agentic_black_box_8d` | 8-dimensional black-box optimisation with unknown landscape. |
| `agentic_supercompressible_3d` | Supercompressible metamaterial design (Bessa 2019 benchmark, 3D). |
| `agentic_supercompressible_7d` | Same benchmark, 7D parameter space. |
| `agentic_project_euler_078` | Coin partitions (smoke test; no f3dasm DOE). |

---

## Authorship

- **Elvis Aguero** (`elvis_alexander_aguero_vera@brown.edu`) — design, architecture, implementation.
- Bessa Research Group, Brown University.

---

## License

Inherits the host project's **BSD-3-Clause** license. See `LICENSE` at the repository root.

---

## Status

This is the **v2 API**: class-based `Agent` hierarchy, declarative `Graph`/`Edge` topology, LangGraph `StateGraph` orchestration, and unified Claude/Ollama adapters. Exercised end-to-end against `agentic_modular_resonance` with Haiku 4.5; the deliverable folder is reproducible via the agent-written `replicate.py`.
