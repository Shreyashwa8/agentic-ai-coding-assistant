# Agentic AI Project

A local agentic AI assistant that talks to a locally running [Ollama](https://ollama.com)
server and autonomously plans and invokes tools to answer questions, research topics on the
web, manage files, search code, and run Python snippets. Works with any model supported by
Ollama — default is `qwen2.5:3b`.

**Tools available to the agent:**
- `web_search` — DuckDuckGo search (no API key required)
- `fetch_url` — fetch and read webpage content
- `list_directory`, `read_file`, `write_file`, `make_directory`, `copy_file`, `move_file` — file management inside a sandboxed `workspace/`
- `search_code` — regex-based code search (like grep)
- `run_python` — sandboxed Python execution

This package contains only the portable Python agent code. Ollama itself is a native app you
install separately per OS (it ships its own Windows/Linux/Mac builds with GPU auto-detection).

## Requirements

- Python 3.9+
- Ollama (installed natively for your OS — see below)

## Setup

### Windows

1. Install Ollama:
   ```
   winget install Ollama.Ollama
   ```
   (or download the installer from https://ollama.com/download)

2. Pull the model (Ollama runs as a background service after install, so this can be run
   from any terminal):
   ```
   ollama pull qwen2.5:3b
   ```

3. Unzip this package and set up the Python side:
   ```
   cd agentic-ai-project
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Run the agent:
   ```
   python -m agent.agent "List the files in the workspace and summarize what's there."
   ```

### Linux / macOS

1. Install Ollama:
   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```

2. Pull the model:
   ```
   ollama pull qwen2.5:3b
   ```

3. Set up and run:
   ```
   cd agentic-ai-project
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python -m agent.agent "List the files in the workspace and summarize what's there."
   ```

## Auto Model Routing

When no model is specified, the agent automatically picks the best model based on your task:

| Task Type | Detected by keywords | Model used |
|---|---|---|
| Coding | `code`, `function`, `bug`, `implement`, `algorithm`, `script` ... | `qwen2.5-coder:7b` |
| Reasoning | `solve`, `calculate`, `math`, `analyze`, `tradeoff`, `explain why` ... | `qwen2.5:7b` |
| Research | `search`, `research`, `latest`, `what is`, `summarize`, `web` ... | `mistral` |
| General | *(no strong signal)* | `qwen2.5:3b` |

```bash
# auto-routed to qwen2.5-coder:7b
python -m agent.agent "Write a binary search function in Python"

# auto-routed to qwen2.5:7b
python -m agent.agent "Explain the tradeoffs between FSDP and DeepSpeed ZeRO"

# auto-routed to mistral
python -m agent.agent "Search for the latest LLM benchmarks and summarize"

# auto-routed to qwen2.5:3b (general)
python -m agent.agent "List the files in workspace"
```

The agent prints `[router] task_type=... model=...` at the start of each run so you can see which model was selected.

You can override routing anytime with `--model` or `AGENT_MODEL` (see below).

> **Note:** Pull any model before using it: `ollama pull <model-name>`

## Switching Models

The agent works with **any model available in Ollama** that supports tool calling.

### Option 1 — CLI flag (per run)
```bash
python -m agent.agent --model llama3.2:3b "Summarize the latest AI news"
python -m agent.agent --model mistral "Write a Python sorting algorithm"
python -m agent.agent --model qwen2.5:7b "Research FSDP sharding strategies"
```

### Option 2 — Environment variable (session default)
```bash
# Linux / macOS
export AGENT_MODEL=llama3.2:3b
python -m agent.agent "Your task here"

# Windows
set AGENT_MODEL=llama3.2:3b
python -m agent.agent "Your task here"
```

### Recommended models for tool calling

| Model | Size | Pull command | Notes |
|---|---|---|---|
| `qwen2.5:3b` | ~1.8 GB | `ollama pull qwen2.5:3b` | Default, fast, good tool use |
| `qwen2.5:7b` | ~4.4 GB | `ollama pull qwen2.5:7b` | Better reasoning |
| `llama3.2:3b` | ~2 GB | `ollama pull llama3.2:3b` | Meta's latest small model |
| `mistral` | ~4 GB | `ollama pull mistral` | Strong general purpose |
| `qwen2.5-coder:7b` | ~4.4 GB | `ollama pull qwen2.5-coder:7b` | Best for coding tasks |

> **Note:** Models must support tool/function calling to work with this agent.
> Check [ollama.com/search](https://ollama.com/search) for the full model list.

## Notes

- GPU is used automatically if present and has enough VRAM (Ollama offloads as many model
  layers to GPU as fit, and the rest to CPU) — no configuration needed. This model's weights
  are ~1.8 GB, so any GPU with 4 GB+ VRAM should fit the whole model.
- `agent/tools.py` sandboxes all file/search/exec operations to the `workspace/` directory;
  paths that try to escape it are rejected.
- If `ollama pull` or the agent's HTTP calls fail, confirm the Ollama service is running:
  `ollama list` should succeed and show your pulled model.
