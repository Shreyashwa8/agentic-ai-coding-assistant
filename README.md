# Agentic AI Project

A local agentic AI assistant that talks to a locally running [Ollama](https://ollama.com)
server (model `qwen2.5:3b`) and autonomously plans and invokes tools to answer questions,
research topics on the web, manage files, search code, and run Python snippets.

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

## Notes

- GPU is used automatically if present and has enough VRAM (Ollama offloads as many model
  layers to GPU as fit, and the rest to CPU) — no configuration needed. This model's weights
  are ~1.8 GB, so any GPU with 4 GB+ VRAM should fit the whole model.
- `agent/tools.py` sandboxes all file/search/exec operations to the `workspace/` directory;
  paths that try to escape it are rejected.
- If `ollama pull` or the agent's HTTP calls fail, confirm the Ollama service is running:
  `ollama list` should succeed and show `qwen2.5:3b`.
