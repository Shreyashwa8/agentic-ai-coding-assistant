import json
from .tools import TOOL_SCHEMAS, TOOL_IMPLS

# ── Specialist agent definitions ──────────────────────────────────────────────

SPECIALIST_AGENTS = {
    "coding": {
        "model": "qwen2.5-coder:7b",
        "description": "Writes, debugs, refactors, and runs Python code.",
        "tool_names": {"list_directory", "read_file", "write_file", "search_code", "run_python"},
        "system_prompt": (
            "You are a coding specialist. Write, debug, refactor, and run Python code. "
            "Use run_python to verify your code works. Always wrap output in print(). "
            "Be concise and return only the final code or result."
        ),
    },
    "research": {
        "model": "mistral",
        "description": "Searches the web and summarizes information from URLs.",
        "tool_names": {"web_search", "fetch_url", "write_file"},
        "system_prompt": (
            "You are a research specialist. Use web_search to find sources, then fetch_url "
            "to read full content. Summarize findings clearly and cite sources."
        ),
    },
    "reasoning": {
        "model": "qwen2.5:7b",
        "description": "Analyzes, compares tradeoffs, explains concepts, solves logic/math problems.",
        "tool_names": {"read_file", "write_file", "run_python"},
        "system_prompt": (
            "You are a reasoning specialist. Analyze problems step by step, compare tradeoffs, "
            "explain concepts clearly, and solve math or logic problems. Show your reasoning."
        ),
    },
    "file_manager": {
        "model": "qwen2.5:3b",
        "description": "Lists, reads, writes, copies, and organizes files in the workspace.",
        "tool_names": {"list_directory", "read_file", "write_file", "make_directory", "copy_file", "move_file"},
        "system_prompt": (
            "You are a file management specialist. List, read, write, copy, and organize files "
            "inside the sandboxed workspace directory. Be precise and confirm actions taken."
        ),
    },
}

# ── Orchestrator tools ─────────────────────────────────────────────────────────

def _build_delegate_tool():
    agent_descriptions = "\n".join(
        f"  - {name}: {spec['description']}"
        for name, spec in SPECIALIST_AGENTS.items()
    )
    return {
        "type": "function",
        "function": {
            "name": "delegate_task",
            "description": (
                f"Delegate a subtask to a specialist agent.\n"
                f"Available agents:\n{agent_descriptions}"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": list(SPECIALIST_AGENTS.keys()),
                        "description": "which specialist agent to delegate to",
                    },
                    "task": {
                        "type": "string",
                        "description": "the subtask to give the specialist agent",
                    },
                },
                "required": ["agent", "task"],
            },
        },
    }


def _delegate_task(agent: str, task: str, verbose: bool = True) -> str:
    from .agent import run_agent
    spec = SPECIALIST_AGENTS.get(agent)
    if spec is None:
        return f"error: unknown agent '{agent}'"

    schemas = [s for s in TOOL_SCHEMAS if s["function"]["name"] in spec["tool_names"]]
    impls   = {k: v for k, v in TOOL_IMPLS.items() if k in spec["tool_names"]}

    if verbose:
        print(f"\n>>> Delegating to [{agent}] ({spec['model']}): {task[:100]}")

    result = run_agent(
        task,
        verbose=verbose,
        model=spec["model"],
        tool_schemas=schemas,
        tool_impls=impls,
        system_prompt=spec["system_prompt"],
        label=agent,
    )
    return result


# ── Orchestrator ───────────────────────────────────────────────────────────────

_ORCHESTRATOR_MODEL = "qwen2.5:7b"

_ORCHESTRATOR_PROMPT = (
    "You are an orchestrator agent that coordinates specialist agents to complete complex tasks. "
    "Break the user's task into subtasks and delegate each to the right specialist using delegate_task. "
    "Available specialists:\n"
    "  - coding: writing/debugging/running code\n"
    "  - research: web search and URL reading\n"
    "  - reasoning: analysis, tradeoffs, math, explanations\n"
    "  - file_manager: listing/reading/writing/organizing files\n\n"
    "Strategy:\n"
    "1. Analyze the task and identify what types of work are needed.\n"
    "2. Delegate each part to the right specialist in logical order.\n"
    "3. Pass results from one agent to the next when needed.\n"
    "4. Once all subtasks are done, synthesize a final answer.\n\n"
    "For simple tasks that only need one specialist, delegate once and return their answer. "
    "Never do the work yourself — always delegate. "
    "When you have all results, give a final synthesized answer with no more tool calls."
)


def run_multi_agent(task: str, verbose: bool = True) -> str:
    import requests, os

    OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
    delegate_tool = _build_delegate_tool()

    messages = [
        {"role": "system", "content": _ORCHESTRATOR_PROMPT},
        {"role": "user", "content": task},
    ]

    if verbose:
        print(f"\n[orchestrator] Starting multi-agent run  model={_ORCHESTRATOR_MODEL}")

    for step in range(1, 20):
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": _ORCHESTRATOR_MODEL,
                "messages": messages,
                "tools": [delegate_tool],
                "stream": False,
                "options": {"num_gpu": -1},
            },
            timeout=300,
        )
        resp.raise_for_status()
        message = resp.json()["message"]
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            if verbose:
                print(f"[orchestrator] step {step} — final synthesis")
            return message.get("content", "")

        messages.append(message)

        for call in tool_calls:
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)

            agent_name = args.get("agent", "")
            subtask    = args.get("task", "")

            result = _delegate_task(agent_name, subtask, verbose=verbose)

            if verbose:
                print(f"[orchestrator] step {step} — [{agent_name}] returned: {result[:200]}\n")

            messages.append({"role": "tool", "content": str(result)})

    return "stopped: orchestrator reached max steps"
