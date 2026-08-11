import json
import os
import requests

from .tools import TOOL_SCHEMAS, TOOL_IMPLS
from .router import route_model

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", None)  # None = auto-route
MAX_STEPS = 12

SYSTEM_PROMPT = (
    "You are a coding, research, and file-management assistant agent. You can call tools to "
    "search the web, fetch webpage content, list directories, read files, write files, create "
    "directories, copy files/directories, move or rename files/directories, search code, and "
    "run Python snippets. File operations are scoped to a sandboxed workspace directory. "
    "For research tasks, use web_search first to find relevant sources, then fetch_url to read "
    "the full content of promising pages before answering. For plain file-management requests "
    "(make a file, make a folder, copy something, move/rename something), call the matching "
    "tool directly -- do not write a Python script to do it. Use tools to gather real "
    "information before answering -- do not guess. When you use run_python to exercise code "
    "from a file you already read, copy the exact class/function names and signatures you saw "
    "-- do not invent names. Each run_python call must be fully self-contained: include all "
    "needed imports and definitions, and always wrap values you want to see in print(...). "
    "Never write 'I will now call X' or describe a tool call in prose -- either actually make "
    "the tool call in this same turn, or give your final answer. There is no separate turn for "
    "'now I will run it': acting and describing are the same turn. When you have enough "
    "information, give a final answer in plain text with no further tool calls."
)


def _call_ollama(messages, model: str, tool_schemas: list):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "tools": tool_schemas,
            "stream": False,
            # -1 = offload as many layers as fit in VRAM, remainder on CPU.
            "options": {"num_gpu": -1},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]


def run_agent(
    task: str,
    verbose: bool = True,
    model: str = None,
    tool_schemas: list = None,
    tool_impls: dict = None,
    system_prompt: str = None,
    label: str = "",
) -> str:
    if model:
        task_type = "manual"
    elif DEFAULT_MODEL:
        model, task_type = DEFAULT_MODEL, "env"
    else:
        model, task_type = route_model(task)

    if verbose:
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}[router] task_type={task_type}  model={model}")

    schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
    impls   = tool_impls   if tool_impls   is not None else TOOL_IMPLS
    prompt  = system_prompt or SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": task},
    ]
    last_call = None

    for step in range(1, MAX_STEPS + 1):
        message = _call_ollama(messages, model, schemas)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            if verbose:
                prefix = f"[{label}] " if label else ""
                print(f"{prefix}[step {step}] final answer")
            return message.get("content", "")

        messages.append(message)

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)

            impl = impls.get(name)
            if impl is None:
                result = f"error: unknown tool '{name}'"
            else:
                try:
                    result = impl(**args)
                except Exception as exc:
                    result = f"error: {exc}"

            this_call = (name, json.dumps(args, sort_keys=True))
            if this_call == last_call:
                result = (
                    f"{result}\n\nNOTE: you already made this exact call and got this exact "
                    "error. Repeating it will not help -- change the code/arguments or stop "
                    "and answer with what you know."
                )
            last_call = this_call

            if verbose:
                prefix = f"[{label}] " if label else ""
                print(f"{prefix}[step {step}] tool={name} args={args}\n  -> {result[:200]}")

            messages.append({"role": "tool", "content": str(result)})

    return "stopped: reached max steps without a final answer"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="*", help="task for the agent")
    parser.add_argument("--model", default=None, help="Ollama model name (e.g. llama3.2:3b, mistral, qwen2.5:7b)")
    parser.add_argument("--multi-agent", action="store_true", help="use multi-agent orchestration")
    args = parser.parse_args()
    task = " ".join(args.task) or "List the files in the workspace and summarize what's there."
    print("\n=== FINAL ANSWER ===")
    if args.multi_agent:
        from .multiagent import run_multi_agent
        print(run_multi_agent(task))
    else:
        print(run_agent(task, model=args.model))
