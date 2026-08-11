import json
import os
import requests

from .tools import TOOL_SCHEMAS, TOOL_IMPLS

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "qwen2.5:3b")
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


def _call_ollama(messages, model: str):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
            # -1 = offload as many layers as fit in VRAM, remainder on CPU.
            "options": {"num_gpu": -1},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]


def run_agent(task: str, verbose: bool = True, model: str = None) -> str:
    model = model or DEFAULT_MODEL
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    last_call = None

    for step in range(1, MAX_STEPS + 1):
        message = _call_ollama(messages, model)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            if verbose:
                print(f"[step {step}] final answer")
            return message.get("content", "")

        messages.append(message)

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)

            impl = TOOL_IMPLS.get(name)
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
                print(f"[step {step}] tool={name} args={args}\n  -> {result[:200]}")

            messages.append({"role": "tool", "content": str(result)})

    return "stopped: reached max steps without a final answer"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", nargs="*", help="task for the agent")
    parser.add_argument("--model", default=None, help="Ollama model name (e.g. llama3.2:3b, mistral, qwen2.5:7b)")
    args = parser.parse_args()
    task = " ".join(args.task) or "List the files in the workspace and summarize what's there."
    print("\n=== FINAL ANSWER ===")
    print(run_agent(task, model=args.model))
