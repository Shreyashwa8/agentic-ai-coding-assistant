import json
import requests

from .tools import TOOL_SCHEMAS, TOOL_IMPLS

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:3b"
MAX_STEPS = 12

SYSTEM_PROMPT = (
    "You are a coding/research/file-management assistant agent. You can call tools to "
    "list directories, read files, write files, create directories, copy files/directories, "
    "move or rename files/directories, search code, and run Python snippets, all scoped to a "
    "sandboxed workspace directory. For plain file-management requests (make a file, make a "
    "folder, copy something, move/rename something), call the matching tool directly -- do "
    "not write a Python script to do it. Use tools to gather real information before answering "
    "-- do not guess about file contents. When you use run_python to exercise code from a "
    "file you already read, copy the exact class/function names and signatures you saw in "
    "that file's contents -- do not invent names. Each run_python call must be fully "
    "self-contained: include all needed imports and class/function definitions in the same "
    "snippet, and always wrap the value you want to see in print(...), since only stdout is "
    "returned to you. Never write 'I will now call X' or describe a tool call in prose -- "
    "either actually make the tool call in this same turn, or give your final answer. There "
    "is no separate turn for 'now I will run it': acting and describing are the same turn. "
    "When you have enough information, give a final answer in plain text with no further "
    "tool calls."
)


def _call_ollama(messages):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
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


def run_agent(task: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    last_call = None

    for step in range(1, MAX_STEPS + 1):
        message = _call_ollama(messages)
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
    import sys
    task = " ".join(sys.argv[1:]) or "List the files in the workspace and summarize what's there."
    print("\n=== FINAL ANSWER ===")
    print(run_agent(task))
