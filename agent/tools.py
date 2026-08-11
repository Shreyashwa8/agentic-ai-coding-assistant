import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)


def _resolve_in_workspace(path: str) -> Path:
    candidate = (WORKSPACE / path).resolve()
    if WORKSPACE not in candidate.parents and candidate != WORKSPACE:
        raise ValueError(f"path '{path}' escapes the workspace sandbox")
    return candidate


def list_directory(path: str = ".") -> str:
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"error: '{path}' does not exist in workspace"
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return "\n".join(entries) if entries else "(empty directory)"


def read_file(path: str, max_chars: int = 4000) -> str:
    target = _resolve_in_workspace(path)
    if not target.is_file():
        return f"error: '{path}' is not a file in workspace"
    text = target.read_text(errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
    return text


def write_file(path: str, content: str) -> str:
    target = _resolve_in_workspace(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} chars to {path}"


def make_directory(path: str) -> str:
    target = _resolve_in_workspace(path)
    target.mkdir(parents=True, exist_ok=True)
    return f"created directory {path}"


def copy_file(source: str, destination: str) -> str:
    src = _resolve_in_workspace(source)
    dst = _resolve_in_workspace(destination)
    if not src.exists():
        return f"error: '{source}' does not exist in workspace"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return f"copied {source} to {destination}"


def move_file(source: str, destination: str) -> str:
    src = _resolve_in_workspace(source)
    dst = _resolve_in_workspace(destination)
    if not src.exists():
        return f"error: '{source}' does not exist in workspace"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"moved {source} to {destination}"


def search_code(pattern: str, path: str = ".") -> str:
    target = _resolve_in_workspace(path)
    if not target.exists():
        return f"error: '{path}' does not exist in workspace"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"error: invalid pattern: {exc}"

    files = [target] if target.is_file() else (p for p in target.rglob("*") if p.is_file())
    matches = []
    for file_path in files:
        try:
            text = file_path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                rel = file_path.relative_to(WORKSPACE)
                matches.append(f"{rel}:{lineno}:{line}")
    return "\n".join(matches) if matches else "no matches"


def web_search(query: str, max_results: int = 5) -> str:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "error: duckduckgo-search not installed. Run: pip install duckduckgo-search"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "no results found"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}\n    URL: {r['href']}\n    {r['body']}")
        return "\n\n".join(lines)
    except Exception as exc:
        return f"error: {exc}"


def fetch_url(url: str, max_chars: int = 4000) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > max_chars:
            return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more chars]"
        return text
    except Exception as exc:
        return f"error: {exc}"


def run_python(code: str, timeout: int = 10) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout, cwd=str(WORKSPACE),
        )
    except subprocess.TimeoutExpired:
        return f"error: execution timed out after {timeout}s"
    out = result.stdout.strip()
    err = result.stderr.strip()
    parts = []
    if out:
        parts.append(f"stdout:\n{out}")
    if err:
        parts.append(f"stderr:\n{err}")
    parts.append(f"exit_code: {result.returncode}")
    return "\n".join(parts)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories at a path inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "relative path, default '.'"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text content of a file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "relative file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file inside the workspace, creating directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "relative file path"},
                    "content": {"type": "string", "description": "text content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_directory",
            "description": "Create a directory (and any missing parent directories) inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "relative directory path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "copy_file",
            "description": "Copy a file or directory to another location inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "relative source path"},
                    "destination": {"type": "string", "description": "relative destination path"},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file or directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "relative source path"},
                    "destination": {"type": "string", "description": "relative destination path"},
                },
                "required": ["source", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a regex pattern across files in the workspace (like grep -rn).",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "regex pattern to search for"},
                    "path": {"type": "string", "description": "relative path to search under, default '.'"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo and return titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "search query"},
                    "max_results": {"type": "integer", "description": "number of results to return, default 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a webpage and return its cleaned text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "full URL to fetch"},
                    "max_chars": {"type": "integer", "description": "max characters to return, default 4000"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": (
                "Execute a short Python snippet in a fresh sandboxed subprocess (cwd = workspace) "
                "and return stdout/stderr. Each call starts a brand-new Python process with no "
                "memory of previous calls or previously read files -- classes/functions from a file "
                "you read earlier are NOT already defined here. To use code from a workspace file, "
                "either import it (e.g. 'from Solution import Solution') or paste the full "
                "definition into this snippet before using it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python source code to execute"}},
                "required": ["code"],
            },
        },
    },
]

TOOL_IMPLS = {
    "list_directory": list_directory,
    "read_file": read_file,
    "write_file": write_file,
    "make_directory": make_directory,
    "copy_file": copy_file,
    "move_file": move_file,
    "search_code": search_code,
    "web_search": web_search,
    "fetch_url": fetch_url,
    "run_python": run_python,
}
