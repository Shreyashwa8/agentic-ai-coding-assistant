MODELS = {
    "coding":    "qwen2.5-coder:7b",
    "reasoning": "qwen2.5:7b",
    "research":  "mistral",
    "general":   "qwen2.5:3b",
}

_CODING_KEYWORDS = {
    "code", "function", "class", "bug", "debug", "implement", "algorithm",
    "script", "program", "refactor", "fix", "error", "exception", "import",
    "module", "test", "loop", "array", "list", "dict", "sort", "compile",
    "syntax", "python", "write a function", "write a class", "decorator",
    "recursion", "api", "endpoint", "library", "package",
}

_REASONING_KEYWORDS = {
    "reason", "solve", "calculate", "math", "proof", "logic", "analyze",
    "compare", "evaluate", "explain why", "how does", "derive", "theorem",
    "equation", "formula", "optimize", "complexity", "big o", "step by step",
    "think through", "tradeoff", "trade-off", "decision", "should i",
    "which is better", "pros and cons",
}

_RESEARCH_KEYWORDS = {
    "search", "research", "find", "what is", "who is", "summarize", "news",
    "latest", "recent", "web", "url", "fetch", "read about", "tell me about",
    "look up", "browse", "article", "paper", "documentation",
}


def route_model(task: str) -> tuple:
    task_lower = task.lower()

    scores = {
        "coding":    sum(1 for kw in _CODING_KEYWORDS if kw in task_lower),
        "reasoning": sum(1 for kw in _REASONING_KEYWORDS if kw in task_lower),
        "research":  sum(1 for kw in _RESEARCH_KEYWORDS if kw in task_lower),
    }

    best_type = max(scores, key=scores.get)

    if scores[best_type] == 0:
        return MODELS["general"], "general"

    return MODELS[best_type], best_type
