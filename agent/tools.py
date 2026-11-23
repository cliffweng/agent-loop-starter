"""Tool registry. The model sees schemas; only this file runs code.

A failure returns "ERROR: ..." so the loop can hand it back as a tool message.
"""

from __future__ import annotations

import ast
import html
import json
import math
import operator
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import quote

import requests

# Exponent cap so 2**100000 cannot hang. The node allowlist is the security
# boundary: names, calls, and attributes never run. No eval/exec.
_MAX_EXPR_LEN = 200
_MAX_EXPONENT = 100
_MAX_RESULT_LEN = 500
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_TAGS = re.compile(r"<[^>]+>")


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema for the arguments object
    fn: Callable[..., str]


def _eval(node):
    """Walk one expression node. Anything that is not arithmetic raises."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        value = node.value
        # bool is a subclass of int, so True/False need an explicit reject.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("only numbers are allowed")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if type(node.op) is ast.Pow and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent exceeds {_MAX_EXPONENT}")
        return _OPS[type(node.op)](left, right)
    raise ValueError(f"disallowed syntax: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Arithmetic only. `/` is true division, so 8/2 is the string 4.0."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("expression must be a non-empty string")
    if len(expression) > _MAX_EXPR_LEN:
        raise ValueError("expression is too long")
    result = _eval(ast.parse(expression, mode="eval"))
    if isinstance(result, float) and not math.isfinite(result):
        raise ValueError("result is not finite")
    text = str(result)
    if len(text) > _MAX_RESULT_LEN:
        raise ValueError("result is too large")
    return text


def web_search(query: str) -> str:
    """Keyless Wikipedia search. A few short hits the model can quote."""
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": 3,
        },
        # Wikipedia rejects the default requests User-Agent.
        headers={"User-Agent": "agent-loop-starter/0.1 (teaching example; educational)"},
        timeout=20,
    )
    response.raise_for_status()
    hits = response.json().get("query", {}).get("search", [])
    lines = []
    for hit in hits[:3]:
        title = hit.get("title", "")
        # Unescape first so encoded tags like &lt;b&gt; are stripped too.
        snippet = _TAGS.sub("", html.unescape(hit.get("snippet", ""))).strip()
        url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
        lines.append(f"- {title}: {snippet} ({url})")
    return ("\n".join(lines) if lines else "No results.")[:1500]


def get_time() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tool(name, description, properties, required, fn) -> Tool:
    schema = {"type": "object", "properties": properties, "required": required}
    return Tool(name, description, schema, fn)


# Allowlist. run_tool will not call a name that is missing here.
TOOLS = {
    "calculator": _tool(
        "calculator",
        "Evaluate arithmetic. Supports + - * / // % **, unary +/-, and parentheses.",
        {"expression": {"type": "string", "description": "Math expression such as (2+3)*4"}},
        ["expression"], calculator,
    ),
    "web_search": _tool(
        "web_search",
        "Search Wikipedia and return up to 3 short results (title, snippet, url).",
        {"query": {"type": "string", "description": "Search query"}},
        ["query"], web_search,
    ),
    "get_time": _tool(
        "get_time",
        "Return the current UTC time as an ISO-8601 string.",
        {}, [], get_time,
    ),
}


def tool_schemas() -> list:
    """OpenAI tool list: {"type": "function", "function": {...}} per tool."""
    return [
        {"type": "function", "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }}
        for tool in TOOLS.values()
    ]


def run_tool(name: str, args_json_str: str) -> str:
    """Parse JSON arguments, call the tool, and always return a string."""
    tool = TOOLS.get(name)
    if tool is None:
        return f"ERROR: unknown tool {name!r}"
    try:
        args = json.loads(args_json_str if args_json_str else "{}")
        if not isinstance(args, dict):
            return "ERROR: arguments must be a JSON object"
        result = tool.fn(**args)
    except Exception as exc:  # the model should see this; it must not escape
        return f"ERROR: {exc}"
    return result if isinstance(result, str) else str(result)
