"""Tool unit tests. No network: web_search's requests.get is monkeypatched."""

import json
from datetime import datetime

import pytest

from agent.tools import calculator, get_time, run_tool, tool_schemas, web_search


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2+3*4", "14"),          # precedence: * before +
        ("(2+3)*4", "20"),        # parentheses
        ("-3 + 2.5", "-0.5"),     # unary minus and float
        ("-(2+3)", "-5"),
        ("3.5 - 1", "2.5"),
        ("1.5 * -2", "-3.0"),
        ("7 // 2", "3"),
        ("8/2", "4.0"),           # true division
        ("2 ** 3", "8"),
        ("+4", "4"),
        ("1991*3", "5973"),
    ],
)
def test_calculator_arithmetic(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os')",
        "x",
        "(1).real",
        "os.system('ls')",
        "__import__('os').system('id')",
    ],
)
def test_calculator_rejects_unsafe_syntax(expr):
    with pytest.raises(ValueError):
        calculator(expr)


def test_calculator_caps_exponent():
    assert calculator("2 ** 10") == "1024"
    with pytest.raises(ValueError):
        calculator("2 ** 101")


def test_run_tool_success_unknown_and_bad_json():
    assert run_tool("calculator", json.dumps({"expression": "1991*3"})) == "5973"
    unknown = run_tool("no_such_tool", "{}")
    bad_json = run_tool("calculator", "not-json")
    divzero = run_tool("calculator", json.dumps({"expression": "1/0"}))
    assert unknown.startswith("ERROR:")
    assert bad_json.startswith("ERROR:")
    assert divzero.startswith("ERROR:")


def test_tool_schemas_openai_shape():
    schemas = tool_schemas()
    assert schemas
    assert {item["type"] for item in schemas} == {"function"}
    by_name = {item["function"]["name"]: item["function"] for item in schemas}
    assert set(by_name) == {"calculator", "web_search", "get_time"}
    assert "expression" in by_name["calculator"]["parameters"]["properties"]
    assert by_name["web_search"]["parameters"]["required"] == ["query"]


def test_get_time_is_utc_iso():
    datetime.strptime(get_time(), "%Y-%m-%dT%H:%M:%SZ")


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


def test_web_search_strips_html_and_builds_urls(monkeypatch):
    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse({
            "query": {
                "search": [
                    {
                        "title": "Python (programming language)",
                        "snippet": "Python is a <span class=\"searchmatch\">language</span> &amp; runtime. &lt;b&gt;ok&lt;/b&gt;",
                    },
                    {
                        "title": "Guido van Rossum",
                        "snippet": "Created <b>Python</b>.",
                    },
                ]
            }
        })

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    text = web_search("python release")

    assert captured["url"] == "https://en.wikipedia.org/w/api.php"
    assert captured["params"]["action"] == "query"
    assert captured["params"]["list"] == "search"
    assert captured["params"]["srsearch"] == "python release"
    assert captured["params"]["format"] == "json"
    assert captured["params"]["srlimit"] == 3
    assert "agent-loop-starter" in captured["headers"]["User-Agent"]
    assert "<span" not in text and "<b>" not in text
    assert "&amp;" not in text and "&lt;" not in text
    assert "language & runtime" in text and "ok" in text
    assert "https://en.wikipedia.org/wiki/Python_%28programming_language%29" in text
    assert "Guido van Rossum" in text
    assert len(text) <= 1500


def test_web_search_empty_and_length_cap(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        snippet = "word " * 800
        hits = [{"title": f"Title {i}", "snippet": snippet} for i in range(3)]
        return _FakeResponse({"query": {"search": hits}})

    monkeypatch.setattr("agent.tools.requests.get", fake_get)
    text = web_search("long")
    assert len(text) <= 1500
    assert text.startswith("- Title 0:")

    def fake_empty(url, params=None, headers=None, timeout=None):
        return _FakeResponse({"query": {"search": []}})

    monkeypatch.setattr("agent.tools.requests.get", fake_empty)
    assert web_search("nothing") == "No results."
