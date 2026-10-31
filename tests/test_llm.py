"""llm.chat contract tests. requests.post is monkeypatched; no network."""

import pytest

from agent.llm import chat


def _silence_dotenv(monkeypatch):
    monkeypatch.setattr("agent.llm.load_local_env", lambda: None)


def test_missing_api_key(monkeypatch):
    _silence_dotenv(monkeypatch)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        chat([{"role": "user", "content": "hi"}], [])


def test_chat_posts_tools_and_returns_message(monkeypatch):
    _silence_dotenv(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    captured = {}

    class FakeResp:
        status_code = 200
        text = '{"choices":[]}'

        def json(self):
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "tool_calls": None,
                    }
                }]
            }

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResp()

    monkeypatch.setattr("agent.llm.requests.post", fake_post)
    message = chat(
        [{"role": "user", "content": "q"}],
        [{"type": "function", "function": {"name": "calculator"}}],
        model="override/model",
    )

    assert message["role"] == "assistant"
    assert message["content"] == "hi"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 60
    assert captured["json"]["model"] == "override/model"
    assert captured["json"]["tool_choice"] == "auto"
    assert captured["json"]["tools"][0]["function"]["name"] == "calculator"


def test_chat_uses_env_model_and_base_url(monkeypatch):
    _silence_dotenv(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_MODEL", "env/model")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1/")
    captured = {}

    class FakeResp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["model"] = json["model"]
        return FakeResp()

    monkeypatch.setattr("agent.llm.requests.post", fake_post)
    assert chat([], [])["content"] == "ok"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["model"] == "env/model"


def test_http_error_includes_response_text(monkeypatch):
    _silence_dotenv(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    class FakeResp:
        status_code = 401
        text = "invalid api key"

        def json(self):
            return {}

    monkeypatch.setattr("agent.llm.requests.post", lambda *a, **k: FakeResp())
    with pytest.raises(RuntimeError, match="invalid api key"):
        chat([], [])
