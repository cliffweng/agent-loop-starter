"""Agent loop tests with a fake LLM. No network and no API key."""

from agent.loop import run_agent


def _call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_tool_call_then_final_answer():
    seen = {"n": 0}

    def fake_llm(messages, tools):
        seen["n"] += 1
        names = [item["function"]["name"] for item in tools]
        assert "calculator" in names
        if seen["n"] == 1:
            assert messages[0]["role"] == "system"
            assert messages[-1]["role"] == "user"
            return {
                "role": "assistant",
                "content": "I'll multiply.",
                "tool_calls": [_call("calculator", '{"expression": "1991*3"}')],
            }
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert tool_msgs and tool_msgs[-1]["tool_call_id"] == "call_1"
        assert tool_msgs[-1]["content"] == "5973"
        return {"role": "assistant", "content": "1991 times 3 is 5973."}

    seen_steps = []
    result = run_agent("What is 1991 times 3?", llm=fake_llm, on_step=seen_steps.append)

    assert result["answer"] == "1991 times 3 is 5973."
    assert [step["type"] for step in result["steps"]] == [
        "thought",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert result["steps"][1]["name"] == "calculator"
    assert "5973" in result["steps"][2]["content"]
    assert seen_steps == result["steps"]
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert seen["n"] == 2


def test_always_calling_a_tool_stops_at_max_steps():
    calls = {"n": 0}

    def fake_llm(messages, tools):
        calls["n"] += 1
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("get_time", "{}", call_id=f"c{calls['n']}")],
        }

    result = run_agent("What time is it?", llm=fake_llm, max_steps=3)
    assert result["answer"] == "Stopped: reached max_steps without a final answer."
    assert calls["n"] == 3
    assert [s["type"] for s in result["steps"]] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert result["steps"][-1]["content"] == result["answer"]
    assistants = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 3


def test_tool_error_is_fed_back_as_a_tool_message():
    def fake_llm(messages, tools):
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if not tool_msgs:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [_call("calculator", "NOT JSON", call_id="bad")],
            }
        assert tool_msgs[-1]["tool_call_id"] == "bad"
        assert tool_msgs[-1]["content"].startswith("ERROR:")
        return {"role": "assistant", "content": "The tool failed, so I will not guess."}

    result = run_agent("break the calculator", llm=fake_llm, max_steps=4)
    assert result["answer"] == "The tool failed, so I will not guess."
    tool_msg = next(m for m in result["messages"] if m["role"] == "tool")
    assert tool_msg["content"].startswith("ERROR:")
    assert tool_msg["tool_call_id"] == "bad"
    assert [s["type"] for s in result["steps"]] == ["tool_call", "tool_result", "final"]
