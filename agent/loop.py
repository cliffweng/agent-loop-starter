"""The agent loop. The message list is the only state.

Call the model with tool schemas. Run any tool_calls and append the results.
No tool_calls means that text is the answer. max_steps stops a model that
would call tools forever.
"""

from __future__ import annotations

import json

from agent.tools import run_tool, tool_schemas

SYSTEM_PROMPT = (
    "You are a concise assistant with three tools: calculator (arithmetic), "
    "web_search (Wikipedia), and get_time (current UTC). "
    "Call a tool when it will make the answer more accurate. "
    "When you know the answer, reply in plain text and do not call a tool."
)


def run_agent(question, llm=None, max_steps=6, on_step=None) -> dict:
    """llm(messages, tools) -> assistant message. Defaults to agent.llm.chat.

    Tests inject a fake, so the loop needs no network and no API key.
    on_step(event) is how the CLI and the UI watch a run.
    Returns {"answer", "steps", "messages"}.
    """
    if llm is None:
        from agent.llm import chat  # lazy: importing the loop needs no key

        llm = chat

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    steps: list = []

    def emit(event: dict) -> None:
        steps.append(event)
        if on_step is not None:
            on_step(event)

    for step in range(1, max_steps + 1):
        message = llm(messages, tool_schemas())
        if not isinstance(message, dict):
            raise RuntimeError("llm() must return the assistant message dict")
        message.setdefault("role", "assistant")
        # Tool results must follow the assistant message that requested them.
        messages.append(message)

        content = message.get("content")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            calls = []
        if not calls:
            answer = content if isinstance(content, str) else ""
            emit({"step": step, "type": "final", "content": answer})
            return {"answer": answer, "steps": steps, "messages": messages}

        # Text beside a tool call is the model's running thought.
        if isinstance(content, str) and content.strip():
            emit({"step": step, "type": "thought", "content": content})
        for call in calls:
            fn = (call or {}).get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") if fn.get("arguments") is not None else ""
            if not isinstance(args, str):
                args = json.dumps(args)
            call_id = (call or {}).get("id") or ""
            emit({
                "step": step,
                "type": "tool_call",
                "name": name,
                "arguments": args,
                "tool_call_id": call_id,
            })
            result = run_tool(name, args)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result})
            emit({
                "step": step,
                "type": "tool_result",
                "name": name,
                "tool_call_id": call_id,
                "content": result,
            })

    answer = "Stopped: reached max_steps without a final answer."
    emit({"step": max_steps, "type": "final", "content": answer})
    return {"answer": answer, "steps": steps, "messages": messages}
