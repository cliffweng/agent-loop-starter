"""Streamlit UI. History is for display; every question starts a fresh loop."""

from __future__ import annotations

import os

import streamlit as st

from agent.llm import DEFAULT_MODEL, load_local_env
from agent.loop import run_agent

st.set_page_config(page_title="Agent loop", page_icon="🤖")
load_local_env()

st.title("Agent loop")
st.caption("An LLM, a few tools, and a loop: call a tool, read the result, then answer or stop.")


def api_key() -> str:
    """Streamlit secrets (Community Cloud) first, then the environment or .env."""
    secret = ""
    try:
        # Missing secrets.toml is normal on a laptop. Do not crash the page.
        secret = str(st.secrets["OPENROUTER_API_KEY"] or "")
    except Exception:
        secret = ""
    return secret or os.environ.get("OPENROUTER_API_KEY", "")


key = api_key()
if not key:
    st.error(
        "Missing OPENROUTER_API_KEY. Locally, copy .env.example to .env, or copy "
        ".streamlit/secrets.toml.example to .streamlit/secrets.toml and paste your key. "
        "On Streamlit Community Cloud, add the secret OPENROUTER_API_KEY in the app settings."
    )
    st.stop()

os.environ["OPENROUTER_API_KEY"] = key

if "model" not in st.session_state:
    st.session_state.model = os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
if "max_steps" not in st.session_state:
    st.session_state.max_steps = 6
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Settings")
    st.text_input("Model", key="model")
    st.slider("max_steps", min_value=1, max_value=12, key="max_steps")
    st.caption("One step is one model call. The loop stops at an answer or at this cap.")

os.environ["AGENT_MODEL"] = st.session_state.model


def render_trace(steps, answer) -> None:
    """One expander per tool call (arguments + result), then the answer."""
    i = 0
    while i < len(steps):
        event, kind, n = steps[i], steps[i].get("type"), steps[i].get("step")
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if kind == "tool_call":
            result = ""
            if nxt and nxt.get("type") == "tool_result":
                result, i = nxt.get("content") or "", i + 1
            with st.expander(f"Step {n} · {event.get('name') or 'tool'}"):
                st.markdown("**Arguments**")
                st.code(event.get("arguments") or "{}", language="json")
                st.markdown("**Result**")
                st.code(result)
        elif kind == "thought":
            with st.expander(f"Step {n} · thought"):
                st.write(event.get("content") or "")
        elif kind == "tool_result":
            with st.expander(f"Step {n} · result"):
                st.code(event.get("content") or "")
        i += 1
    if isinstance(answer, str) and answer.startswith("Stopped:"):
        st.warning(answer)
    else:
        st.markdown(answer or "")


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_trace(turn["steps"], turn["answer"])

prompt = st.chat_input("Ask a question")
if prompt:
    with st.spinner("Running the loop…"):
        result = run_agent(prompt, max_steps=int(st.session_state.max_steps))
    st.session_state.history.append({
        "question": prompt,
        "steps": result["steps"],
        "answer": result["answer"],
    })
    st.rerun()
