#!/usr/bin/env python3
"""Run one question and print each step.

    python agent.py "What year was Python first released, and what is that year times 3?"
    python agent.py --max-steps 4 --model x-ai/grok-4.3
"""

from __future__ import annotations

import argparse
import os
import sys

from agent.llm import load_local_env
from agent.loop import run_agent


def print_step(event: dict) -> None:
    """Pretty-print one event. The step log is how you debug the loop."""
    n, kind = event.get("step"), event.get("type")
    if kind == "thought":
        print(f"[step {n}] 🧠 thought: {event.get('content', '')}")
    elif kind == "tool_call":
        print(f"[step {n}] 🔧 tool_call {event.get('name')}({event.get('arguments', '')})")
    elif kind == "tool_result":
        print(f"[step {n}] 📎 result: {event.get('content', '')}")
    elif kind == "final":
        print(f"[step {n}] ✅ final: {event.get('content', '')}")


def main(argv=None) -> int:
    load_local_env()
    parser = argparse.ArgumentParser(description="Ask the tool-using agent one question.")
    parser.add_argument("question", nargs="*", help="Question. Prompts if omitted.")
    parser.add_argument("--max-steps", type=int, default=6, help="Cap on model calls.")
    parser.add_argument("--model", default=None, help="Override AGENT_MODEL for this run.")
    args = parser.parse_args(argv)

    if args.max_steps < 1:
        print("--max-steps must be at least 1", file=sys.stderr)
        return 2
    if args.model:
        os.environ["AGENT_MODEL"] = args.model

    question = " ".join(args.question).strip()
    if not question:
        try:
            question = input("Question: ").strip()
        except EOFError:
            question = ""
    if not question:
        print("No question given.", file=sys.stderr)
        return 2

    run_agent(question, max_steps=args.max_steps, on_step=print_step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
