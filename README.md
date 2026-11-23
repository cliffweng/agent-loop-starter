# agent-loop-starter

A minimal teaching repo for the core LLM agent loop, in plain Python.
No LangChain, LangGraph, CrewAI, or LlamaIndex. One model call, a few tools,
and a loop you can read in one sitting.

Frameworks are optional later, once this shape is obvious. Background reading:
[docs/research-notes.md](docs/research-notes.md).

## What an agent is

An agent here is three pieces:

1. **An LLM** that reads a transcript and writes the next message.
2. **Tools** it may ask for (math, search, clock).
3. **A loop** that runs those tools and feeds the results back until the model answers.

The model proposes a tool name and a JSON argument object. Your code runs the
function. The model never sees your API key and never executes Python. It
only sees text.

```mermaid
flowchart TD
    start[User question] --> messages["messages = system + user"]
    messages --> call[Call the LLM with tool schemas]
    call --> decision{tool_calls?}
    decision -->|yes| run[Run each tool]
    run --> append[Append tool results to messages]
    append --> cap{Under max_steps?}
    cap -->|yes| call
    cap -->|no| stopped["Stop: max_steps"]
    decision -->|no| final[Final answer]
```

## Repo layout

```
agent-loop-starter/
├── agent.py                 # CLI: python agent.py "question"
├── app.py                   # Streamlit UI
├── agent/
│   ├── __init__.py          # exports run_agent
│   ├── llm.py               # one HTTP call to OpenRouter
│   ├── tools.py             # registry, schemas, safe execution
│   └── loop.py              # the loop itself
├── tests/                   # fake LLM, no network, no API key
├── docs/research-notes.md
├── requirements.txt
├── .env.example
└── .streamlit/              # theme + secrets.toml.example
```

## How the pieces fit

### Messages are the state

`run_agent` keeps one list. There is no other memory.

```text
system     instructions and the tool policy
user       the question
assistant  text and, sometimes, tool_calls
tool       the string result of one call
assistant  the next decision, or the final answer
```

The next call receives the whole list. A fact the model should remember has
to be written into a message. Later, short-term memory is this list plus a
summary of older turns.

The assistant message that requested a tool must sit in the list before the
tool result. Chat APIs match `tool_call_id` on the tool message to that
turn. `loop.py` appends them in that order.

### Tool schemas are the contract

`tool_schemas()` builds the OpenAI tools list:

```json
{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
```

The model uses the name and description to choose, and the JSON schema to
shape arguments. It does not see `calculator`'s source. A vague description
gets vague calls. Write it the way you would brief a new teammate.

`llm.chat` sends the schemas with `tool_choice: "auto"`, so the model may
call a tool or answer in text. `"none"` bans tools. `"required"` forces a
call. Auto matches this loop: stop when the model is done talking to tools.

### Tool execution is an allowlist

`TOOLS` maps a name to a `Tool` (description, parameter schema, function).
`run_tool` looks up the name, parses the argument string, and calls the
function. Unknown names, bad JSON, and exceptions come back as
`"ERROR: ..."`. The loop appends that string like any other result. The
model can retry, switch tools, or explain the failure.

| Tool | What it does | Why it is in a teaching demo |
| --- | --- | --- |
| `calculator` | Arithmetic via `ast` | Allowlist of nodes. No `eval`, no names, no calls. Exponent size is capped. |
| `web_search` | Wikipedia search API | Keyless. Output trimmed to 1500 characters. Wikipedia requires a descriptive User-Agent. |
| `get_time` | Current UTC time, ISO-8601 | No arguments. Stops the model from inventing "now". |

`/` is true division, so `8/2` is `4.0`. `//` is floor division.

### When the loop stops

- The assistant message has **no** `tool_calls`. Its `content` is the answer.
- **`max_steps`** model calls have already happened. The answer is
  `Stopped: reached max_steps without a final answer.`

A model that calls a tool every turn would otherwise run until you kill it.
The cap bounds cost. It does not make a weak prompt smarter. Raise it for
multi-step questions.

### Errors go back to the model

`run_tool` is the boundary that must not raise into the loop. A tool message
that says `ERROR: disallowed syntax: Call` lets the model change its request.
Read the step log when a final answer looks wrong. The answer alone hides
whether a tool ran.

### Observability

`run_agent(..., on_step=fn)` emits one small dict per event:

| `type` | When |
| --- | --- |
| `thought` | The model wrote text and also requested a tool |
| `tool_call` | Name, arguments JSON, `tool_call_id` |
| `tool_result` | The string `run_tool` returned |
| `final` | The answer, including the max-steps stop |

The CLI prints these. Streamlit puts each tool call, its arguments, and its
result in an expander. The loop itself does not print. Same events, two views.

`llm` is an argument. Production passes `agent.llm.chat`. Tests pass a
function that returns canned messages, so `pytest` needs no key and no network.

## A run, narrated

Quickstart question:

> What year was Python first released, and what is that year times 3?

A tool-using model often searches, then multiplies:

```text
[step 1] 🧠 thought: I'll look up the release year.
[step 1] 🔧 tool_call web_search({"query":"Python programming language first release"})
[step 1] 📎 result: - Python (programming language): ... (https://en.wikipedia.org/wiki/...)
[step 2] 🔧 tool_call calculator({"expression":"1991*3"})
[step 2] 📎 result: 5973
[step 3] ✅ final: Python was first released in 1991. 1991 × 3 = 5973.
```

Some models already know 1991 and only call `calculator`, or they answer
from memory. Both are legal under `tool_choice: "auto"`. The trace shows
which happened. Edit `SYSTEM_PROMPT` in `agent/loop.py` to push toward tools.
The multiply step appends another assistant message and another tool message.
The last assistant message has text and no `tool_calls`, and the loop returns.

## Quickstart

Python 3.10 or newer. Dependencies are `requests`, `streamlit`, and `pytest`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENROUTER_API_KEY` ([openrouter.ai/keys](https://openrouter.ai/keys)).
The file is gitignored. Do not put the key in code or in a committed file.

```bash
python agent.py "What year was Python first released, and what is that year times 3?"
streamlit run app.py
pytest
```

```bash
python agent.py --max-steps 3 "What time is it in UTC?"
python agent.py --model x-ai/grok-4.3 "What is (20+5)*3?"
```

With no question, the CLI prompts. `AGENT_MODEL` defaults to `x-ai/grok-4.3`.
`OPENROUTER_BASE_URL` overrides the API base (`https://openrouter.ai/api/v1`).

The Streamlit sidebar shows the model and a `max_steps` slider. Each question
starts a new loop. Older turns stay on screen and are not sent back as
context. Chat memory is a next step, left unwired on purpose.

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub.
2. In [Streamlit Community Cloud](https://streamlit.io/cloud), create an app.
   Set the main file to `app.py`.
3. Add a secret in the app settings:

```toml
OPENROUTER_API_KEY = "your-key-here"
```

The app reads `st.secrets["OPENROUTER_API_KEY"]`, then the environment.
If both are missing it stops with setup instructions. Copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (gitignored)
when you want that file instead of `.env`.

## Exercises

1. **Add a tool.** Register a unit converter or word counter in `TOOLS` with a description the model can tell apart from `calculator`. Ask for it and read the trace.
2. **Hit the brake.** `--max-steps 1` on a question that needs a tool should show a result, then the stop sentence. Compare `2` and `6`.
3. **Swap the model.** Change `AGENT_MODEL` or the sidebar. See whether the new model searches, calculates, or answers from memory.
4. **Read the transcript.** Print `result["messages"]` and match each `tool_call_id` to its tool message.
5. **Cause an error.** Ask the calculator for `__import__('os')`. The model should receive `ERROR: ...` and the process should stay up.
6. **Tighten the prompt.** Change `SYSTEM_PROMPT` so the final answer must name the tool it used. The loop stays the same.

## Next steps

**Memory.** Short-term memory is the message list. When it gets long, summarize older turns and keep recent ones verbatim. Long-term memory lives outside the window: files, a database, or a vector store behind a tool such as `search_notes`. Add the tool first.

**Planning.** This loop is already close to ReAct: thought, action, observation, repeat. Plan-and-execute writes a plan, then carries out each step. Reflection critiques the answer and tries again. All three are still "call the model, maybe call a tool, read the text."

**Multi-agent orchestration.** A router picks a specialist. An orchestrator splits work across workers and merges results. Use that when one prompt is doing too many jobs. Each worker is this same loop with a narrower tool list.

**Evals and guardrails.** Score the traces you already emit: right tool, answer matches a known value, stayed inside `max_steps`. Guardrails reject a bad argument, cap output size (calculator and search already do), or require a second look before a risky action.

**Frameworks, later.** [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) is a graph of steps with explicit state, checkpoints, and human approval. [CrewAI](https://docs.crewai.com/) is role-based crews: agents, tasks, and a process. Both are easier to judge once you can draw this flowchart from memory. Move when the `for` loop is what is in the way.

## Tests

```bash
pytest
```

`tests/test_tools.py` checks precedence, negatives, and floats; rejects
`__import__`, names, and attribute access; and checks that unknown tools
and bad JSON become `ERROR:` strings. `web_search` monkeypatches
`requests.get`. `tests/test_loop.py` uses a fake LLM: tool then answer,
always-tool hits `max_steps`, and a tool error is appended as a tool
message. `tests/test_llm.py` checks the HTTP payload with `requests.post`
monkeypatched. No test calls OpenRouter.
