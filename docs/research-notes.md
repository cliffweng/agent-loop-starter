# Research notes: the agent loop and tool calling

Short reading list for this repo. Read `agent/loop.py` first, then these.

The shared pattern: the model chooses an action, something outside the model
runs it, and the result comes back as text. Training and orchestration
differ. The message loop does not.

## OpenAI — Function calling

<https://platform.openai.com/docs/guides/function-calling>

- You send `tools` (name, description, JSON schema) on the chat request.
- The model does not run the function. It returns `tool_calls` with a name
  and an arguments string.
- Your app runs the call and appends `{"role": "tool", "tool_call_id": ...}`.
  Then you call the model again.
- `tool_choice` can be `auto`, `none`, `required`, or one function. This
  repo sends `auto`.
- One assistant message may contain several calls. `run_agent` runs all of
  them before the next model turn.

## OpenRouter — Tool calling

<https://openrouter.ai/docs/features/tool-calling>

- The HTTP API matches OpenAI chat completions, so `agent/llm.py` is one
  `requests` POST and not a provider SDK.
- `tools`, `tool_calls`, and `role: "tool"` use that same shape.
  `OPENROUTER_BASE_URL` can point at another compatible host.
- Support varies by model. A model that ignores `tools` answers in
  `content`, and this loop treats that text as final.
- `AGENT_MODEL` defaults to `x-ai/grok-4.3`. Pick a model that lists tools.

## ReAct — reason, then act

Yao et al., 2022. <https://arxiv.org/abs/2210.03629>

- Interleave a reasoning trace, an action, and the observation that follows.
- On QA and interactive tasks, that mix beat chain-of-thought alone and
  actions alone. The trace gives the model somewhere to update its plan.
- In this repo: `thought` is assistant text beside a tool call, the action
  is `tool_call`, the observation is the tool message.
- The paper used few-shot prompting, before native function calling was
  common. The control flow is the same either way.

## Anthropic — Building effective agents

<https://www.anthropic.com/engineering/building-effective-agents>

- Prefer the smallest pattern that fits: a prompt chain, a router, parallel
  workers, orchestrator-workers, an evaluator, or one agent loop.
- Their "agent" is a model using tools in a loop with a clear stop. That is
  this repository.
- Extra agents add latency and cost. Add one when a single loop is mixing
  jobs, not before.
- Tools need sharp descriptions and errors the model can use. A confusing
  schema fails inside any framework.

## Toolformer

Schick et al., 2023. <https://arxiv.org/abs/2302.04761>

- A model can be trained to insert API calls (calculator, search, calendar) and to keep a call only when the result helps the next tokens.
- "Do I need a tool here?" can be learned, not only prompted. This repo does not train. The piece to copy: the result lands in the same context the model reads next.

## Plan-and-Solve

Wang et al., 2023. <https://arxiv.org/abs/2305.04091>

- Plain zero-shot chain-of-thought often skips a step. Asking for a plan first, then a solution, reduces those misses.
- Try it by editing `SYSTEM_PROMPT`: write a short plan, then use one tool at a time. A first call that only plans, plus later calls that execute, is still just messages.

## Reflexion

Shinn et al., 2023. <https://arxiv.org/abs/2303.11366>

- After a failed attempt the agent writes a reflection into a buffer. The next attempt reads that note. Weights stay frozen.
- It needs a failure signal: a test, a checker, or a person. The `steps` list is the hook: an evaluator can append a note and call `run_agent` again.

## Frameworks, after the loop feels small

**LangGraph** — <https://docs.langchain.com/oss/python/langgraph/overview>

- Nodes and edges, including cycles, plus checkpoints so a run can pause for a human or resume after a crash.
- Reach for it when you outgrow "append a message and continue": branching, durable retries, approval before a tool.

**CrewAI** — <https://docs.crewai.com/>

- Agents have a role, a goal, and tools. Tasks plus a process form a crew, pictured as a small team with one task list.
- Each member is still an LLM, a tool list, and a stop rule.

## Keep

- The model proposes. Your process runs an allowlist.
- The transcript is the state. Anything missing from it is invisible.
- Stop when there is no tool call, and stop at `max_steps`. Log every step.
- Tool failures are text. The next call is the recovery path.
- Add planning, memory, or another agent when this loop is the bottleneck.
