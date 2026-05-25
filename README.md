# SwarmCoder

**Omnipotent local coding agent + parallel swarm — 100% offline, zero cloud costs.**

SwarmCoder is a conversational coding assistant that runs entirely on your machine using [Ollama](https://ollama.ai). It combines a powerful tool-calling agent loop with a dynamic multi-agent swarm: you describe what kind of expert you need, it generates one, then you can run multiple agents in parallel or have them communicate in a roundtable.

Inspired by [Aider](https://github.com/paul-gauthier/aider), [SWE-agent](https://github.com/princeton-nlp/SWE-agent), [Qwen-Agent](https://github.com/QwenLM/Qwen-Agent), and [OpenHands](https://github.com/All-Hands-AI/OpenHands).

---

## Features

- **31 tools** — file read/write/edit (surgical search-replace), grep, repo map, git, shell, Python execution, web fetch, session memory
- **8 built-in agents** — architect, implementer, reviewer, tester, optimizer, debugger, documenter, security
- **Dynamic agent factory** — describe any expert role, the LLM generates a powerful system prompt automatically
- **Parallel swarm** — run multiple agents simultaneously on independent subtasks
- **Roundtable** — agents see each other's outputs and respond across multiple rounds
- **Persistent agents** — custom agents saved to `~/.swarm_agents.json`, survive restarts
- **Persistent history** — conversation auto-saved to `~/.swarm_history.json`
- **SWARM.md** — drop a `SWARM.md` in your project root, SwarmCoder reads it as project memory at startup

---

## Requirements

- macOS / Linux
- [Ollama](https://ollama.ai) installed and running
- Python 3.10+

---

## Setup

```bash
git clone https://github.com/Leonardo-Corte/swarm-coder.git
cd swarm-coder

# Create Python env
python3 -m venv venv
venv/bin/pip install openai

# Pull models (choose based on your RAM)
ollama pull qwen-agent:latest      # orchestrator — has native tool calling
ollama pull qwen2.5-coder:14b      # sub-agents — optimized for code (needs ~10GB)

# Make launcher executable
chmod +x swarm.sh qcode.sh

# Optional: add to PATH
ln -s $(pwd)/swarm.sh /usr/local/bin/swarm-coder
ln -s $(pwd)/qcode.sh /usr/local/bin/qcode
```

---

## Usage

```bash
swarm-coder
```

### Conversation flow

```
you > analyze the project at ./my-app and tell me where we are
swarm > [calls analyze_project, reads files, git log...] here's the summary...

you > my goal is to add JWT auth. create a task plan
swarm > [saves plan to session notes] here are the 8 steps...

you > execute the task plan
swarm > [step by step: reads files, invokes implementer+tester in parallel, writes code, runs tests, commits]

you > create an agent called ios_llm_engineer who is an iOS expert in CoreML and MLX
swarm > [calls create_agent, LLM generates expert prompt] Agent ready.

you > create a ux_designer and a backend_engineer
swarm > [creates both]

you > run a roundtable with ios_llm_engineer, ux_designer, backend_engineer. 2 rounds
swarm > [all 3 run in parallel, then see each other's outputs and respond]
```

### Commands

| Command | Effect |
|---------|--------|
| `/agents` | list built-in agents |
| `/myagents` | list custom agents |
| `/tools` | list all 31 tools |
| `/notes` | list session notes |
| `/save` | save history manually |
| `/resume` | reload last session |
| `/clear` | clear history |
| `/model NAME` | switch orchestrator model |
| `/submodel NAME` | switch sub-agent model |
| `/swarm` | show SWARM.md |

---

## Project memory (SWARM.md)

Copy `SWARM.md.template` into your project root as `SWARM.md` and fill it in. SwarmCoder reads it automatically at startup so it always knows the project context without you having to re-explain it.

```bash
cp /path/to/swarm-coder/SWARM.md.template ./my-project/SWARM.md
```

---

## Architecture

```
swarm/
  main.py          # Conversational agent loop: 31 tools, dynamic agents, roundtable
  orchestrator.py  # Batch pipeline mode (non-interactive)
swarm.sh           # Launcher (interactive by default, --batch for pipeline)
qcode.sh           # qwen-code wrapper pointed at Ollama
SWARM.md.template  # Project memory template
requirements.txt
```

### Tool-call parsing (3-layer resilience)

SwarmCoder works with any Ollama model regardless of native tool-calling support:

1. **Native** — structured `tool_calls` from the OpenAI-compatible API
2. **XML tags** — `<tool_call>{...}</tool_call>` in model text
3. **JSON extraction** — bracket-balanced JSON object detection from any text output

### Models

| Role | Default model | Why |
|------|-------------|-----|
| Orchestrator | `qwen-agent:latest` | Native tool-calling template |
| Sub-agents | `qwen2.5-coder:14b` | Best open coding model at 14B |

Switch any time with `/model` and `/submodel` in the chat.

---

## License

MIT
