"""
SwarmCoder v2 - Omnipotent conversational coding agent + parallel swarm.
Inspired by: Aider (search/replace, repo map), SWE-agent (ACI tools),
             Qwen-Agent (tool registry), OpenHands (rich environment).
"""

import os, sys, json, re, subprocess, difflib, urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Any

# ── Instructor structured extraction model ──────────────────────────────────
class ToolCallExtract(BaseModel):
    name: str = Field(description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

# ── Config ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL    = os.environ.get("SWARM_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY     = os.environ.get("SWARM_API_KEY", "ollama")
# qwen-agent:latest has native tool-calling template; qwen2.5-coder:14b is used for sub-agents
ORCHESTRATOR_MODEL = os.environ.get("SWARM_MODEL", "qwen-agent:latest")
SUB_AGENT_MODEL    = os.environ.get("SWARM_SUB_MODEL", "qwen2.5-coder:14b")
MAX_FILE_CHARS     = 12000
MAX_SHELL_CHARS    = 6000
MAX_ITERATIONS     = 30

# ── Reflexion prompt ────────────────────────────────────────────────────────
REFLEXION_PROMPT = """\
You just completed a task. Reflect critically and briefly:
1. What worked well?
2. What mistakes or inefficiencies occurred?
3. What would you do differently next time?
4. Any project-specific patterns or gotchas to remember?

Be specific and concise. Max 150 words. Write as bullet points.
Format your response as:
## Learnings — {date}
- [learning 1]
- [learning 2]
..."""

# ── History persistence ─────────────────────────────────────────────────────
HISTORY_FILE = Path.home() / ".swarm_history.json"

def _save_history(history: list, path: Path = HISTORY_FILE):
    try:
        path.write_text(json.dumps(history, indent=2, ensure_ascii=False))
    except Exception:
        pass

def _load_history(path: Path = HISTORY_FILE) -> list:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return []

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# ── ChromaDB globals (lazy-init) ────────────────────────────────────────────
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None

# ── Colors ─────────────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
BLUE="\033[94m"; GREEN="\033[92m"; YELLOW="\033[93m"
CYAN="\033[96m"; MAGENTA="\033[95m"; RED="\033[91m"
def c(col,t): return f"{col}{t}{R}"

# ── Session memory (in-memory, survives within one run) ───────────────────
_NOTES: dict[str, str] = {}

# ── Built-in sub-agent personas ────────────────────────────────────────────
AGENTS = {
    "architect":   "You are an elite software architect. Produce clear implementation plans: components, interfaces, file structure, ordered steps. Save your plan with save_note(). Be decisive and specific.",
    "implementer": "You are a senior software engineer and EXECUTOR. Write complete production-ready code and write every file to disk using write_file(). Never output code as text. After writing: verify with run_shell() or run_python(). Fix any failures immediately.",
    "reviewer":    "You are a rigorous code reviewer. Read all relevant files with read_file(), then list numbered findings: bugs, security issues, performance problems, missing edge cases. Be specific with file:line references.",
    "tester":      "You are a QA engineer and EXECUTOR. Write comprehensive pytest tests covering unit tests, edge cases, and integration. Write the test file to disk with write_file(), then run with run_shell('python -m pytest ...'). Report pass/fail.",
    "optimizer":   "You are a performance engineer. Read code with read_file(), identify bottlenecks, then use edit_file() to apply improvements without changing behavior. Run benchmarks to verify speedup.",
    "debugger":    "You are a debugging expert. Read error messages and code with read_file(). Trace root causes precisely. Apply minimal fixes with edit_file(). Re-run to confirm the fix works.",
    "documenter":  "You are a technical writer. Read code with read_file() and write concise docstrings, README sections, and API docs to disk with write_file(). Markdown format. Clear and accurate.",
    "security":    "You are a security engineer. Read all code with read_file(). Identify OWASP top-10 issues, injection flaws, auth bypasses, secrets exposure. Report with file:line references and exact fix recommendations.",
}

EXECUTOR_MANDATE = """\
EXECUTOR MANDATE — ABSOLUTE RULES:
1. NEVER output code as markdown to the user. Code MUST go to disk via write_file().
2. When you produce code for any file: call write_file() IMMEDIATELY — no preamble.
3. After writing files: call run_shell() or run_python() to verify they work.
4. Only respond in plain text AFTER all files are written AND verified.
5. If tests fail: fix with edit_file() and re-run. Never give up after one failure.
WRONG: "Here is the code: ```python def foo(): ...```"
RIGHT: <tool_call>{"name": "write_file", "arguments": {"path": "foo.py", "content": "def foo(): ..."}}</tool_call>"""

ARCHITECT_SYSTEM_PROMPT = """\
You are an elite software architect. Your job is PLANNING ONLY — never write implementation code.

Given a coding task, produce a complete project blueprint:

1. FILE STRUCTURE — list every file to create with exact path:
   - path/to/file.py — one-line description of its role

2. MODULE INTERFACES — for each file, specify:
   - Functions/classes it exposes (name, params, return type)
   - What it imports from other modules in this project

3. IMPLEMENTATION ORDER — numbered list of files in dependency order
   (files with no internal deps first)

4. KEY DECISIONS — 2-3 architectural choices made and why

FORMAT RULES:
- Use exact Python syntax for signatures: def foo(x: int, y: str) -> list[str]:
- Be complete — the implementer will follow this exactly, cannot ask questions
- Do NOT write any function bodies or implementation code
- Do NOT use placeholders like "# TODO" or "..."
"""

# Names of tools sub-agents are allowed to use
SUB_AGENT_TOOLS_NAMES = {
    "read_file", "write_file", "edit_file", "append_file", "list_directory",
    "grep", "find_files", "run_shell", "run_python",
    "git_status", "git_diff", "save_note", "get_note",
    "plan_project", "implement_plan",
}

# ── Dynamic custom agents (created at runtime by the user) ─────────────────
# { name: { system_prompt, role, skills, created_at } }
CUSTOM_AGENTS: dict[str, dict] = {}

AGENTS_FILE = Path.home() / ".swarm_agents.json"

def _load_agents_from_disk():
    if AGENTS_FILE.exists():
        try:
            data = json.loads(AGENTS_FILE.read_text())
            CUSTOM_AGENTS.update(data)
        except Exception:
            pass

_load_agents_from_disk()

def _save_agents_to_disk():
    AGENTS_FILE.write_text(json.dumps(CUSTOM_AGENTS, indent=2, ensure_ascii=False))

def _get_system(name: str) -> str | None:
    """Return system prompt for built-in or custom agent."""
    if name in CUSTOM_AGENTS:
        return CUSTOM_AGENTS[name]["system_prompt"]
    if name in AGENTS:
        return AGENTS[name]
    return None

def _run_agent_with_system(system: str, task: str, context: str = "") -> str:
    """Run a sub-agent with a full tool-calling loop (up to 12 iterations)."""
    # SUB_AGENT_TOOLS is defined after TOOLS_SCHEMA — access lazily via module globals
    sub_tools = globals().get("SUB_AGENT_TOOLS", [])
    full_system = EXECUTOR_MANDATE + "\n\n" + system
    user = f"Context:\n{context}\n\nTask:\n{task}" if context else task
    messages = [
        {"role": "system", "content": full_system},
        {"role": "user",   "content": user},
    ]
    output_parts = []
    MAX_SUB_ITER = 12
    for iteration in range(MAX_SUB_ITER):
        try:
            kwargs = dict(
                model=SUB_AGENT_MODEL,
                messages=messages,
                temperature=0.25,
                max_tokens=4096,
            )
            if sub_tools:
                kwargs["tools"] = sub_tools
                kwargs["tool_choice"] = "auto"
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            return f"ERROR: {e}"

        msg = resp.choices[0].message
        content = msg.content or ""

        # ── Layer 1: Native structured tool_calls ─────────────────────────
        if getattr(msg, "tool_calls", None):
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                fn = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                print(c(DIM, f"    [sub-tool] {fn}({_fmt_args(args)})"), flush=True)
                result = dispatch(fn, args)
                short = result[:200].replace("\n", " ")
                print(c(DIM, f"    [sub-tool] → {short}"), flush=True)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "content": result[:MAX_SHELL_CHARS]})
            continue

        # ── Layer 2: XML/JSON tool call in text ───────────────────────────
        parsed_calls = _parse_tool_from_text(content)
        if parsed_calls:
            fake_tool_calls = []
            for pc in parsed_calls:
                fid = f"sub_call_{pc['name']}_{iteration}"
                fake_tool_calls.append({
                    "id": fid, "type": "function",
                    "function": {"name": pc["name"], "arguments": json.dumps(pc["arguments"])}
                })
            messages.append({"role": "assistant", "content": "", "tool_calls": fake_tool_calls})
            for pc, ftc in zip(parsed_calls, fake_tool_calls):
                print(c(DIM, f"    [sub-tool²] {pc['name']}({_fmt_args(pc['arguments'])})"), flush=True)
                result = dispatch(pc["name"], pc["arguments"])
                short = result[:200].replace("\n", " ")
                print(c(DIM, f"    [sub-tool²] → {short}"), flush=True)
                messages.append({"role": "tool", "tool_call_id": ftc["id"],
                                  "content": result[:MAX_SHELL_CHARS]})
            continue

        # ── Layer 3: Plain text — sub-agent is done ───────────────────────
        if content:
            output_parts.append(content)
        return "\n\n".join(output_parts) if output_parts else content

    return "\n\n".join(output_parts) or "ERROR: sub-agent reached max iterations."

# ─── META-PROMPT for agent creation ────────────────────────────────────────

META_PROMPT = """You are an elite AI systems designer who creates highly specialized AI agent personas.

Generate an EXTREMELY POWERFUL, EXPERT-LEVEL system prompt for the following agent:

Name: {name}
Role: {role}
Domain: {domain}
Required skills: {skills}
Extra instructions: {extra}

Rules for the system prompt you write:
1. Start directly with the agent's identity ("You are...")
2. Establish DEEP domain expertise — name specific frameworks, tools, APIs, methodologies
3. Define a clear problem-solving process specific to this domain
4. Specify exact output formats and quality standards
5. Include domain-specific best practices, constraints, and gotchas
6. Make the agent AUTONOMOUS and DECISIVE — it executes, not asks
7. If it involves code: specify languages, libraries, patterns to prefer/avoid
8. If it involves design/UX: specify deliverable formats (wireframes, user flows, copy)
9. If it involves communication: specify tone, structure, channels
10. Length: 250–450 words. Dense with expertise. No fluff.

Output ONLY the system prompt. Nothing else. Start with "You are"."""

# ═══════════════════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _shell(cmd: str, cwd: str = None, timeout: int = 60) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout, cwd=cwd,
                           env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"})
        out = (r.stdout + r.stderr)[:MAX_SHELL_CHARS]
        return f"exit={r.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"

# ── File tools ──────────────────────────────────────────────────────────────

def tool_read_file(path: str, start_line: int = None, end_line: int = None) -> str:
    p = Path(path).expanduser()
    if not p.exists(): return f"ERROR: not found: {p}"
    text = p.read_text(errors="replace")
    lines = text.splitlines()
    if start_line or end_line:
        s = (start_line or 1) - 1
        e = end_line or len(lines)
        lines = lines[s:e]
        text = "\n".join(f"{s+i+1:4} | {l}" for i, l in enumerate(lines))
        return text
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS] + f"\n... [truncated — {len(p.read_bytes())} bytes total]"
    # Add line numbers
    numbered = "\n".join(f"{i+1:4} | {l}" for i, l in enumerate(text.splitlines()))
    return numbered

def tool_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {p}"

def tool_edit_file(path: str, old_string: str, new_string: str) -> str:
    """Surgical search-and-replace inside a file (from Aider's editblock approach)."""
    p = Path(path).expanduser()
    if not p.exists(): return f"ERROR: not found: {p}"
    original = p.read_text(encoding="utf-8")

    # 1. Exact match
    if old_string in original:
        result = original.replace(old_string, new_string, 1)
        p.write_text(result, encoding="utf-8")
        return f"OK: replaced in {p}"

    # 2. Normalized whitespace match
    norm_orig = "\n".join(l.rstrip() for l in original.splitlines())
    norm_old  = "\n".join(l.rstrip() for l in old_string.splitlines())
    if norm_old in norm_orig:
        idx = norm_orig.index(norm_old)
        # find the actual position in original
        result = original[:idx] + new_string + original[idx + len(norm_old):]
        p.write_text(result, encoding="utf-8")
        return f"OK: replaced (normalized) in {p}"

    # 3. Fuzzy match using difflib
    lines_orig = original.splitlines()
    lines_old  = old_string.splitlines()
    matcher = difflib.SequenceMatcher(None, lines_orig, lines_old, autojunk=False)
    best = matcher.find_longest_match(0, len(lines_orig), 0, len(lines_old))
    if best.size >= max(1, len(lines_old) * 0.85):
        result_lines = lines_orig[:best.a] + new_string.splitlines() + lines_orig[best.a + best.size:]
        p.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
        return f"OK: replaced (fuzzy, match={best.size}/{len(lines_old)}) in {p}"

    return (f"ERROR: could not find the old_string in {p}.\n"
            f"File has {len(lines_orig)} lines. "
            f"Tip: use read_file to see current content and adjust old_string.")

def tool_append_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(content)
    return f"OK: appended {len(content)} chars to {p}"

def tool_delete_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists(): return f"ERROR: not found: {p}"
    p.unlink()
    return f"OK: deleted {p}"

def tool_list_directory(path: str = ".", show_hidden: bool = False) -> str:
    p = Path(path).expanduser()
    if not p.exists(): return f"ERROR: not found: {p}"
    items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
    lines = []
    for item in items:
        if not show_hidden and item.name.startswith("."): continue
        kind = "/" if item.is_dir() else ""
        try:
            size = f"{item.stat().st_size:>8}" if item.is_file() else "     dir"
        except Exception:
            size = "       ?"
        lines.append(f"{size}  {item.name}{kind}")
    return f"{p}\n" + "\n".join(lines)

# ── Search tools ────────────────────────────────────────────────────────────

def tool_grep(pattern: str, path: str = ".", file_glob: str = None,
              case_sensitive: bool = True, context_lines: int = 2) -> str:
    flags = [] if case_sensitive else ["-i"]
    glob_flags = [f"--include={file_glob}"] if file_glob else []
    cmd = ["grep", "-rn", "--color=never"] + flags + glob_flags + [f"-C{context_lines}", pattern, path]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        out = r.stdout[:MAX_SHELL_CHARS]
        if not out.strip():
            return f"No matches for '{pattern}' in {path}"
        return out
    except Exception as e:
        return f"ERROR: {e}"

def tool_find_files(directory: str = ".", pattern: str = "*",
                    file_type: str = None, max_results: int = 100) -> str:
    cmd = ["find", directory, "-name", pattern]
    if file_type == "file":   cmd += ["-type", "f"]
    if file_type == "dir":    cmd += ["-type", "d"]
    cmd += ["-not", "-path", "*/.git/*", "-not", "-path", "*/node_modules/*",
            "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/venv/*",
            "-not", "-path", "*/dist/*", "-not", "-path", "*/build/*"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        lines = r.stdout.strip().splitlines()[:max_results]
        return "\n".join(lines) or "No files found"
    except Exception as e:
        return f"ERROR: {e}"

def tool_repo_map(directory: str = ".") -> str:
    """Lightweight code map: extract all function/class/method definitions."""
    path = Path(directory).expanduser().resolve()
    lines_out = [f"# Repo map: {path}\n"]

    # Patterns per language
    patterns = {
        "*.py":    [r"^(class\s+\w+|def\s+\w+|async def\s+\w+)"],
        "*.js":    [r"(^|\s)(function\s+\w+|class\s+\w+|const\s+\w+\s*=.*=>|exports\.\w+)"],
        "*.ts":    [r"(^|\s)(function\s+\w+|class\s+\w+|interface\s+\w+|type\s+\w+\s*=|export)"],
        "*.go":    [r"^func\s+\w+"],
        "*.rs":    [r"^(pub\s+)?(fn|struct|enum|impl|trait)\s+\w+"],
        "*.java":  [r"(public|private|protected|static)\s+\w+\s+\w+\s*\("],
        "*.rb":    [r"^(def|class|module)\s+\w+"],
    }

    found = {}
    for glob, pats in patterns.items():
        for fpath in sorted(path.rglob(glob)):
            rel = fpath.relative_to(path)
            skip_dirs = {".git","node_modules","__pycache__","venv","dist","build",".tox"}
            if any(p in rel.parts for p in skip_dirs): continue
            try:
                text = fpath.read_text(errors="replace")
                hits = []
                for i, line in enumerate(text.splitlines(), 1):
                    for pat in pats:
                        if re.search(pat, line):
                            hits.append(f"  {i:4}: {line.strip()}")
                            break
                if hits:
                    found[str(rel)] = hits
            except Exception:
                continue

    if not found:
        return "No code symbols found (empty repo or unsupported languages)"

    for fpath, hits in sorted(found.items()):
        lines_out.append(f"\n{fpath}")
        lines_out.extend(hits[:40])  # cap per file

    return "\n".join(lines_out)

# ── Execution tools ──────────────────────────────────────────────────────────

def tool_run_shell(command: str, cwd: str = None, timeout: int = 60) -> str:
    BLOCKED = ["rm -rf /", "mkfs", "dd if=", ":(){:", "curl|sh", "wget|sh"]
    for b in BLOCKED:
        if b in command:
            return f"ERROR: blocked dangerous command pattern: {b}"
    return _shell(command, cwd=cwd, timeout=timeout)

def tool_run_python(code: str, timeout: int = 30) -> str:
    """Execute Python code in a subprocess and return stdout+stderr."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(
            [sys.executable, fname], capture_output=True, text=True, timeout=timeout
        )
        out = (r.stdout + r.stderr)[:MAX_SHELL_CHARS]
        return f"exit={r.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {timeout}s"
    finally:
        Path(fname).unlink(missing_ok=True)

# ── Git tools ───────────────────────────────────────────────────────────────

def tool_git_status(path: str = ".") -> str:
    return _shell(f"git -C {path!r} status --short && git -C {path!r} log --oneline -5")

def tool_git_diff(path: str = ".", staged: bool = False) -> str:
    flag = "--cached" if staged else ""
    return _shell(f"git -C {path!r} diff {flag} --stat && git -C {path!r} diff {flag}")[:MAX_SHELL_CHARS]

def tool_git_log(path: str = ".", n: int = 20) -> str:
    return _shell(f"git -C {path!r} log --oneline --graph -n {n}")

def tool_git_add(files: str, path: str = ".") -> str:
    return _shell(f"git -C {path!r} add {files}")

def tool_git_commit(message: str, path: str = ".", add_all: bool = False) -> str:
    if add_all:
        _shell(f"git -C {path!r} add -A")
    return _shell(f"git -C {path!r} commit -m {message!r}")

def tool_git_create_branch(branch: str, path: str = ".") -> str:
    return _shell(f"git -C {path!r} checkout -b {branch!r}")

def tool_git_checkout(ref: str, path: str = ".") -> str:
    return _shell(f"git -C {path!r} checkout {ref!r}")

def tool_git_clone(url: str, destination: str = ".") -> str:
    return _shell(f"git clone {url!r} {destination!r}", timeout=120)

# ── Web tools ────────────────────────────────────────────────────────────────

def tool_web_fetch(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return the text content (strips HTML tags)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SwarmCoder/2.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # Strip HTML tags
        clean = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL)
        clean = re.sub(r"<script[^>]*>.*?</script>", "", clean, flags=re.DOTALL)
        clean = re.sub(r"<[^>]+>", " ", clean)
        clean = re.sub(r"\s{3,}", "\n\n", clean).strip()
        return clean[:max_chars]
    except Exception as e:
        return f"ERROR fetching {url}: {e}"

# ── Memory tools ─────────────────────────────────────────────────────────────

def tool_save_note(key: str, content: str) -> str:
    _NOTES[key] = content
    return f"OK: saved note '{key}' ({len(content)} chars)"

def tool_get_note(key: str) -> str:
    if key not in _NOTES:
        return f"ERROR: no note named '{key}'. Available: {list(_NOTES.keys())}"
    return _NOTES[key]

def tool_list_notes() -> str:
    if not _NOTES:
        return "No notes saved."
    return "\n".join(f"  {k}: {v[:60]}{'...' if len(v)>60 else ''}" for k,v in _NOTES.items())

# ── Project analysis ─────────────────────────────────────────────────────────

def tool_analyze_project(directory: str, max_depth: int = 3) -> str:
    path = Path(directory).expanduser().resolve()
    if not path.exists(): return f"ERROR: not found: {path}"
    parts = [f"# Project: {path}\n"]

    # Tree (excluding clutter)
    r = subprocess.run(
        ["find", str(path), "-maxdepth", str(max_depth),
         "-not", "-path", "*/.git/*", "-not", "-path", "*/node_modules/*",
         "-not", "-path", "*/__pycache__/*", "-not", "-path", "*/venv/*",
         "-not", "-path", "*/dist/*"],
        capture_output=True, text=True, timeout=10
    )
    parts.append("## Tree\n```\n" + r.stdout[:2500] + "\n```")

    # Git
    gl = _shell(f"git -C {str(path)!r} log --oneline -8 2>/dev/null")
    gs = _shell(f"git -C {str(path)!r} status --short 2>/dev/null")
    if "exit=0" in gl:
        parts.append(f"## Git log\n```\n{gl}\n```")
        parts.append(f"## Git status\n```\n{gs}\n```")

    # Key files
    for fname in ["README.md","CLAUDE.md","SWARM.md","package.json","pyproject.toml",
                  "requirements.txt","Makefile","Dockerfile",".env.example"]:
        fp = path / fname
        if fp.exists():
            content = fp.read_text(errors="replace")[:1500]
            parts.append(f"## {fname}\n```\n{content}\n```")

    return "\n\n".join(parts)

# ── Sub-agent tools ───────────────────────────────────────────────────────────

def _run_agent(agent: str, task: str, context: str = "") -> dict:
    """Run a named sub-agent with the full tool-calling loop."""
    system = _get_system(agent) or AGENTS["implementer"]
    try:
        output = _run_agent_with_system(system, task, context)
        return {"agent": agent, "output": output, "error": None}
    except Exception as e:
        return {"agent": agent, "output": None, "error": str(e)}

def tool_invoke_agent(agent: str, task: str, context: str = "") -> str:
    all_agents = {**AGENTS, **{k: v["system_prompt"] for k,v in CUSTOM_AGENTS.items()}}
    if agent not in all_agents:
        return f"ERROR: unknown agent '{agent}'. Built-in: {list(AGENTS.keys())}. Custom: {list(CUSTOM_AGENTS.keys())}"
    print(c(CYAN, f"  ↳ [{agent}] working..."), flush=True)
    r = _run_agent(agent, task, context)
    if r["error"]: return f"[{agent}] ERROR: {r['error']}"
    print(c(GREEN, f"  ✓ [{agent}]"), flush=True)
    return f"[{agent}]:\n{r['output']}"

def tool_invoke_agents_parallel(tasks: list) -> str:
    """Run multiple agents (built-in or custom) simultaneously."""
    names = [t.get("agent","?") for t in tasks]
    print(c(CYAN, f"  ↳ parallel: {names}"), flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as ex:
        futures = {ex.submit(_run_agent, t["agent"], t["task"], t.get("context","")): t["agent"]
                   for t in tasks}
        for f in as_completed(futures):
            r = f.result()
            if r["error"]:
                results.append(f"[{r['agent']}] ERROR: {r['error']}")
            else:
                print(c(GREEN, f"  ✓ [{r['agent']}]"), flush=True)
                results.append(f"[{r['agent']}]:\n{r['output']}")
    return ("\n\n" + "─"*40 + "\n\n").join(results)

# ── Dynamic agent factory tools ─────────────────────────────────────────────

def tool_create_agent(name: str, role: str, skills: list = None,
                      domain: str = "", extra: str = "") -> str:
    """Generate a powerful custom agent. The LLM writes the expert system prompt."""
    print(c(MAGENTA, f"  ✦ generating agent '{name}'..."), flush=True)
    skills_str = "\n".join(f"- {s}" for s in (skills or []))
    meta = META_PROMPT.format(
        name=name, role=role, domain=domain or role,
        skills=skills_str or "(infer from role)", extra=extra or "None"
    )
    try:
        resp = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[{"role":"user","content":meta}],
            temperature=0.7, max_tokens=800,
        )
        system_prompt = resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR generating agent: {e}"

    CUSTOM_AGENTS[name] = {
        "system_prompt": system_prompt,
        "role": role,
        "skills": skills or [],
        "domain": domain,
        "created_at": datetime.now().isoformat(),
    }
    _save_agents_to_disk()
    print(c(GREEN, f"  ✓ agent '{name}' ready"), flush=True)
    return (f"Agent '{name}' created and saved.\n\n"
            f"{'─'*50}\n{system_prompt}\n{'─'*50}")

def tool_list_custom_agents() -> str:
    """List all dynamically created agents."""
    if not CUSTOM_AGENTS:
        return "No custom agents yet. Use create_agent to create one."
    lines = [f"Custom agents ({len(CUSTOM_AGENTS)}):"]
    for name, data in CUSTOM_AGENTS.items():
        skills = ", ".join(data.get("skills", [])) or "—"
        lines.append(f"\n  {c(MAGENTA, name)}\n    Role: {data['role']}\n    Skills: {skills}")
    return "\n".join(lines)

def tool_inspect_agent(name: str) -> str:
    """Show the full system prompt of a custom agent."""
    if name in CUSTOM_AGENTS:
        data = CUSTOM_AGENTS[name]
        return (f"Agent: {name}\nRole: {data['role']}\n"
                f"Created: {data['created_at']}\n\n"
                f"System prompt:\n{'─'*50}\n{data['system_prompt']}")
    if name in AGENTS:
        return f"Built-in agent '{name}':\n{AGENTS[name]}"
    return f"ERROR: agent '{name}' not found."

def tool_delete_agent(name: str) -> str:
    """Remove a custom agent."""
    if name not in CUSTOM_AGENTS:
        return f"ERROR: custom agent '{name}' not found."
    del CUSTOM_AGENTS[name]
    _save_agents_to_disk()
    return f"OK: deleted agent '{name}'"

def tool_agent_roundtable(agents: list, task: str,
                           rounds: int = 1, share_context: bool = True) -> str:
    """
    Multi-agent discussion: agents work on the same task and can see each other's outputs.
    Each round, every agent gets the previous round's outputs as context.
    Great for: design review, architecture debates, cross-discipline collaboration.

    Example: [ios_engineer, ux_designer, backend_dev] discuss an API design.
    """
    all_names = list(AGENTS.keys()) + list(CUSTOM_AGENTS.keys())
    bad = [a for a in agents if a not in all_names]
    if bad:
        return f"ERROR: unknown agents: {bad}. Available: {all_names}"

    print(c(MAGENTA, f"\n  ⬡ Roundtable: {agents} × {rounds} round(s)"), flush=True)

    transcript = []   # list of {agent, round, output}
    prev_outputs = {} # agent -> last output

    for rnd in range(1, rounds + 1):
        print(c(DIM, f"\n  — Round {rnd}/{rounds} —"), flush=True)
        round_tasks = []
        for agent_name in agents:
            if share_context and prev_outputs:
                ctx_lines = [f"[{a}]:\n{o[:800]}" for a, o in prev_outputs.items() if a != agent_name]
                ctx = f"Other agents' views:\n\n" + "\n\n".join(ctx_lines)
            else:
                ctx = ""
            round_tasks.append((agent_name, task, ctx))

        # Run all agents in parallel for this round
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            futures = {
                ex.submit(_run_agent_with_system,
                          _get_system(a) or AGENTS["implementer"], task, ctx): (a, rnd)
                for a, task_str, ctx in round_tasks
                for a in [a]
            }
            # Rebuild correctly
            futures = {}
            for agent_name, task_str, ctx in round_tasks:
                system = _get_system(agent_name) or AGENTS["implementer"]
                f = ex.submit(_run_agent_with_system, system, task_str, ctx)
                futures[f] = (agent_name, rnd)

            for f in as_completed(futures):
                agent_name, rnd_num = futures[f]
                output = f.result()
                prev_outputs[agent_name] = output
                transcript.append({"agent": agent_name, "round": rnd_num, "output": output})
                print(c(GREEN, f"  ✓ [{agent_name}] rnd {rnd_num}"), flush=True)

    # Format transcript
    out_lines = [f"\n{'═'*60}", f"ROUNDTABLE: {' ↔ '.join(agents)}  ({rounds} round(s))", f"TASK: {task[:100]}", f"{'═'*60}"]
    for entry in sorted(transcript, key=lambda x: (x["round"], agents.index(x["agent"]))):
        col = MAGENTA if entry["agent"] in CUSTOM_AGENTS else CYAN
        out_lines.append(f"\n{c(col, BOLD+'['+entry['agent']+']'+R+c(col,''))} (round {entry['round']})\n{entry['output']}")
        out_lines.append("─" * 50)
    return "\n".join(out_lines)

# ── ChromaDB semantic search tools ──────────────────────────────────────────

def _get_or_create_chroma(cwd: str):
    """Lazy-init ChromaDB collection for the current project."""
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    import chromadb
    from chromadb.utils import embedding_functions

    db_path = Path.home() / ".swarm_chroma" / Path(cwd).name
    db_path.mkdir(parents=True, exist_ok=True)

    _CHROMA_CLIENT = chromadb.PersistentClient(path=str(db_path))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
        name="codebase", embedding_function=ef
    )
    return _CHROMA_COLLECTION

def tool_index_codebase(directory: str = ".") -> str:
    """Index all code files into ChromaDB for semantic search."""
    path = Path(directory).expanduser().resolve()
    coll = _get_or_create_chroma(str(path))

    extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".rb", ".c", ".cpp", ".h"}
    skip_dirs = {".git", "node_modules", "__pycache__", "venv", "dist", "build"}

    docs, ids, metas = [], [], []
    for fpath in sorted(path.rglob("*")):
        if fpath.suffix not in extensions:
            continue
        if any(s in fpath.parts for s in skip_dirs):
            continue
        try:
            text = fpath.read_text(errors="replace")
            if len(text) < 10:
                continue
            # Chunk into 50-line segments
            lines = text.splitlines()
            for i in range(0, len(lines), 50):
                chunk = "\n".join(lines[i:i+50])
                chunk_id = f"{fpath}:{i}"
                docs.append(chunk)
                ids.append(chunk_id)
                metas.append({"file": str(fpath.relative_to(path)), "line_start": i})
        except Exception:
            continue

    if not docs:
        return "No code files found to index."

    # Upsert in batches
    batch = 100
    for i in range(0, len(docs), batch):
        coll.upsert(documents=docs[i:i+batch], ids=ids[i:i+batch], metadatas=metas[i:i+batch])

    return f"OK: indexed {len(docs)} chunks from {len({m['file'] for m in metas})} files."

def tool_semantic_search(query: str, n_results: int = 5, directory: str = ".") -> str:
    """Semantic search over the indexed codebase. Finds conceptually related code."""
    try:
        coll = _get_or_create_chroma(str(Path(directory).expanduser().resolve()))
        results = coll.query(query_texts=[query], n_results=min(n_results, 10))
        if not results["documents"][0]:
            return "No results. Run index_codebase first."
        lines = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            score = round(1 - dist, 3)
            lines.append(f"\n[score={score}] {meta['file']}:{meta['line_start']}\n{doc[:300]}")
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}. Run index_codebase first."

# ── Reflexion ────────────────────────────────────────────────────────────────

def _reflect_on_task(user_msg: str, assistant_reply: str, cwd: str) -> str | None:
    """Ask the model to reflect on the just-completed task. Appends learnings to SWARM.md."""
    # Only reflect on substantial tasks (not short questions)
    if len(user_msg) < 30 or len(assistant_reply) < 100:
        return None
    try:
        resp = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": REFLEXION_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"))},
                {"role": "user", "content": f"Task I was given:\n{user_msg[:500]}\n\nMy response:\n{assistant_reply[:1000]}"},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        reflection = resp.choices[0].message.content.strip()
        if not reflection:
            return None

        # Append to SWARM.md
        swarm_md = Path(cwd) / "SWARM.md"
        divider = "\n\n---\n"
        if swarm_md.exists():
            swarm_md.write_text(swarm_md.read_text() + divider + reflection + "\n")
        else:
            swarm_md.write_text(f"# SwarmCoder Project Memory\n\n{reflection}\n")

        return reflection
    except Exception:
        return None

# ── Architect mode tools ─────────────────────────────────────────────────────

def tool_plan_project(task: str, directory: str = ".") -> str:
    """
    Architect phase: produce a complete file structure + interface plan BEFORE writing any code.
    Always call this before implement_plan() for multi-file projects.
    """
    print(c(MAGENTA, "  ✦ architect planning..."), flush=True)

    # Read existing project context
    ctx_parts = []
    path = Path(directory).expanduser().resolve()
    if path.exists():
        # Add repo map for context
        repo = tool_repo_map(directory)
        if "No code symbols" not in repo:
            ctx_parts.append(f"Existing codebase:\n{repo[:2000]}")

    context = "\n\n".join(ctx_parts) if ctx_parts else "New project — no existing code."

    try:
        resp = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=[
                {"role": "system", "content": ARCHITECT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nTask:\n{task}"},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        plan = resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR in architect planning: {e}"

    # Save plan to session notes
    tool_save_note("architect_plan", plan)
    tool_save_note("architect_task", task)
    tool_save_note("architect_dir", directory)

    print(c(GREEN, "  ✓ architect plan ready"), flush=True)
    return f"ARCHITECT PLAN:\n{'─'*50}\n{plan}\n{'─'*50}\nPlan saved. Now call implement_plan() to execute it."


def tool_implement_plan(plan: str = "", directory: str = "") -> str:
    """
    Editor phase: implement the architect's plan file by file.
    If plan is empty, reads from session notes (set by plan_project).
    """
    # Load from notes if not provided
    if not plan:
        plan = _NOTES.get("architect_plan", "")
    if not directory:
        directory = _NOTES.get("architect_dir", ".")
    if not plan:
        return "ERROR: no plan found. Call plan_project() first."

    task = _NOTES.get("architect_task", "implement the plan")
    print(c(CYAN, "  ↳ implementer executing plan..."), flush=True)

    IMPLEMENTER_PROMPT = EXECUTOR_MANDATE + """

You are a senior software engineer implementing an architect's plan.
Follow the plan EXACTLY. Implement files in the specified order.
For each file:
1. Call write_file() with the complete implementation
2. Call run_python() or run_shell() to verify it has no syntax errors
3. Fix any errors immediately with edit_file()

Do NOT deviate from the plan's interfaces — other files depend on them."""

    result = _run_agent_with_system(
        system=IMPLEMENTER_PROMPT,
        task=f"Implement this architect plan:\n\n{plan}\n\nOriginal task: {task}\nWorking directory: {directory}",
        context="",
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY & DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

TOOL_MAP = {
    # File
    "read_file":               tool_read_file,
    "write_file":              tool_write_file,
    "edit_file":               tool_edit_file,
    "append_file":             tool_append_file,
    "delete_file":             tool_delete_file,
    "list_directory":          tool_list_directory,
    # Search
    "grep":                    tool_grep,
    "find_files":              tool_find_files,
    "repo_map":                tool_repo_map,
    # Execution
    "run_shell":               tool_run_shell,
    "run_python":              tool_run_python,
    # Git
    "git_status":              tool_git_status,
    "git_diff":                tool_git_diff,
    "git_log":                 tool_git_log,
    "git_add":                 tool_git_add,
    "git_commit":              tool_git_commit,
    "git_create_branch":       tool_git_create_branch,
    "git_checkout":            tool_git_checkout,
    "git_clone":               tool_git_clone,
    # Web
    "web_fetch":               tool_web_fetch,
    # Memory
    "save_note":               tool_save_note,
    "get_note":                tool_get_note,
    "list_notes":              tool_list_notes,
    # Project
    "analyze_project":         tool_analyze_project,
    # Built-in agents
    "invoke_agent":            tool_invoke_agent,
    "invoke_agents_parallel":  tool_invoke_agents_parallel,
    # Dynamic agent factory ← NEW
    "create_agent":            tool_create_agent,
    "list_custom_agents":      tool_list_custom_agents,
    "inspect_agent":           tool_inspect_agent,
    "delete_agent":            tool_delete_agent,
    "agent_roundtable":        tool_agent_roundtable,
    # Semantic search (ChromaDB)
    "index_codebase":          tool_index_codebase,
    "semantic_search":         tool_semantic_search,
    # Architect mode
    "plan_project":            tool_plan_project,
    "implement_plan":          tool_implement_plan,
}

TOOLS_SCHEMA = [
    {"type":"function","function":{"name":"read_file","description":"Read a file with line numbers. Supports windowed reading with start_line/end_line.","parameters":{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}},"required":["path"]}}},
    {"type":"function","function":{"name":"write_file","description":"Write (create or overwrite) a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"edit_file","description":"Surgically replace old_string with new_string inside a file. Supports fuzzy matching. PREFER this over write_file for changes to existing files.","parameters":{"type":"object","properties":{"path":{"type":"string"},"old_string":{"type":"string","description":"Exact (or near-exact) text to find and replace"},"new_string":{"type":"string","description":"Replacement text"}},"required":["path","old_string","new_string"]}}},
    {"type":"function","function":{"name":"append_file","description":"Append content to the end of a file.","parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"delete_file","description":"Delete a file.","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"list_directory","description":"List files in a directory.","parameters":{"type":"object","properties":{"path":{"type":"string","default":"."},"show_hidden":{"type":"boolean","default":False}},"required":[]}}},
    {"type":"function","function":{"name":"grep","description":"Search for a pattern in files (like grep -rn).","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string","default":"."},"file_glob":{"type":"string","description":"e.g. '*.py'"},"case_sensitive":{"type":"boolean","default":True},"context_lines":{"type":"integer","default":2}},"required":["pattern"]}}},
    {"type":"function","function":{"name":"find_files","description":"Find files by name pattern.","parameters":{"type":"object","properties":{"directory":{"type":"string","default":"."},"pattern":{"type":"string","default":"*"},"file_type":{"type":"string","enum":["file","dir"],"description":"Optional filter"}},"required":[]}}},
    {"type":"function","function":{"name":"repo_map","description":"Generate a lightweight code map of the repo: all functions, classes, methods with line numbers.","parameters":{"type":"object","properties":{"directory":{"type":"string","default":"."}},"required":[]}}},
    {"type":"function","function":{"name":"run_shell","description":"Execute a shell command. Returns stdout+stderr+exit code.","parameters":{"type":"object","properties":{"command":{"type":"string"},"cwd":{"type":"string"},"timeout":{"type":"integer","default":60}},"required":["command"]}}},
    {"type":"function","function":{"name":"run_python","description":"Execute Python code in a subprocess. Returns output.","parameters":{"type":"object","properties":{"code":{"type":"string"},"timeout":{"type":"integer","default":30}},"required":["code"]}}},
    {"type":"function","function":{"name":"git_status","description":"Show git status and recent log.","parameters":{"type":"object","properties":{"path":{"type":"string","default":"."}},"required":[]}}},
    {"type":"function","function":{"name":"git_diff","description":"Show git diff (staged or unstaged).","parameters":{"type":"object","properties":{"path":{"type":"string","default":"."},"staged":{"type":"boolean","default":False}},"required":[]}}},
    {"type":"function","function":{"name":"git_log","description":"Show git log.","parameters":{"type":"object","properties":{"path":{"type":"string","default":"."},"n":{"type":"integer","default":20}},"required":[]}}},
    {"type":"function","function":{"name":"git_add","description":"Stage files for commit.","parameters":{"type":"object","properties":{"files":{"type":"string","description":"Files to add, e.g. '.' or 'src/main.py'"},"path":{"type":"string","default":"."}},"required":["files"]}}},
    {"type":"function","function":{"name":"git_commit","description":"Create a git commit.","parameters":{"type":"object","properties":{"message":{"type":"string"},"path":{"type":"string","default":"."},"add_all":{"type":"boolean","default":False,"description":"Stage all changes before committing"}},"required":["message"]}}},
    {"type":"function","function":{"name":"git_create_branch","description":"Create and checkout a new git branch.","parameters":{"type":"object","properties":{"branch":{"type":"string"},"path":{"type":"string","default":"."}},"required":["branch"]}}},
    {"type":"function","function":{"name":"git_checkout","description":"Checkout a branch or commit.","parameters":{"type":"object","properties":{"ref":{"type":"string"},"path":{"type":"string","default":"."}},"required":["ref"]}}},
    {"type":"function","function":{"name":"git_clone","description":"Clone a git repository.","parameters":{"type":"object","properties":{"url":{"type":"string"},"destination":{"type":"string","default":"."}},"required":["url"]}}},
    {"type":"function","function":{"name":"web_fetch","description":"Fetch a URL and return text content (HTML stripped). Use for docs, APIs, GitHub raw files.","parameters":{"type":"object","properties":{"url":{"type":"string"},"max_chars":{"type":"integer","default":8000}},"required":["url"]}}},
    {"type":"function","function":{"name":"save_note","description":"Save a note to session memory. Use for task plans, decisions, findings.","parameters":{"type":"object","properties":{"key":{"type":"string"},"content":{"type":"string"}},"required":["key","content"]}}},
    {"type":"function","function":{"name":"get_note","description":"Retrieve a saved note by key.","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}},
    {"type":"function","function":{"name":"list_notes","description":"List all saved notes.","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"analyze_project","description":"Deep analysis of a project: file tree, git history, key config files. Always call this first on a new project.","parameters":{"type":"object","properties":{"directory":{"type":"string"},"max_depth":{"type":"integer","default":3}},"required":["directory"]}}},
    {"type":"function","function":{"name":"invoke_agent","description":"Invoke a specialized sub-agent: architect, implementer, reviewer, tester, optimizer, debugger, documenter, security.","parameters":{"type":"object","properties":{"agent":{"type":"string","enum":list(AGENTS.keys())},"task":{"type":"string"},"context":{"type":"string"}},"required":["agent","task"]}}},
    {"type":"function","function":{"name":"invoke_agents_parallel","description":"Run MULTIPLE agents (built-in or custom) simultaneously. Each needs agent+task. Use for independent subtasks.","parameters":{"type":"object","properties":{"tasks":{"type":"array","items":{"type":"object","properties":{"agent":{"type":"string"},"task":{"type":"string"},"context":{"type":"string"}},"required":["agent","task"]}}},"required":["tasks"]}}},
    # ── Dynamic agent factory ──────────────────────────────────────────────
    {"type":"function","function":{"name":"create_agent","description":"CREATE a new custom AI agent by describing its role and skills. The system generates a powerful expert system prompt automatically. The agent is then available for invoke_agent, invoke_agents_parallel, and agent_roundtable. Agents persist across sessions.","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Short unique name for the agent (snake_case, e.g. 'ios_llm_engineer')"},"role":{"type":"string","description":"Clear description of the agent's role and expertise"},"skills":{"type":"array","items":{"type":"string"},"description":"List of specific skills, frameworks, tools this agent masters"},"domain":{"type":"string","description":"Domain context (e.g. 'iOS mobile development', 'web design', 'marketing')"},"extra":{"type":"string","description":"Any extra instructions or constraints for this agent"}},"required":["name","role"]}}},
    {"type":"function","function":{"name":"list_custom_agents","description":"List all custom agents created so far (persisted across sessions).","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"inspect_agent","description":"Show the full system prompt and details of any agent (built-in or custom).","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}},
    {"type":"function","function":{"name":"delete_agent","description":"Delete a custom agent.","parameters":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}},
    {"type":"function","function":{"name":"agent_roundtable","description":"Run a MULTI-AGENT DISCUSSION where agents see each other's outputs and respond. Perfect for cross-discipline collaboration (e.g. iOS engineer + UX designer + backend dev debating an architecture). Each agent gets previous round outputs as context. Multiple rounds = deeper refinement.","parameters":{"type":"object","properties":{"agents":{"type":"array","items":{"type":"string"},"description":"List of agent names (built-in or custom) to include in roundtable"},"task":{"type":"string","description":"The topic/task all agents work on and discuss"},"rounds":{"type":"integer","default":1,"description":"Number of discussion rounds (1=each speaks once, 2=each responds to others, etc.)"},"share_context":{"type":"boolean","default":True,"description":"Whether each agent sees the others' outputs"}},"required":["agents","task"]}}},
    # ── Semantic search (ChromaDB) ────────────────────────────────────────
    {"type":"function","function":{"name":"index_codebase","description":"Index the entire codebase into a local vector database for semantic search. Run this once per project before using semantic_search.","parameters":{"type":"object","properties":{"directory":{"type":"string","default":"."}},"required":[]}}},
    {"type":"function","function":{"name":"semantic_search","description":"Semantic search over the indexed codebase. Finds conceptually related code even without exact keyword match. E.g. 'authentication logic', 'database connection', 'error handling'.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Natural language description of code to find"},"n_results":{"type":"integer","default":5},"directory":{"type":"string","default":"."}},"required":["query"]}}},
    # ── Architect mode ────────────────────────────────────────────────────
    {"type":"function","function":{"name":"plan_project","description":"ARCHITECT PHASE: Produce a complete project blueprint BEFORE writing any code. Lists all files, function signatures, imports, and implementation order. Call this first for ANY multi-file project, then call implement_plan() to execute.","parameters":{"type":"object","properties":{"task":{"type":"string","description":"What to build — be specific and complete"},"directory":{"type":"string","default":".","description":"Working directory"}},"required":["task"]}}},
    {"type":"function","function":{"name":"implement_plan","description":"EDITOR PHASE: Implement the architect plan from plan_project(). Reads the saved plan and implements each file in order, verifying each one works. Call this after plan_project().","parameters":{"type":"object","properties":{"plan":{"type":"string","description":"The plan text (optional — if empty, reads from session notes set by plan_project)"},"directory":{"type":"string","description":"Working directory (optional — reads from session notes if empty)"}},"required":[]}}},
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS_SCHEMA}

# ── Sub-agent tool schema (subset of TOOLS_SCHEMA) ────────────────────────
# Built lazily after TOOLS_SCHEMA is complete; used by _run_agent_with_system
SUB_AGENT_TOOLS = [t for t in TOOLS_SCHEMA if t["function"]["name"] in SUB_AGENT_TOOLS_NAMES]

def dispatch(name: str, args: dict) -> str:
    fn = TOOL_MAP.get(name)
    if not fn: return f"ERROR: unknown tool '{name}'"
    try:
        return str(fn(**args))
    except TypeError as e:
        return f"ERROR calling {name}: {e}"
    except Exception as e:
        return f"ERROR in {name}: {e}"

# ═══════════════════════════════════════════════════════════════════════════
# TOOL CALL PARSING (3-layer resilience)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_json_objects(text: str) -> list[dict]:
    """Extract all valid top-level JSON objects from text using bracket balancing."""
    objects = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            in_str = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                elif ch == '\\' and in_str:
                    escape = True
                elif ch == '"':
                    in_str = not in_str
                elif not in_str:
                    if ch == '{': depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    objects.append(obj)
                            except Exception:
                                pass
                            i = j + 1
                            break
            else:
                i += 1
        else:
            i += 1
    return objects


def _parse_tool_from_text(text: str) -> list[dict]:
    """
    Extract tool calls from model text output. Handles:
      1. <tool_call>{...}</tool_call> XML tags
      2. Any JSON object with "name" matching a known tool
    """
    calls = []

    # Format 1: explicit XML tags (highest priority)
    for m in re.finditer(r"<tool_call>\s*([\s\S]*?)\s*</tool_call>", text):
        for obj in _extract_json_objects(m.group(1)):
            name = obj.get("name") or obj.get("function", {}).get("name")
            args = obj.get("arguments") or obj.get("parameters") or obj.get("function", {}).get("arguments", {})
            if name in TOOL_NAMES and isinstance(args, dict):
                calls.append({"name": name, "arguments": args})
    if calls: return calls

    # Format 2: any JSON object in text (strip markdown fences first)
    cleaned = re.sub(r"```(?:json|python)?\s*", "", text).replace("```", "")
    for obj in _extract_json_objects(cleaned):
        name = obj.get("name") or obj.get("function", {}).get("name")
        args = obj.get("arguments") or obj.get("parameters") or {}

        if isinstance(args, dict):
            if name in TOOL_NAMES:
                # Standard match
                calls.append({"name": name, "arguments": args})
            elif name and name not in TOOL_NAMES and "role" in args:
                # Model used agent name as tool name — treat as create_agent call
                calls.append({"name": "create_agent",
                               "arguments": {"name": name, **args}})

    return calls


def _instructor_extract_tool(content: str) -> list[dict] | None:
    """Layer 4: use instructor to extract tool call from unstructured text."""
    try:
        import instructor
        iclient = instructor.from_openai(
            OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY),
            mode=instructor.Mode.JSON,
        )
        extraction = iclient.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            response_model=ToolCallExtract,
            messages=[
                {"role": "system", "content": f"Extract the tool call from this text. Valid tool names: {sorted(TOOL_NAMES)}"},
                {"role": "user", "content": content},
            ],
            max_retries=2,
            temperature=0,
            max_tokens=512,
        )
        if extraction.name in TOOL_NAMES:
            return [{"name": extraction.name, "arguments": extraction.arguments}]
    except Exception:
        pass
    return None


def _extract_code_blocks_and_write(text: str, cwd: str) -> list[str]:
    """
    When the model returns code in markdown instead of using write_file,
    auto-extract code blocks with filename hints and write them to disk.
    Returns list of written paths.
    """
    written = []
    # Pattern: optional filename before a fenced code block
    pattern = re.compile(
        r'(?:^|\n)'                              # start of line
        r'(?:[*#`\s]*)?'                         # optional formatting
        r'([\w./\-]+\.(?:py|js|ts|go|rs|sh|yaml|json|toml|md|txt))'  # filename
        r'[:`*\s]*\n'                            # separator
        r'```(?:\w+)?\n([\s\S]+?)```',           # fenced code block
        re.MULTILINE
    )
    for m in pattern.finditer(text):
        filename = m.group(1).strip()
        code = m.group(2)
        # Skip filenames that look like random words
        if '/' not in filename and '.' not in filename:
            continue
        path = Path(cwd) / filename
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code, encoding="utf-8")
            written.append(str(path))
        except Exception:
            pass
    return written


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════

def _build_system_prompt(cwd: str = ".") -> str:
    """Build system prompt. Injects SWARM.md project memory if present."""
    swarm_md = Path(cwd) / "SWARM.md"
    project_ctx = ""
    if swarm_md.exists():
        content = swarm_md.read_text(errors="replace")[:3000]
        project_ctx = f"\n\n═══ PROJECT MEMORY (SWARM.md) ═══\n{content}"

    builtin  = ", ".join(AGENTS.keys())
    custom   = ", ".join(CUSTOM_AGENTS.keys()) or "none yet"
    n_tools  = len(TOOL_MAP)
    today    = datetime.now().strftime("%Y-%m-%d")

    return f"""You are SwarmCoder, an omnipotent autonomous coding agent.
Tools: {n_tools} | Built-in agents: {builtin} | Custom agents: {custom}
Orchestrator: {ORCHESTRATOR_MODEL} | Sub-agents: {SUB_AGENT_MODEL} | Date: {today}{project_ctx}

═══ CORE WORKFLOW ═══
1. PROJECT START → call analyze_project() first to understand the codebase.
2. EXPLORATION → grep, find_files, repo_map, read_file to understand details.
3. PLANNING → save_note("task_plan", "...") to record your numbered step plan.
4. MULTI-FILE TASKS → ALWAYS use two-pass architect pattern:
   a. plan_project(task) — architect designs complete file structure + interfaces
   b. implement_plan()   — editor implements file by file in order
   NEVER write multi-file code without planning first.
5. EXECUTION → work step by step, using tools to read/write/run/test.
6. EDITING → ALWAYS prefer edit_file (surgical) over write_file (full rewrite).
7. VERIFICATION → after writing code, run it with run_python or run_shell.
8. GIT → commit progress at logical milestones with git_commit.

═══ AGENT SYSTEM ═══
DYNAMIC AGENT FACTORY:
- create_agent(name, role, skills, domain) → LLM generates a powerful expert system prompt
- invoke_agent(name, task) → invoke any built-in or custom agent
- invoke_agents_parallel([{{agent, task}}, ...]) → run multiple agents simultaneously
- agent_roundtable([agents], task, rounds) → agents see each other's output and respond

WHEN USER ASKS TO CREATE A SPECIALIZED AGENT: call create_agent() immediately.

═══ TOOL CALL FORMAT ═══
CRITICAL: wrap every tool call in <tool_call> tags. "name" = TOOL name, not agent name.

Analyze a project:
<tool_call>
{{"name": "analyze_project", "arguments": {{"directory": "/path/to/project"}}}}
</tool_call>

Create a custom agent:
<tool_call>
{{"name": "create_agent", "arguments": {{"name": "ios_llm_engineer", "role": "iOS engineer expert in on-device LLMs", "skills": ["CoreML", "MLX", "Metal"], "domain": "iOS mobile"}}}}
</tool_call>

Parallel agents:
<tool_call>
{{"name": "invoke_agents_parallel", "arguments": {{"tasks": [{{"agent": "implementer", "task": "write the API"}}, {{"agent": "tester", "task": "write the tests"}}]}}}}
</tool_call>

Roundtable:
<tool_call>
{{"name": "agent_roundtable", "arguments": {{"agents": ["ios_llm_engineer", "ux_funnel_designer"], "task": "design the onboarding flow", "rounds": 2}}}}
</tool_call>

Multi-file project (ALWAYS use architect pattern):
<tool_call>
{{"name": "plan_project", "arguments": {{"task": "create a REST API with auth, models, and tests", "directory": "./my_api"}}}}
</tool_call>
Then after reviewing the plan:
<tool_call>
{{"name": "implement_plan", "arguments": {{}}}}
</tool_call>

Chain as many tool calls as needed. Give final response as plain text when done.

═══ EXECUTOR MANDATE ═══
{EXECUTOR_MANDATE}

═══ PRINCIPLES ═══
- Read before writing. Minimal edits. Verify after changes. Parallel when independent.
- NEVER output code as text. ALWAYS write_file() + run_shell() to verify. You are an EXECUTOR."""

# ═══════════════════════════════════════════════════════════════════════════
# AGENT TURN
# ═══════════════════════════════════════════════════════════════════════════

def agent_turn(history: list, cwd: str = ".") -> str:
    messages = [{"role": "system", "content": _build_system_prompt(cwd)}] + history

    for iteration in range(MAX_ITERATIONS):
        resp = client.chat.completions.create(
            model=ORCHESTRATOR_MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.3,
            max_tokens=4096,
        )
        msg = resp.choices[0].message
        content = msg.content or ""

        # ── Layer 1: Native structured tool_calls ─────────────────────────
        if msg.tool_calls:
            messages.append({
                "role": "assistant", "content": content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                fn = tc.function.name
                try: args = json.loads(tc.function.arguments)
                except: args = {}
                print(c(DIM, f"  [tool] {fn}({_fmt_args(args)})"), flush=True)
                result = dispatch(fn, args)
                messages.append({"role":"tool","tool_call_id":tc.id,"content":result[:MAX_SHELL_CHARS]})
            continue

        # ── Layer 2: XML tag parsing ──────────────────────────────────────
        parsed_calls = _parse_tool_from_text(content)
        if parsed_calls:
            fake_tool_calls = []
            for pc in parsed_calls:
                fid = f"call_{pc['name']}_{iteration}"
                fake_tool_calls.append({
                    "id": fid, "type": "function",
                    "function": {"name": pc["name"], "arguments": json.dumps(pc["arguments"])}
                })
            messages.append({"role": "assistant", "content": "", "tool_calls": fake_tool_calls})
            for pc, ftc in zip(parsed_calls, fake_tool_calls):
                print(c(DIM, f"  [tool²] {pc['name']}({_fmt_args(pc['arguments'])})"), flush=True)
                result = dispatch(pc["name"], pc["arguments"])
                messages.append({"role":"tool","tool_call_id":ftc["id"],"content":result[:MAX_SHELL_CHARS]})
            continue

        # ── Layer 3: Plain text reply ─────────────────────────────────────
        # If the model outputted code in markdown instead of using write_file,
        # auto-extract and write those files, then continue the loop.
        if content:
            written = _extract_code_blocks_and_write(content, cwd)
            if written:
                file_list = "\n".join(f"  - {p}" for p in written)
                print(c(YELLOW, f"  ⚡ auto-wrote {len(written)} file(s) from response"), flush=True)
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Files auto-extracted and written to disk:\n{file_list}\n\n"
                        "Now run them to verify they work. Use run_shell or run_python. "
                        "If tests pass, confirm completion. If errors, fix with edit_file and re-run."
                    )
                })
                continue

        # ── Layer 4: instructor structured extraction ──────────────────────
        if content and len(content) > 50:  # only try if there's substantial content
            instructor_calls = _instructor_extract_tool(content)
            if instructor_calls:
                print(c(DIM, f"  [tool⁴] instructor extracted: {instructor_calls[0]['name']}"), flush=True)
                fid = f"call_{instructor_calls[0]['name']}_{iteration}_instr"
                fake_tool_calls = [
                    {
                        "id": fid, "type": "function",
                        "function": {
                            "name": instructor_calls[0]["name"],
                            "arguments": json.dumps(instructor_calls[0]["arguments"]),
                        },
                    }
                ]
                messages.append({"role": "assistant", "content": "", "tool_calls": fake_tool_calls})
                result = dispatch(instructor_calls[0]["name"], instructor_calls[0]["arguments"])
                messages.append({"role": "tool", "tool_call_id": fid, "content": result[:MAX_SHELL_CHARS]})
                continue

        return content

    return "ERROR: reached max iterations without final reply."

def _fmt_args(args: dict) -> str:
    return ", ".join(f"{k}={repr(v)[:40]}" for k, v in list(args.items())[:4])

# ═══════════════════════════════════════════════════════════════════════════
# CHAT REPL
# ═══════════════════════════════════════════════════════════════════════════

def _print_banner():
    custom_count = len(CUSTOM_AGENTS)
    print(f"""{BOLD}{CYAN}
╔══════════════════════════════════════════════════╗
║   SwarmCoder v5  —  Omnipotent Coding Agent      ║
╚══════════════════════════════════════════════════╝{R}
Orchestrator : {YELLOW}{ORCHESTRATOR_MODEL}{R}
Sub-agents   : {YELLOW}{SUB_AGENT_MODEL}{R}
Tools        : {GREEN}{len(TOOL_MAP)}{R}  Built-in agents: {GREEN}{len(AGENTS)}{R}  Custom agents: {MAGENTA}{custom_count}{R}
Intelligence : {GREEN}instructor + reflexion + semantic RAG + architect mode{R}
{DIM}
Commands:
  /exit            quit
  /clear           clear conversation history
  /resume          reload last saved history
  /save            save history now
  /agents          list built-in agents
  /myagents        list custom agents
  /tools           list all tools
  /notes           list session notes
  /model NAME      switch orchestrator model
  /submodel NAME   switch sub-agent model
  /swarm           show SWARM.md if present
  /reflect         reflect on last exchange → update SWARM.md
  /index           index codebase into ChromaDB for semantic search{R}
""")


def main():
    _print_banner()
    global ORCHESTRATOR_MODEL, SUB_AGENT_MODEL

    cwd = os.getcwd()
    print(c(DIM, f"cwd: {cwd}"))

    # Check SWARM.md
    swarm_md = Path(cwd) / "SWARM.md"
    if swarm_md.exists():
        print(c(YELLOW, f"  ★ SWARM.md found — project memory loaded"))

    # Check Ollama
    try:
        client.models.list()
    except Exception as e:
        print(c(RED, f"\nERROR: Cannot reach Ollama at {OLLAMA_BASE_URL}"))
        print(c(RED, f"Run: ollama serve\n({e})"))
        sys.exit(1)

    # Load history
    history = _load_history()
    if history:
        print(c(DIM, f"  ↩ {len(history)} messages from last session loaded (/clear to reset)"))
    print()

    while True:
        try:
            raw = input(f"{BOLD}{BLUE}you >{R} ").strip()
        except (EOFError, KeyboardInterrupt):
            _save_history(history)
            print(c(DIM, "\nSession saved. Goodbye."))
            break

        if not raw:
            continue

        # ── Slash commands ─────────────────────────────────────────────────
        if raw == "/exit":
            _save_history(history)
            print(c(DIM, "Session saved. Goodbye."))
            break
        elif raw == "/clear":
            history.clear()
            _save_history(history)
            print(c(YELLOW, "History cleared."))
            continue
        elif raw == "/resume":
            history = _load_history()
            print(c(YELLOW, f"Loaded {len(history)} messages from disk."))
            continue
        elif raw == "/save":
            _save_history(history)
            print(c(YELLOW, f"Saved {len(history)} messages to {HISTORY_FILE}"))
            continue
        elif raw == "/agents":
            print(f"\n{BOLD}Built-in agents:{R}")
            for k, v in AGENTS.items():
                print(f"  {c(CYAN,k)}: {v[:70]}")
            continue
        elif raw == "/myagents":
            print(tool_list_custom_agents())
            continue
        elif raw == "/tools":
            cols = list(TOOL_MAP.keys())
            for i in range(0, len(cols), 4):
                print("  " + "  ".join(c(GREEN, x) for x in cols[i:i+4]))
            continue
        elif raw == "/notes":
            print(tool_list_notes())
            continue
        elif raw == "/swarm":
            if swarm_md.exists():
                print(swarm_md.read_text())
            else:
                print(c(DIM, "No SWARM.md in current directory."))
            continue
        elif raw == "/reflect":
            if len(history) >= 2:
                last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                last_asst = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
                r = _reflect_on_task(last_user, last_asst, cwd)
                print(c(MAGENTA, r or "Nothing to reflect on."))
            else:
                print(c(DIM, "No conversation history to reflect on."))
            continue
        elif raw == "/index":
            print(c(DIM, "  Indexing codebase into ChromaDB..."), flush=True)
            result = tool_index_codebase(cwd)
            print(c(GREEN, f"  {result}"))
            continue
        elif raw.startswith("/model "):
            ORCHESTRATOR_MODEL = raw[7:].strip()
            print(c(YELLOW, f"Orchestrator → {ORCHESTRATOR_MODEL}"))
            continue
        elif raw.startswith("/submodel "):
            SUB_AGENT_MODEL = raw[10:].strip()
            print(c(YELLOW, f"Sub-agents → {SUB_AGENT_MODEL}"))
            continue

        # ── Agent turn ─────────────────────────────────────────────────────
        history.append({"role": "user", "content": raw})
        print(f"\n{c(CYAN,'SwarmCoder')} {c(DIM,'...')}", flush=True)

        try:
            reply = agent_turn(history, cwd=cwd)
        except KeyboardInterrupt:
            print(c(YELLOW, "\n[interrupted]"))
            history.pop()  # remove unanswered user message
            continue
        except Exception as e:
            import traceback; traceback.print_exc()
            reply = f"ERROR: {e}"

        history.append({"role": "assistant", "content": reply})

        # Reflexion: reflect on completed task, update SWARM.md
        reflection = _reflect_on_task(raw, reply, cwd)
        if reflection:
            print(c(DIM, f"  ✦ reflexion saved to SWARM.md"), flush=True)

        # Auto-save every turn
        _save_history(history)

        print(f"\n{BOLD}{CYAN}swarm >{R}\n{reply}")
        print(c(DIM, "─" * 60 + "\n"))


if __name__ == "__main__":
    main()
