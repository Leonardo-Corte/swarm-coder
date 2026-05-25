"""
SwarmCoder - Coding agent swarm orchestrator.
Runs multiple specialized agents in parallel via Ollama.
"""

import os
import sys
import json
import asyncio
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"
DEFAULT_MODEL = "qwen2.5-coder:14b"

client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

# ─── Agent definitions ──────────────────────────────────────────────────────

AGENTS = {
    "architect": {
        "system": (
            "You are a software architect. Given a task, produce a clear, "
            "structured implementation plan: components, interfaces, file "
            "structure, and the order of implementation. Be concise and technical."
        ),
        "color": "\033[94m",  # blue
    },
    "implementer": {
        "system": (
            "You are a senior software engineer. Given a task and optional "
            "architecture notes, write complete, production-ready code. "
            "Include only working code, no placeholders. Use best practices."
        ),
        "color": "\033[92m",  # green
    },
    "reviewer": {
        "system": (
            "You are a code reviewer. Given code or a plan, identify bugs, "
            "security issues, performance problems, and missing edge cases. "
            "Be precise and actionable. Output findings as a numbered list."
        ),
        "color": "\033[93m",  # yellow
    },
    "tester": {
        "system": (
            "You are a QA engineer. Given a task and optional code, write "
            "comprehensive tests: unit tests, edge cases, and integration "
            "scenarios. Use pytest. Include setup and teardown where needed."
        ),
        "color": "\033[95m",  # magenta
    },
    "optimizer": {
        "system": (
            "You are a performance engineer. Given code, identify and fix "
            "bottlenecks. Improve time complexity, memory usage, and "
            "readability without changing functionality."
        ),
        "color": "\033[96m",  # cyan
    },
}

RESET = "\033[0m"
BOLD = "\033[1m"


# ─── Core LLM call ──────────────────────────────────────────────────────────

def call_agent(agent_name: str, task: str, context: str = "", model: str = DEFAULT_MODEL) -> dict:
    agent = AGENTS[agent_name]
    messages = [{"role": "system", "content": agent["system"]}]

    user_content = task
    if context:
        user_content = f"Context:\n{context}\n\nTask:\n{task}"

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=4096,
        )
        return {
            "agent": agent_name,
            "output": response.choices[0].message.content,
            "error": None,
        }
    except Exception as e:
        return {"agent": agent_name, "output": None, "error": str(e)}


# ─── Parallel execution ─────────────────────────────────────────────────────

def run_agents_parallel(agents: list[str], task: str, context: str = "", model: str = DEFAULT_MODEL) -> dict:
    results = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as executor:
        futures = {
            executor.submit(call_agent, agent, task, context, model): agent
            for agent in agents
        }
        for future in as_completed(futures):
            result = future.result()
            results[result["agent"]] = result
            agent_color = AGENTS[result["agent"]]["color"]
            status = "done" if not result["error"] else f"ERROR: {result['error']}"
            print(f"  {agent_color}[{result['agent']}]{RESET} {status}")
    return results


# ─── Pipeline modes ─────────────────────────────────────────────────────────

def pipeline_full(task: str, model: str = DEFAULT_MODEL) -> dict:
    """Full pipeline: architect → parallel(implement+test) → review → optimize."""
    print(f"\n{BOLD}=== SwarmCoder Pipeline ==={RESET}")
    print(f"Task: {task[:80]}{'...' if len(task) > 80 else ''}\n")

    # Phase 1: Architecture
    print(f"{BOLD}Phase 1: Architecture{RESET}")
    arch_result = call_agent("architect", task, model=model)
    arch_output = arch_result["output"] or ""
    _print_result(arch_result)

    # Phase 2: Implementation + Tests in parallel
    print(f"\n{BOLD}Phase 2: Implementation + Tests [parallel]{RESET}")
    phase2 = run_agents_parallel(
        ["implementer", "tester"],
        task,
        context=f"Architecture:\n{arch_output}",
        model=model,
    )
    impl_output = phase2.get("implementer", {}).get("output") or ""
    test_output = phase2.get("tester", {}).get("output") or ""
    _print_result(phase2.get("implementer", {}))
    _print_result(phase2.get("tester", {}))

    # Phase 3: Review + Optimize in parallel
    print(f"\n{BOLD}Phase 3: Review + Optimize [parallel]{RESET}")
    review_context = f"Implementation:\n{impl_output}\n\nTests:\n{test_output}"
    phase3 = run_agents_parallel(
        ["reviewer", "optimizer"],
        task,
        context=review_context,
        model=model,
    )
    _print_result(phase3.get("reviewer", {}))
    _print_result(phase3.get("optimizer", {}))

    return {
        "architecture": arch_output,
        "implementation": impl_output,
        "tests": test_output,
        "review": phase3.get("reviewer", {}).get("output") or "",
        "optimized": phase3.get("optimizer", {}).get("output") or "",
    }


def pipeline_quick(task: str, model: str = DEFAULT_MODEL) -> dict:
    """Quick mode: architect + implementer + reviewer in parallel."""
    print(f"\n{BOLD}=== SwarmCoder Quick ==={RESET}")
    print(f"Task: {task[:80]}{'...' if len(task) > 80 else ''}\n")

    print(f"{BOLD}Running: architect + implementer + reviewer [parallel]{RESET}")
    results = run_agents_parallel(
        ["architect", "implementer", "reviewer"],
        task,
        model=model,
    )
    for r in results.values():
        _print_result(r)
    return {k: v.get("output") or "" for k, v in results.items()}


def pipeline_single(agent: str, task: str, model: str = DEFAULT_MODEL) -> dict:
    """Run a single named agent."""
    if agent not in AGENTS:
        print(f"Unknown agent '{agent}'. Available: {', '.join(AGENTS.keys())}")
        sys.exit(1)
    print(f"\n{BOLD}=== SwarmCoder [{agent}] ==={RESET}")
    result = call_agent(agent, task, model=model)
    _print_result(result)
    return {agent: result.get("output") or ""}


# ─── Output ─────────────────────────────────────────────────────────────────

def _print_result(result: dict):
    if not result:
        return
    agent = result.get("agent", "?")
    color = AGENTS.get(agent, {}).get("color", "")
    if result.get("error"):
        print(f"\n{color}[{agent}]{RESET} ERROR: {result['error']}")
        return
    output = result.get("output") or ""
    print(f"\n{color}{BOLD}[{agent}]{RESET}")
    print(output)
    print(f"{color}{'─' * 60}{RESET}")


def save_output(results: dict, output_file: str):
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{BOLD}Saved to:{RESET} {output_file}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SwarmCoder - Parallel coding agent swarm (Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orchestrator.py "Build a REST API for user auth with JWT"
  python orchestrator.py --mode quick "Implement a LRU cache in Python"
  python orchestrator.py --agent implementer "Write a binary search function"
  python orchestrator.py --model qwen-agent:latest "Refactor this for async"
  python orchestrator.py --list-agents
        """,
    )
    parser.add_argument("task", nargs="?", help="Coding task description")
    parser.add_argument(
        "--mode",
        choices=["full", "quick"],
        default="full",
        help="Pipeline mode (default: full)",
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENTS.keys()),
        help="Run a single specific agent",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--output",
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List available agents",
    )

    args = parser.parse_args()

    if args.list_agents:
        print(f"\n{BOLD}Available agents:{RESET}")
        for name, cfg in AGENTS.items():
            print(f"  {cfg['color']}{name}{RESET}")
        return

    if not args.task:
        parser.print_help()
        sys.exit(1)

    # Check Ollama is running
    try:
        models = client.models.list()
    except Exception as e:
        print(f"\nERROR: Cannot connect to Ollama at {OLLAMA_BASE_URL}")
        print(f"Make sure Ollama is running: ollama serve")
        print(f"Details: {e}")
        sys.exit(1)

    if args.agent:
        results = pipeline_single(args.agent, args.task, model=args.model)
    elif args.mode == "quick":
        results = pipeline_quick(args.task, model=args.model)
    else:
        results = pipeline_full(args.task, model=args.model)

    if args.output:
        save_output(results, args.output)


if __name__ == "__main__":
    main()
