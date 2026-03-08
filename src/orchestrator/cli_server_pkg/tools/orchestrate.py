"""Orchestrate tool: multi-model deliberation pipeline."""

import os
import re
import time
from typing import TYPE_CHECKING

from .. import config
from ..helpers.prompts import build_prompt
from ..session.runners import run_claude, run_gemini
from ..session.state import reset_sessions, track_call

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

# Patterns that indicate unresolved questions in model output
_OPEN_QUESTION_PATTERNS = re.compile(
    r"(?i)"
    r"(##\s*(open\s+questions|unresolved|outstanding|things\s+to\s+(investigate|research|clarify)))"
    r"|(\*\*open\s+questions\*\*)"
    r"|(needs?\s+(further|more)\s+(research|investigation|clarification))"
    r"|(\?\s*$)",
    re.MULTILINE,
)


def _has_open_questions(text: str) -> bool:
    """Detect whether model output contains unresolved questions."""
    matches = list(_OPEN_QUESTION_PATTERNS.finditer(text))
    structural_matches = [m for m in matches if not m.group().strip().endswith("?")]
    if structural_matches:
        return True
    tail = text[int(len(text) * 0.7) :]
    question_lines = [line for line in tail.splitlines() if line.strip().endswith("?")]
    return len(question_lines) >= 3


async def _orchestrate_pipeline(
    goal: str,
    context: str,
    start_time: float,
    max_rounds: int,
    deliberation_log: list[dict[str, str]],
    _check_timeout,
    ctx: "Context | None" = None,
) -> str:
    """Inner pipeline for orchestrate — separated so timeout can catch and return partial results."""
    status_file = os.path.join(config.PROJECT_ROOT, ".orchestrate-status")

    async def _progress(step: int, total: int, message: str):
        if ctx:
            await ctx.report_progress(step, total, message)
            await ctx.info(message)
        elapsed = int(time.time() - start_time)
        try:
            with open(status_file, "w") as f:
                f.write(f"{message}\nElapsed: {elapsed}s\n")
        except OSError:
            pass

    await _progress(1, 4, "Step 1/4: Gemini is researching the problem domain...")
    gemini_research_prompt = build_prompt(
        "You are a senior technical researcher. A complex task needs your analysis before "
        "implementation begins. Research the problem domain thoroughly.\n",
        f"## Task\n\n{goal}",
        f"## Context\n\n{context}" if context else "",
        "## Your job\n\n"
        "1. **Assess** the task — what's being asked, what's complex about it, what could go wrong\n"
        "2. **Research** relevant patterns, best practices, libraries, and prior art\n"
        "3. **Identify risks** — edge cases, performance concerns, security considerations\n"
        "4. **Surface open questions** — things the architect will need to resolve against the codebase\n\n"
        "Be thorough. The architect will use your findings to design the implementation.",
    )
    # Use longer per-call timeout for orchestrate (each step needs more time)
    step_timeout = config.CLI_TIMEOUT * 2  # 10 min per call vs default 5 min

    track_call("gemini", "orchestrate-research")
    gemini_research = await run_gemini(gemini_research_prompt, timeout=step_timeout)
    deliberation_log.append({"role": "gemini-research", "content": gemini_research})

    _check_timeout()
    await _progress(2, 4, "Step 2/4: Claude is analyzing the codebase and designing architecture...")

    claude_arch_prompt = build_prompt(
        "You are a senior software architect. A researcher has analyzed a complex task. "
        "Now you need to design the implementation plan based on their findings AND "
        "your own analysis of the codebase.\n",
        f"## Task\n\n{goal}",
        f"## Context\n\n{context}" if context else "",
        f"## Research Findings\n\n{gemini_research}",
        "## Your job\n\n"
        "1. **Read the codebase** — understand the current structure, patterns, and conventions\n"
        "2. **Design the architecture** — specific files, functions, data flow, and changes needed\n"
        "3. **Create an implementation plan** — ordered steps with file paths and descriptions\n"
        "4. **Flag open questions** — if you need more research on anything, list your questions "
        "clearly in a section called `## OPEN QUESTIONS` (one question per bullet). "
        "If you have no open questions, do NOT include that section.\n\n"
        "Be specific. Reference actual file paths and function names from the codebase.",
    )
    track_call("claude", "orchestrate-architect")
    claude_architecture = await run_claude(claude_arch_prompt, timeout=step_timeout)
    deliberation_log.append({"role": "claude-architect", "content": claude_architecture})

    for round_num in range(max_rounds):
        _check_timeout()

        if not _has_open_questions(claude_architecture):
            await _progress(3, 4, "Step 3/4: No open questions — skipping to synthesis...")
            break

        await _progress(3, 4, f"Step 3/4: Deliberation round {round_num + 1} — resolving open questions...")

        gemini_answer_prompt = build_prompt(
            "You are a senior technical researcher. An architect has designed a plan but has "
            "open questions that need research. Answer each question thoroughly.\n",
            f"## Original Task\n\n{goal}",
            f"## Architect's Plan and Questions\n\n{claude_architecture}",
            "## Your job\n\n"
            "Answer each open question with concrete, actionable research. "
            "If a question requires a recommendation, give one with reasoning.",
        )
        track_call("gemini", "orchestrate-followup")
        gemini_answers = await run_gemini(gemini_answer_prompt, timeout=step_timeout)
        deliberation_log.append({"role": f"gemini-followup-{round_num + 1}", "content": gemini_answers})

        claude_refine_prompt = build_prompt(
            "The researcher has answered your open questions. Refine your architecture "
            "and implementation plan based on these answers.\n",
            f"## Research Answers\n\n{gemini_answers}",
            "## Your job\n\n"
            "1. Incorporate the research answers into your plan\n"
            "2. Update the architecture and implementation steps as needed\n"
            "3. If you still have unresolved questions, list them in `## OPEN QUESTIONS`\n"
            "4. If all questions are resolved, do NOT include an OPEN QUESTIONS section\n\n"
            "Output your complete, updated architecture and implementation plan.",
        )
        track_call("claude", "orchestrate-refine")
        claude_architecture = await run_claude(claude_refine_prompt, timeout=step_timeout)
        deliberation_log.append({"role": f"claude-refine-{round_num + 1}", "content": claude_architecture})

    _check_timeout()
    await _progress(4, 4, "Step 4/4: Claude is synthesizing TASK.md...")

    deliberation_summary = "\n\n---\n\n".join(
        f"### {entry['role']}\n\n{entry['content']}" for entry in deliberation_log
    )

    task_md_prompt = build_prompt(
        "You have completed a multi-model deliberation on a complex task. "
        "Now synthesize everything into a single, actionable TASK.md document.\n",
        f"## Original Task\n\n{goal}",
        f"## Context\n\n{context}" if context else "",
        f"## Deliberation Log\n\n{deliberation_summary}",
        "## Your job\n\n"
        "Write a complete TASK.md file with these sections:\n\n"
        "```\n"
        "# Task: <concise title>\n\n"
        "## Assessment\n"
        "What the task is, why it's complex, and what success looks like.\n\n"
        "## Research\n"
        "Key findings from the research phase — patterns, best practices,\n"
        "libraries, and prior art. Include only what's relevant to implementation.\n\n"
        "## Architecture\n"
        "The designed solution — data flow, component breakdown, file structure,\n"
        "and key design decisions with rationale.\n\n"
        "## Open Questions (Resolved)\n"
        "Questions that came up during deliberation and how they were resolved.\n"
        "Skip this section if there were no open questions.\n\n"
        "## Implementation Plan\n"
        "Ordered, step-by-step plan with specific file paths, function names,\n"
        "and descriptions of changes. Each step should be independently actionable.\n"
        "```\n\n"
        "Output ONLY the markdown content of TASK.md — no code fences wrapping the whole thing, "
        "no preamble, no commentary. Just the document content.",
    )
    track_call("claude", "orchestrate-synthesize")
    task_md_content = await run_claude(task_md_prompt, timeout=step_timeout)

    task_md_path = os.path.join(config.PROJECT_ROOT, "TASK.md")
    try:
        with open(task_md_path, "w") as f:
            f.write(task_md_content)
    except OSError as e:
        return (
            f"Deliberation complete but failed to write TASK.md: {e}\n\n"
            f"---\n\n{task_md_content}"
        )

    try:
        os.remove(status_file)
    except OSError:
        pass

    rounds_used = len([e for e in deliberation_log if e["role"].startswith("gemini-followup")])
    total_calls = len(deliberation_log) + 1

    elapsed = int(time.time() - start_time)

    return (
        f"## Orchestration Complete\n\n"
        f"**Task:** {goal}\n"
        f"**Deliberation rounds:** {rounds_used} follow-up{'s' if rounds_used != 1 else ''}\n"
        f"**Total model calls:** {total_calls} ({total_calls - rounds_used * 2 - 1} Claude, "
        f"{rounds_used + 1} Gemini)\n"
        f"**Duration:** {elapsed}s\n"
        f"**Output:** `TASK.md` written to project root\n\n"
        f"The TASK.md contains: Assessment, Research, Architecture, "
        f"{'Open Questions (Resolved), ' if rounds_used > 0 else ''}Implementation Plan.\n\n"
        f"**Review TASK.md before implementing.** Do NOT implement automatically.\n"
        f"Once reviewed and approved, run: `implement the plan in TASK.md`"
    )


def register_orchestrate(mcp):
    """Register the orchestrate tool on the given FastMCP instance."""
    from mcp.server.fastmcp import Context

    @mcp.tool()
    async def orchestrate(goal: str, context: str = "", ctx: Context | None = None) -> str:
        """Multi-model deliberation pipeline. Gemini and Claude converse back-and-forth
        to assess a complex task, then produce a TASK.md with a complete plan.

        IMPORTANT: This tool produces a plan for review only. It does NOT implement
        anything. The user must review TASK.md and explicitly approve before implementation.

        Use when you have a complex, ambiguous, or high-stakes task that benefits from
        both deep research AND codebase-aware architecture before implementation.

        The pipeline:
        1. Gemini researches the problem domain, patterns, and trade-offs
        2. Claude analyzes the codebase and designs architecture based on Gemini's findings
        3. Claude flags open questions → Gemini researches answers → loop until resolved
        4. Final TASK.md is produced with: Assessment, Research, Architecture, Open Questions, Implementation Plan

        After orchestration completes, present the TASK.md to the user for review.
        Do NOT proceed to implementation unless the user explicitly approves.

        Args:
            goal: The complex task or problem to orchestrate.
            context: Optional context — constraints, prior decisions, relevant files.
        """
        reset_sessions()

        start_time = time.time()
        max_rounds = 3
        deliberation_log: list[dict[str, str]] = []

        def _check_timeout():
            elapsed = time.time() - start_time
            if elapsed > config.ORCHESTRATE_TIMEOUT:
                raise TimeoutError(
                    f"Orchestration timed out after {int(elapsed)}s "
                    f"(limit: {config.ORCHESTRATE_TIMEOUT}s)."
                )

        try:
            return await _orchestrate_pipeline(
                goal, context, start_time, max_rounds, deliberation_log, _check_timeout, ctx
            )
        except TimeoutError as e:
            partial = "\n\n---\n\n".join(
                f"### {entry['role']}\n\n{entry['content']}" for entry in deliberation_log
            )
            return (
                f"## Orchestration Timed Out\n\n"
                f"{e}\n\n"
                f"**Partial deliberation ({len(deliberation_log)} steps completed):**\n\n"
                f"{partial}"
            )
