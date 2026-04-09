"""
Revision Agent (RA) — corrects a rejected or failed edit based on feedback.

Has tools (search_code, read_function, find_symbol, get_callers, get_call_details)
so it can search for alternative injection points if the feedback suggests the
original approach was wrong.

Called after:
  - JA rejects the GA's edit (feedback_source="judgment")
  - EAA diagnoses a runtime error (feedback_source="error_analysis")
  - Schedule diff detects a constraint violation (feedback_source="constraint_violation")
"""

from __future__ import annotations

from typing import List

from google.adk import Agent
from google.genai.types import GenerateContentConfig

_LLM_CONFIG = GenerateContentConfig(temperature=0)

_INSTRUCTION = """\
You generate corrected code edits based on feedback from a reviewer or error diagnosis.

You will receive:
  - The original constraint to implement
  - The previous edit diff that was rejected or failed
  - Structured feedback explaining what went wrong

Your job:
1. Read the feedback carefully — understand exactly what was wrong.
2. Do NOT repeat the same mistake. If the feedback says wrong variable, wrong
   function, or wrong approach — use your tools to find the right one.
3. Use search_code, read_function, get_callers to verify your fix before outputting.
4. Output the corrected edit using XML tags:
   <relative_path>file.py</relative_path>
   <old_text>exact lines to replace</old_text>
   <new_text>replacement with fix applied</new_text>
   <explanation>what was changed and why this addresses the feedback</explanation>

Rules:
- old_text must be EXACT verbatim lines from the CURRENT file (it may differ from the
  original if a previous edit was already applied)
- Follow api_examples patterns from the feedback context
- Use .get() for dict access
- Hardcode constraint values directly

You have tools: search_code, read_function, find_symbol, get_callers, get_call_details.
Use them if the feedback suggests a different function or approach is needed.

{constraints}
{schema}
"""


def create_revision_agent(
    model,
    tools: List,
    constraint_content: str,
    schema_content: str,
) -> Agent:
    """Create the Revision Agent (RA).

    Args:
        model: ADK model identifier or LiteLlm instance.
        tools: List of callable tool functions (same tools as Main Agent).
        constraint_content: Contents of constraints.txt for the solver.
        schema_content: JSON schema describing the solver API.
    """
    instruction = _INSTRUCTION.format(
        constraints=f"<constraints>\n{constraint_content}\n</constraints>",
        schema=f"<schema>\n{schema_content}\n</schema>",
    )
    return Agent(
        name="ra",
        model=model,
        description="Corrects rejected or failed edits based on reviewer feedback.",
        instruction=instruction,
        tools=tools,
        generate_content_config=_LLM_CONFIG,
    )
