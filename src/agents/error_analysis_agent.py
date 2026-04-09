"""
Error Analysis Agent (EAA) — diagnoses test failures and produces a corrected edit.

Has tools (search_code, read_function, find_symbol, find_referencing_symbols)
so it can actively search for the function that controls the failure mode, not just
the function that was originally edited.

The EAA reads the test error, identifies the root cause, finds the controlling
function via tool calls, and outputs an XML-formatted corrected edit.
"""

from __future__ import annotations

from typing import List

from google.adk import Agent
from google.genai.types import GenerateContentConfig

_LLM_CONFIG = GenerateContentConfig(temperature=0)

_INSTRUCTION = """\
A constraint modification was applied but the toy test failed. Your job:

1. Read the error to understand the FAILURE MODE (not the constraint).
   Example: 'stops on different buses' → the ASSIGNMENT LOOP put them on different buses.
   Example: 'wrong order' → the ASSIGNMENT LOOP inserted them independently.
   Example: 'capacity exceeded' → the LOAD CHECK threshold is wrong.

2. CRITICAL: Find the function that CONTROLS the failure mode — the one that
   makes the DECISION, not the one that scores or evaluates a single option.
   - If the error is about WHICH bus a stop lands on → find the function with the
     loop that iterates over stops and builds the CANDIDATE LIST of buses.
     Filtering the candidate list BEFORE evaluation is the right fix.
   - Do NOT edit a leaf scoring/evaluation function. The fix belongs in the
     CALLER that assembles candidates or assigns stops in a loop.
   - Use find_referencing_symbols to trace from the evaluation function UP to the loop.

3. Read the controlling function. Look for where candidate lists are built,
   where stops are iterated, and where assignment decisions are made.

4. Output an edit that fixes the failure mode:
   <relative_path>file.py</relative_path>
   <old_text>exact lines from the file</old_text>
   <new_text>corrected lines</new_text>
   <explanation>why this fixes the failure mode</explanation>

IMPORTANT: If the error says items are on DIFFERENT groups when they should be
on the SAME group, the fix must go in the function that DECIDES which group
each item is assigned to — the main assignment loop that builds candidate lists.
Do NOT fix this in scoring, evaluation, or constraint-checking functions.
Use find_referencing_symbols to trace UP to the assignment loop and filter the
candidate list so the second item can only be assigned to the group containing the first.
Copy variable names EXACTLY from the source — do not guess or pluralize.

You have tools: search_code, read_function, find_symbol, find_referencing_symbols.
Use them to find the code responsible for the failure.

{constraints}
{schema}
"""


def create_error_analysis_agent(
    model,
    tools: List,
    constraint_content: str,
    schema_content: str,
    static_instruction: str = "",
) -> Agent:
    """Create the Error Analysis Agent (EAA).

    Args:
        model: ADK model identifier or LiteLlm instance.
        tools: List of callable tool functions (same tools as Main Agent).
        constraint_content: Contents of constraints.txt for the solver.
        schema_content: JSON schema describing the solver API.
        static_instruction: Optional stable context to cache (e.g. function sources).
    """
    instruction = _INSTRUCTION.format(
        constraints=f"<constraints>\n{constraint_content}\n</constraints>",
        schema=f"<schema>\n{schema_content}\n</schema>",
    )
    return Agent(
        name="eaa",
        model=model,
        description="Diagnoses test failures by searching for the code that caused the failure.",
        static_instruction=static_instruction or None,
        instruction=instruction,
        tools=tools,
        generate_content_config=_LLM_CONFIG,
    )
