"""
Verifier Agent (JA) — validates a proposed edit before execution.

Tool-free, single-turn. Reviews the full edit text against the constraint and
the original function source. Returns a structured verdict with specific,
actionable feedback so the Generator can fix exactly what's wrong.
"""

from __future__ import annotations

from google.adk import Agent
from google.genai.types import GenerateContentConfig

_LLM_CONFIG = GenerateContentConfig(temperature=0)

_INSTRUCTION = """\
You are a strict code reviewer for school-bus routing policy edits.

You will receive several proposed edits (numbered 1, 2, 3, ...) that each attempt
to implement the same constraint. Your job is to evaluate all of them and pick the
BEST one that is correct and safe to execute.

Evaluate each edit on these criteria:
1. CORRECTNESS — Does the edit fully implement the stated constraint?
   - For must_assign ("stop X on bus Y"): stop is forced onto bus Y AND blocked from
     all other buses. Use null-safe access (e.g. `(data.bus_current_arrivals or {}).get(bus_id)`)
     to check bus Y exists before blocking all others — so the constraint is a no-op
     when bus Y is missing from the scenario.
     EXCEPTION: if the constraint text contains "instead of bus Z" (e.g. "on bus Y instead of bus Z"):
       - The edit should ONLY block bus Z (do NOT force onto bus Y, do NOT block other buses).
       - No null-safe check needed — you are just blocking bus Z.
       - Example: "stop 20006 on bus 2 instead of bus 3" → ONLY block bus 3.
       - This exception takes priority. If the edit forces onto bus Y, it FAILS this check.
   - For must_not_assign: stop is blocked from the target bus
   - For ordering: both stops are on the same sequence with enforced order
   - For same_bus/not_same_bus: pairing constraint is actually enforced
2. PLACEMENT   — Is the constraint enforced BEFORE or DURING solving?
   - REJECT any edit that filters or modifies the final solution/output after the
     solver returns. Post-solution filtering does NOT enforce constraints.
   - For optimization-based solvers: constraint registration must happen BEFORE
     the model is finalized/closed. If the edit adds constraint registration after
     a call to a function that finalizes the model, that is WRONG.
   - REJECT any edit that modifies data-preparation or preprocessing functions to
     filter out vehicles/routes. Filtering at the data-prep stage corrupts indices
     used by the model, causing infeasibility. Constraints belong in the model-building
     function (where model.constraint() calls live), not in data setup.
3. SAFETY      — Does it preserve all existing logic? Respects postconditions in the docstring?
4. SYNTAX      — Is the Python syntactically valid? Is indentation consistent?
5. SCOPE       — Are all referenced variables actually in scope at the edit location?
6. MINIMALITY  — Is the change minimal (guard clause / condition, no unnecessary params)?

Your response MUST be exactly two lines:

choice: <N>
jud: None

where N is the number of the best valid edit (1, 2, 3, ...). If NO edit is correct, output:

choice: None
jud: <specific description of what is wrong across all edits and exactly what code is needed>

No preamble. No explanation before the verdict. Two lines only.
"""


def create_judgment_agent(model, static_instruction: str = "") -> Agent:
    """Create the Verifier Agent (JA).

    Args:
        model: ADK model identifier or LiteLlm instance.
        static_instruction: Optional stable context (e.g. function sources) to place
            in static_instruction for caching. Subsequent calls only send the proposed
            fix as the dynamic delta.
    """
    return Agent(
        name="ja",
        model=model,
        description="Validates a proposed code edit before execution.",
        static_instruction=static_instruction or None,
        instruction=_INSTRUCTION,
        generate_content_config=_LLM_CONFIG,
    )
