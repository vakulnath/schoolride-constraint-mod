"""
Multi-agent pipeline for constraint modification.

Four specialized agents collaborate (AFL-style):
  GA  — Generation Agent:      produces code edits via tool-calling
  JA  — Judgment Agent:        validates edits before execution (tool-free)
  EAA — Error Analysis Agent:  diagnoses runtime errors (tool-free)
  RA  — Revision Agent:        corrects edits using JA / EAA feedback
"""

from agents.generation_agent import run_generation_agent
from agents.judgment_agent import run_judgment_agent
from agents.error_analysis_agent import run_error_analysis_agent
from agents.revision_agent import run_revision_agent

__all__ = [
    "run_generation_agent",
    "run_judgment_agent",
    "run_error_analysis_agent",
    "run_revision_agent",
]
