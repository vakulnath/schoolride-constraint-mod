"""
Multi-agent pipeline for constraint modification.

Five agents, each in its own file, all defined using Google ADK:
  MainAgent  — explores codebase with tools to find injection points
  GA         — generates minimal code edits (no tools, single-turn)
  JA         — validates edits pre-execution (no tools, single-turn)
  EAA        — diagnoses test failures with tools, produces corrected edit
  RA         — revises rejected/failed edits with tools
"""

from src.agents.main_agent import create_main_agent
from src.agents.generation_agent import create_generation_agent
from src.agents.judgment_agent import create_judgment_agent
from src.agents.error_analysis_agent import create_error_analysis_agent
from src.agents.revision_agent import create_revision_agent

__all__ = [
    "create_main_agent",
    "create_generation_agent",
    "create_judgment_agent",
    "create_error_analysis_agent",
    "create_revision_agent",
]
