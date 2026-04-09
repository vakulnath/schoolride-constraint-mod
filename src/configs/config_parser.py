"""
Configuration parser for constraint-modification project.

Reads config.ini and environment variables to provide project configuration.
All paths are resolved from project root (constraint_modification/).
"""

from pathlib import Path
import configparser
import importlib.util
import os
from dotenv import load_dotenv


def _env_path(key: str, project_root: Path | None = None) -> Path | None:
    """Get a path from environment variable if set.

    If the path is relative, resolve it from project_root.
    If project_root is None, return the path as-is.
    """
    value = os.getenv(key, "").strip()
    if not value:
        return None
    path = Path(value)
    # Resolve relative paths from project root
    if not path.is_absolute() and project_root:
        path = project_root / path
    return path


def _find_package_dir(package: str) -> Path | None:
    """Find installed package directory.

    Try multiple strategies:
    1. importlib.util.find_spec (standard Python discovery)
    2. Check site-packages directly (fallback for venv-installed packages)
    """
    # Strategy 1: importlib.util.find_spec
    try:
        spec = importlib.util.find_spec(package)
    except Exception:
        spec = None

    if spec is not None:
        if spec.submodule_search_locations:
            loc = next(iter(spec.submodule_search_locations), None)
            return Path(loc) if loc else None
        if spec.origin:
            return Path(spec.origin).parent

    # Strategy 2: Check site-packages directly (fallback for venv packages)
    try:
        import site
        for site_dir in site.getsitepackages() + [site.getusersitepackages()]:
            pkg_dir = Path(site_dir) / package
            if pkg_dir.exists() and pkg_dir.is_dir():
                return pkg_dir
    except Exception:
        pass

    return None


def load_config() -> dict:
    """Load and parse configuration from config.ini and environment variables."""

    # Get project root (src/configs/config_parser.py -> constraint_modification/)
    project_root = Path(__file__).parent.parent.parent

    # Load .env file first
    env_file = project_root / ".env"
    load_dotenv(env_file)

    # Read config.ini
    config_file = Path(__file__).parent / "config.ini"
    config = configparser.ConfigParser()
    config.read(config_file)

    # Build result dictionary
    result = {}

    # ── Project paths ────────────────────────────────────────────────────
    result["CONSTRAINTS_FILE"] = Path(config.get("paths", "constraints_file"))
    result["QUERIES_FILE"] = Path(config.get("paths", "queries_file"))
    result["ENV_FILE"] = config.get("paths", "env_file")

    # ── Workspace paths ──────────────────────────────────────────────────
    result["CONTEXT_RUNS_ROOT"] = config.get("paths", "context_runs_root")
    result["INDEX_WORKSPACE"] = Path(config.get("paths", "index_workspace"))
    result["HEXALY_INDEX_WORKSPACE"] = Path(config.get("paths", "hexaly_index_workspace"))

    # ── Output directory structure ───────────────────────────────────────
    result["VANILLA_BASELINE_DIR"] = config.get("paths", "vanilla_baseline_dir")
    result["TEMP_CODE_DIR"] = config.get("paths", "temp_code_dir")
    result["OUTPUT_DIR"] = config.get("paths", "output_dir")
    result["POLICY_DIR"] = config.get("paths", "policy_dir")
    result["LOGS_DIR"] = config.get("paths", "logs_dir")

    # ── Log file names ───────────────────────────────────────────────────
    result["MAIN_AGENT_LOG"] = config.get("paths", "main_agent_log")
    result["AGENT_LOG"] = config.get("paths", "agent_log")
    result["PANEL_JUDGMENT_LOG"] = config.get("paths", "panel_judgment_log")
    result["LLM_EDIT_LOG"] = config.get("paths", "llm_edit_log")
    result["SCHEDULE_DIFF_LOG"] = config.get("paths", "schedule_diff_log")
    result["QUERY_SUMMARY_LOG"] = config.get("paths", "query_summary_log")

    # ── Solver-specific directories ──────────────────────────────────────
    result["INSERTION_SOLVER_DIR"] = config.get("paths", "insertion_solver_dir")
    result["HEXALY_SOLVER_DIR"] = config.get("paths", "hexaly_solver_dir")
    result["INSERTION_TEMP_CODE_DIR"] = config.get("paths", "insertion_temp_code_dir")
    result["HEXALY_TEMP_CODE_DIR"] = config.get("paths", "hexaly_temp_code_dir")

    # ── Indexing files ───────────────────────────────────────────────────
    insertion_files = config.get("indexing", "insertion_files").split(",")
    result["INDEX_FILES"] = [f.strip() for f in insertion_files]

    hexaly_files = config.get("indexing", "hexaly_files").split(",")
    result["HEXALY_INDEX_FILES"] = [f.strip() for f in hexaly_files]

    # ── Model configuration ───────────────────────────────────────────────
    result["DEFAULT_MODEL"] = config.get("models", "default_model")

    nversion_raw = config.get("models", "nversion_models").split(",")
    result["NVERSION_MODELS"] = [m.strip() for m in nversion_raw]

    labels_raw = config.get("models", "nversion_labels").split(",")
    result["NVERSION_LABELS"] = [label.strip() for label in labels_raw]

    fallback_raw = config.get("models", "fallback_chain").split(",")
    result["FALLBACK_CHAIN"] = {}
    for pair in fallback_raw:
        pair = pair.strip()
        if ":" in pair:
            src, dst = pair.split(":", 1)
            result["FALLBACK_CHAIN"][src.strip()] = dst.strip()

    # ── External package paths (with env var overrides) ──────────────────
    # SOLVERS configuration
    result["SOLVERS_ROOT"] = _env_path("SOLVERS_ROOT", project_root)
    result["SOLVERS_SRC"] = _env_path("SOLVERS_SRC", project_root)
    result["SOLVERS_POLICY_ROOT"] = _env_path("SOLVERS_POLICY_ROOT", project_root)

    if result["SOLVERS_SRC"] is None:
        if result["SOLVERS_ROOT"] is not None:
            result["SOLVERS_SRC"] = result["SOLVERS_ROOT"] / "src"
        else:
            policy_dir = _find_package_dir("policy")
            if policy_dir:
                # If policy is in site-packages, use that location directly
                result["SOLVERS_SRC"] = policy_dir.parent if policy_dir else None

    if result["SOLVERS_POLICY_ROOT"] is None:
        if result["SOLVERS_SRC"] is not None and (result["SOLVERS_SRC"] / "policy").exists():
            result["SOLVERS_POLICY_ROOT"] = result["SOLVERS_SRC"] / "policy"
        elif (policy_dir := _find_package_dir("policy")):
            # If policy package is installed directly, use that
            result["SOLVERS_POLICY_ROOT"] = policy_dir

    # Derive SOLVERS_ROOT from SOLVERS_SRC if not set (for local/editable installs)
    if result["SOLVERS_ROOT"] is None and result["SOLVERS_SRC"] is not None:
        result["SOLVERS_ROOT"] = result["SOLVERS_SRC"].parent if result["SOLVERS_SRC"].name == "src" else result["SOLVERS_SRC"]

    # SIMULATOR configuration
    result["SIM_ROOT"] = _env_path("SIM_ROOT", project_root)
    result["SIM_SRC"] = _env_path("SIM_SRC", project_root)

    if result["SIM_SRC"] is None:
        if result["SIM_ROOT"] is not None:
            result["SIM_SRC"] = result["SIM_ROOT"] / "src"
        else:
            toy_dir = _find_package_dir("toy_generator")
            if toy_dir:
                # If toy_generator is in site-packages, use that location directly
                result["SIM_SRC"] = toy_dir.parent if toy_dir else None

    if result["SIM_ROOT"] is None and result["SIM_SRC"] is not None:
        result["SIM_ROOT"] = result["SIM_SRC"].parent if result["SIM_SRC"].name == "src" else result["SIM_SRC"]

    # TOY_TEST_PATH
    result["TOY_TEST_PATH"] = _env_path("TOY_TEST_PATH", project_root)
    if result["TOY_TEST_PATH"] is None and result["SIM_SRC"] is not None:
        result["TOY_TEST_PATH"] = result["SIM_SRC"] / "toy_generator" / "test.py"

    # SBR_VENV_PY (school bus routing venv python)
    result["SBR_VENV_PY"] = _env_path("SBR_VENV_PY", project_root)
    if result["SBR_VENV_PY"] is None and result["SIM_ROOT"] is not None:
        unix_py = result["SIM_ROOT"] / "venv" / "bin" / "python"
        win_py = result["SIM_ROOT"] / "venv" / "Scripts" / "python.exe"
        if unix_py.exists():
            result["SBR_VENV_PY"] = unix_py
        elif win_py.exists():
            result["SBR_VENV_PY"] = win_py

    # ── Validation ───────────────────────────────────────────────────────
    if result["SOLVERS_ROOT"] is None or result["SOLVERS_POLICY_ROOT"] is None or result["SIM_ROOT"] is None:
        raise RuntimeError(
            "school-solvers/school-simulator paths are not configured. "
            "Set SOLVERS_SRC, SOLVERS_POLICY_ROOT, SIM_ROOT/SIM_SRC in environment, "
            "or ensure school-solvers and school-simulator are installed."
        )

    return result


# Load configuration once at module import time
_config = load_config()

# Export all configuration values
# Project paths
CONSTRAINTS_FILE = _config["CONSTRAINTS_FILE"]
QUERIES_FILE = _config["QUERIES_FILE"]
ENV_FILE = _config["ENV_FILE"]

# Workspace paths
CONTEXT_RUNS_ROOT = _config["CONTEXT_RUNS_ROOT"]
INDEX_WORKSPACE = _config["INDEX_WORKSPACE"]
HEXALY_INDEX_WORKSPACE = _config["HEXALY_INDEX_WORKSPACE"]

# Output directory structure
VANILLA_BASELINE_DIR = _config["VANILLA_BASELINE_DIR"]
TEMP_CODE_DIR = _config["TEMP_CODE_DIR"]
OUTPUT_DIR = _config["OUTPUT_DIR"]
POLICY_DIR = _config["POLICY_DIR"]
LOGS_DIR = _config["LOGS_DIR"]

# Log file names
MAIN_AGENT_LOG = _config["MAIN_AGENT_LOG"]
AGENT_LOG = _config["AGENT_LOG"]
PANEL_JUDGMENT_LOG = _config["PANEL_JUDGMENT_LOG"]
LLM_EDIT_LOG = _config["LLM_EDIT_LOG"]
SCHEDULE_DIFF_LOG = _config["SCHEDULE_DIFF_LOG"]
QUERY_SUMMARY_LOG = _config["QUERY_SUMMARY_LOG"]

# Indexing files
INDEX_FILES = _config["INDEX_FILES"]
HEXALY_INDEX_FILES = _config["HEXALY_INDEX_FILES"]

# Solver-specific directories
INSERTION_SOLVER_DIR = _config["INSERTION_SOLVER_DIR"]
HEXALY_SOLVER_DIR = _config["HEXALY_SOLVER_DIR"]
INSERTION_TEMP_CODE_DIR = _config["INSERTION_TEMP_CODE_DIR"]
HEXALY_TEMP_CODE_DIR = _config["HEXALY_TEMP_CODE_DIR"]

# Model configuration
DEFAULT_MODEL = _config["DEFAULT_MODEL"]
NVERSION_MODELS = _config["NVERSION_MODELS"]
NVERSION_LABELS = _config["NVERSION_LABELS"]
FALLBACK_CHAIN = _config["FALLBACK_CHAIN"]

# External package paths
SOLVERS_ROOT = _config["SOLVERS_ROOT"]
SOLVERS_SRC = _config["SOLVERS_SRC"]
SOLVERS_POLICY_ROOT = _config["SOLVERS_POLICY_ROOT"]
SIM_ROOT = _config["SIM_ROOT"]
SIM_SRC = _config["SIM_SRC"]
SBR_VENV_PY = _config["SBR_VENV_PY"]
TOY_TEST_PATH = _config["TOY_TEST_PATH"]
