"""
Toy test runner — runs the simulator's toy test suite against a modified policy directory.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict

from src.configs import config_parser as cfg


def run_toy_test(query_id: int, solver_type: str, temp_code_dir: str) -> Dict[str, Any]:
    """Run toy test with the modified policy code.

    Args:
        query_id: Unique identifier for this query (used to name output dirs).
        solver_type: "insertion" or "hexaly".
        temp_code_dir: Directory containing the modified policy/ subdirectory.

    Returns:
        dict with keys: passed (bool), output (str).
    """
    toy_test = str(cfg.TOY_TEST_PATH) if cfg.TOY_TEST_PATH else None
    if not toy_test or not os.path.exists(toy_test):
        venv_test = os.path.join(".venv", "lib", "python3.13", "site-packages",
                                 "toy_generator", "test.py")
        if os.path.isfile(venv_test):
            toy_test = os.path.abspath(venv_test)
    if not toy_test:
        return {"passed": False, "output": "No toy test found"}

    abs_temp = os.path.realpath(os.path.abspath(temp_code_dir))
    wrapper = (
        "import sys\n"
        f"sys.path.insert(0, {abs_temp!r})\n"
        "for _k in list(sys.modules):\n"
        "    if _k.startswith('policy'):\n"
        "        del sys.modules[_k]\n"
        "import policy\n"
        f"import pytest\nsys.exit(pytest.main(['-x','-s','--tb=long',{toy_test!r}]))"
    )

    env = os.environ.copy()
    env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(env_file):
        with open(env_file) as _ef:
            for _line in _ef:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    env.setdefault(_k.strip(), _v.strip())

    policy_dir = os.path.join(temp_code_dir, "policy")
    env["CONSTRAINT_QUERY_ID"] = str(query_id)
    env["TOY_DATA_ROOT"] = os.path.abspath(os.path.join(policy_dir, "..", "output"))
    env["TOY_DATA_DIRNAME"] = f"query_{query_id}_modified"
    env["TOY_POLICY"] = solver_type
    if cfg.SIM_ROOT:
        env["SIM_ROOT"] = str(cfg.SIM_ROOT)
    os.makedirs(env["TOY_DATA_ROOT"], exist_ok=True)

    try:
        r = subprocess.run(
            [".venv/bin/python3", "-c", wrapper],
            env=env, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "output": "TIMEOUT"}

    output = r.stdout + "\n" + r.stderr
    passed = r.returncode == 0 and "Traceback" not in output
    return {"passed": passed, "output": output[-3000:]}
