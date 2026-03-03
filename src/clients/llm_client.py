"""
LLM client with provider + task-based model selection.

Supported providers:
- Gemini API (requires GEMINI_API_KEY)
"""

import json
import os
from typing import Any, Dict, List, Optional

import requests


_SUPPORTED_PROVIDERS = {"gemini"}

# Task→model defaults for Gemini API model IDs
# Keep default on Flash to fit free-tier experimentation.
_GEMINI_TASK_MODEL_MAP = {
    "search_query": "gemini-2.5-flash",
    "validation": "gemini-2.5-flash",
    "assessment": "gemini-2.5-flash",
    "edit": "gemini-2.5-flash",
    "default": "gemini-2.5-flash",
}


def _normalize_provider(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _pick_provider(env: Dict[str, str]) -> str:
    """
    Resolve active LLM provider.

    Priority:
    1. Explicit LLM_PROVIDER
    2. GEMINI_API_KEY present -> gemini
    """
    explicit = _normalize_provider(env.get("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "")))
    if explicit:
        if explicit not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported LLM_PROVIDER='{explicit}'. "
                f"Expected one of: {sorted(_SUPPORTED_PROVIDERS)}"
            )
        return explicit

    gemini_key = env.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
    if gemini_key:
        return "gemini"

    raise ValueError(
        "No LLM provider configured. Set GEMINI_API_KEY "
        "or set LLM_PROVIDER explicitly."
    )


def _model_override_key(provider: str, task_type: str) -> str:
    task_key = task_type.strip().upper().replace("-", "_")
    return f"{provider.upper()}_MODEL_{task_key}"


def _get_model_for_task(provider: str, task_type: str, env: Dict[str, str]) -> str:
    # Provider/task-specific override
    override_key = _model_override_key(provider, task_type)
    if env.get(override_key):
        return env[override_key].strip()
    if os.getenv(override_key):
        return os.getenv(override_key, "").strip()

    # Provider-wide override
    provider_key = f"{provider.upper()}_MODEL"
    if env.get(provider_key):
        return env[provider_key].strip()
    if os.getenv(provider_key):
        return os.getenv(provider_key, "").strip()

    return _GEMINI_TASK_MODEL_MAP.get(task_type, _GEMINI_TASK_MODEL_MAP["default"])


def _extract_gemini_text(payload: Dict[str, Any]) -> str:
    """
    Extract concatenated text from Gemini generateContent response.
    """
    texts: List[str] = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content", {}) or {}
        for part in content.get("parts", []) or []:
            text = part.get("text", "")
            if text:
                texts.append(text)
    return "".join(texts).strip()


def _call_gemini(
    prompt: str,
    model: str,
    system_prompt: Optional[str],
    env: Dict[str, str],
    max_tokens: int,
    temperature: float,
) -> str:
    api_key = env.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set. Cannot call Gemini.")

    base_url = env.get(
        "GEMINI_BASE_URL",
        os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
    ).rstrip("/")

    url = f"{base_url}/models/{model}:generateContent?key={api_key}"

    body: Dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": int(max_tokens),
        },
    }
    if system_prompt:
        body["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    resp = requests.post(
        url,
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    text = _extract_gemini_text(data)
    if text:
        return text

    block_reason = ((data.get("promptFeedback") or {}).get("blockReason") or "").strip()
    if block_reason:
        raise RuntimeError(f"Gemini response blocked: {block_reason}")

    raise RuntimeError(f"Gemini returned no text candidates for model '{model}'.")


def call_llm(
    prompt: str,
    task_type: str = "default",
    system_prompt: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
) -> str:
    """
    Call an LLM with automatic provider + model selection.

    Args:
        prompt: User prompt text.
        task_type: One of 'search_query', 'validation', 'assessment', 'edit', 'default'.
        system_prompt: Optional system prompt.
        env: Environment variables (for API keys and provider/model selection).
        max_tokens: Max response tokens.
        temperature: Sampling temperature.

    Returns:
        The model's text response.
    """
    env = env or os.environ.copy()
    provider = _pick_provider(env)
    model = _get_model_for_task(provider, task_type, env)

    return _call_gemini(
        prompt=prompt,
        model=model,
        system_prompt=system_prompt,
        env=env,
        max_tokens=max_tokens,
        temperature=temperature,
    )
