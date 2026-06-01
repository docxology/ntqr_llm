"""Minimal LLM client that dispatches to a local Ollama or a hosted API.

Stdlib only (urllib + json) so the method module adds no dependencies. Two
backends:

* ``ollama`` (default) -- talks to a locally running Ollama daemon. The default
  model is ``granite4.1:3b``. ``ensure_ollama`` does a *real* check that the
  daemon is up and the model is pulled, and raises an actionable error if not.
* ``anthropic`` -- posts to the Anthropic Messages API using ``ANTHROPIC_API_KEY``
  from the environment. Provided as a drop-in alternative; the demo defaults to
  Ollama so it runs fully locally and free.

The client supports JSON-mode completions and Ollama JSON Schema completions so
personas can return structured output we can parse deterministically.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite4.1:3b")
DEFAULT_ANTHROPIC_MODEL = os.environ.get(
    "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
)


class LLMUnavailable(RuntimeError):
    """Raised when the requested backend/model cannot be reached."""


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_tags() -> list[str]:
    """Return the list of locally available Ollama model names (or raise)."""
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/tags", timeout=3
        ) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError) as exc:
        raise LLMUnavailable(
            f"Ollama not reachable at {OLLAMA_URL}. Is it running? "
            f"Start it with `ollama serve`. ({exc})"
        ) from exc
    return [m.get("name", "") for m in data.get("models", [])]


def ensure_ollama(model: str = DEFAULT_OLLAMA_MODEL) -> None:
    """Verify the Ollama daemon is up and ``model`` is pulled."""
    names = ollama_tags()
    # Accept exact match or the bare family (e.g. "granite4.1:3b" vs base).
    if model not in names and not any(n.startswith(model) for n in names):
        raise LLMUnavailable(
            f"Ollama model {model!r} not found. Available: {names or '[]'}. "
            f"Pull it with `ollama pull {model}`."
        )


class LLMClient:
    """Dispatches chat completions to the chosen backend."""

    def __init__(
        self,
        backend: str = "ollama",
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ):
        self.backend = backend
        self.temperature = temperature
        self.timeout = timeout
        if backend == "ollama":
            self.model = model or DEFAULT_OLLAMA_MODEL
        elif backend == "anthropic":
            self.model = model or DEFAULT_ANTHROPIC_MODEL
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def preflight(self) -> None:
        """Fail fast with a clear message if the backend is not usable."""
        if self.backend == "ollama":
            ensure_ollama(self.model)
        elif self.backend == "anthropic" and not os.environ.get(
            "ANTHROPIC_API_KEY"
        ):
            raise LLMUnavailable(
                "ANTHROPIC_API_KEY is not set; cannot use the anthropic backend."
            )

    def complete(
        self,
        system: str,
        user: str,
        json_mode: bool = False,
        schema: dict | None = None,
    ) -> str:
        if self.backend == "ollama":
            return self._ollama_complete(system, user, json_mode, schema)
        return self._anthropic_complete(system, user, json_mode, schema)

    def complete_json(self, system: str, user: str) -> dict:
        """Complete in JSON mode and parse the first JSON object found."""
        text = self.complete(system, user, json_mode=True)
        return _extract_json(text)

    def _ollama_complete(
        self,
        system: str,
        user: str,
        json_mode: bool,
        schema: dict | None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        try:
            data = _post_json(
                f"{OLLAMA_URL}/api/chat", payload, {}, self.timeout
            )
        except (urllib.error.URLError, OSError) as exc:
            raise LLMUnavailable(f"Ollama chat failed: {exc}") from exc
        return data.get("message", {}).get("content", "")

    def _anthropic_complete(
        self,
        system: str,
        user: str,
        json_mode: bool,
        schema: dict | None,
    ) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set.")
        if schema is not None or json_mode:
            user = (
                user + "\n\nRespond with a single JSON object, nothing else."
            )
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "temperature": self.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers,
            self.timeout,
        )
        blocks = data.get("content", [])
        return blocks[0].get("text", "") if blocks else ""


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip code fences / surrounding prose: grab the outermost {...}.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"no JSON object found in model response: {text!r}")
