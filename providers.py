"""Provider-agnostic model calls for Ethoculus.

Every probe in this repo talks to models only through `generate()`. Which
platform answers is controlled entirely by configuration, never by code:

    TARGET_PROVIDER=anthropic      TARGET_MODEL=claude-opus-5-5
    JUDGE_PROVIDER=openai          JUDGE_MODEL=gpt-5.6-luna

Three wire formats cover nearly every hosted or local model:

    openai_responses  OpenAI's Responses API (native OpenAI, reasoning models)
    openai_chat       The OpenAI-compatible /chat/completions format, which
                      most other platforms also accept (Gemini, xAI, Groq,
                      Mistral, DeepSeek, Together, OpenRouter, Ollama, vLLM)
    anthropic         Anthropic's Messages API

Presets below map a short provider name to a wire format, a default base
URL, and the environment variable that holds that platform's API key. Any
setting can be overridden with <PREFIX>_BASE_URL, <PREFIX>_API_KEY_ENV,
<PREFIX>_API_FORMAT, and so on (PREFIX is TARGET or JUDGE).

Calls are plain HTTPS through `requests`, so no vendor SDK is required.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass

import requests

# name: (api_format, default_base_url, default_api_key_env)
# Base URLs reflect each vendor's published endpoint as of this writing;
# confirm against the vendor's current docs if a call returns 404.
PRESETS: dict[str, tuple[str, str, str | None]] = {
    "openai": ("openai_responses", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openai_chat": ("openai_chat", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
    "gemini": ("openai_chat", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "xai": ("openai_chat", "https://api.x.ai/v1", "XAI_API_KEY"),
    "groq": ("openai_chat", "https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "mistral": ("openai_chat", "https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "deepseek": ("openai_chat", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "together": ("openai_chat", "https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "openrouter": ("openai_chat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("openai_chat", "http://localhost:11434/v1", None),
    "custom": ("openai_chat", "", "CUSTOM_API_KEY"),
}

ANTHROPIC_VERSION = "2023-06-01"


class ProviderError(RuntimeError):
    """The platform returned an HTTP error or an unreadable response."""


@dataclass
class ProviderConfig:
    provider: str
    model: str
    api_format: str
    base_url: str
    api_key_env: str | None
    timeout: int = 120
    json_mode: str = "auto"            # auto | schema | prompt
    max_tokens_param: str = "max_tokens"  # openai_chat only

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None

    def check_key(self) -> None:
        if self.api_key_env and not self.api_key:
            raise SystemExit(
                f"{self.api_key_env} is not set (needed for {self}). "
                "Add it to your local .env file."
            )


def config_from_env(prefix: str, provider: str | None = None, model: str | None = None,
                    base_url: str | None = None) -> ProviderConfig:
    """Build a config for TARGET or JUDGE. CLI values win over env values."""
    provider = (provider or os.getenv(f"{prefix}_PROVIDER") or "openai").lower()
    if provider not in PRESETS:
        raise SystemExit(f"Unknown provider '{provider}'. Choose from: {', '.join(PRESETS)}")
    fmt, url, key_env = PRESETS[provider]
    model = model or os.getenv(f"{prefix}_MODEL")
    if not model:
        raise SystemExit(f"No model set. Use --{prefix.lower()}-model or {prefix}_MODEL in .env.")
    base = base_url or os.getenv(f"{prefix}_BASE_URL") or url
    if not base:
        raise SystemExit(f"Provider '{provider}' needs {prefix}_BASE_URL.")
    return ProviderConfig(
        provider=provider,
        model=model,
        api_format=os.getenv(f"{prefix}_API_FORMAT", fmt),
        base_url=base.rstrip("/"),
        api_key_env=os.getenv(f"{prefix}_API_KEY_ENV", key_env or "") or None,
        timeout=int(os.getenv(f"{prefix}_TIMEOUT", "120")),
        json_mode=os.getenv(f"{prefix}_JSON_MODE", "auto"),
        max_tokens_param=os.getenv(f"{prefix}_MAX_TOKENS_PARAM", "max_tokens"),
    )


@dataclass
class LLMResult:
    text: str
    incomplete: bool                 # True when the token budget ran out
    stop_reason: str | None
    response_id: str | None
    resolved_model: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: int


def _schema_body(json_schema: dict) -> dict:
    """Accepts either the OpenAI text.format wrapper or a bare JSON Schema."""
    return json_schema.get("schema", json_schema)


def _json_instruction(json_schema: dict) -> str:
    return (
        "Respond with only one JSON object that matches this JSON Schema. "
        "No prose, no markdown, no code fences.\n"
        + json.dumps(_schema_body(json_schema))
    )


def parse_json(text: str) -> dict | None:
    """Lenient JSON extraction: tolerates code fences or stray prose."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _post(cfg: ProviderConfig, url: str, headers: dict, body: dict) -> dict:
    resp = requests.post(url, headers=headers, json=body, timeout=cfg.timeout)
    if resp.status_code >= 400:
        raise ProviderError(f"{cfg} HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        return resp.json()
    except ValueError as e:
        raise ProviderError(f"{cfg} returned non-JSON: {resp.text[:300]}") from e


def _openai_responses(cfg, messages, instructions, json_schema, max_tokens):
    body = {"model": cfg.model, "input": messages, "max_output_tokens": max_tokens}
    if instructions:
        body["instructions"] = instructions
    if json_schema:
        if cfg.json_mode == "prompt":
            body["instructions"] = (instructions or "") + "\n\n" + _json_instruction(json_schema)
        else:
            body["text"] = {"format": json_schema}
    data = _post(cfg, f"{cfg.base_url}/responses",
                 {"Authorization": f"Bearer {cfg.api_key}"}, body)
    parts = []
    for item in data.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    usage = data.get("usage") or {}
    reason = (data.get("incomplete_details") or {}).get("reason")
    return ("".join(parts), data.get("status") == "incomplete", reason or data.get("status"),
            data.get("id"), data.get("model"),
            usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def _openai_chat(cfg, messages, instructions, json_schema, max_tokens):
    msgs = list(messages)
    system = instructions or ""
    use_schema = json_schema and cfg.json_mode == "schema"
    if json_schema and not use_schema:
        system = (system + "\n\n" + _json_instruction(json_schema)).strip()
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    body = {"model": cfg.model, "messages": msgs, cfg.max_tokens_param: max_tokens}
    if use_schema:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": json_schema.get("name", "output"),
                            "strict": True, "schema": _schema_body(json_schema)},
        }
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    data = _post(cfg, f"{cfg.base_url}/chat/completions", headers, body)
    choice = (data.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
    usage = data.get("usage") or {}
    return (text, finish == "length", finish, data.get("id"), data.get("model"),
            usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))


def _anthropic(cfg, messages, instructions, json_schema, max_tokens):
    system_parts = [instructions] if instructions else []
    convo = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            convo.append({"role": m["role"], "content": m["content"]})
    if json_schema:
        system_parts.append(_json_instruction(json_schema))
    body = {"model": cfg.model, "max_tokens": max_tokens, "messages": convo}
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    data = _post(cfg, f"{cfg.base_url}/v1/messages",
                 {"x-api-key": cfg.api_key or "", "anthropic-version": ANTHROPIC_VERSION},
                 body)
    text = "".join(b.get("text", "") for b in data.get("content", []) or []
                   if b.get("type") == "text")
    usage = data.get("usage") or {}
    stop = data.get("stop_reason")
    return (text, stop == "max_tokens", stop, data.get("id"), data.get("model"),
            usage.get("input_tokens", 0), usage.get("output_tokens", 0))


_ADAPTERS = {
    "openai_responses": _openai_responses,
    "openai_chat": _openai_chat,
    "anthropic": _anthropic,
}


def generate(cfg: ProviderConfig, messages: list[dict], instructions: str | None = None,
             json_schema: dict | None = None, max_output_tokens: int = 600) -> LLMResult:
    """One model call. `messages` is a list of {'role', 'content'} dicts using
    roles system/user/assistant; each adapter translates them as needed."""
    adapter = _ADAPTERS.get(cfg.api_format)
    if adapter is None:
        raise SystemExit(f"Unknown api_format '{cfg.api_format}'. Choose from: {', '.join(_ADAPTERS)}")
    started = time.perf_counter()
    text, incomplete, stop, rid, rmodel, tin, tout = adapter(
        cfg, messages, instructions, json_schema, max_output_tokens)
    return LLMResult(
        text=(text or "").strip(), incomplete=incomplete, stop_reason=stop,
        response_id=rid, resolved_model=rmodel,
        input_tokens=int(tin or 0), output_tokens=int(tout or 0),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
