"""A small LLM gateway: tiering, caching, retries and cost accounting.

A four-agent debate over several rounds multiplies calls quickly, so the
routing is not decoration. Retrieval-shaped turns go to the cheap tier and only
the judge is escalated. Identical requests are served from an on-disk cache,
which matters because agents re-ask the same sub-questions constantly.

Every call is recorded, so a debate can report cost per verdict.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "llm_cache"
API = "https://generativelanguage.googleapis.com/v1beta"

# Accepted names for the key, in priority order. Google AI Studio hands out
# both "AIza..." API keys and "AQ..." tokens; both authenticate as ?key=.
KEY_NAMES = ("GOOGLE_API_KEY", "GOOGLE_AI_STUDIO", "GEMINI_API_KEY")


def load_env(path: Path | None = None) -> dict[str, str]:
    """Minimal .env reader. No dependency, no surprises, never logs values."""
    path = path or ROOT / ".env"
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("\"'")
    return out


def candidate_keys() -> list[tuple[str, str]]:
    """Every plausible key, as (source, value), most explicit first.

    `.env` is listed BEFORE the environment on purpose. A stale
    GOOGLE_API_KEY exported in a shell will otherwise shadow the good key the
    user just wrote to .env, and the only symptom is a 401 that looks like the
    new key is bad. That exact bug cost real debugging time here.
    """
    out: list[tuple[str, str]] = []
    env_file = load_env()
    for n in KEY_NAMES:
        if env_file.get(n):
            out.append((f".env:{n}", env_file[n]))
    for k, v in env_file.items():
        if v.startswith(("AIza", "AQ.")) and not any(v == x for _, x in out):
            out.append((f".env:{k}", v))
    for n in KEY_NAMES:
        if os.environ.get(n) and not any(os.environ[n] == x for _, x in out):
            out.append((f"env:{n}", os.environ[n]))
    return out


def verify_key(key: str, timeout: int = 20) -> bool:
    try:
        return requests.get(f"{API}/models?key={key}", timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def find_api_key(verify: bool = True) -> str | None:
    """First candidate that actually authenticates."""
    cands = candidate_keys()
    if not cands:
        return None
    if not verify:
        return cands[0][1]
    for _src, key in cands:
        if verify_key(key):
            return key
    return None

# Tier -> ordered fallback chain, best first.
#
# Hardcoding one model per tier broke badly: /models cheerfully *lists* models
# that 404 on use ("no longer available to new users"), and Pro tiers can be
# quota-blocked (429) on a given key while Flash works fine. So each tier is a
# chain, resolved once against the live API and cached.
TIER_CHAINS = {
    "cheap": [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-2.5-flash",
    ],
    "strong": [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-pro-latest",
        "gemini-flash-latest",
    ],
}

# USD per 1M tokens (input, output). Reporting only, and best-effort: unknown
# models price at 0 rather than inventing a number.
PRICES = {
    "gemini-3.7-flash": (0.30, 2.50),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-pro-latest": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
}

RESOLVED_CACHE = CACHE / "_resolved_models.json"


def _redact(text: str, key: str | None) -> str:
    """Never let an API key reach a log, a traceback or a terminal."""
    if key and key in text:
        text = text.replace(key, f"{key[:6]}…REDACTED")
    return re.sub(r"key=[A-Za-z0-9._\-]+", "key=REDACTED", text)


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, model: str, pin: int, pout: int) -> None:
        self.calls += 1
        self.input_tokens += pin
        self.output_tokens += pout
        ci, co = PRICES.get(model, (0.0, 0.0))
        self.cost_usd += pin / 1e6 * ci + pout / 1e6 * co
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def summary(self) -> str:
        return (
            f"{self.calls} calls ({self.cached} cache hits), "
            f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens, "
            f"${self.cost_usd:.4f}"
        )


class Gateway:
    def __init__(self, api_key: str | None = None, use_cache: bool = True):
        self.api_key = api_key or find_api_key()
        if not self.api_key:
            raise RuntimeError(
                "No API key. Set GOOGLE_API_KEY, or put it in howzat/.env "
                "(see .env.example)."
            )
        self.use_cache = use_cache
        self.usage = Usage()
        CACHE.mkdir(parents=True, exist_ok=True)
        self._resolved: dict[str, str] = {}
        if RESOLVED_CACHE.exists():
            try:
                self._resolved = json.loads(RESOLVED_CACHE.read_text())
            except json.JSONDecodeError:
                self._resolved = {}

    def probe(self, model: str) -> bool:
        """Does this model actually answer? Listing it is not enough."""
        url = f"{API}/models/{model}:generateContent?key={self.api_key}"
        try:
            r = requests.post(
                url,
                json={
                    "contents": [{"role": "user", "parts": [{"text": "OK"}]}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": 5},
                },
                timeout=60,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    def resolve(self, tier: str) -> str:
        """First model in the tier's chain that actually works, cached."""
        if tier in self._resolved:
            return self._resolved[tier]
        for model in TIER_CHAINS[tier]:
            if self.probe(model):
                self._resolved[tier] = model
                RESOLVED_CACHE.write_text(json.dumps(self._resolved, indent=2))
                return model
        raise RuntimeError(
            f"No working model for tier {tier!r}. Tried: {TIER_CHAINS[tier]}"
        )

    def list_models(self) -> list[str]:
        r = requests.get(f"{API}/models?key={self.api_key}", timeout=30)
        r.raise_for_status()
        return [
            m["name"].split("/")[-1]
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]

    def _key(self, payload: dict, model: str) -> Path:
        blob = json.dumps({"m": model, "p": payload}, sort_keys=True).encode()
        return CACHE / f"{hashlib.sha256(blob).hexdigest()[:24]}.json"

    def generate(
        self,
        prompt: str,
        *,
        tier: str = "cheap",
        system: str | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
        temperature: float = 0.3,
        max_retries: int = 4,
    ) -> dict[str, Any]:
        # Try the resolved model, then fall through the rest of the tier chain.
        # A transient 503 on one model should degrade to the next, not abort a
        # debate that has already done all its evidence-gathering.
        primary = self.resolve(tier)
        chain = [primary] + [m for m in TIER_CHAINS[tier] if m != primary]
        errors: list[str] = []
        for model in chain:
            try:
                return self._generate_one(
                    model, prompt, system=system, tools=tools, history=history,
                    temperature=temperature, max_retries=max_retries,
                )
            except RuntimeError as e:
                errors.append(f"{model}: {str(e)[:120]}")
                self._resolved.pop(tier, None)  # stop trusting the cached choice
        raise RuntimeError(
            f"every model in tier {tier!r} failed:\n  " + "\n  ".join(errors)
        )

    def _generate_one(
        self,
        model: str,
        prompt: str,
        *,
        system: str | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
        temperature: float = 0.3,
        max_retries: int = 4,
    ) -> dict[str, Any]:
        contents = list(history or [])
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": tools}]

        cache_path = self._key(payload, model)
        if self.use_cache and cache_path.exists():
            self.usage.cached += 1
            return json.loads(cache_path.read_text())

        url = f"{API}/models/{model}:generateContent?key={self.api_key}"
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.post(url, json=payload, timeout=180)
                if r.status_code in (429, 500, 502, 503, 504):
                    # 429 is quota, not a blip: back off much harder than a 5xx.
                    wait = (2 ** attempt) * (15 if r.status_code == 429 else 3)
                    time.sleep(wait)
                    last_err = f"HTTP {r.status_code}"
                    continue
                if 400 <= r.status_code < 500:
                    # Retrying a 400/403/404 just wastes four round trips and
                    # buries the real message.
                    raise RuntimeError(
                        f"{model}: HTTP {r.status_code} "
                        f"{_redact(r.text[:300], self.api_key)}"
                    )
                r.raise_for_status()
                data = r.json()
                um = data.get("usageMetadata", {})
                self.usage.add(
                    model,
                    um.get("promptTokenCount", 0),
                    um.get("candidatesTokenCount", 0),
                )
                if self.use_cache:
                    cache_path.write_text(json.dumps(data))
                return data
            except requests.RequestException as e:
                last_err = _redact(str(e), self.api_key)
                time.sleep(2 ** attempt * 2)
        raise RuntimeError(
            f"gateway failed after {max_retries} attempts: "
            f"{_redact(str(last_err), self.api_key)}"
        )

    # -- response helpers -------------------------------------------------

    @staticmethod
    def parts(resp: dict) -> list[dict]:
        cands = resp.get("candidates") or []
        if not cands:
            return []
        return cands[0].get("content", {}).get("parts", []) or []

    @classmethod
    def text(cls, resp: dict) -> str:
        return "".join(p.get("text", "") for p in cls.parts(resp)).strip()

    @classmethod
    def calls(cls, resp: dict) -> list[dict]:
        return [p["functionCall"] for p in cls.parts(resp) if "functionCall" in p]


if __name__ == "__main__":
    gw = Gateway()
    print(f"{len(gw.list_models())} models listed by the API")
    for tier in TIER_CHAINS:
        print(f"  {tier:7s} -> {gw.resolve(tier)}")
