"""Client for a local llama.cpp ``llama-server`` (OpenAI-compatible chat API).

Every request:
- goes to this computer only (the base URL must be localhost), so nothing leaves the
  machine and nothing costs money;
- turns Qwen3.5's "thinking" off (``chat_template_kwargs``), which makes answers short;
- asks for JSON that follows a schema (``response_format``) for edits and lessons.
  llama.cpp turns the schema into a grammar, so the answer is JSON of that shape. Full
  rewrites instead come back as a plain ```python block (see ``parse_rewrite``);
- reuses the server's cached prompt prefix (``cache_prompt``).

``mock_transport`` replaces the server with a fake one driven by the random mutator, so
the whole pipeline can run offline (tests, CI and ``heuriforge demo``).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx
import numpy as np
from pydantic import BaseModel, Field, ValidationError

from heuriforge.config import LOCAL_HOSTS, LLMSettings, ModelEndpoint


class LLMError(RuntimeError):
    """The model call failed. ``fatal`` means the server cannot be reached at all."""

    def __init__(self, message: str, fatal: bool = False, text: str = ""):
        super().__init__(message)
        self.fatal = fatal
        self.text = text  # the raw answer, when there was one


class Edit(BaseModel):
    search: str
    replace: str


class DiffAnswer(BaseModel):
    idea: str = ""
    edits: list[Edit] = Field(min_length=1, max_length=3)


class RewriteAnswer(BaseModel):
    idea: str = ""
    code: str


class LessonAnswer(BaseModel):
    lesson: str


ANSWERS = {"diff": DiffAnswer, "rewrite": RewriteAnswer, "lesson": LessonAnswer}


@dataclass
class Reply:
    data: dict
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    timings: dict = field(default_factory=dict)


def check_local(url: str) -> None:
    host = urlparse(url).hostname or ""
    if host not in LOCAL_HOSTS and f"[{host}]" not in LOCAL_HOSTS:
        raise ValueError(
            f"model server URL {url!r} is not on this computer. HeuriForge only talks to a "
            "local llama-server (127.0.0.1 / localhost)."
        )


def parse_json(text: str) -> dict:
    """The JSON object in a reply (tolerates a <think> block or ``` fences around it)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise json.JSONDecodeError("expected a JSON object", text, 0)
    return value


_FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n(.*?)(?:\n```|\Z)", re.S)


def parse_rewrite(text: str, cut_off: bool = False) -> dict:
    """{"idea", "code"} from "Plan: ..." (or "Idea: ...") followed by a ```python block."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    m = _FENCE.search(text)
    if m is None:
        raise ValueError("no ```python code block in the answer")
    if cut_off and not text.rstrip().endswith("```"):
        raise ValueError("the code block was never closed")
    code = m.group(1).strip("\n")
    if not code.strip():
        raise ValueError("the code block is empty")
    plan = re.sub(r"^[\s*#_]*(plan|idea)[\s*_]*:[\s*_]*", "", text[: m.start()], flags=re.I)
    return {"idea": " ".join(plan.replace("*", " ").split()), "code": code}


class LLMClient:
    def __init__(self, settings: LLMSettings, transport: httpx.AsyncBaseTransport | None = None):
        for endpoint in settings.models:
            check_local(endpoint.base_url)
        self.settings = settings
        self.client = httpx.AsyncClient(
            transport=transport, timeout=httpx.Timeout(settings.timeout_s, connect=10.0)
        )

    async def close(self) -> None:
        await self.client.aclose()

    def endpoint(self, name: str) -> ModelEndpoint:
        for e in self.settings.models:
            if e.name == name:
                return e
        raise KeyError(name)

    async def health(self, endpoint: ModelEndpoint) -> bool:
        root = endpoint.base_url.rstrip("/").removesuffix("/v1")
        try:
            r = await self.client.get(f"{root}/health", timeout=10.0)
        except httpx.HTTPError:
            return False
        return r.status_code == 200

    def request_body(
        self, endpoint: ModelEndpoint, messages: list[dict], schema: dict | None, seed: int | None
    ) -> dict:
        s = self.settings
        body = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "top_k": s.top_k,
            "min_p": s.min_p,
            "presence_penalty": s.presence_penalty,
            "max_tokens": s.max_tokens,
            "chat_template_kwargs": {"enable_thinking": bool(s.thinking)},
            "cache_prompt": True,
        }
        if schema is not None:
            # Both spellings: the OpenAI form, and llama.cpp's own form.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
                "schema": schema,
            }
        if seed is not None:
            body["seed"] = int(seed)
        return body

    async def complete(
        self,
        endpoint: ModelEndpoint,
        messages: list[dict],
        kind: str,
        schema: dict | None,
        seed: int | None = None,
    ) -> Reply:
        """Send one chat request and return the validated JSON answer."""
        url = endpoint.base_url.rstrip("/") + "/chat/completions"
        body = self.request_body(endpoint, messages, schema, seed)
        retries = max(0, self.settings.retries)
        for attempt in range(retries + 1):
            t0 = time.perf_counter()
            try:
                r = await self.client.post(url, json=body)
            except httpx.TimeoutException as exc:
                if attempt < retries:
                    continue
                raise LLMError(f"the model server did not answer in time ({exc!r})") from exc
            except httpx.TransportError as exc:
                if attempt < retries:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                raise LLMError(
                    f"cannot reach the model server at {endpoint.base_url} ({exc!r}). Is it running? "
                    "Start it with: make server",
                    fatal=True,
                ) from exc
            if r.status_code in (429, 500, 502, 503) and attempt < retries:
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise LLMError(f"the model server answered {r.status_code}: {r.text[:300]}")
            return self._reply(r.json(), kind, time.perf_counter() - t0)
        raise LLMError("no answer after retries")

    def _reply(self, payload: dict, kind: str, seconds: float) -> Reply:
        try:
            choice = payload["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected reply from the server: {str(payload)[:200]}") from exc
        usage = payload.get("usage") or {}
        timings = payload.get("timings") or {}
        cached = timings.get("cache_n")
        if cached is None:
            cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        reply = Reply(
            data={},
            text=text,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(cached or 0),
            seconds=seconds,
            timings=timings,
        )
        cut_off = choice.get("finish_reason") == "length"
        try:
            raw = parse_rewrite(text, cut_off) if kind == "rewrite" else parse_json(text)
            reply.data = ANSWERS[kind].model_validate(raw).model_dump()
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            cut = " (the answer was cut off: raise llm.max_tokens)" if cut_off else ""
            raise LLMError(
                f"the answer was not in the requested format{cut}: {str(exc)[:200]}",
                text=text,
            ) from exc
        return reply


# --------------------------------------------------------------------------- offline mock
def mock_transport(task, seed: int = 0, fault_rate: float = 0.15) -> httpx.MockTransport:
    """A fake llama-server: answers with random mutations of the program in the prompt.

    A fraction of the answers is deliberately bad (an edit that does not match, a forbidden
    import), so the offline runs also exercise the error paths.
    """
    from heuriforge.mutators import RandomMutator

    rng = np.random.default_rng(seed)
    mutator = RandomMutator(task, rng)
    fence = re.compile(r"## Current program.*?```python\n(.*?)\n```", re.S)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        body = json.loads(request.content)
        prompt = "\n".join(m["content"] for m in body["messages"])
        fmt = body.get("response_format")
        props = fmt["json_schema"]["schema"]["properties"] if fmt else {}
        if "lesson" in props:
            answer = {
                "lesson": "Programs that combine distance with how full the truck is do better."
            }
        else:
            m = fence.search(prompt)
            block = m.group(1) if m else ""
            new_block, idea = mutator.mutate(block)
            r = rng.random()
            if r < fault_rate / 2:
                new_block = "import os\n" + new_block  # a forbidden import: the audit must catch it
                idea = "use the operating system"
            if "edits" in props:
                search = (
                    block if rng.random() >= fault_rate / 2 else block + "\n# not in the program"
                )
                answer = {"idea": idea, "edits": [{"search": search, "replace": new_block}]}
            else:
                answer = f"Plan: {idea}\n```python\n{new_block}\n```"
        content = answer if isinstance(answer, str) else json.dumps(answer)
        prompt_tokens = len(prompt) // 4
        cache = int(prompt_tokens * 0.6)
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(content) // 4},
            "timings": {
                "cache_n": cache,
                "prompt_n": prompt_tokens - cache,
                "predicted_n": len(content) // 4,
                "predicted_per_second": 0.0,
            },
        }
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)
