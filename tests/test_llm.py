import asyncio
import json

import httpx
import pytest

from heuriforge.config import LLMSettings, ModelEndpoint
from heuriforge.database import Program
from heuriforge.llm import LLMClient, LLMError, check_local, mock_transport, parse_json
from heuriforge.prompts import SCHEMAS, mutation_messages
from heuriforge.tasks import get_task

TASK = get_task("cvrp_construct")
ENDPOINT = ModelEndpoint(name="m", base_url="http://127.0.0.1:8080/v1", model="local")


def settings(**kw) -> LLMSettings:
    return LLMSettings(models=[ENDPOINT], retries=0, **kw)


def messages(mutation="diff"):
    parent = Program(island=0, code=TASK.initial_block.strip(), train_score=30.0, id=1)
    return mutation_messages(TASK, parent, mutation, [], [])


def test_only_local_servers_are_allowed():
    for url in ("http://127.0.0.1:8080/v1", "http://localhost:8081/v1", "http://[::1]:8080/v1"):
        check_local(url)
    for url in (
        "https://api.openai.com/v1",
        "http://192.168.1.5:8080/v1",
        "http://evil.example/v1",
    ):
        with pytest.raises(ValueError, match="not on this computer"):
            check_local(url)
    with pytest.raises(ValueError):
        LLMClient(
            LLMSettings(models=[ModelEndpoint(name="x", base_url="https://api.example.com/v1")])
        )


def test_request_turns_thinking_off_and_asks_for_json():
    client = LLMClient(settings())
    body = client.request_body(ENDPOINT, messages(), SCHEMAS["diff"], seed=5)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMAS["diff"]
    assert body["cache_prompt"] is True and body["seed"] == 5 and body["top_k"] == 20
    asyncio.run(client.close())


def test_mock_server_round_trip():
    async def go():
        client = LLMClient(settings(), mock_transport(TASK, seed=3, fault_rate=0.0))
        assert await client.health(ENDPOINT)
        diff = await client.complete(ENDPOINT, messages("diff"), "diff", SCHEMAS["diff"])
        rewrite = await client.complete(
            ENDPOINT, messages("rewrite"), "rewrite", SCHEMAS["rewrite"]
        )
        await client.close()
        return diff, rewrite

    diff, rewrite = asyncio.run(go())
    assert diff.data["edits"][0]["search"].startswith("def score_candidates")
    assert "def score_candidates" in rewrite.data["code"]
    assert diff.prompt_tokens > 0 and diff.cached_tokens > 0


def reply_with(content: str, finish: str = "stop") -> httpx.MockTransport:
    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": finish}],
                "usage": {},
            },
        )

    return httpx.MockTransport(handler)


def test_bad_answers_raise_clear_errors():
    async def ask(transport):
        client = LLMClient(settings(), transport)
        try:
            return await client.complete(ENDPOINT, messages(), "diff", SCHEMAS["diff"])
        finally:
            await client.close()

    with pytest.raises(LLMError, match="not in the requested format"):
        asyncio.run(ask(reply_with("I think you should")))
    with pytest.raises(LLMError, match="cut off"):
        asyncio.run(ask(reply_with('{"idea": "x", "edits": [', finish="length")))
    with pytest.raises(LLMError, match="not in the requested format"):
        asyncio.run(ask(reply_with('{"idea": "x", "edits": []}')))  # needs at least one edit
    ok = asyncio.run(
        ask(
            reply_with(
                '<think>hmm</think>```json\n{"idea": "x", "edits": [{"search": "a", "replace": "b"}]}\n```'
            )
        )
    )
    assert ok.data["edits"] == [{"search": "a", "replace": "b"}]


def test_unreachable_server_is_fatal():
    async def go():
        dead = ModelEndpoint(name="m", base_url="http://127.0.0.1:9/v1")
        client = LLMClient(LLMSettings(models=[dead], retries=0))
        try:
            await client.complete(dead, messages(), "diff", SCHEMAS["diff"])
        finally:
            await client.close()

    with pytest.raises(LLMError) as info:
        asyncio.run(go())
    assert info.value.fatal


def test_parse_json_variants():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure! {"a": 1} hope this helps') == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        parse_json("[1, 2]")


def test_rewrites_come_back_as_a_code_block():
    from heuriforge.llm import parse_rewrite

    client = LLMClient(settings())
    body = client.request_body(ENDPOINT, messages("rewrite"), SCHEMAS["rewrite"], seed=1)
    assert "response_format" not in body  # no JSON grammar for full rewrites
    asyncio.run(client.close())
    text = 'Idea: go far first\n```python\ndef f(a):\n    return "x"\n```'
    assert parse_rewrite(text) == {"idea": "go far first", "code": 'def f(a):\n    return "x"'}
    assert parse_rewrite("**Idea:** b\n```\ndef f():\n    pass\n")["idea"] == "b"
    plan = (
        "Plan: Visit far customers first.\nThen fill the truck.\n```python\ndef f():\n    pass\n```"
    )
    assert parse_rewrite(plan)["idea"] == "Visit far customers first. Then fill the truck."
    with pytest.raises(ValueError, match="never closed"):
        parse_rewrite("Idea: x\n```python\ndef f():\n    while", cut_off=True)
    with pytest.raises(ValueError, match="no ```python"):
        parse_rewrite("I would change the scores.")
