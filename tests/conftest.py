"""A mock OpenAI-compatible server, so the load-test client is tested without a GPU."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

ANSWER = '{"color": "red", "shape": "circle"}'
STATS = {"in_flight": 0, "max_in_flight": 0, "served": 0}


async def chat(request: Request) -> JSONResponse:
    body = await request.json()
    model = body.get("model")
    STATS["in_flight"] += 1  # one event loop, so plain ints are safe
    STATS["max_in_flight"] = max(STATS["max_in_flight"], STATS["in_flight"])
    try:
        if model == "fail":
            return JSONResponse({"error": "boom"}, status_code=500)
        if model == "garbage":
            return JSONResponse({"unexpected": True})
        await asyncio.sleep(2.0 if model == "slow" else 0.05)
        STATS["served"] += 1
        content = ANSWER
        fmt = body.get("response_format") or {}
        if fmt.get("type") == "json_schema" and model != "nogrammar":
            # Behave like an enforced grammar: the first allowed value of each field.
            props = fmt["json_schema"]["schema"]["properties"]
            content = json.dumps({k: v["enum"][0] for k, v in props.items()})
        choice: dict = {"message": {"role": "assistant", "content": content}}
        if body.get("logprobs"):
            # Structural characters near-certain (log-prob 0), value characters less so.
            values = [v for v in json.loads(content).values()]
            tokens = []
            for ch in content:
                tokens.append({"token": ch, "logprob": 0.0})
            text = content
            for v in values:
                start = text.find(f'"{v}"') + 1
                for i in range(start, start + len(v)):
                    tokens[i]["logprob"] = -0.3
            choice["logprobs"] = {"content": tokens}
        return JSONResponse(
            {
                "choices": [choice],
                "usage": {"prompt_tokens": 300, "completion_tokens": 12},
            }
        )
    finally:
        STATS["in_flight"] -= 1


async def health(_: Request) -> JSONResponse:
    return JSONResponse({})


async def models(_: Request) -> JSONResponse:
    return JSONResponse({"data": [{"id": "base"}, {"id": "smoke-lora"}, {"id": "nogrammar"}]})


async def version(_: Request) -> JSONResponse:
    return JSONResponse({"version": "0.0-mock"})


async def reset(_: Request) -> JSONResponse:
    STATS.update(in_flight=0, max_in_flight=0, served=0)
    return JSONResponse(STATS)


async def stats(_: Request) -> JSONResponse:
    return JSONResponse(STATS)


APP = Starlette(
    routes=[
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/health", health),
        Route("/v1/models", models),
        Route("/version", version),
        Route("/reset", reset, methods=["POST"]),
        Route("/stats", stats),
    ]
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mock_server() -> str:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(APP, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "mock server did not start"
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def closed_port_url() -> str:
    return f"http://127.0.0.1:{free_port()}"
