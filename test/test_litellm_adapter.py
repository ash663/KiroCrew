"""LiteLLM ACP adapter conformance (issue #1693).

Covers the pure helpers and the ACP protocol handlers without needing
``litellm`` / ``mcp`` installed (both are guarded module-level imports that fall
back to ``None``), so these run under the base install.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiro_crew.acp_adapters import litellm_server as L

# ── pure helpers ────────────────────────────────────────────────────────────


def test_build_litellm_params_ollama() -> None:
    assert L.build_litellm_params("ollama", "qwen3:32b", base_url="http://h:11434") == (
        "ollama_chat/qwen3:32b",
        {"api_base": "http://h:11434"},
    )


def test_build_litellm_params_openai_default_key() -> None:
    model, kwargs = L.build_litellm_params(
        "openai_compatible", "gpt-4o-mini", base_url="https://x/v1"
    )
    assert model == "openai/gpt-4o-mini"
    assert kwargs["api_base"] == "https://x/v1"
    assert kwargs["api_key"] == "not-needed"


def test_build_litellm_params_bedrock() -> None:
    model, kwargs = L.build_litellm_params(
        "bedrock", "anthropic.claude-3-5-sonnet", bedrock_profile="p", bedrock_region="us-east-1"
    )
    assert model == "bedrock/anthropic.claude-3-5-sonnet"
    assert kwargs == {"aws_profile_name": "p", "aws_region_name": "us-east-1"}


def test_build_litellm_params_requires_model() -> None:
    with pytest.raises(ValueError):
        L.build_litellm_params("ollama", "")


def test_build_litellm_params_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError):
        L.build_litellm_params("mystery", "m")


def test_sanitize_tool_name() -> None:
    assert L.sanitize_tool_name("weird name!/x") == "weird_name__x"
    assert len(L.sanitize_tool_name("a" * 100)) == 64


def test_mcp_tools_to_openai() -> None:
    ot = L.mcp_tools_to_openai(
        [{"name": "cron_add", "description": "d", "inputSchema": {"type": "object"}}]
    )
    assert ot[0]["type"] == "function"
    assert ot[0]["function"]["name"] == "cron_add"
    assert ot[0]["function"]["parameters"] == {"type": "object"}


def test_extract_prompt_text() -> None:
    blocks = [{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}]
    assert L.extract_prompt_text(blocks) == "ab"
    assert L.extract_prompt_text("hi") == "hi"


def test_tool_call_accumulator_reassembles_stream() -> None:
    acc = L._ToolCallAccumulator()
    acc.add({"index": 0, "id": "call_1", "function": {"name": "foo", "arguments": '{"a":'}})
    acc.add({"index": 0, "function": {"arguments": "1}"}})
    assert acc.finalize() == [
        {"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": '{"a":1}'}}
    ]


# ── ACP protocol handlers ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_initialize_and_session_new() -> None:
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sent: list = []

    async def cap(o: dict) -> None:
        sent.append(o)

    srv._send = cap  # type: ignore[assignment]
    await srv._dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    await srv._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": "/tmp", "mcpServers": []},
        }
    )

    init = sent[0]["result"]
    assert init["protocolVersion"] == "2025-08-22"
    assert init["agentCapabilities"]["loadSession"] is False
    assert init["agentCapabilities"]["promptCapabilities"]["image"] is False

    snew = sent[1]["result"]
    assert "sessionId" in snew
    # No `modes` key at all — an empty availableModes list would read as
    # advertised=True with no ids and fail the session closed. See
    # test_session_new_result_reads_as_modes_not_advertised.
    assert "modes" not in snew


@pytest.mark.asyncio
async def test_session_cancel_sets_flag() -> None:
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sent: list = []

    async def cap(o: dict) -> None:
        sent.append(o)

    srv._send = cap  # type: ignore[assignment]
    await srv._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/new",
            "params": {"cwd": "/tmp", "mcpServers": []},
        }
    )
    sid = sent[-1]["result"]["sessionId"]
    await srv._dispatch(
        {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": sid}}
    )
    assert srv._sessions[sid].cancel.is_set()


@pytest.mark.asyncio
async def test_request_permission_outcomes() -> None:
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sess = L._Session("s", "/tmp", bridge=None)  # type: ignore[arg-type]

    async def allow(method: str, params: dict) -> dict:
        return {"result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}}}

    srv._request = allow  # type: ignore[assignment]
    assert await srv._request_permission(sess, "tc", "tool", {}) is True

    async def cancelled(method: str, params: dict) -> dict:
        return {"result": {"outcome": {"outcome": "cancelled"}}}

    srv._request = cancelled  # type: ignore[assignment]
    assert await srv._request_permission(sess, "tc", "tool", {}) is False


# ── trusted tool identity for host-side MCP governance ──────────────────────
#
# AcpClient keys per-tool MCP denies on ``_meta.kiro.{mcpServerName,toolName}``
# and caches them from the ``tool_call`` notification, because the later
# session/request_permission payload carries no ``_meta``. If the adapter omits
# them, an MCP-served call is classified as a generic command/tool and a
# configured deny never matches — so these lock the metadata in place.


class _FakeBridge:
    """Stands in for McpToolBridge with a fixed tool registry."""

    def __init__(self, identity: tuple[str, str] = ("", "")) -> None:
        self._identity = identity
        self.called: list[tuple[str, dict]] = []
        self.openai_tools: list[dict] = []

    def identity(self, tool_name: str) -> tuple[str, str]:
        return self._identity

    async def call(self, name: str, args: dict) -> str:
        self.called.append((name, args))
        return "ok"


async def _capture_tool_call(bridge: _FakeBridge) -> dict:
    """Run one tool call through the adapter and return the tool_call update."""
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sess = L._Session("s", "/tmp", bridge=bridge)  # type: ignore[arg-type]
    sent: list[dict] = []

    async def cap(o: dict) -> None:
        sent.append(o)

    async def allow(method: str, params: dict) -> dict:
        return {"result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}}}

    srv._send = cap  # type: ignore[assignment]
    srv._request = allow  # type: ignore[assignment]
    await srv._execute_tool_call(sess, {"id": "tc1", "function": {"name": "cron_add"}})
    updates = [
        m["params"]["update"]
        for m in sent
        if m.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call"
    ]
    assert len(updates) == 1
    return updates[0]


@pytest.mark.asyncio
async def test_tool_call_carries_trusted_mcp_identity() -> None:
    update = await _capture_tool_call(_FakeBridge(("kirocrew-cron", "cron_add")))
    assert update["_meta"]["kiro"] == {
        "mcpServerName": "kirocrew-cron",
        "toolName": "cron_add",
    }


@pytest.mark.asyncio
async def test_unregistered_tool_cannot_forge_mcp_identity() -> None:
    # A name the bridge never registered gets no MCP provenance, so a
    # hallucinated tool cannot dress itself up as an MCP-served call.
    update = await _capture_tool_call(_FakeBridge(("", "")))
    assert "_meta" not in update


def test_bridge_identity_is_empty_for_unknown_tool() -> None:
    bridge = L.McpToolBridge()
    assert bridge.identity("never_registered") == ("", "")


def test_bridge_identity_returns_server_and_original_name() -> None:
    bridge = L.McpToolBridge()
    bridge._sessions["kirocrew_cron_cron_add"] = object()
    bridge._orig_name["kirocrew_cron_cron_add"] = "cron_add"
    bridge._server_of["kirocrew_cron_cron_add"] = "kirocrew-cron"
    assert bridge.identity("kirocrew_cron_cron_add") == ("kirocrew-cron", "cron_add")


# ── per-session model isolation ─────────────────────────────────────────────
#
# The adapter can host N concurrent sessions (parent + subagents share the one
# process). The model used to live on the server, so a session/set_model from
# any session retargeted every other session's completions.


@pytest.mark.asyncio
async def test_set_model_is_scoped_to_the_requesting_session(monkeypatch) -> None:
    monkeypatch.setenv("KIROCREW_LLM_MODEL", "base-model")
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sent: list[dict] = []

    async def cap(o: dict) -> None:
        sent.append(o)

    srv._send = cap  # type: ignore[assignment]

    async def new_session() -> str:
        await srv._dispatch(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "session/new",
                "params": {"cwd": "/tmp", "mcpServers": []},
            }
        )
        return sent[-1]["result"]["sessionId"]

    a = await new_session()
    b = await new_session()
    assert srv._sessions[a].model == "base-model"

    await srv._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/set_model",
            "params": {"sessionId": a, "modelId": "session-a-model"},
        }
    )
    assert srv._sessions[a].model == "session-a-model"
    assert srv._sessions[b].model == "base-model"  # untouched
    assert srv._default_model == "base-model"  # default not mutated


@pytest.mark.asyncio
async def test_set_model_for_unknown_session_is_ignored(monkeypatch) -> None:
    monkeypatch.setenv("KIROCREW_LLM_MODEL", "base-model")
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]

    async def cap(o: dict) -> None:
        pass

    srv._send = cap  # type: ignore[assignment]
    await srv._dispatch(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "session/set_model",
            "params": {"sessionId": "nope", "modelId": "x"},
        }
    )
    assert srv._default_model == "base-model"


# ── session/new must NOT advertise an empty mode list ───────────────────────
#
# `availableModes: []` is not "no modes": parse_session_modes() reports
# advertised=True with no ids, and AcpClient then fails closed with
# AcpError("Agent mode ... is not available"), killing every session before its
# first prompt. Omitting the key yields advertised=False, the compatible path.


@pytest.mark.asyncio
async def test_session_new_omits_modes() -> None:
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    result = await srv._session_new({"cwd": "/tmp", "mcpServers": []})
    assert "sessionId" in result
    assert "modes" not in result


@pytest.mark.asyncio
async def test_session_new_result_reads_as_modes_not_advertised() -> None:
    # Assert against the HOST's own parser, not our expectation of it.
    from kiro_crew.acp._dispatch import parse_session_modes

    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    result = await srv._session_new({"cwd": "/tmp", "mcpServers": []})
    ids, _current, advertised = parse_session_modes(result)
    assert advertised is False
    assert ids == []


# ── the canonical mcp__ title is what the governance gate keys on ────────────
#
# classify_tool_title() (platform/governance.py) derives ("mcp", "@server/tool")
# ONLY from a title starting with `mcp__`. An unprefixed title falls through to
# ("commands", …) + ("tools", …), so an `@kirocrew-cron/cron_add` deny never
# matches and the call is auto-approved. `_meta` alone does not cover this: it
# drives the app-own-server auto-approve, a different path.


@pytest.mark.asyncio
async def test_tool_call_title_is_the_canonical_mcp_name() -> None:
    update = await _capture_tool_call(_FakeBridge(("kirocrew-cron", "cron_add")))
    assert update["title"] == "mcp__kirocrew-cron__cron_add"


@pytest.mark.asyncio
async def test_governance_classifies_the_emitted_title_as_mcp() -> None:
    # Assert through the HOST's own classifier rather than our expectation of it.
    from kiro_crew.platform.governance import classify_tool_title

    update = await _capture_tool_call(_FakeBridge(("kirocrew-cron", "cron_add")))
    assert classify_tool_title(update["title"]) == (("mcp", "@kirocrew-cron/cron_add"),)


@pytest.mark.asyncio
async def test_permission_request_carries_the_canonical_title() -> None:
    # The permission payload has no `_meta`, so its title is the only signal.
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sess = L._Session("s", "/tmp", bridge=_FakeBridge(("kirocrew-cron", "cron_add")))  # type: ignore[arg-type]
    seen: list[dict] = []

    async def cap(o: dict) -> None:
        pass

    async def req(method: str, params: dict) -> dict:
        seen.append(params)
        return {"result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}}}

    srv._send = cap  # type: ignore[assignment]
    srv._request = req  # type: ignore[assignment]
    await srv._execute_tool_call(sess, {"id": "tc1", "function": {"name": "cron_add"}})
    assert seen[0]["toolCall"]["title"] == "mcp__kirocrew-cron__cron_add"


@pytest.mark.asyncio
async def test_non_mcp_tool_keeps_its_bare_title() -> None:
    update = await _capture_tool_call(_FakeBridge(("", "")))
    assert update["title"] == "cron_add"
    assert "_meta" not in update


# ── every tool_call must get a matching tool result ─────────────────────────
#
# `messages` IS the session history (loadSession is False, nothing repairs it).
# An assistant message whose tool_calls lack responses is re-sent on every later
# prompt and an OpenAI-compatible endpoint 400s it, bricking the session.


def _unanswered(sess) -> set[str]:
    """tool_call ids on assistant messages with no matching tool result."""
    wanted, answered = set(), set()
    for m in sess.messages:
        for tc in m.get("tool_calls") or []:
            wanted.add(tc["id"])
        if m.get("role") == "tool":
            answered.add(m.get("tool_call_id"))
    return wanted - answered


class _CancelBridge(_FakeBridge):
    """Sets the session's cancel flag the moment the first tool runs."""

    def __init__(self, sess_holder: list) -> None:
        super().__init__(("kirocrew-cron", "cron_add"))
        self._holder = sess_holder

    async def call(self, name: str, args: dict) -> str:
        self._holder[0].cancel.set()
        return "ok"


class _FakeDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChunk:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


class _FakeStream:
    """Async-iterable stand-in for litellm's streaming response."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        async def gen():
            for c in self._chunks:
                yield c

        return gen()

    async def aclose(self):
        return None


def _stub_litellm(monkeypatch, calls: list[dict], then_text: str = "done"):
    """Patch L.litellm so the REAL _run_turn executes.

    First round streams `calls` as tool-call deltas; later rounds stream plain
    text so the loop terminates.
    """
    state = {"round": 0}

    async def acompletion(**kwargs):
        state["round"] += 1
        if state["round"] == 1:
            deltas = [
                _FakeChunk(
                    _FakeDelta(
                        tool_calls=[
                            {
                                "index": i,
                                "id": c["id"],
                                "function": c["function"],
                            }
                        ]
                    )
                )
                for i, c in enumerate(calls)
            ]
            return _FakeStream(deltas)
        return _FakeStream([_FakeChunk(_FakeDelta(content=then_text))])

    monkeypatch.setattr(L, "litellm", type("M", (), {"acompletion": staticmethod(acompletion)}))


@pytest.mark.asyncio
async def test_cancel_mid_tool_calls_leaves_history_balanced(monkeypatch) -> None:
    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    holder: list = []
    sess = L._Session("s", "/tmp", bridge=_CancelBridge(holder))  # type: ignore[arg-type]
    holder.append(sess)

    async def cap(o: dict) -> None:
        pass

    async def allow(method: str, params: dict) -> dict:
        return {"result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}}}

    srv._send = cap  # type: ignore[assignment]
    srv._request = allow  # type: ignore[assignment]
    srv._provider = "ollama"
    sess.model = "qwen3:32b"

    calls = [
        {"id": "a", "function": {"name": "cron_add", "arguments": "{}"}},
        {"id": "b", "function": {"name": "cron_add", "arguments": "{}"}},
        {"id": "c", "function": {"name": "cron_add", "arguments": "{}"}},
    ]
    _stub_litellm(monkeypatch, calls)
    stop = await srv._run_turn(sess)
    assert stop == "cancelled"
    assert _unanswered(sess) == set(), "cancelled calls left the history unbalanced"


@pytest.mark.asyncio
async def test_raising_tool_call_leaves_history_balanced(monkeypatch) -> None:
    class _BoomBridge(_FakeBridge):
        async def call(self, name: str, args: dict) -> str:
            raise RuntimeError("backend exploded")

    srv = L.LiteLLMAcpServer(reader=None, writer=None)  # type: ignore[arg-type]
    sess = L._Session("s", "/tmp", bridge=_BoomBridge(("kirocrew-cron", "cron_add")))  # type: ignore[arg-type]

    async def cap(o: dict) -> None:
        pass

    async def allow(method: str, params: dict) -> dict:
        return {"result": {"outcome": {"outcome": "selected", "optionId": "allow_once"}}}

    srv._send = cap  # type: ignore[assignment]
    srv._request = allow  # type: ignore[assignment]
    srv._provider = "ollama"
    sess.model = "qwen3:32b"

    calls = [{"id": "a", "function": {"name": "cron_add", "arguments": "{}"}}]
    _stub_litellm(monkeypatch, calls)
    await srv._run_turn(sess)
    assert _unanswered(sess) == set()
    err = [m for m in sess.messages if m.get("role") == "tool"][-1]
    assert "backend exploded" in err["content"]


@pytest.mark.asyncio
async def test_call_id_is_stable_when_the_model_omits_one() -> None:
    # The synthesized error result must reuse the id the tool_call announced,
    # not mint a second one — otherwise it answers nothing.
    call: dict = {"function": {"name": "cron_add", "arguments": "{}"}}
    first = L._call_id(call)
    assert first and L._call_id(call) == first
    assert call["id"] == first


# ── the protocol fd must be immune to library stdout writes ─────────────────
#
# fd 1 IS the JSON-RPC channel. LiteLLM prints an error banner to stdout on its
# own initiative ("Give Feedback / Get Help: ...", "LiteLLM.Info: ..."), which
# was observed live against Bedrock interleaving with session/update frames and
# breaking line-by-line JSON parsing. _open_stdio() dups fd 1 for the protocol
# and repoints fd 1 at stderr, so no print -- Python or C level -- can desync it.

_STDIO_ISOLATION_PROBE = r"""
import asyncio, json, os, sys
sys.path.insert(0, "__SRC__")
from kiro_crew.acp_adapters.litellm_server import _open_stdio

async def main():
    reader, writer = await _open_stdio()
    # Exactly what litellm does: print to stdout, at both the Python and fd level.
    print("BANNER-VIA-PRINT")
    os.write(1, b"BANNER-VIA-FD1\n")
    writer.write((json.dumps({"jsonrpc": "2.0", "id": 1, "result": {}}) + "\n").encode())
    await writer.drain()

asyncio.run(main())
"""


def test_stdout_writes_cannot_corrupt_the_protocol_stream(tmp_path) -> None:
    import subprocess
    import sys as _sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    probe = tmp_path / "probe.py"
    probe.write_text(_STDIO_ISOLATION_PROBE.replace("__SRC__", src))

    r = subprocess.run(
        [_sys.executable, str(probe)], capture_output=True, timeout=60, text=True
    )
    # fd 1 (the protocol channel) must carry ONLY the JSON-RPC frame.
    proto_lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    assert proto_lines == ['{"jsonrpc": "2.0", "id": 1, "result": {}}'], (
        f"protocol fd polluted: {proto_lines!r}"
    )
    for ln in proto_lines:
        json.loads(ln)  # every line must parse

    # The banners are not lost — they are diverted to stderr, which the host drains.
    assert "BANNER-VIA-PRINT" in r.stderr
    assert "BANNER-VIA-FD1" in r.stderr
