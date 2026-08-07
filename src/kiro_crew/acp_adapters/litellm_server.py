"""LiteLLM-backed ACP server adapter (issue #1693).

Bridges Kiro Crew's ACP *client* to any provider `LiteLLM
<https://github.com/BerriAI/litellm>`_ supports — Ollama / OpenAI-compatible
endpoints / Amazon Bedrock. ``AcpClient._spawn`` launches this module as
``python -m kiro_crew.acp_adapters.litellm_server`` when ``agent.provider`` is
not ``"acp"``. It speaks the kiro ACP dialect (``protocolVersion "2025-08-22"``)
over newline-delimited JSON-RPC on stdio:

    initialize → session/new → session/prompt (streamed) → session/cancel
    + server-initiated session/request_permission before each tool call.

Provider configuration arrives via ``KIROCREW_LLM_*`` env vars (set by
``KiroCrewConfig._litellm_provider_env``). Tools arrive as stdio ``mcpServers``
in ``session/new`` (Kiro Crew's managed core servers, injected by
``AcpClient._litellm_session_mcp_servers``); the adapter is their MCP *client*.

``litellm`` and ``mcp`` are optional dependencies (the ``providers`` extra), so
they are imported at module scope behind ``try/except ImportError`` and left as
``None`` when absent — the base install is unaffected, and the pure helpers
below are unit-testable without either package.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import uuid
from contextlib import AsyncExitStack
from typing import Any

# Optional dependencies — the ``providers`` extra
# (``pip install "kirocrew[providers]"``). Absent in a base install, so each is
# guarded and the callers below degrade with a warning instead of raising on
# import. See AUTOSDE `top-level-imports`, "Optional dependencies".
try:
    import litellm
except ImportError:  # pragma: no cover - exercised only without the extra
    litellm = None  # type: ignore[assignment]

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:  # pragma: no cover - exercised only without the extra
    ClientSession = None  # type: ignore[assignment,misc]
    StdioServerParameters = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment]

logger = logging.getLogger("kiro_crew.acp_adapters.litellm_server")

PROTOCOL_VERSION = "2025-08-22"
_ADAPTER_NAME = "kirocrew-litellm-adapter"

# ACP permission option ids we advertise on session/request_permission.
_OPT_ALLOW_ONCE = "allow_once"
_OPT_ALLOW_ALWAYS = "allow_always"
_OPT_REJECT = "reject_once"
_ALLOW_OPTS = frozenset({_OPT_ALLOW_ONCE, _OPT_ALLOW_ALWAYS, "allow"})


# ---------------------------------------------------------------------------
# Pure helpers (no litellm / mcp import — unit-testable directly)
# ---------------------------------------------------------------------------


def build_litellm_params(
    provider: str,
    model: str,
    *,
    base_url: str = "",
    api_key: str = "",
    bedrock_profile: str = "",
    bedrock_region: str = "",
) -> tuple[str, dict[str, Any]]:
    """Return ``(litellm_model_string, extra_kwargs)`` for a provider.

    LiteLLM keys a backend off the model-string prefix (``ollama_chat/…``,
    ``openai/…``, ``bedrock/…``) plus per-provider kwargs. Raises ``ValueError``
    when no model is configured or the provider is unknown.
    """
    if not model:
        raise ValueError(
            "no model configured: set agent.model to a provider-native id "
            "(e.g. 'qwen3:32b', 'gpt-4o-mini', "
            "'anthropic.claude-3-5-sonnet-20240620-v1:0')"
        )
    kwargs: dict[str, Any] = {}
    if provider == "ollama":
        # ollama_chat/ uses the /api/chat endpoint, which does tool-calling;
        # the older ollama/ prefix routes to /api/generate and does not.
        model_str = (
            model if model.startswith(("ollama/", "ollama_chat/")) else f"ollama_chat/{model}"
        )
        if base_url:
            kwargs["api_base"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
    elif provider == "openai_compatible":
        model_str = model if model.startswith("openai/") else f"openai/{model}"
        if base_url:
            kwargs["api_base"] = base_url
        # Many OpenAI-compatible servers ignore the key but the SDK requires a
        # non-empty value; a placeholder keeps local proxies happy.
        kwargs["api_key"] = api_key or "not-needed"
    elif provider == "bedrock":
        model_str = model if model.startswith("bedrock/") else f"bedrock/{model}"
        if bedrock_profile:
            kwargs["aws_profile_name"] = bedrock_profile
        if bedrock_region:
            kwargs["aws_region_name"] = bedrock_region
    else:
        raise ValueError(f"unsupported provider {provider!r}")
    return model_str, kwargs


def sanitize_tool_name(name: str) -> str:
    """Coerce an MCP tool name into the OpenAI ``^[A-Za-z0-9_-]{1,64}$`` shape."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:64]
    return cleaned or "tool"


def mcp_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    """Convert MCP ``Tool`` objects to OpenAI/LiteLLM ``tools`` schema.

    Accepts objects with ``.name`` / ``.description`` / ``.inputSchema`` (the
    MCP SDK shape) or equivalent dicts.
    """
    out: list[dict[str, Any]] = []
    for t in tools:
        name = getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)
        if not name:
            continue
        desc = getattr(t, "description", None)
        if desc is None and isinstance(t, dict):
            desc = t.get("description")
        schema = getattr(t, "inputSchema", None)
        if schema is None and isinstance(t, dict):
            schema = t.get("inputSchema") or t.get("input_schema")
        out.append(
            {
                "type": "function",
                "function": {
                    "name": sanitize_tool_name(str(name)),
                    "description": str(desc or ""),
                    "parameters": schema or {"type": "object", "properties": {}},
                },
            }
        )
    return out


def extract_prompt_text(prompt: Any) -> str:
    """Join the text blocks of an ACP ``session/prompt`` ``prompt`` array."""
    if isinstance(prompt, str):
        return prompt
    if not isinstance(prompt, list):
        return ""
    parts: list[str] = []
    for block in prompt:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _make_response(req_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _make_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _session_update(session_id: str, update: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": "session/update",
        "params": {"sessionId": session_id, "update": update},
    }


# ---------------------------------------------------------------------------
# MCP tool bridge (lazy-imports the `mcp` SDK)
# ---------------------------------------------------------------------------


class McpToolBridge:
    """Connects to the stdio ``mcpServers`` from ``session/new`` and exposes
    their tools to LiteLLM, dispatching tool calls back to the owning session.
    """

    def __init__(self) -> None:
        self._stack: Any = None  # contextlib.AsyncExitStack
        self._sessions: dict[str, Any] = {}  # sanitized tool name -> ClientSession
        self._orig_name: dict[str, str] = {}  # sanitized -> original tool name
        # sanitized tool name -> the ``mcpServers`` entry name that served it.
        # This is the trusted "served by an MCP server" discriminator the host's
        # governance gate keys on; it is derived from the session/new payload the
        # host itself sent, never from anything the model authored.
        self._server_of: dict[str, str] = {}
        self.openai_tools: list[dict[str, Any]] = []

    def identity(self, tool_name: str) -> tuple[str, str]:
        """Trusted ``(mcp_server_name, original_tool_name)`` for a tool call.

        Returns ``("", "")`` for a name the bridge never registered, so a
        hallucinated tool name cannot forge MCP provenance.
        """
        server = self._server_of.get(tool_name, "")
        if not server:
            return "", ""
        return server, self._orig_name.get(tool_name, tool_name)

    async def connect(self, servers: list[dict[str, Any]]) -> None:
        if not servers:
            return
        if stdio_client is None:  # pragma: no cover - only when extra not installed
            logger.warning(
                "the 'mcp' package is not installed; the LiteLLM adapter will "
                "run with no tools. Install with: pip install 'kirocrew[providers]'"
            )
            return

        self._stack = AsyncExitStack()
        for entry in servers:
            name = entry.get("name") or "?"
            command = entry.get("command")
            if not command:
                continue
            env = {**os.environ}
            for pair in entry.get("env") or []:
                if isinstance(pair, dict) and "name" in pair:
                    env[str(pair["name"])] = str(pair.get("value", ""))
            params = StdioServerParameters(
                command=command, args=list(entry.get("args") or []), env=env
            )
            try:
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                listed = await session.list_tools()
                for tool in listed.tools:
                    sanitized = sanitize_tool_name(tool.name)
                    if sanitized in self._sessions:
                        logger.warning("tool name collision on %r; last wins", sanitized)
                    self._sessions[sanitized] = session
                    self._orig_name[sanitized] = tool.name
                    self._server_of[sanitized] = name
                self.openai_tools.extend(mcp_tools_to_openai(listed.tools))
                logger.info("connected MCP server %s (%d tools)", name, len(listed.tools))
            except Exception:
                logger.warning("failed to connect MCP server %s", name, exc_info=True)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        session = self._sessions.get(tool_name)
        if session is None:
            return f"error: unknown tool {tool_name!r}"
        result = await session.call_tool(self._orig_name.get(tool_name, tool_name), arguments)
        return _mcp_result_to_text(result)

    async def close(self) -> None:
        if self._stack is not None:
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            self._stack = None


def _mcp_result_to_text(result: Any) -> str:
    """Flatten an MCP ``call_tool`` result into text for the model."""
    content = getattr(result, "content", None)
    if content is None:
        return json.dumps(result, default=str)
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        else:
            parts.append(json.dumps(getattr(block, "__dict__", str(block)), default=str))
    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# The ACP server
# ---------------------------------------------------------------------------


def _call_id(call: dict[str, Any]) -> str:
    """The stable ``tool_call_id`` for a streamed call, minted once and cached.

    A model may omit ``id``. ``_execute_tool_call`` and ``_run_turn``'s balancing
    paths MUST agree on the value, so it is written back into ``call`` instead of
    regenerated per call site — otherwise a synthesized error result would carry
    an id that matches no ``tool_call`` and leave the history just as unbalanced
    as omitting it.
    """
    cid = call.get("id")
    if not cid:
        cid = str(uuid.uuid4())
        call["id"] = cid
    return str(cid)


def _has_tool_result(sess: "_Session", call_id: str) -> bool:
    """Whether ``messages`` already carries a tool result for ``call_id``.

    Guards the error path against double-appending when a call failed *after*
    recording its own result.
    """
    return any(
        m.get("role") == "tool" and m.get("tool_call_id") == call_id for m in sess.messages
    )


class _Session:
    def __init__(self, session_id: str, cwd: str, bridge: McpToolBridge, model: str = "") -> None:
        self.id = session_id
        self.cwd = cwd
        self.bridge = bridge
        # Per-session model. The adapter can host N concurrent sessions (parent
        # + subagents), so this must NOT live on the server: a session/set_model
        # from one session would otherwise retarget every other session's
        # completions. Seeded from the server default at session/new.
        self.model = model
        self.messages: list[dict[str, Any]] = []
        self.cancel = asyncio.Event()


class LiteLLMAcpServer:
    """Newline-delimited JSON-RPC ACP server over stdio."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer
        self._write_lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future] = {}  # server-initiated request id -> future
        self._sessions: dict[str, _Session] = {}
        # Server-wide DEFAULT only; each session gets its own copy (see
        # _Session.model) so set_model never leaks across concurrent sessions.
        self._default_model = os.environ.get("KIROCREW_LLM_MODEL", "")
        self._provider = os.environ.get("KIROCREW_LLM_PROVIDER", "")
        self._closing = False

    # -- transport --------------------------------------------------------

    async def _send(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj) + "\n").encode()
        async with self._write_lock:
            self._writer.write(data)
            await self._writer.drain()

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a server-initiated request and await the client's response."""
        req_id = str(uuid.uuid4())
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut
        await self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        return await fut

    async def serve(self) -> None:
        while not self._closing:
            line = await self._reader.readline()
            if not line:
                break  # EOF: client closed the pipe
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("dropping non-JSON line from client")
                continue
            if isinstance(msg, dict) and msg.get("method"):
                asyncio.create_task(self._dispatch(msg))
            elif isinstance(msg, dict) and "id" in msg:
                fut = self._pending.pop(msg["id"], None)
                if fut is not None and not fut.done():
                    fut.set_result(msg)

    # -- dispatch ---------------------------------------------------------

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}
        try:
            if method == "initialize":
                await self._send(_make_response(req_id, self._initialize_result()))
            elif method == "session/new":
                await self._send(_make_response(req_id, await self._session_new(params)))
            elif method == "session/prompt":
                await self._session_prompt(req_id, params)
            elif method in ("session/set_model", "session/set_config_option"):
                mid = params.get("modelId") or params.get("value")
                if method == "session/set_model" and mid:
                    # Scope to the requesting session — see _Session.model. An
                    # unknown/absent sessionId is ignored rather than applied
                    # globally, so it cannot retarget other live sessions.
                    target = self._sessions.get(str(params.get("sessionId") or ""))
                    if target is not None:
                        target.model = str(mid)
                await self._send(_make_response(req_id, {}))
            elif method == "session/set_mode":
                await self._send(_make_response(req_id, {}))
            elif method == "session/cancel":
                sess = self._sessions.get(str(params.get("sessionId") or ""))
                if sess is not None:
                    sess.cancel.set()  # notification: no response
            elif req_id is not None:
                await self._send(_make_error(req_id, -32601, f"method not found: {method}"))
        except Exception as exc:  # keep the server alive; surface as an error frame
            logger.exception("error handling %s", method)
            if req_id is not None:
                await self._send(_make_error(req_id, -32603, f"{type(exc).__name__}: {exc}"))

    def _initialize_result(self) -> dict[str, Any]:
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "agentInfo": {"name": _ADAPTER_NAME},
            # No image prompt support; no session load. Empty modes => the client
            # will not attempt session/set_mode. We omit `models` so it never
            # calls session/set_model (env KIROCREW_LLM_MODEL is source of truth).
            "agentCapabilities": {
                "loadSession": False,
                "promptCapabilities": {"image": False},
            },
        }

    async def _session_new(self, params: dict[str, Any]) -> dict[str, Any]:
        session_id = str(uuid.uuid4())
        bridge = McpToolBridge()
        await bridge.connect(params.get("mcpServers") or [])
        self._sessions[session_id] = _Session(
            session_id, params.get("cwd", ""), bridge, model=self._default_model
        )
        # Deliberately OMIT `modes`. An empty ``availableModes: []`` is NOT the
        # same as no modes: parse_session_modes() reports advertised=True with no
        # ids, which the client treats as "this backend genuinely offers modes and
        # yours is absent" and fails closed with AcpError("Agent mode ... is not
        # available") — killing every session before its first prompt. Omitting
        # the key yields advertised=False ("unknown"), the backward-compatible
        # path. Agent modes are a kiro-cli concept the adapter has no analogue
        # for; the client also skips set_mode for this backend.
        return {"sessionId": session_id}

    # -- the agentic loop -------------------------------------------------

    async def _session_prompt(self, req_id: Any, params: dict[str, Any]) -> None:
        sess = self._sessions.get(str(params.get("sessionId") or ""))
        if sess is None:
            await self._send(_make_error(req_id, -32602, "unknown sessionId"))
            return
        sess.cancel.clear()
        sess.messages.append({"role": "user", "content": extract_prompt_text(params.get("prompt"))})
        try:
            stop = await self._run_turn(sess)
        except Exception as exc:
            logger.exception("prompt turn failed")
            await self._send(
                _session_update(
                    sess.id,
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"\n[adapter error: {exc}]"},
                    },
                )
            )
            stop = "refusal"
        await self._send(_make_response(req_id, {"stopReason": stop}))

    async def _run_turn(self, sess: _Session) -> str:
        if litellm is None:  # pragma: no cover - only when extra not installed
            raise RuntimeError(
                "the 'litellm' package is required for agent.provider "
                f"{self._provider!r}. Install with: pip install 'kirocrew[providers]'"
            )

        model_str, extra = build_litellm_params(
            self._provider,
            sess.model,
            base_url=os.environ.get("KIROCREW_LLM_BASE_URL", ""),
            api_key=os.environ.get("KIROCREW_LLM_API_KEY", ""),
            bedrock_profile=os.environ.get("KIROCREW_LLM_BEDROCK_PROFILE", ""),
            bedrock_region=os.environ.get("KIROCREW_LLM_BEDROCK_REGION", ""),
        )
        tools = sess.bridge.openai_tools or None

        # Bounded tool-use loop: model → (tools) → results → model → …
        for _ in range(50):
            if sess.cancel.is_set():
                return "cancelled"
            text_parts: list[str] = []
            tool_calls = _ToolCallAccumulator()
            stream = await litellm.acompletion(
                model=model_str,
                messages=sess.messages,
                tools=tools,
                stream=True,
                **extra,
            )
            async for chunk in stream:
                if sess.cancel.is_set():
                    with contextlib.suppress(Exception):
                        await stream.aclose()
                    return "cancelled"
                delta = _chunk_delta(chunk)
                if delta is None:
                    continue
                piece = getattr(delta, "content", None)
                if piece:
                    text_parts.append(piece)
                    await self._send(
                        _session_update(
                            sess.id,
                            {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": piece},
                            },
                        )
                    )
                for tc in getattr(delta, "tool_calls", None) or []:
                    tool_calls.add(tc)

            calls = tool_calls.finalize()
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if calls:
                assistant_msg["tool_calls"] = calls
            sess.messages.append(assistant_msg)

            if not calls:
                return "end_turn"

            # EVERY tool_call must get a matching {"role": "tool"} result before
            # this returns. `messages` is the whole session history (loadSession
            # is False and nothing repairs it), so an assistant message whose
            # tool_calls have no responses is re-sent on every later prompt —
            # an OpenAI-compatible endpoint rejects that with a 400, and the
            # session is bricked for the rest of its life. Both escape routes
            # below therefore close out the calls they skip:
            #   * cancel — the remaining, never-executed calls
            #   * a raising _execute_tool_call (bridge.call -> session.call_tool);
            #     _session_prompt's except would otherwise swallow it and leave
            #     the hole.
            for i, call in enumerate(calls):
                if sess.cancel.is_set():
                    for pending in calls[i:]:
                        sess.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": _call_id(pending),
                                "content": "Tool call cancelled.",
                            }
                        )
                    return "cancelled"
                try:
                    await self._execute_tool_call(sess, call)
                except Exception as exc:
                    logger.warning("tool call failed", exc_info=True)
                    cid = _call_id(call)
                    if not _has_tool_result(sess, cid):
                        sess.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": cid,
                                "content": f"error: {type(exc).__name__}: {exc}",
                            }
                        )
        # Loop ceiling hit — treat as a completed turn rather than spinning.
        return "end_turn"

    async def _execute_tool_call(self, sess: _Session, call: dict[str, Any]) -> None:
        fn = call.get("function", {})
        name = fn.get("name", "")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        tool_call_id = _call_id(call)

        # Trusted tool identity for the host's governance gate, in BOTH forms it
        # consults — they are not interchangeable:
        #
        #  * the canonical ``mcp__<server>__<tool>`` TITLE. classify_tool_title()
        #    (platform/governance.py) branches on the ``mcp__`` prefix to derive
        #    ("mcp", "@server/tool"); an unprefixed title falls through to
        #    ("commands", …) + ("tools", …) instead, so a deny like
        #    ``@kirocrew-cron/cron_add`` never matches and the call is
        #    auto-approved. kiro-cli emits this same prefixed form as its raw ACP
        #    title, so this matches the host's existing convention.
        #  * ``_meta.kiro.{mcpServerName,toolName}``, which AcpClient caches from
        #    THIS notification (the later session/request_permission payload
        #    carries no ``_meta``) and which drives the app-own-server
        #    auto-approve specifically.
        #
        # Both derive from the bridge's own registry, built from the host's
        # session/new ``mcpServers`` — never from the model-authored call — so an
        # unregistered name yields ("", ""), keeps the bare title, and is treated
        # as non-MCP rather than being able to forge provenance.
        mcp_server_name, orig_tool_name = sess.bridge.identity(name)
        title = f"mcp__{mcp_server_name}__{orig_tool_name}" if mcp_server_name else name

        update: dict[str, Any] = {
            "sessionUpdate": "tool_call",
            "toolCallId": tool_call_id,
            "title": title,
            "kind": "other",
            "rawInput": args,
        }
        if mcp_server_name:
            update["_meta"] = {
                "kiro": {"mcpServerName": mcp_server_name, "toolName": orig_tool_name}
            }

        await self._send(_session_update(sess.id, update))

        # Permission carries the same canonical title: this payload has no
        # ``_meta``, so the title is the only signal the general gate can key on.
        allowed = await self._request_permission(sess, tool_call_id, title, args)
        if not allowed:
            result_text = "Tool call rejected by the user."
        else:
            # Dispatch still uses the model-facing (sanitized) name — that is the
            # bridge's registry key; only the governance-facing title is prefixed.
            result_text = await sess.bridge.call(name, args)

        await self._send(
            _session_update(
                sess.id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": tool_call_id,
                    "status": "completed",
                    "content": [{"content": {"type": "text", "text": result_text}}],
                },
            )
        )
        sess.messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result_text})

    async def _request_permission(
        self, sess: _Session, tool_call_id: str, name: str, args: dict[str, Any]
    ) -> bool:
        reply = await self._request(
            "session/request_permission",
            {
                "toolCall": {
                    "toolCallId": tool_call_id,
                    "title": name,
                    "kind": "other",
                    "input": args,
                },
                "options": [
                    {"optionId": _OPT_ALLOW_ONCE, "name": "Allow", "kind": _OPT_ALLOW_ONCE},
                    {
                        "optionId": _OPT_ALLOW_ALWAYS,
                        "name": "Always allow",
                        "kind": _OPT_ALLOW_ALWAYS,
                    },
                    {"optionId": _OPT_REJECT, "name": "Reject", "kind": _OPT_REJECT},
                ],
            },
        )
        outcome = (reply.get("result") or {}).get("outcome") or {}
        if outcome.get("outcome") != "selected":
            return False  # "cancelled" == rejection
        return outcome.get("optionId") in _ALLOW_OPTS

    async def shutdown(self) -> None:
        self._closing = True
        for sess in self._sessions.values():
            await sess.bridge.close()


class _ToolCallAccumulator:
    """Reassembles streamed OpenAI tool-call deltas keyed by ``index``."""

    def __init__(self) -> None:
        self._by_index: dict[int, dict[str, Any]] = {}

    def add(self, tc: Any) -> None:
        index = getattr(tc, "index", None)
        if index is None and isinstance(tc, dict):
            index = tc.get("index", 0)
        index = index or 0
        slot = self._by_index.setdefault(
            index, {"id": None, "function": {"name": "", "arguments": ""}}
        )
        tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
        if tc_id:
            slot["id"] = tc_id
        fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
        if fn is not None:
            fname = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
            fargs = getattr(fn, "arguments", None) or (
                fn.get("arguments") if isinstance(fn, dict) else None
            )
            if fname:
                slot["function"]["name"] = fname
            if fargs:
                slot["function"]["arguments"] += fargs

    def finalize(self) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for index in sorted(self._by_index):
            slot = self._by_index[index]
            if not slot["function"]["name"]:
                continue
            calls.append(
                {
                    "id": slot["id"] or str(uuid.uuid4()),
                    "type": "function",
                    "function": slot["function"],
                }
            )
        return calls


def _chunk_delta(chunk: Any) -> Any:
    choices = getattr(chunk, "choices", None)
    if not choices and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return None
    first = choices[0]
    return getattr(first, "delta", None) or (
        first.get("delta") if isinstance(first, dict) else None
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


async def _open_stdio() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Bind the JSON-RPC transport, then take stdout away from everyone else.

    fd 1 IS the protocol channel: the host reads it line-by-line as JSON-RPC and
    a single stray line breaks the stream. LiteLLM prints to stdout on its own
    initiative -- an error banner ("Give Feedback / Get Help: ...",
    "LiteLLM.Info: If you need to debug this error, ...") lands mid-turn and
    corrupts frames. Observed live against Bedrock, where it interleaved with
    session/update notifications.

    So: dup fd 1 to a private fd for the protocol writer, then point fd 1 (and
    ``sys.stdout``) at fd 2. Redirecting at the FD level, not just rebinding
    ``sys.stdout``, also covers C-level and subprocess writes. Diagnostics still
    reach the host, which already drains our stderr -- they just cannot desync
    the protocol.
    """
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)

    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    proto_out = os.fdopen(proto_fd, "wb", buffering=0)

    w_transport, w_protocol = await loop.connect_write_pipe(
        asyncio.streams.FlowControlMixin, proto_out
    )
    writer = asyncio.StreamWriter(w_transport, w_protocol, reader, loop)
    return reader, writer


async def main() -> None:
    logging.basicConfig(
        level=os.environ.get("KIROCREW_LLM_LOG_LEVEL", "WARNING"), stream=sys.stderr
    )
    reader, writer = await _open_stdio()
    # Belt and braces alongside the fd redirect: ask litellm not to print its
    # banner at all, so the noise does not even reach stderr on every error.
    if litellm is not None:
        with contextlib.suppress(Exception):
            litellm.suppress_debug_info = True
    server = LiteLLMAcpServer(reader, writer)
    try:
        await server.serve()
    finally:
        await server.shutdown()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
