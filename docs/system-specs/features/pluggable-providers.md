# Pluggable model providers (Ollama / OpenAI-compatible / Amazon Bedrock)

Status: **accepted** — implemented by [issue #1693](https://github.com/kirodotdev/KiroCrew/issues/1693).

## Summary

Make `agent.provider` selectable beyond the fixed `acp` backend so Kiro Crew can
drive the **agent** model against:

1. **Ollama / local models** — `http://localhost:11434` or any OpenAI-compatible
   base URL (llama.cpp, LM Studio, vLLM, LiteLLM proxy).
2. **Amazon Bedrock** — via the standard AWS credential chain + explicit
   `profile` / `region`.
3. **Generic OpenAI-compatible** — a `base_url` + API key (OpenRouter, Together,
   Groq, self-hosted proxies).

## What a non-`acp` provider does NOT get (v1 scope)

`acp` (kiro-cli) stays the DEFAULT and the only fully-featured path. A
non-`acp` provider is deliberately partial, and the gaps are load-bearing enough
to state up front rather than discover at runtime:

| gap | consequence |
|---|---|
| **No file or shell tools.** The adapter bridges only the managed MCP servers (spawn / messaging / artifacts / skills / cron / computer). kiro-cli's `fs_read` / `fs_write` / `execute_bash` are process built-ins with no MCP equivalent. | The agent cannot read or edit files or run commands. This limits the issue's own "local privacy over file contents" motivation. |
| **No Kiro Crew system prompt.** The agent-spec prompt kiro-cli loads never reaches the adapter; sessions start from a bare user message. | No persona, no memory or skills conventions, no injected-message handling. |
| **No compaction, and history is RAM-only** (`loadSession: false`). | A long run exhausts the context window with no recovery, and any respawn (crash, idle reap, gateway restart) silently drops the conversation. This is exactly the "long autonomous runs" workload the issue cites, so treat crons/monitors on a non-`acp` provider as unsupported for now. |
| **No session sharing.** One adapter process per session (see the routing table below). | Subagent fan-out costs one process each. |
| **No effort / Tool Search / slash commands.** kiro-cli features with no adapter analogue. | `change_effort` reports unsupported; slash commands arrive as prompt text. |

`agent.provider`'s config help carries the same warning, and selecting a
non-`acp` provider without the `providers` extra or without `agent.model` fails
at spawn with an actionable error rather than mid-turn.

## Approach: an ACP-server adapter sidecar (not a host-side agent loop)

Today the agent's tool-execution loop lives **inside the backend subprocess**
(`kiro-cli`), which speaks the Agent Client Protocol (ACP) over stdio.
`AcpProvider.stream()` merely relays `AcpClient.stream_events()`; Kiro Crew
forwards `EVENT_TOOL_CALL` / `EVENT_TOOL_RESULT` / `EVENT_PERMISSION_REQUEST`
events and sends back `approve_tool` / `reject_tool`. The removed `claude_code`
provider worked the same way — it was a second **ACP-speaking subprocess**
(`claude-agent-acp`) driven by the same client behind the dormant
`ACP_BACKEND_CLAUDE` seam.

Rather than re-implement a full agent harness in the host (conversation state,
MCP-tool bridge, permission mediation, context accounting, compaction), we add a
**third ACP backend**: a small in-package Python ACP *server* that wraps
[LiteLLM](https://github.com/BerriAI/litellm). One LiteLLM adapter covers Ollama,
OpenAI-compatible, and Bedrock in a single implementation. Kiro Crew's existing
`AcpClient` drives it unchanged, so the entire event / permission / compaction /
MCP-injection pipeline is reused.

```
AcpClient (unchanged)  ──stdio ACP──▶  litellm_server (new)  ──▶  LiteLLM  ──▶  Ollama / OpenAI-compat / Bedrock
        │                                     │
        └── session/new mcpServers ───────────┴──▶ MCP client ──▶ Kiro Crew tools (gateway)
```

### Which host path starts it

`AcpProvider` now hosts three backends, and they split two ways:

| backend | hosted by | session sharing |
|---|---|---|
| kiro-cli (`acp`) | `AcpRuntime` — one process, N multiplexed sessions | eligible |
| claude-agent-acp | `AcpClient` — one process per session | never |
| LiteLLM adapter | `AcpClient` — one process per session | never |

So `start()` keys on `is_kiro_cli_backend`, **not** on `not is_claude_backend`:
the adapter is neither claude nor kiro-cli, and routing it by the absence of
claude sent it to `_start_kiro_runtime()` → `AcpRuntime.spawn()` → `kiro-cli`,
which launched the wrong child entirely and silently served prompts from the
kiro backend instead of the configured provider. Everything kiro-cli-SPECIFIC
keys on the same predicate: the `cli.json` effort and Tool Search overlays, the
`/effort` slash command, and `stream_command`'s `_kiro.dev/commands/execute`.
`is_session_sharing_eligible` likewise — the adapter *can* serve concurrent
sessions, but only `AcpRuntime` multiplexes them, so subagents take the
per-process path.

Sharing is a deliberate non-goal for v1 rather than a limitation to route
around: teaching `AcpRuntime` a second child would mean threading a non-kiro
argv through its `--agent` / `--model` pinning, `is_kiro_cli=True` sandbox
wrapping, and mode-discovery handshake, and each adapter session connects its
own MCP servers anyway, so a shared adapter process would still multiply the
tool subprocesses it was meant to save.

## The ACP contract the adapter must implement

`AcpClient` is the ACP **client**; the adapter is the ACP **server**. Wire format
is newline-delimited JSON-RPC 2.0 over stdio. kiro dialect: string
`protocolVersion "2025-08-22"`. Load-bearing methods only:

| Method | Direction | Adapter behavior |
|---|---|---|
| `initialize` | client→server | reply `{"protocolVersion":"2025-08-22","agentCapabilities":{"loadSession":false,"promptCapabilities":{"image":false}}}` |
| `session/new` | client→server | params `{"cwd","mcpServers":[…]}`; connect to `mcpServers`, list tools, reply `{"sessionId":…}`. MUST NOT include `modes` — an empty `availableModes: []` reads as advertised-with-no-ids and the client fails the session closed; omitting the key is the compatible "unknown" path |
| `session/prompt` | client→server | params `{"sessionId","prompt":[{"type":"text","text":…}]}` (key is **`prompt`**, not `content`); run the LiteLLM loop, stream `session/update`, then reply `{"stopReason":"end_turn"}` |
| `session/cancel` | client→server (notification, no id) | set cancel flag; current prompt replies `{"stopReason":"cancelled"}` |
| `session/request_permission` | server→client (request) | emitted before each tool call; await `{"outcome":{"outcome":"selected","optionId":…}}` or `{"outcome":{"outcome":"cancelled"}}` |
| `session/set_model` | client→server | optional; switch model for later turns. Scoped to `params.sessionId` — the adapter hosts N concurrent sessions, so it must not mutate a server-wide default |

`session/update` notifications the adapter emits (inner `params.update`
discriminated by `sessionUpdate`):
- `agent_message_chunk` → `{"content":{"type":"text","text":delta}}` (thinking → `type:"thinking"`)
- `tool_call` → `{"toolCallId","title","kind","rawInput":{…}}`
- `tool_call_update` → `{"toolCallId","status":"completed","content":[{"content":{"type":"text","text":…}}]}`

All `_kiro.dev/*`, `set_mode`, steer, terminate, metadata, and compaction are
optional and degrade gracefully; the adapter omits them for v1.

### The agentic loop (per `session/prompt`)

1. Append the user prompt to the session's message history.
2. Loop: call `litellm.acompletion(model, messages, tools=<mcp tools as OpenAI schema>, stream=True)`.
3. Stream text deltas out as `agent_message_chunk`; collect any `tool_calls`.
4. No tool calls → reply `{"stopReason":"end_turn"}`.
5. Tool calls → for each: emit `tool_call`, send `session/request_permission`,
   await the outcome; on allow, invoke the MCP tool and emit `tool_call_update`
   (append the result to `messages`); on reject, append a rejection result.
   Then continue the loop so the model can react to results.
6. Cancel flag set at any await point → reply `{"stopReason":"cancelled"}`.

The `tool_call` notification MUST carry BOTH the canonical
`mcp__<server>__<tool>` **title** and `_meta.kiro.{mcpServerName,toolName}` for
any MCP-served call, and `session/request_permission` MUST repeat that title.
They cover different gates and neither substitutes for the other:
`classify_tool_title()` derives `("mcp", "@server/tool")` only from the `mcp__`
title prefix — an unprefixed title falls through to `("commands", …)` +
`("tools", …)`, so an `@kirocrew-cron/cron_add` deny never matches — while `AcpClient` caches the `_meta` pair from this notification (the
permission payload carries no `_meta`) to drive the app-own-server auto-approve. Omit it and the host classifies the call as a generic command/tool, so
a configured deny (e.g. `@kirocrew-cron/cron_add`) never matches. The pair is
read from the bridge's own registry — built from the host's `session/new`
`mcpServers` — never from the model-authored call, so an unregistered tool name
yields no provenance and cannot forge one.

### Tool-calling fidelity (issue's main risk)

Smaller local models emit tool calls inconsistently. **PLANNED (not in v1):** a
`kirocrew doctor` capability probe — a one-shot completion with a trivial tool;
if the model does not emit a well-formed tool call, print a loud warning. It
would be advisory, not a hard gate. Until then a model with weak tool-calling
fidelity degrades silently, one more reason `acp` stays the default.

### Tool-call history must stay balanced

`messages` IS the session history (`loadSession: false`, nothing repairs it), so
EVERY `tool_call` must receive a matching `{"role": "tool"}` result before the
turn returns — including the calls skipped by a cancel and any whose execution
raised. An assistant message whose `tool_calls` lack responses is re-sent on
every later prompt, and an OpenAI-compatible endpoint rejects it with a 400,
bricking the session for its remaining life. Ids come from `_call_id()`, which
mints one once and caches it on the call, so a synthesized result cannot answer
an id the model never saw.

## MCP tools

`session/new` already carries `mcpServers` — `AcpClient` builds it as
`[*_claude_session_mcp_servers(), *_pooled_mcp_servers()]`, and
`_pooled_mcp_servers()` injects the real shared-gateway broker-stub servers.
So when the MCP gateway is enabled the adapter receives Kiro Crew's tools with no
client change. Gateway-disabled tool exposure is a documented follow-on (mirrors
the claude backend, which needs a seam override to see tools).

## Config shape

```jsonc
{
  "agent": {
    "provider": "ollama",            // acp | ollama | openai_compatible | bedrock
    "model": "qwen3:32b",            // LiteLLM/provider-native model id
    "base_url": "http://localhost:11434",   // ollama / openai_compatible
    "api_key_env": "OPENAI_API_KEY", // NAME of env var holding the key — never the key itself
    "bedrock": { "profile": "default", "region": "us-east-1" }
  }
}
```

Design choices:
- **`api_key_env` holds an env-var *name*, not the secret.** Config files are not
  a secret store; the adapter reads `os.environ[api_key_env]` at spawn.
- `provider` default stays `"acp"` — zero behavior change for existing users.
- Per-role providers (`role_providers`, mirroring `role_models`) are a documented
  follow-on; v1 selects one provider for the whole agent.

## File-by-file changes

- `src/kiro_crew/config/loader.py`
  - Widen `AgentConfig.provider` enum → `["acp","ollama","openai_compatible","bedrock"]`.
  - Add `AgentConfig.base_url`, `AgentConfig.api_key_env`, and a nested
    `BedrockConfig(profile, region)` field; parse them in `load()`.
  - `create_provider_factory()`: branch on `self.agent.provider`. Non-`acp`
    providers build an `extra_env` (`KIROCREW_LLM_*`) and pass
    `acp_backend=ACP_BACKEND_LITELLM` into `AcpProvider(...)`.
- `src/kiro_crew/acp/types.py` — add `ACP_BACKEND_LITELLM = "litellm"`.
- `src/kiro_crew/acp/client.py` — in `_spawn`, add an
  `elif self.backend == ACP_BACKEND_LITELLM:` branch whose argv is
  `[sys.executable, "-m", "kiro_crew.acp_adapters.litellm_server"]`. All provider
  config rides the already-merged `extra_env`, so no other client surgery.
- `src/kiro_crew/acp_adapters/__init__.py`, `.../litellm_server.py` — the adapter.
- `setup.cfg` — new optional extra `providers = litellm>=1.50,<2 ; mcp>=1.0,<2`.
- `config/loader.py` validation + `docs/system-specs/modules/providers.md` — drop
  the "provider enum is `[\"acp\"]`, there is no provider to choose" constraint.
- `docs/system-specs/modules/providers.md` — replace the "provider is fixed to
  `acp`, there is no provider to choose" constraint with the widened enum + the
  LiteLLM factory branch. **(done)**
- `docs/guides/getting-started.md` — the issue's secondary item (a stale
  "Ollama for local vector-memory embeddings" line) is **already resolved
  upstream**: no such line exists in the current tree (README or docs).

## Live verification (real provider)

Protocol and streaming conformance is stub-tested, but the LiteLLM path was also
exercised end-to-end against a **real Amazon Bedrock model**
(`us.amazon.nova-micro-v1:0`, us-west-2), driving the adapter as a real
subprocess over ACP stdio. Two things only a live run could surface:

1. **LiteLLM writes to stdout.** Its error banner (`Give Feedback / Get Help:
   ...`, `LiteLLM.Info: ...`) printed straight onto fd 1 — the JSON-RPC channel —
   interleaving with `session/update` frames and breaking line-by-line parsing.
   `_open_stdio()` now dups fd 1 for the protocol writer and repoints fd 1 (and
   `sys.stdout`) at stderr, so no library write can desync the stream. Covered by
   `test_stdout_writes_cannot_corrupt_the_protocol_stream`.
2. **Bedrock on-demand needs an inference profile.** A bare model id
   (`amazon.nova-micro-v1:0`) is rejected: *"Invocation of model ID … with
   on-demand throughput isn't supported. Retry your request with the ID or ARN of
   an inference profile"*. Set `agent.model` to the regional profile id — the
   `us.`-prefixed form (`us.amazon.nova-micro-v1:0`).

A full tool-calling round trip also passed live: MCP bridge → tools as OpenAI
schema → model-issued call → `session/request_permission` → MCP dispatch →
result consumed by the follow-up turn, with the canonical
`mcp__<server>__<tool>` title present on both the `tool_call` and the permission
request.

## Testing

- Config: enum accepts the four values; `base_url`/`api_key_env`/`bedrock` parse
  and round-trip through `to_dict()`/`load()`.
- Factory: `provider="ollama"` yields an `AcpProvider` with
  `acp_backend=="litellm"` and the expected `KIROCREW_LLM_*` `extra_env`;
  `provider="acp"` is byte-identical to today.
- Adapter (async, `@pytest.mark.asyncio`): drive it over an in-memory stdio pair
  with `litellm.acompletion` stubbed — assert the `initialize` / `session/new` /
  `session/prompt` happy path, a tool-call round trip through the permission
  handshake, and `session/cancel` → `stopReason:"cancelled"`.
- Gate: `black && isort && flake8 && mypy && pytest`.
