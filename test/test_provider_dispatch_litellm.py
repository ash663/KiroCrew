"""Factory dispatch for the LiteLLM ACP backend (issue #1693)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from kiro_crew.config.loader import KiroCrewConfig


def _load(data: dict) -> KiroCrewConfig:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.json"
        p.write_text(json.dumps(data))
        with mock.patch("kiro_crew.config.loader.config_path", return_value=p):
            return KiroCrewConfig.load()


def _client(cfg: KiroCrewConfig):
    # Build a provider from the factory and reach its underlying AcpClient.
    # This does NOT spawn a subprocess — construction only.
    return cfg.create_provider_factory()(session_key="t")._client


def test_acp_provider_unchanged() -> None:
    c = _client(_load({"agent": {"provider": "acp"}}))
    assert c.backend == ""
    assert not (c._extra_env or {}).get("KIROCREW_LLM_PROVIDER")


def test_unknown_provider_falls_back_to_kiro_backend() -> None:
    # A provider outside the litellm set degrades safely to the kiro-cli backend
    # rather than spawning the adapter with a bad config.
    c = _client(_load({"agent": {"provider": "banana", "model": "m"}}))
    assert c.backend == ""
    assert not (c._extra_env or {}).get("KIROCREW_LLM_PROVIDER")


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible", "bedrock"])
def test_non_acp_uses_litellm_backend(provider: str) -> None:
    c = _client(_load({"agent": {"provider": provider, "model": "m"}}))
    assert c.backend == "litellm"
    env = c._extra_env or {}
    assert env["KIROCREW_LLM_PROVIDER"] == provider
    assert env["KIROCREW_LLM_MODEL"] == "m"


def test_ollama_base_url_env() -> None:
    c = _client(
        _load(
            {
                "agent": {
                    "provider": "ollama",
                    "model": "qwen3:32b",
                    "base_url": "http://localhost:11434",
                }
            }
        )
    )
    assert c._extra_env["KIROCREW_LLM_BASE_URL"] == "http://localhost:11434"


def test_api_key_resolved_from_named_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MYKEY", "sk-secret")
    c = _client(
        _load(
            {
                "agent": {
                    "provider": "openai_compatible",
                    "model": "gpt-4o-mini",
                    "api_key_env": "MYKEY",
                }
            }
        )
    )
    env = c._extra_env
    assert env["KIROCREW_LLM_API_KEY_ENV"] == "MYKEY"
    assert env["KIROCREW_LLM_API_KEY"] == "sk-secret"


def test_bedrock_profile_region_env() -> None:
    c = _client(
        _load(
            {
                "agent": {
                    "provider": "bedrock",
                    "model": "anthropic.claude-3-5-sonnet",
                    "bedrock": {"profile": "p", "region": "us-west-2"},
                }
            }
        )
    )
    assert c._extra_env["KIROCREW_LLM_BEDROCK_PROFILE"] == "p"
    assert c._extra_env["KIROCREW_LLM_BEDROCK_REGION"] == "us-west-2"


@pytest.mark.asyncio
async def test_managed_mcp_servers_injected_only_for_litellm() -> None:
    # async because the builder is offloaded with asyncio.to_thread — its cold
    # path does filesystem traversal + executable lookup and must not run on the
    # gateway event loop.
    litellm_c = _client(_load({"agent": {"provider": "ollama", "model": "m"}}))
    names = {e["name"] for e in await litellm_c._litellm_session_mcp_servers()}
    assert {"kirocrew-core", "kirocrew-cron"} <= names

    acp_c = _client(_load({"agent": {"provider": "acp"}}))
    assert await acp_c._litellm_session_mcp_servers() == []


# ── backend routing: the adapter must reach AcpClient, not AcpRuntime ────────
#
# AcpProvider.start() used to branch on `not is_claude_backend`, which sent the
# litellm backend down _start_kiro_runtime() -> AcpRuntime.spawn() -> kiro-cli.
# The adapter was never launched and the prompt went to the kiro backend, so
# `provider = "ollama"` silently talked to kiro instead of Ollama. Routing keys
# on is_kiro_cli_backend now; these lock that in without spawning anything.


def _provider(cfg: KiroCrewConfig):
    return cfg.create_provider_factory()(session_key="t")


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible", "bedrock"])
def test_litellm_backend_uses_the_client_path_not_the_kiro_runtime(provider: str) -> None:
    p = _provider(_load({"agent": {"provider": provider, "model": "m"}}))
    assert p.is_litellm_backend
    assert not p.is_kiro_cli_backend
    # uses_legacy_client == "start() calls _client.ensure_ready()"
    assert p.uses_legacy_client


def test_kiro_backend_still_uses_the_runtime_path() -> None:
    p = _provider(_load({"agent": {"provider": "acp"}}))
    assert p.is_kiro_cli_backend
    assert not p.uses_legacy_client
    assert not p.is_litellm_backend


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible", "bedrock"])
def test_litellm_backend_is_not_session_sharing_eligible(provider: str) -> None:
    # Sharing requires AcpRuntime's multiplexed process. The adapter runs on
    # AcpClient (one process per session), so reporting eligible would send
    # subagent creation down the shared path against a provider that cannot
    # host it. Eligibility keys on is_kiro_cli_backend, not "not claude".
    p = _provider(_load({"agent": {"provider": provider, "model": "m"}}))
    assert p.is_session_sharing_eligible is False


def test_kiro_backend_remains_session_sharing_eligible() -> None:
    assert _provider(_load({"agent": {"provider": "acp"}})).is_session_sharing_eligible is True


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible", "bedrock"])
@pytest.mark.asyncio
async def test_litellm_backend_reports_effort_unsupported(provider: str) -> None:
    # /effort is a kiro-cli slash command and the adapter advertises no config
    # options, so effort must report unsupported rather than pushing a command
    # the adapter would answer with "method not found".
    p = _provider(_load({"agent": {"provider": provider, "model": "m"}}))
    with mock.patch("kiro_crew.providers.acp.model_supports_effort", return_value=True):
        assert await p.change_effort("high") is False
        assert await p.clear_effort() is False


# ── fail fast at spawn, not at the first prompt ──────────────────────────────
#
# Selecting a non-"acp" provider is valid config on a base install, so without a
# spawn-time check the adapter starts cleanly, the session opens, and the user
# only learns the `providers` extra is missing when their first message dies
# inside the turn.


def _litellm_client(provider: str = "ollama", model: str = "m"):
    cfg = _load({"agent": {"provider": provider, "model": model}})
    return cfg.create_provider_factory()(session_key="t")._client


@pytest.mark.asyncio
async def test_spawn_fails_fast_when_litellm_missing() -> None:
    from kiro_crew.acp.client import AcpError

    c = _litellm_client()
    with mock.patch("kiro_crew.acp.client.importlib.util.find_spec", return_value=None):
        with pytest.raises(AcpError) as ei:
            await c._spawn()
    msg = str(ei.value)
    assert "litellm" in msg
    assert "kirocrew[providers]" in msg  # names the fix


@pytest.mark.asyncio
async def test_spawn_fails_fast_when_model_is_empty() -> None:
    from kiro_crew.acp.client import AcpError

    c = _litellm_client(model="")
    # Pretend the extra IS installed so we isolate the model check.
    with mock.patch("kiro_crew.acp.client.importlib.util.find_spec", return_value=object()):
        with pytest.raises(AcpError) as ei:
            await c._spawn()
    assert "agent.model" in str(ei.value)


@pytest.mark.asyncio
async def test_managed_mcp_discovery_runs_off_the_event_loop() -> None:
    # The builder's cold path does synchronous filesystem traversal + executable
    # lookup. Run inline it stalls every other chat and heartbeat task for the
    # duration of a session start, so it must go through asyncio.to_thread —
    # asserted by checking it executes on a non-main thread.
    import threading

    seen: dict = {}

    def _spy() -> list:
        seen["thread"] = threading.current_thread().name
        return [{"name": "kirocrew-core"}]

    c = _client(_load({"agent": {"provider": "ollama", "model": "m"}}))
    with mock.patch("kiro_crew.acp.client.managed_mcp_acp_entries", _spy):
        await c._litellm_session_mcp_servers()
    assert seen["thread"] != threading.main_thread().name
