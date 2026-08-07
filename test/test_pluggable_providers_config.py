"""Config surface for pluggable model providers (issue #1693)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from kiro_crew.config.loader import (
    BedrockConfig,
    KiroCrewConfig,
    _coerce_bedrock,
)


def _load(data: dict) -> KiroCrewConfig:
    """Write *data* to a temp config file and load it through the real loader."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "config.json"
        p.write_text(json.dumps(data))
        with mock.patch("kiro_crew.config.loader.config_path", return_value=p):
            return KiroCrewConfig.load()


@pytest.mark.parametrize("provider", ["acp", "ollama", "openai_compatible", "bedrock"])
def test_provider_enum_accepts_new_values(provider: str) -> None:
    assert _load({"agent": {"provider": provider}}).agent.provider == provider


def test_unknown_provider_falls_back_to_default_on_load() -> None:
    # `provider` is enum-validated, so an unrecognized value is rejected by the
    # schema pass and reset to the default rather than stored as-is. The factory
    # therefore only ever sees a validated provider id (see
    # test_provider_dispatch_litellm).
    assert _load({"agent": {"provider": "banana"}}).agent.provider == "acp"


def test_default_provider_is_acp() -> None:
    assert _load({}).agent.provider == "acp"


def test_new_fields_parse_and_roundtrip() -> None:
    cfg = _load(
        {
            "agent": {
                "provider": "openai_compatible",
                "model": "gpt-4o-mini",
                "base_url": "https://api.example.com/v1",
                "api_key_env": "MYKEY",
                "bedrock": {"profile": "p", "region": "us-east-1"},
            }
        }
    )
    a = cfg.agent
    assert a.base_url == "https://api.example.com/v1"
    assert a.api_key_env == "MYKEY"
    assert (a.bedrock.profile, a.bedrock.region) == ("p", "us-east-1")

    rt = cfg.to_dict()["agent"]
    assert rt["base_url"] == "https://api.example.com/v1"
    assert rt["api_key_env"] == "MYKEY"
    assert rt["bedrock"] == {"profile": "p", "region": "us-east-1"}


def test_bedrock_defaults_and_coercion() -> None:
    assert _load({"agent": {"provider": "bedrock"}}).agent.bedrock == BedrockConfig()
    assert _coerce_bedrock({"region": "eu-west-1"}).region == "eu-west-1"
    assert _coerce_bedrock(None) == BedrockConfig()
    assert _coerce_bedrock(BedrockConfig(profile="x")).profile == "x"
