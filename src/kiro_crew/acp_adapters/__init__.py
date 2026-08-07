"""ACP-server adapters that let Kiro Crew's ACP client drive non-kiro backends.

Currently ships one adapter, :mod:`kiro_crew.acp_adapters.litellm_server`, which
wraps LiteLLM to reach Ollama / OpenAI-compatible endpoints / Amazon Bedrock
(issue #1693). Adapters are launched as ``python -m <module>`` subprocesses by
``kiro_crew.acp.client.AcpClient`` and speak the kiro ACP dialect over stdio.
"""
