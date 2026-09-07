import pytest

from apps.api.adapters.azure_openai_llm_client import AzureOpenAILLMClient
from apps.api.adapters.openai_llm_client import OpenAILLMClient
from apps.api.deps import _build_real_llm_client


def test_defaults_to_openai_when_llm_provider_unset(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    llm = _build_real_llm_client()

    assert isinstance(llm, OpenAILLMClient)


def test_selects_openai_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    llm = _build_real_llm_client()

    assert isinstance(llm, OpenAILLMClient)


def test_selects_azure_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "azure_openai")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "fake-deployment")

    llm = _build_real_llm_client()

    assert isinstance(llm, AzureOpenAILLMClient)


def test_selection_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Azure_OpenAI")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "fake-deployment")

    llm = _build_real_llm_client()

    assert isinstance(llm, AzureOpenAILLMClient)


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")

    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        _build_real_llm_client()
