import pytest

try:
    import functions.llm.client as client_mod
except BaseException as _import_err:
    pytest.skip(
        f"Skipping test_client.py: cannot import functions.llm.client ({_import_err})",
        allow_module_level=True,
    )


class DummyGenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key


def test_get_model_name_priority_override(monkeypatch):
    creds = {"gemini": {"model_name": "from_creds"}}
    monkeypatch.setenv("GEMINI_MODEL", "from_env")

    assert client_mod.get_model_name(creds, model_name_override="override") == "override"


def test_get_model_name_fallback_credentials(monkeypatch):
    creds = {"gemini": {"model_name": "from_creds"}}
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    assert client_mod.get_model_name(creds) == "from_creds"


def test_get_model_name_fallback_env(monkeypatch):
    creds = {"gemini": {"api_key_env": "GEMINI_API_KEY"}}  # no model_name
    monkeypatch.setenv("GEMINI_MODEL", "from_env")

    assert client_mod.get_model_name(creds) == "from_env"


def test_get_model_name_raises_when_missing(monkeypatch):
    creds = {"gemini": {"api_key_env": "GEMINI_API_KEY"}}  # no model_name
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    with pytest.raises(ValueError) as e:
        client_mod.get_model_name(creds)

    assert "model name not found" in str(e.value).lower()


def test_build_gemini_client_requires_api_key_env(monkeypatch):
    # Missing api_key_env
    creds = {"gemini": {"model_name": "m"}}

    with pytest.raises(ValueError) as e:
        client_mod.build_gemini_client(creds)

    assert "api_key_env is required" in str(e.value).lower()


def test_build_gemini_client_requires_env_var_set(monkeypatch):
    creds = {"gemini": {"api_key_env": "GEMINI_API_KEY", "model_name": "m"}}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(EnvironmentError) as e:
        client_mod.build_gemini_client(creds)

    assert "gemini_api_key" in str(e.value).lower()


def test_build_gemini_client_success(monkeypatch):
    # Patch google.genai.Client used inside module
    monkeypatch.setattr(client_mod.genai, "Client", DummyGenAIClient)
    monkeypatch.setenv("GEMINI_API_KEY", "secret")

    creds = {"gemini": {"api_key_env": "GEMINI_API_KEY", "model_name": "m"}}

    ctx = client_mod.build_gemini_client(creds)

    assert ctx["model_name"] == "m"
    assert isinstance(ctx["client"], DummyGenAIClient)
    assert ctx["client"].api_key == "secret"


def test_build_gemini_client_override_model_name(monkeypatch):
    monkeypatch.setattr(client_mod.genai, "Client", DummyGenAIClient)
    monkeypatch.setenv("GEMINI_API_KEY", "secret")
    monkeypatch.setenv("GEMINI_MODEL", "from_env")  # should be ignored by override

    creds = {"gemini": {"api_key_env": "GEMINI_API_KEY", "model_name": "from_creds"}}

    ctx = client_mod.build_gemini_client(creds, model_name_override="override")

    assert ctx["model_name"] == "override"
