from __future__ import annotations

from typing import Any, Dict

from functions.online.pipeline_4b_generate import run_pipeline_4b_generate


def test_pipeline_4b_online_wrapper_calls_core(monkeypatch):
    # fake params object/dict returned by load_parameters
    def fake_load_parameters(_: str):
        return {"dummy": True}

    def fake_repo_root_from_parameters_path(_: str):
        return "/repo_root"

    called: Dict[str, Any] = {}

    def fake_core(**kwargs):
        called.update(kwargs)
        return {"ok": True, "cache_id": "p4b__x"}

    monkeypatch.setattr("functions.online.pipeline_4b_generate.load_parameters", fake_load_parameters)
    monkeypatch.setattr(
        "functions.online.pipeline_4b_generate.repo_root_from_parameters_path",
        fake_repo_root_from_parameters_path,
    )
    monkeypatch.setattr("functions.online.pipeline_4b_generate.run_pipeline_4b_generate_core", fake_core)

    out = run_pipeline_4b_generate(
        query="q",
        top_k=10,
        debug=True,
        parameters_path="configs/parameters.yaml",
        credentials_path="configs/credentials.yaml",
        schema_model_name="Output",
        include_retrieval_results=False,
        top_k_vector=20,
        top_k_bm25=20,
    )

    assert out["ok"] is True
    assert called["query"] == "q"
    assert called["top_k"] == 10
    assert called["debug"] is True
    assert called["parameters_path"] == "configs/parameters.yaml"
    assert called["credentials_path"] == "configs/credentials.yaml"
    assert called["schema_model_name"] == "Output"
    assert called["include_retrieval_results"] is False
    assert called["top_k_vector"] == 20
    assert called["top_k_bm25"] == 20
    assert called["repo_root"] == "/repo_root"
    assert called["params"] == {"dummy": True}