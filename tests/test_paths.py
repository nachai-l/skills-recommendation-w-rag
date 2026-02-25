# tests/test_paths.py

from pathlib import Path

from functions.utils.paths import repo_root_from_parameters_path, resolve_path


def test_repo_root_from_parameters_path(tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    params = configs / "parameters.yaml"
    params.write_text("", encoding="utf-8")

    root = repo_root_from_parameters_path(params)
    assert root == tmp_path


def test_repo_root_from_relative_parameters_path(monkeypatch, tmp_path: Path):
    configs = tmp_path / "configs"
    configs.mkdir()
    params = configs / "parameters.yaml"
    params.write_text("", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    root = repo_root_from_parameters_path("configs/parameters.yaml")
    assert root == tmp_path


def test_resolve_path_relative(tmp_path: Path):
    result = resolve_path("prompts/gen.yaml", base_dir=tmp_path)
    assert result == (tmp_path / "prompts" / "gen.yaml").resolve()


def test_resolve_path_absolute():
    p = Path("/absolute/path/to/file.txt")
    result = resolve_path(p, base_dir="/some/base")
    assert result == p
