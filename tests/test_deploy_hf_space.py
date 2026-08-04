import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "deploy-hf-space.py"
SPEC = importlib.util.spec_from_file_location("deploy_hf_space", MODULE_PATH)
deploy_hf_space = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(deploy_hf_space)


def test_deploy_tokens_can_be_read_from_local_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_hf_space, "ROOT", tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("BGG_API_TOKEN", raising=False)

    (tmp_path / ".secrets").write_text(
        "\n".join(
            [
                "# local-only deploy credentials",
                "BGG_API_TOKEN=bgg-test-token",
                "HF_TOKEN=hf_test_token",
            ]
        ),
        encoding="utf-8",
    )

    assert deploy_hf_space._read_hf_token() == "hf_test_token"
    assert deploy_hf_space._read_bgg_token() == "bgg-test-token"


def test_deploy_hf_token_prefers_environment_over_local_secrets(tmp_path, monkeypatch):
    monkeypatch.setattr(deploy_hf_space, "ROOT", tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_env_token")
    (tmp_path / ".secrets").write_text("HF_TOKEN=hf_file_token\n", encoding="utf-8")

    assert deploy_hf_space._read_hf_token() == "hf_env_token"
