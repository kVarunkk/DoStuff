"""First-run setup: template rendering, .env editing, and when setup triggers."""
import yaml
import pytest
from typer.testing import CliRunner

from dostuff import config as config_module
from dostuff import setup_wizard
from dostuff.cli import app
from dostuff.config import Config


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    monkeypatch.setattr(config_module.Path, "home", classmethod(lambda cls: home))
    monkeypatch.chdir(proj)
    for var in ("MODEL", "MODEL_PROVIDER", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(var, "")  # records the original so keys setup writes are undone
        monkeypatch.delenv(var)
    config_module.reset_config()
    yield home
    config_module.reset_config()


def test_templates_ship_with_package():
    assert "models:" in setup_wizard.read_template("config.example.yaml")
    assert "OPENAI_API_KEY" in setup_wizard.read_template("env.example")


def test_render_config_replaces_models_and_keeps_other_sections():
    raw = yaml.safe_load(setup_wizard.render_config("gemini/gemini-3.1-flash-lite", "gemini"))
    assert raw["models"] == [{"name": "gemini/gemini-3.1-flash-lite", "provider": "gemini", "active": True}]
    assert raw["permissions"]["mode"] == "ask"
    assert raw["loop"]["max_iterations"] > 0


def test_set_env_var_uncomments_replaces_and_appends(tmp_path):
    env = tmp_path / ".env"
    setup_wizard.set_env_var(env, "OPENAI_API_KEY", "sk-1")
    assert "\nOPENAI_API_KEY=sk-1\n" in env.read_text()
    assert "# OPENAI_API_KEY=" not in env.read_text()
    setup_wizard.set_env_var(env, "OPENAI_API_KEY", "sk-2")
    assert env.read_text().count("OPENAI_API_KEY=") == 1 and "sk-2" in env.read_text()
    setup_wizard.set_env_var(env, "CUSTOM_KEY", "a b")
    assert env.read_text().endswith('CUSTOM_KEY="a b"\n')


def test_needs_setup(home, monkeypatch):
    assert setup_wizard.needs_setup()
    monkeypatch.setenv("MODEL", "openai/gpt-4o-mini")
    assert not setup_wizard.needs_setup()
    monkeypatch.delenv("MODEL")
    (home / ".dostuff").mkdir()
    (home / ".dostuff" / "config.yaml").write_text("models: []\n")
    assert not setup_wizard.needs_setup()


def test_setup_command_writes_config_and_key(home):
    # provider 3 (gemini), default model, a key, then decline the test request
    result = CliRunner().invoke(app, ["setup"], input="3\n\nAIza-test\nn\n")
    assert result.exit_code == 0, result.output
    assert Config().model_name == "gemini/gemini-3.1-flash-lite"
    assert "GEMINI_API_KEY=AIza-test" in (home / ".dostuff" / ".env").read_text()


def test_setup_command_keeps_existing_config_when_declined(home):
    (home / ".dostuff").mkdir()
    (home / ".dostuff" / "config.yaml").write_text("models: []\n")
    result = CliRunner().invoke(app, ["setup"], input="n\n")
    assert result.exit_code == 1
    assert (home / ".dostuff" / "config.yaml").read_text() == "models: []\n"


def test_config_and_doctor_create_no_files(home):
    runner = CliRunner()
    runner.invoke(app, ["config"])
    result = runner.invoke(app, ["doctor"])
    assert "dostuff setup" in result.output and result.exit_code == 1
    assert not (home / ".dostuff").exists()
