from click.testing import CliRunner
from voice_rag.cli.__main__ import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "voice-rag" in result.output.lower() or "usage" in result.output.lower()


def test_cli_doctor_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["doctor"])
    assert result.exit_code == 0


def test_cli_init_creates_config(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    config_file = tmp_path / "voice-rag.yaml"
    assert config_file.exists()


def test_pyproject_has_elevenlabs_extra():
    """Ensure elevenlabs is a declared optional dependency."""
    import importlib.metadata as meta
    extras = meta.requires("voice-rag") or []
    elevenlabs_deps = [d for d in extras if "elevenlabs" in d.lower()]
    assert elevenlabs_deps, "Expected 'elevenlabs' in optional dependencies"


from unittest.mock import MagicMock, patch

def test_ingest_shows_total_and_calls_agent(tmp_path):
    """ingest_cmd should call agent.ingest() and print total chunk count."""
    runner = CliRunner()
    with patch("voice_rag.cli.commands.KnowledgeAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.ingest.return_value = 42
        mock_agent_cls.return_value = mock_agent

        result = runner.invoke(cli, ["ingest", str(tmp_path)])

        assert result.exit_code == 0
        assert "Total: 42 chunks" in result.output
        mock_agent.ingest.assert_called_once_with(str(tmp_path), recreate=False)


def test_serve_has_reload_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["serve", "--help"])
    assert "--reload" in result.output

def test_serve_banner_uses_localhost_for_0000():
    """When host is 0.0.0.0 the banner should show localhost."""
    from voice_rag.cli.commands import _format_serve_url
    assert _format_serve_url("0.0.0.0", 8000) == "http://localhost:8000/v1"
    assert _format_serve_url("192.168.1.1", 8000) == "http://192.168.1.1:8000/v1"


def test_query_command_exists():
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])
    assert result.exit_code == 0
    assert "--config" in result.output
    assert "--limit" in result.output

def test_query_command_registered():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert "query" in result.output
