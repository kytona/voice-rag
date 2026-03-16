from click.testing import CliRunner
from unittest.mock import MagicMock, patch

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
    assert "local_path" in config_file.read_text()


def test_pyproject_has_elevenlabs_extra():
    """Ensure elevenlabs is a declared optional dependency."""
    import importlib.metadata as meta
    extras = meta.requires("voice-rag") or []
    elevenlabs_deps = [d for d in extras if "elevenlabs" in d.lower()]
    assert elevenlabs_deps, "Expected 'elevenlabs' in optional dependencies"

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


def test_doctor_reports_local_embedded_qdrant(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "voice-rag.yaml"
    config_path.write_text("vector_store:\n  local_path: .qdrant\n  url: \"\"\n")

    result = runner.invoke(cli, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "local embedded Qdrant at .qdrant" in result.output
    assert f"Config file found: {config_path}" in result.output


def test_doctor_checks_remote_qdrant_when_url_is_set(tmp_path):
    runner = CliRunner()
    config_path = tmp_path / "voice-rag.yaml"
    config_path.write_text("vector_store:\n  url: http://example.com:6333\n")

    with patch("qdrant_client.QdrantClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client_cls.return_value = mock_client

        result = runner.invoke(cli, ["doctor", "--config", str(config_path)])

    assert result.exit_code == 0
    assert "remote Qdrant at http://example.com:6333" in result.output
    assert "Qdrant reachable at http://example.com:6333" in result.output
