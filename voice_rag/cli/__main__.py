import click
from voice_rag.cli.commands import init_cmd, ingest_cmd, serve_cmd, inspect_cmd, doctor_cmd, query_cmd


@click.group()
@click.version_option(package_name="voice-rag")
def cli():
    """voice-rag: Provider-agnostic voice RAG pipeline."""
    pass


cli.add_command(init_cmd, "init")
cli.add_command(ingest_cmd, "ingest")
cli.add_command(serve_cmd, "serve")
cli.add_command(inspect_cmd, "inspect")
cli.add_command(doctor_cmd, "doctor")
cli.add_command(query_cmd, "query")


if __name__ == "__main__":
    cli()
