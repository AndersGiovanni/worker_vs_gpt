"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Worker vs. GPT."""


if __name__ == "__main__":
    main(prog_name="worker_vs_gpt")  # pragma: no cover
