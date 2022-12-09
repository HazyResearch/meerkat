import subprocess
import typer

from meerkat.interactive.startup import start

app = typer.Typer()


@app.command()
def init():
    typer.echo("Initializing Meerkat")


@app.command()
def run(
    script_path: str,
    dev: bool = False,
    shareable: bool = False,
    port: int = 5000,
):
    typer.echo(f"Running {script_path}")
    network_info = start(shareable=shareable, api_port=port, dev=dev)
    typer.echo(f"dev: {dev}")
    typer.echo(f"shareable: {shareable}")
    typer.echo(f"port: {port}")
    subprocess.run(["python", script_path])


@app.command()
def install():
    typer.echo("Installing Meerkat")


if __name__ == "__main__":
    app()
