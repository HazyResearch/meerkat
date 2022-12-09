import typer

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
    typer.echo(f"dev: {dev}")
    typer.echo(f"shareable: {shareable}")
    typer.echo(f"port: {port}")


@app.command()
def install():
    typer.echo("Installing Meerkat")


if __name__ == "__main__":
    app()
