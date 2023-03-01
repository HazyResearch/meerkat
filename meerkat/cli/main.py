import os
import shutil
import subprocess
import time
from enum import Enum

import rich
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from meerkat.constants import (
    MEERKAT_DEMO_DIR,
    MEERKAT_INTERNAL_APP_DIR,
    MEERKAT_NPM_PACKAGE,
    App,
    MeerkatApp,
    PackageManager,
    PathHelper,
    SystemHelper,
)
from meerkat.interactive.server import API_PORT, FRONTEND_PORT
from meerkat.interactive.startup import run_frontend, run_script
from meerkat.state import APIInfo, state
from meerkat.tools.collect_env import collect_env_info

cli = typer.Typer()


def _unwrap_enum(value):
    """Unwrap an Enum value if it's an Enum, otherwise return the value."""
    return value.value if isinstance(value, Enum) else value


@cli.command()
def init(
    name: str = typer.Option(
        "meerkat_app",
        help="Name of the app",
    ),
    package_manager: PackageManager = typer.Option(
        "npm",
        show_choices=True,
        help="Package manager to use",
    ),
):
    """Create a new Meerkat app. This will create a new folder called `app` in
    the current directory and install all the necessary packages.

    Internally, Meerkat uses SvelteKit to create the app, and adds all
    the setup required by Meerkat to the app.
    """
    # This is a no-op, but it's here for clarity.
    os.chdir(PathHelper().rundir)

    # Create a MeerkatApp object to represent the app
    package_manager = _unwrap_enum(package_manager)
    app = MeerkatApp(
        appdir=PathHelper().appdir,
        appname=name,
        package_manager=package_manager,
    )

    if app.exists():
        # Check if app exists, and tell the user to delete it if it does
        rich.print(
            f"[red]An app already exists at {app.appdir}. "
            "Please delete it and rerun this command.[/red]"
        )
        raise typer.Exit(1)

    rich.print(
        f":seedling: Creating [purple]Meerkat[/purple] app: [green]{name}[/green]"
    )

    with Progress(
        SpinnerColumn(spinner_name="material"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        # Install prerequisites: package manager
        progress.add_task(description="Installing system prerequisites...", total=None)
        try:
            if package_manager == "bun":
                # Install the bun package manager
                SystemHelper().install_bun()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        progress.add_task(description="Creating app...", total=None)
        try:
            # Create the Meerkat app.
            app.create()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        progress.add_task(description="Installing packages...", total=None)
        try:
            # Install packages in the new app.
            app.setup_mk_dependencies()
            app.setup_mk_build_command()
            app.install()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        progress.add_task(description="Getting tailwind...", total=None)
        try:
            # Install TailwindCSS.
            app.install_tailwind()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Final setup for the app.
        app.setup()

    # Print a message.
    app.print_finish_message()


@cli.command()
def run(
    script_path: str = typer.Argument(
        ..., help="Path to a Python script to run in Meerkat"
    ),
    dev: bool = typer.Option(False, "--dev/--prod", help="Run in development mode"),
    api_port: int = typer.Option(API_PORT, help="Meerkat API port"),
    frontend_port: int = typer.Option(FRONTEND_PORT, help="Meerkat frontend port"),
    host: str = typer.Option("127.0.0.1", help="Host to run on"),
    target: str = typer.Option("page", help="Target to run in script"),
    package_manager: PackageManager = typer.Option(
        "npm", show_choices=True, help="Package manager to use"
    ),
    shareable: bool = typer.Option(False, help="Run in public sharing mode"),
    subdomain: str = typer.Option(
        "app", help="Subdomain to use for public sharing mode"
    ),
    debug: bool = typer.Option(False, help="Enable debug logging mode"),
    skip_build: bool = typer.Option(True, help="Skip building the app."),
):
    """Launch a Meerkat app, given a path to a Python script."""
    _run(
        script_path=script_path,
        dev=dev,
        host=host,
        api_port=api_port,
        frontend_port=frontend_port,
        target=target,
        package_manager=_unwrap_enum(package_manager),
        shareable=shareable,
        subdomain=subdomain,
        debug=debug,
        skip_build=skip_build,
    )


def _run(
    script_path: str,
    dev: bool = False,
    host: str = "127.0.0.1",
    api_port: int = API_PORT,
    frontend_port: int = FRONTEND_PORT,
    target: str = "page",
    package_manager: PackageManager = PackageManager.npm,
    shareable: bool = False,
    subdomain: str = "app",
    debug: bool = False,
    skip_build: bool = True,
):
    # Pretty print information to console
    rich.print(f":rocket: Running [bold violet]{script_path}[/bold violet]")
    if dev:
        rich.print(
            ":wrench: Dev mode is [bold violet]on[/bold violet]\n"
            ":hammer: Live reload is [bold violet]enabled[/bold violet]"
        )
    else:
        rich.print(":wrench: Production mode is [bold violet]on[/bold violet]")
    rich.print(":x: To stop the app, press [bold violet]Ctrl+C[/bold violet]")
    rich.print()

    # Run the frontend
    # TODO: make the dummy API info take in the actual hostname
    dummy_api_info = APIInfo(api=None, port=api_port, name="127.0.0.1")
    frontend_info = run_frontend(
        package_manager=package_manager,
        port=frontend_port,
        dev=dev,
        shareable=shareable,
        subdomain=subdomain,
        apiurl=dummy_api_info.url,
        appdir=PathHelper().appdir,
        skip_build=skip_build,
    )

    # Run the uvicorn server
    # Set the logging level to debug if debug is enabled
    api_info = run_script(
        script_path,
        server_name=host,
        port=api_port,
        dev=dev,
        target=target,
        frontend_url=frontend_info.url,
        apiurl=dummy_api_info.url,
        debug=debug,
    )

    # Put them into the global state so the exit handler can use them to clean up
    # the processes when the user exits this script.
    state.api_info = api_info
    state.frontend_info = frontend_info

    while (api_info.process.poll() is None) or (frontend_info.process.poll() is None):
        # Exit on Ctrl+C
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            rich.print()
            break


@cli.command()
def update():
    """Update the Meerkat npm package to the latest version."""
    # Check if there's an app/ folder in the current directory
    if os.path.exists("app"):
        # Run `npm i MEERKAT_NPM_PACKAGE` in the app/ folder
        subprocess.run(["npm", "i", MEERKAT_NPM_PACKAGE], cwd="app")
        rich.print(":tada: Updated Meerkat npm package to the latest version!")
    else:
        rich.print(
            ":x: Could not find [purple]app[/purple] folder in the current directory."
        )


@cli.command()
def install(
    package_manager: PackageManager = typer.Option(
        "npm", show_choices=True, help="Package manager to use"
    ),
    run_dev: bool = typer.Option(False, help="Run `npm run dev` after installation"),
):
    """Install npm and other dependencies for interactive Meerkat."""
    SystemHelper().install_node()
    package_manager = _unwrap_enum(package_manager)
    app = App(appdir=MEERKAT_INTERNAL_APP_DIR, package_manager=package_manager)
    app.install()
    if run_dev:
        app.run_dev()


DemoScript = Enum(
    "DemoScript",
    {
        k: k
        for k in [
            x.split(".py")[0] for x in os.listdir(MEERKAT_DEMO_DIR) if x.endswith(".py")
        ]
    }
    if MEERKAT_DEMO_DIR
    else {},
)


@cli.command()
def demo(
    script: DemoScript = typer.Argument(
        ..., show_choices=True, help="Demo script to run"
    ),
    run: bool = typer.Option(True, help="Run the demo script"),
    api_port: int = typer.Option(API_PORT, help="Meerkat API port"),
    frontend_port: int = typer.Option(FRONTEND_PORT, help="Meerkat frontend port"),
    dev: bool = typer.Option(False, "--dev/--prod", help="Run in development mode"),
    copy: bool = typer.Option(
        False, help="Copy the demo script to the current directory"
    ),
    debug: bool = typer.Option(False, help="Enable debug logging mode"),
):
    """Run a demo script."""
    # Get the path to the demo script
    script = script.value
    script_path = os.path.join(MEERKAT_DEMO_DIR, f"{script}.py")

    # Optional: Copy the demo script to the current directory.
    if copy:
        shutil.copy(script_path, f"./{script}.py")
        rich.print(f"Copied [purple]{script}.py[/purple] to the current directory.")
        script_path = f"{script}.py"

    # Optional: Run the demo script.
    if run:
        _run(
            script_path=script_path,
            api_port=api_port,
            frontend_port=frontend_port,
            dev=dev,
            debug=debug,
        )


@cli.command()
def collect_env():
    print(collect_env_info())


if __name__ == "__main__":
    cli()
