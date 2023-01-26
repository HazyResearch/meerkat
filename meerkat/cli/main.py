import os
import shutil
import subprocess
import time
from enum import Enum

import rich
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from meerkat.interactive.server import API_PORT, FRONTEND_PORT
from meerkat.interactive.startup import run_frontend, run_script
from meerkat.interactive.svelte import SvelteWriter
from meerkat.state import APIInfo, state

cli = typer.Typer()


class PackageManager(str, Enum):
    npm = "npm"
    bun = "bun"


@cli.command()
def init(
    name: str = typer.Option(
        "meerkat_app",
        help="Name of the app",
    ),
    package_manager: PackageManager = typer.Option(
        "npm", show_choices=True, help="Package manager to use"
    ),
):
    """Create a new Meerkat app. This will create a new folder called `app` in
    the current directory and install all the necessary packages.

    Internally, Meerkat uses SvelteKit to create the app, and adds all
    the setup required by Meerkat to the app.
    """
    # Unwrap the Enum
    package_manager = (
        package_manager.value
        if isinstance(package_manager, PackageManager)
        else package_manager
    )

    # Check if app folder exists, and tell the user to delete it if it does
    if os.path.exists("app"):
        rich.print(
            f"[red]An app folder already exists in this directory. "
            "Please delete it before running this command.[/red]"
        )
        raise typer.Exit(1)

    rich.print(
        f":seedling: Creating [purple]Meerkat[/purple] app: [green]{name}[/green]"
    )

    # Manually pass in _appdir to SvelteWriter, since we don't have an app yet
    # (we're creating it)
    svelte_writer = SvelteWriter(
        appname=name,
        _appdir=os.path.join(os.getcwd(), "app"),
        package_manager=package_manager,
    )

    with Progress(
        SpinnerColumn(spinner_name="material"),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        # Install prerequisites: package manager and create-svelte
        progress.add_task(description="Setting up installer...", total=None)
        try:
            svelte_writer.install_bun()
            svelte_writer.add_svelte()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Create an installer that will call create-svelte
        # Rather than directly calling create-svelte, we use an installer
        # to make it easier to add additional setup steps (programatically)
        # in the future
        svelte_writer.write_installer()

        # Run the installer to create the app
        progress.add_task(description="Creating app...", total=None)
        try:
            svelte_writer.install_packages(cwd="installer")
            svelte_writer.delete_installer()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Rename the app folder (`name`) to `app` before proceeding
        os.rename(name, "app")
        svelte_writer.update_package_json()

        # Install packages for the new app
        progress.add_task(description="Installing packages...", total=None)
        try:
            svelte_writer.install_packages()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Install TailwindCSS
        progress.add_task(description="Getting tailwind...", total=None)
        try:
            svelte_writer.install_tailwind()
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Configure and setup the app for Meerkat
        svelte_writer.create_app()

    # Pretty print information to console
    rich.print(f":tada: Created [purple]{name}[/purple]!")

    # Print instructions
    # 1. The new app is at name
    rich.print(f":arrow_right: The new app is at [purple]./app[/purple]")
    # 2. To see an example of how to make a component, see src/lib/components/ExampleComponent.svelte
    rich.print(
        ":arrow_right: To see an example of how to make a custom component, see "
        "[purple]./app/src/lib/components/ExampleComponent.svelte[/purple] "
        "and [purple]./app/src/lib/components/__init__.py[/purple]"
    )


@cli.command()
def run(
    script_path: str = typer.Argument(
        ..., help="Path to a Python script to run in Meerkat"
    ),
    dev: bool = typer.Option(True, "--dev/--prod", help="Run in development mode"),
    shareable: bool = typer.Option(False, help="Run in public sharing mode"),
    api_port: int = typer.Option(API_PORT, help="Meerkat API port"),
    frontend_port: int = typer.Option(FRONTEND_PORT, help="Meerkat frontend port"),
    target: str = typer.Option("interface", help="Target to run in script"),
    package_manager: PackageManager = typer.Option(
        "npm", show_choices=True, help="Package manager to use"
    ),
    subdomain: str = typer.Option(
        "app", help="Subdomain to use for public sharing mode"
    ),
    debug: bool = typer.Option(False, help="Enable debug logging mode"),
):
    """Launch a Meerkat app, given a path to a Python script."""
    package_manager = (
        package_manager.value
        if isinstance(package_manager, PackageManager)
        else package_manager
    )
    _run(
        script_path=script_path,
        dev=dev,
        shareable=shareable,
        api_port=api_port,
        frontend_port=frontend_port,
        target=target,
        package_manager=package_manager,
        subdomain=subdomain,
        debug=debug,
    )


def _run(
    script_path: str,
    dev: bool = True,
    shareable: bool = False,
    api_port: int = 8000,
    frontend_port: int = 3000,
    target: str = "interface",
    package_manager: str = "npm",
    subdomain: str = "app",
    debug: bool = False,
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
    rich.print(f":x: To stop the app, press [bold violet]Ctrl+C[/bold violet]")
    rich.print()

    # Dump wrapper Component subclasses, ComponentContext
    svelte_writer = SvelteWriter()
    svelte_writer.init_run()

    # Run the frontend
    # TODO: make the dummy API info take in the actual hostname
    dummy_api_info = APIInfo(api=None, port=api_port, name="127.0.0.1")
    frontend_info = run_frontend(
        package_manager,
        frontend_port,
        dev,
        shareable,
        subdomain,
        dummy_api_info.url,
        svelte_writer.appdir,
    )

    # Run the uvicorn server
    # Set the logging level to debug if debug is enabled
    api_info = run_script(
        script_path,
        port=api_port,
        dev=dev,
        target=target,
        frontend_url=frontend_info.url,
        apiurl=dummy_api_info.url,
        debug=debug,
    )

    # output_startup_message(frontend_info.url, api_info.docs_url)

    # Put it in the global state
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
    """Update the Meerkat CLI to the latest version."""
    # Check if there's an app/ folder in the current directory
    if os.path.exists("app"):
        # Run `npm i @meerkat-ml/meerkat` in the app/ folder
        subprocess.run(["npm", "i", "@meerkat-ml/meerkat"], cwd="app")
        rich.print(":tada: Updated Meerkat npm package to the latest version!")
    else:
        rich.print(
            ":x: Could not find [purple]app[/purple] folder in the current directory."
        )


@cli.command()
def install(
    package_manager: str = typer.Option(
        "npm", show_choices=["npm", "bun"], help="Package manager to use"
    ),
    run_dev: bool = typer.Option(True, help="Run `npm run dev` after installation"),
):
    """Install npm and other dependencies for interactive Meerkat."""
    svelte_writer = SvelteWriter(
        package_manager=package_manager,
    )
    svelte_writer.install_node()
    svelte_writer.install_mk_app()
    if run_dev:
        svelte_writer.npm_run_dev()


_DEMO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "demo")

DemoScript = Enum(
    "DemoScript",
    {
        k: k
        for k in [x.split(".py")[0] for x in os.listdir(_DEMO_DIR) if x.endswith(".py")]
    },
)


@cli.command()
def demo(
    script: DemoScript = typer.Argument(
        ..., show_choices=True, help="Demo script to run"
    ),
    run: bool = typer.Option(True, help="Run the demo script"),
    api_port: int = typer.Option(API_PORT, help="Meerkat API port"),
    frontend_port: int = typer.Option(FRONTEND_PORT, help="Meerkat frontend port"),
    copy: bool = typer.Option(
        False, help="Copy the demo script to the current directory"
    ),
):
    """Run a demo script."""
    # Get the path to the demo script
    script = script.value
    script_path = os.path.join(_DEMO_DIR, f"{script}.py")

    # Optional: Copy the demo script to the current directory.
    if copy:
        shutil.copy(script_path, f"./{script}.py")
        rich.print(f"Copied [purple]{script}.py[/purple] to the current directory.")
        script_path = f"{script}.py"

    # Optional: Run the demo script.
    if run:
        _run(script_path=script_path, api_port=api_port, frontend_port=frontend_port)


if __name__ == "__main__":
    cli()
