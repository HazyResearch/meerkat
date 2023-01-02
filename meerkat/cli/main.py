import importlib
import os
import subprocess
import sys

import rich
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from meerkat.interactive.interface import Interface
from meerkat.interactive.server import API_PORT, FRONTEND_PORT
from meerkat.interactive.startup import (
    get_subclasses_recursive,
    output_startup_message,
    run_frontend,
    run_script,
    to_py_module_name,
    wrap_all_components,
)
from meerkat.interactive.svelte import SvelteWriter
from meerkat.constants import APP_DIR
from meerkat.state import APIInfo, state

cli = typer.Typer()


@cli.command()
def init(
    name: str = typer.Option(
        "meerkat_app",
        help="Name of the app",
    ),
):
    """
    Create a new Meerkat app. This will create a new folder called `app` in
    the current directory and install all the necessary packages.

    Internally, Meerkat uses SvelteKit to create the app, and adds all the
    setup required by Meerkat to the app.
    """

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

    svelte_writer = SvelteWriter(appname=name, _appdir=os.path.join(os.getcwd(), "app"))

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

        # These need to be done first, .mk allows Meerkat
        # to recognize the app as a Meerkat app
        svelte_writer.write_libdir()  # src/lib
        svelte_writer.write_dot_mk()  # .mk file

        # Write an ExampleComponent.svelte and __init__.py file
        # and a script example.py that uses the component
        svelte_writer.write_example_component()
        svelte_writer.write_example_py()

        svelte_writer.write_app_css()  # app.css
        svelte_writer.write_constants_js()  # constants.js
        # TODO: add adapter static to svelte.config.js
        svelte_writer.write_svelte_config()  # svelte.config.js
        svelte_writer.write_tailwind_config()  # tailwind.config.cjs

        svelte_writer.import_app_components()
        svelte_writer.write_all_component_wrappers()  # src/lib/components/wrappers
        svelte_writer.write_component_context()  # ComponentContext.svelte
        svelte_writer.write_layout()  # +layout.svelte, layout.js
        svelte_writer.write_slug_route()  # [slug]/+page.svelte

        svelte_writer.write_gitignore()  # .gitignore
        svelte_writer.write_setup_py()  # setup.py

        svelte_writer.copy_banner_small()  # banner_small.png
        svelte_writer.copy_favicon()  # favicon.png
        # TODO add example_ipynb

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
    package_manager: str = typer.Option(
        "npm", show_choices=["npm", "bun"], help="Package manager to use"
    ),
    subdomain: str = typer.Option(
        "app", help="Subdomain to use for public sharing mode"
    ),
):
    """
    Launch a Meerkat app, given a path to a Python script.
    """
    # Pretty print information to console
    rich.print(
        f"[green][Log][/green] :rocket: Running [bold violet]{script_path}[/bold violet]"
    )
    if dev:
        rich.print(
            "[green][Log][/green] :wrench: Dev mode is [bold violet]on[/bold violet]"
        )
        rich.print(
            "[green][Log][/green] :hammer: Live reload is [bold violet]enabled[/bold violet]"
        )
    else:
        rich.print(
            "[green][Log][/green] :wrench: Production mode is [bold violet]on[/bold violet]"
        )
    rich.print()

    # Dump wrapper Component subclasses, ComponentContext
    svelte_writer = SvelteWriter()
    svelte_writer.import_app_components()
    svelte_writer.write_all_component_wrappers()
    svelte_writer.write_component_context()

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
    api_info = run_script(
        script_path,
        port=api_port,
        dev=dev,
        target=target,
        frontend_url=frontend_info.url,
        apiurl=dummy_api_info.url,
    )

    output_startup_message(frontend_info.url, api_info.docs_url)

    # Put it in the global state
    state.api_info = api_info
    state.frontend_info = frontend_info

    reload_index = 1
    while (api_info.process.poll() is None) or (frontend_info.process.poll() is None):
        api_stdout = api_info.process.stdout.readline().decode("utf-8")
        if "Reloading..." in api_stdout:
            rich.print(
                f"[purple][Reload #{reload_index}][/purple] {api_stdout.lstrip('WARNING:  ')}",
                end="",
            )
            reload_index += 1
        else:
            if api_stdout:
                rich.print(
                    f"[medium_purple1][Script][/medium_purple1] {api_stdout}", end=""
                )


@cli.command()
def update():
    """
    Update the Meerkat CLI to the latest version.
    """
    # Check if there's an app/ folder in the current directory
    if os.path.exists("app"):
        # Run `npm i @meerkat-ml/meerkat` in the app/ folder
        subprocess.run(["npm", "i", "@meerkat-ml/meerkat"], cwd="app")
        rich.print(":tada: Updated Meerkat npm package to the latest version!")
    else:
        rich.print(
            ":x: Could not find [purple]app[/purple] folder in the current directory."
        )


if __name__ == "__main__":
    cli()
