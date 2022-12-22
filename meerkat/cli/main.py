import json
import os
import shutil
import subprocess

import rich
import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from meerkat.interactive import start

cli = typer.Typer()


@cli.command()
def init(
    name: str = typer.Option(
        "meerkat_app",
        help="Name of the app",
    ),
):
    
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
    with Progress(
        SpinnerColumn(spinner_name="material"),
        TextColumn("[progress.description]{task.description}"),
        # transient=True,
    ) as progress:

        # Create a new Svelte app, but only output the stdout if there is an error
        progress.add_task(description="Setting up installer...", total=None)
        try:
            subprocess.run(
                ["curl https://bun.sh/install | sh"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
            )
            subprocess.run(
                ["bun", "add", "create-svelte@latest"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        os.makedirs("installer", exist_ok=True)

        with open("installer/installer.js", "w") as f:
            f.write(
                f"""\
import {{ create }} from 'create-svelte';

await create("../{name}", {{
    name: '{name}',
    template: 'skeleton',
    types: 'typescript',
    prettier: true,
    eslint: true,
    playwright: false,
    vitest: false
}});
"""
            )
        # Create a package.json for the installer
        with open("installer/package.json", "w") as f:
            json.dump(
                {
                    "name": "meerkat-installer",
                    "main": "installer.js",
                    "type": "module",
                    "scripts": {"install": "node installer.js"},
                    "dependencies": {"create-svelte": "latest"},
                },
                f,
            )

        # Run the installer to create the app
        progress.add_task(description="Creating app...", total=None)

        try:
            subprocess.run(
                # ["npm", "install"],
                ["bun", "install"],
                cwd="installer",
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Delete the installer
        subprocess.run(["rm", "-rf", "installer"])

        # Install dependencies for the new app
        progress.add_task(description="Installing dependencies...", total=None)

        # Update the package.json
        #   Add "@meerkat-ml/meerkat" to dependencies
        #   Add "tailwindcss" "postcss" "autoprefixer" to devDependencies
        with open(f"{name}/package.json") as f:
            package = json.load(f)

        if "dependencies" not in package:
            package["dependencies"] = {}
        package["dependencies"]["@meerkat-ml/meerkat"] = "latest"

        package["devDependencies"] = {
            **package["devDependencies"],
            **{
                "tailwindcss": "latest",
                "postcss": "latest",
                "autoprefixer": "latest",
            },
        }

        with open(f"{name}/package.json", "w") as f:
            json.dump(package, f)

        # Run the npm install for the new app
        try:
            subprocess.run(
                # ["npm", "install"],
                ["bun", "install"],
                cwd=name,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Install Tailwind
        progress.add_task(description="Getting tailwind...", total=None)
        try:
            subprocess.run(
                ["npx", "tailwindcss", "init", "tailwind.config.cjs", "-p"],
                cwd=name,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            rich.print(e.stdout.decode("utf-8"))
            raise e

        # Write svelte.config.js
        with open(f"{name}/svelte.config.js", "w") as f:
            f.write(
                f"""\
import adapter from '@sveltejs/adapter-auto';
import {{ vitePreprocess }} from '@sveltejs/kit/vite';

/** @type {{import('@sveltejs/kit').Config}} */
const config = {{
kit: {{
    adapter: adapter()
}},
preprocess: vitePreprocess()
}};

export default config;
"""
            )

        # Write tailwind.config.cjs
        with open(f"{name}/tailwind.config.cjs", "w") as f:
            f.write(
                f"""\
/** @type {{import('tailwindcss').Config}} */
module.exports = {{
content: ['./src/**/*.{{html,js,svelte,ts}}'],
theme: {{
    extend: {{}}
}},
plugins: []
}};
"""
            )

        # Add an app.css file inside the src directory
        with open(f"{name}/src/app.css", "w") as f:
            f.write(
                f"""\
@tailwind base;
@tailwind components;
@tailwind utilities;
"""
            )

        # Create "src/routes/+layout.svelte" file
        with open(f"{name}/src/routes/+layout.svelte", "w") as f:
            f.write(
                f"""\
<script>
    import "../app.css";
</script>

<slot />
"""
            )

        # Get path to favicon.png, at "../interactive/app/static/favicon.png"
        favicon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "interactive",
            "app",
            "static",
            "favicon.png",
        )

        # Copy favicon.png to the new app
        shutil.copy(favicon_path, f"{name}/static/favicon.png")

        # Create a "src/lib/components" directory
        os.makedirs(f"{name}/src/lib/components", exist_ok=True)

        # Create an ExampleComponent.svelte file
        with open(f"{name}/src/lib/components/ExampleComponent.svelte", "w") as f:
            f.write(
                f"""\
<script lang="ts">
    export let name: string = "World";
</script>

<h1 class="text-center text-xl underline bg-purple-200">Hello {{name}}!</h1>
"""
            )
        # Create an __init__.py file
        with open(f"{name}/src/lib/components/__init__.py", "w") as f:
            f.write(
                f"""\
import meerkat as mk

class ExampleComponent(mk.gui.Component):
    name: str = "World"
"""
            )
        
        # Create a test script example.py
        with open(f"example.py", "w") as f:
            f.write(
                f"""\
import meerkat as mk
from app.src.lib.components import ExampleComponent

# Import and use the ExampleComponent
example_component = ExampleComponent(name="Meerkat")

# Launch the Meerkat GUI
mk.gui.start()
mk.gui.Interface(component=example_component).launch()
""")
        # Add a config file called meerkat.config.yaml
        with open(f"meerkat.config.yaml", "w") as f:
            f.write(
                f"""\
# This is the Meerkat config file
APP_DIR: app/
""")
        
        
        # Rename the app folder to app
        os.rename(f"{name}", f"app")

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
    dev: bool = typer.Option(False, "--dev/--prod", help="Run in development mode"),
    shareable: bool = typer.Option(False, help="Run in public sharing mode"),
    port: int = typer.Option(5000, help="Meerkat API port"),
    
):
    """
    Launch a Meerkat app, given a path to a Python script.
    """
    
    rich.print(f"Running {script_path}")
    # start(shareable=shareable, api_port=port, dev=dev)
    # breakpoint()
    subprocess.run(["python", script_path])
    


@cli.command()
def install():
    rich.print("Installing Meerkat")


if __name__ == "__main__":
    cli()
