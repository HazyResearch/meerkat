import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

import rich
from jinja2 import Environment, FileSystemLoader
from tabulate import tabulate

from meerkat.tools.singleton import Singleton

if TYPE_CHECKING:
    import nbformat as nbf

logger = logging.getLogger(__name__)


# This file is meerkat/meerkat/constants.py
# Assert that the path to this file ends with "meerkat/meerkat/constants.py"
assert os.path.abspath(__file__).endswith("meerkat/constants.py"), (
    "This file should end with 'meerkat/meerkat/constants.py'. "
    f"Got {__file__}. "
    "If it was moved, update the assert"
    " and the path to the MEERKAT_BASE_DIR below."
)

# Base directory is meerkat/ (two levels up)
MEERKAT_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the (internal to the Python package) meerkat/interactive/app directory.
MEERKAT_INTERNAL_APP_DIR = os.path.join(
    MEERKAT_BASE_DIR, "meerkat", "interactive", "app"
)

# Path to the (internal to Python package) meerkat/interactive/app/build directory.
MEERKAT_INTERNAL_APP_BUILD_DIR = os.path.join(MEERKAT_INTERNAL_APP_DIR, "build")

# Environment variable to set the path to an app directory.
# See PathHelper.appdir for more details.
MEERKAT_APP_DIR = os.environ.get("MEERKAT_APP_DIR", None)

# Name of the Meerkat npm package.
MEERKAT_NPM_PACKAGE = "@meerkat-ml/meerkat"

# Path to the meerkat/interactive/templates directory.
MEERKAT_TEMPLATES_DIR = os.path.join(
    MEERKAT_BASE_DIR, "meerkat", "interactive", "templates"
)

# Meerkat demo directory.
MEERKAT_DEMO_DIR = os.path.join(MEERKAT_BASE_DIR, "meerkat-demo")
if not os.path.exists(MEERKAT_DEMO_DIR):  # pypi install
    MEERKAT_DEMO_DIR = os.path.join(MEERKAT_BASE_DIR, "demo")
if not os.path.exists(MEERKAT_DEMO_DIR):  # repo install
    MEERKAT_DEMO_DIR = None

# Environment variables that should primarily be used inside the
# subprocess run by `mk run` (i.e. the subprocess that is used to
# run the script that is passed to `mk run`).

# A flag to indicate whether we are running in a subprocess of `mk run`.
MEERKAT_RUN_SUBPROCESS = int(os.environ.get("MEERKAT_RUN_SUBPROCESS", 0))

# The `script_path` that was passed to `mk run`, if we are running in a
# subprocess of `mk run`.
MEERKAT_RUN_SCRIPT_PATH = os.environ.get("MEERKAT_RUN_SCRIPT_PATH", None)

# A unique ID for the current run, available only in the subprocess of `mk run`.
MEERKAT_RUN_ID = os.environ.get("MEERKAT_RUN_ID", uuid.uuid4().hex)

# A flag to indicate whether we are in a `mk` CLI script.
MEERKAT_CLI_PROCESS = os.path.basename(sys.argv[0]) == "mk"

# A flag to indicate whether we are specifically in the `mk run` script.
MEERKAT_RUN_PROCESS = (os.path.basename(sys.argv[0]) == "mk") and (
    len(sys.argv) > 1 and (sys.argv[1] == "run" or sys.argv[1] == "demo")
)

# A flag to indicate whether we are specifically in the `mk init` script.
MEERKAT_INIT_PROCESS = (os.path.basename(sys.argv[0]) == "mk") and (
    len(sys.argv) > 1 and sys.argv[1] == "init"
)

# Create a Jinja2 environment.
JINJA_ENV = Environment(loader=FileSystemLoader(MEERKAT_TEMPLATES_DIR))


def write_file(path: str, content: str) -> None:
    with open(path, "w") as f:
        f.write(content)


def is_notebook() -> bool:
    """Check if the current environment is a notebook.

        Taken from
        https://stackoverflow.com/questions/15411967/how-can-i-check-if-code\
    -is-executed-in-the-ipython-notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class PackageManager(str, Enum):
    npm = "npm"
    bun = "bun"


class PathHelper(metaclass=Singleton):
    """Information about important paths."""

    def __init__(self) -> None:
        # Cache properties.
        self._rundir = None
        self._scriptpath = None
        self._appdir = None

    @property
    def rundir(self):
        """The directory from which the current script was run.

        This is the directory from which the `python <script>` or `mk`
        command was run.
        """
        if self._rundir is None:
            self._rundir = os.getcwd()
        return self._rundir

    @property
    def scriptpath(self):
        """The path to the current script being run.

        This is the path to the script as it is passed to `python
        <script>`. This will return the correct path even if the script
        is run with `mk run`.
        """
        if self._scriptpath is None:
            if MEERKAT_RUN_PROCESS:
                # If we are in the `mk run` process, then we can use sys.argv[2].
                # This is the path to the script as it is passed to `mk run`.
                self._scriptpath = sys.argv[2]
            elif MEERKAT_RUN_SUBPROCESS:
                # If we are running in a subprocess of `mk run`, then we can't use
                # sys.argv[0] because it will be the path to the `mk` script, not
                # the path to the script that was passed to `mk run`.
                # Instead, we use the MEERKAT_RUN_SCRIPT_PATH environment variable.
                assert (
                    MEERKAT_RUN_SCRIPT_PATH is not None
                ), "Something is wrong. MEERKAT_RUN_SCRIPT_PATH should be set"
                " by `mk run`."
                self._scriptpath = MEERKAT_RUN_SCRIPT_PATH
            elif is_notebook():
                # If we are running inside a notebook, then we return None.
                self._scriptpath = None
            else:
                # If we are not running in a subprocess of `mk run`, then we
                # can use sys.argv[0]. This is the path to the script as it is
                # passed to `python <script>`.
                self._scriptpath = sys.argv[0]

        return self._scriptpath

    @property
    def scriptpath_abs(self):
        """The absolute path to the current script.

        See `scriptpath` for more information.
        """
        if is_notebook():
            return None
        return os.path.abspath(self.scriptpath)

    @property
    def scriptdir_abs(self):
        """The directory containing the current script.

        See `scriptpath` for more information.
        """
        if is_notebook():
            return None
        return os.path.abspath(os.path.dirname(self.scriptpath_abs))

    @property
    def scriptname(self):
        """The name of the current script."""
        if is_notebook():
            return None
        return os.path.basename(self.scriptpath_abs)

    @property
    def appdir(self):
        """The absolute path to the app/ directory.

        Rules for determining the app directory:

        1. By default, the app directory points to the internal app directory,
            which is MEERKAT_INTERNAL_APP_DIR.
        2. This is not the case if the script or notebook that is being run is
            next to an app/ directory. In this case, Meerkat should infer that
            this directory should be used as the app directory.
        3. This is also not the case if the user explicitly sets the app directory.
            Either by
                passing in a flag to `mk run` on the CLI, or
                passing in an argument to `mk.gui.start` in a Python script
                or notebook, or setting the MEERKAT_APP_DIR environment variable.
        4. Finally, none of these rules apply if the user is running the `mk init`
            script. In this case, the app directory will be created, and we should
            point to this location.
        """
        if self._appdir is not None:
            return self._appdir

        # If the user is running the `mk init` script, then the app directory
        # will be created.
        if MEERKAT_INIT_PROCESS:
            # `rundir` is the directory from which the `mk` script was run.
            self._appdir = os.path.join(PathHelper().rundir, "app")
            return self._appdir

        # Set the default app directory.
        self._appdir = MEERKAT_INTERNAL_APP_DIR

        if is_notebook():
            # If we are running inside a notebook, and the notebook is next to
            # an app/ directory, then use that as the app directory.
            candidate_path = os.path.join(os.getcwd(), "app")
            if os.path.exists(candidate_path):
                # The notebook is next to an app/ directory. Point to this directory.
                self._appdir = candidate_path
            return self._appdir

        # If the script is next to an app/ directory, then use that
        # as the app directory.
        candidate_path = os.path.join(PathHelper().scriptdir_abs, "app")
        if os.path.exists(candidate_path):
            # The script is next to an app/ directory. Point to this directory.
            self._appdir = candidate_path

        # If the user has explicitly set the app directory, then use
        # that as the app directory.
        if MEERKAT_APP_DIR is not None:
            # The user has explicitly set the app directory.
            # Point to this directory.
            self._appdir = MEERKAT_APP_DIR

        # Use the absolute path.
        self._appdir = os.path.abspath(self._appdir)

        return self._appdir

    @property
    def is_user_app(self) -> bool:
        """Returns True if the app directory being used does not point to the
        internal Meerkat app directory."""
        return self.appdir != MEERKAT_INTERNAL_APP_DIR

    def __repr__(self) -> str:
        return f"""\
{self.__class__.__name__}(
    rundir={self.rundir},
    scriptpath={self.scriptpath},
    scriptpath_abs={self.scriptpath_abs},
    scriptdir_abs={self.scriptdir_abs},
    scriptname={self.scriptname},
    appdir={self.appdir},
)\
"""


# A counter to indicate how many times the script has been reloaded.
if os.path.exists(f"{PathHelper().appdir}/.{MEERKAT_RUN_ID}.reload"):
    MEERKAT_RUN_RELOAD_COUNT = int(
        open(f"{PathHelper().appdir}/.{MEERKAT_RUN_ID}.reload", "r").read()
    )
else:
    MEERKAT_RUN_RELOAD_COUNT = 0


class App:
    def __init__(
        self,
        appdir: str,
        appname: Optional[str] = None,
        package_manager: PackageManager = "npm",
    ):
        self.appdir = os.path.abspath(appdir)
        self.appname = appname
        self.package_manager = package_manager

    def set_package_manager(self, package_manager: PackageManager):
        self.package_manager = package_manager

    @property
    def is_user_app(self) -> bool:
        """Returns True if the app directory being used does not point to the
        internal Meerkat app directory."""
        return self.appdir != MEERKAT_INTERNAL_APP_DIR

    def create(self):
        """Create the app directory."""
        os.makedirs(self.appdir, exist_ok=True)

    def exists(self):
        """Check if the app directory exists."""
        return os.path.exists(self.appdir)

    def filter_installed_libraries(self, libraries: List[str]) -> List[str]:
        """Given a list of libraries, return the libraries that are installed
        in the app directory.

        Args:
            libraries (List[str]): List of libraries to check

        Returns:
            List[str]: List of libraries that are installed
        """
        # Check in node_modules
        installed_libraries = []
        for library in libraries:
            if os.path.exists(os.path.join(self.appdir, "node_modules", library)):
                installed_libraries.append(library)
        logger.debug(f"Installed libraries: {installed_libraries}")
        return installed_libraries

        # KG: This is the correct way to check if a library is installed, but
        # it takes around 0.4s, which is too slow for it to be done on every
        # import. Leave this code here for now, but we should find a way to
        # use it in the future (or perhaps use package-lock.json). (TODO)
        p = subprocess.run(
            ["npm", "list"] + list(libraries) + ["--parseable", "--silent"],
            cwd=self.appdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = p.stdout.decode("utf-8").splitlines()
        # output consists of a list of paths to the installed package and
        # omits libraries that are not installed
        installed_libraries = [p for p in libraries for o in output if p in o]
        return installed_libraries

    def get_mk_package_info(self) -> List[str]:
        """Get the list of components available in the (currently) installed
        Meerkat package. This is used to exclude components that cannot be used
        in the app, specifically when writing ComponentContext.svelte.

        Uses a heuristic that goes through the index.js file of the
        Meerkat package and extracts components with a regex. It's not a
        problem if extra imports (that are not components) are included
        in this list, as long as all components are included.
        """
        package_path = os.path.join(self.appdir, "node_modules", MEERKAT_NPM_PACKAGE)
        index_js_path = os.path.join(package_path, "index.js")
        with open(index_js_path, "r") as f:
            index_js = f.read()

        components = re.findall(
            r"export \{ default as (\w+) \}",
            index_js,
        )
        # Exclude commented out components.
        remove_components = re.findall(
            r"(?!\/\/) export \{ default as (\w+) \}",
            index_js,
        )
        components = [c for c in components if c not in remove_components]
        return components

    def install(self, dev=False):
        """Run e.g. `npm install` on an app directory."""
        return subprocess.run(
            [self.package_manager, "install"] + (["--dev"] if dev else []),
            cwd=self.appdir,
            check=True,
            # TODO: check that this is how we should handle outputs.
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def install_tailwind(self):
        return subprocess.run(
            ["npx", "tailwindcss", "init", "tailwind.config.cjs", "-p"],
            cwd=self.appdir,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def pm_run(self, command: str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT):
        """Run e.g. `npm run <command>` on the app directory."""
        return subprocess.run(
            [self.package_manager, "run", command],
            cwd=self.appdir,
            check=True,
            stdout=stdout,
            stderr=stderr,
        )

    def run_dev(self):
        """Run e.g. `npm run dev` on the internal Meerkat app directory."""
        # TODO: check that this is how we should handle stdout and stderr.
        return self.pm_run("dev", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def run_build(self):
        """Run e.g. `npm run build` on the internal Meerkat app directory."""
        return self.pm_run("build")

    def upgrade_meerkat(self):
        """Run e.g. `npm i MEERKAT_NPM_PACKAGE` in the app."""
        subprocess.run(
            [self.package_manager, "i", MEERKAT_NPM_PACKAGE], cwd=self.appdir
        )

    def update_build_command(self, command: str):
        """Update the build command in package.json."""
        with open(os.path.join(self.appdir, "package.json")) as f:
            package = json.load(f)

        package["scripts"]["build"] = command

        with open(os.path.join(self.appdir, "package.json"), "w") as f:
            # Format the JSON nicely.
            json.dump(package, f, indent=4)

    def update_dependencies(self, deps: dict, dev=False):
        """Update the dependencies in package.json."""
        with open(os.path.join(self.appdir, "package.json")) as f:
            package = json.load(f)

        if dev:
            if "devDependencies" not in package:
                package["devDependencies"] = {}
            package["devDependencies"] = {
                **package["devDependencies"],
                **deps,
            }
        else:
            if "dependencies" not in package:
                package["dependencies"] = {}
            package["dependencies"] = {
                **package["dependencies"],
                **deps,
            }

        with open(os.path.join(self.appdir, "package.json"), "w") as f:
            # Format the JSON nicely.
            json.dump(package, f, indent=4)

    def write_file(self, file: str, content: str):
        """Write a file to the app directory.

        Uses a relative path to the file.
        """
        write_file(os.path.join(self.appdir, file), content)


class MeerkatApp(App):
    def create(self):
        """Run an installer that will call create-svelte in order to create a
        new app."""
        installer_app = CreateSvelteInstallerApp(
            appdir="./installer",
            appname=self.appname,
            package_manager=self.package_manager,
        )
        installer_app.create()
        installer_app.install()
        installer_app.delete()

    def print_finish_message(self):
        # Pretty print information to console
        rich.print(f":tada: Created [purple]{self.appname}[/purple]!")

        # Print instructions
        # 1. The new app is at name
        rich.print(":arrow_right: The new app is at [purple]./app[/purple]")
        # 2. To see an example of how to make a component,
        # see src/lib/components/ExampleComponent.svelte
        rich.print(
            ":arrow_right: To see an example of how to make a custom component, see "
            "[purple]./app/src/lib/components/ExampleComponent.svelte[/purple] "
            "and [purple]./app/src/lib/components/__init__.py[/purple]"
        )

    def setup(self):
        # These need to be done first, .mk allows Meerkat
        # to recognize the app as a Meerkat app
        self.write_libdir()  # src/lib
        self.write_dot_mk()  # .mk file

        # Write an ExampleComponent.svelte and __init__.py file
        # and a script example.py that uses the component
        self.write_example_component()
        self.write_example_py()
        self.write_example_ipynb()

        self.write_app_css()  # app.css
        self.write_constants_js()  # constants.js
        self.write_svelte_config()  # svelte.config.js
        self.write_tailwind_config()  # tailwind.config.cjs

        self.write_layout()  # +layout.svelte, layout.js
        self.write_root_route_alternate()  # +page.svelte
        self.write_slug_route()  # [slug]/+page.svelte

        self.write_gitignore()  # .gitignore
        self.write_setup_py()  # setup.py

        self.copy_assets()

    def setup_mk_build_command(self):
        """Update the build command in package.json."""
        self.update_build_command(
            "VITE_API_URL_PLACEHOLDER=http://meerkat.dummy vite build"
        )

    def setup_mk_dependencies(self):
        self.update_dependencies(
            deps={
                # TODO: use the actual version number.
                MEERKAT_NPM_PACKAGE: "latest",
            },
            dev=False,
        )
        self.update_dependencies(
            deps={
                "tailwindcss": "latest",
                "postcss": "latest",
                "autoprefixer": "latest",
                "@tailwindcss/typography": "latest",
                "@sveltejs/adapter-static": "latest",
            },
            dev=True,
        )

    def copy_assets(self):
        """Copy assets from the Meerkat app to the new app."""
        self.copy_banner_small()  # banner_small.png
        self.copy_favicon()  # favicon.png

    def copy_banner_small(self):
        """Copy the Meerkat banner to the new app."""
        banner_path = os.path.join(
            MEERKAT_INTERNAL_APP_DIR,
            "src",
            "lib",
            "assets",
            "banner_small.png",
        )

        # Copy banner to the new app
        dir = f"{self.appdir}/src/lib/assets"
        os.makedirs(dir, exist_ok=True)
        shutil.copy(banner_path, f"{dir}/banner_small.png")

    def copy_favicon(self):
        """Copy the Meerkat favicon to the new app."""
        favicon_path = os.path.join(
            MEERKAT_INTERNAL_APP_DIR,
            "static",
            "favicon.png",
        )

        # Copy favicon.png to the new app
        shutil.copy(favicon_path, f"{self.appdir}/static/favicon.png")

    def render_example_ipynb(self) -> "nbf.NotebookNode":
        import nbformat as nbf

        nb = nbf.v4.new_notebook()
        text = """# Interactive Notebook Example"""

        code_1 = """\
import meerkat as mk
from app.src.lib.components import ExampleComponent

# Launch the Meerkat GUI servers
mk.gui.start(api_port=5000, frontend_port=8000)"""

        code_2 = """\
# Import and use the ExampleComponent
example_component = ExampleComponent(name="Meerkat")

# Run the page (startup may take a few seconds)
page = mk.gui.Page(component=example_component, id="example", height="200px")
page.launch()"""

        nb["cells"] = [
            nbf.v4.new_markdown_cell(text),
            nbf.v4.new_code_cell(code_1),
            nbf.v4.new_code_cell(code_2),
        ]

        return nb

    def write_app_css(self):
        self.write_file("src/lib/app.css", JINJA_ENV.get_template("app.css").render())

    def write_constants_js(self):
        self.write_file(
            "src/lib/constants.js",
            JINJA_ENV.get_template("constants.js").render(),
        )

    def write_dot_mk(self):
        self.write_file(".mk", "")

    def write_example_component(self):
        os.makedirs(f"{self.appdir}/src/lib/components", exist_ok=True)
        self.write_file(
            "src/lib/components/ExampleComponent.svelte",
            JINJA_ENV.get_template("ExampleComponent.svelte").render(),
        )
        self.write_file(
            "src/lib/components/__init__.py",
            JINJA_ENV.get_template("components.init.py").render(),
        )

    def write_example_py(self):
        self.write_file("../example.py", JINJA_ENV.get_template("example.py").render())

    def write_example_ipynb(self):
        import nbformat as nbf

        with open("example.ipynb", "w") as f:
            nbf.write(self.render_example_ipynb(), f)

    def write_gitignore(self):
        # TODO: Should these be gitignored?
        # For users to publish apps, this probably shouldn't be.
        #   src/lib/wrappers
        #   src/lib/ComponentContext.svelte
        self.write_file(
            ".gitignore",
            JINJA_ENV.get_template("gitignore.jinja").render(),
        )

    def write_layout(self):
        self.write_file(
            "src/routes/+layout.svelte",
            JINJA_ENV.get_template("layout.svelte").render(),
        )
        self.write_file(
            "src/routes/+layout.js",
            JINJA_ENV.get_template("+layout.js").render(),
        )

    def write_libdir(self):
        os.makedirs(f"{self.appdir}/src/lib", exist_ok=True)

    def write_root_route_alternate(self):
        self.write_file(
            "src/routes/+page.svelte",
            JINJA_ENV.get_template("page.root.alternate.svelte").render(),
        )

    def write_setup_py(self):
        self.write_file("../setup.py", JINJA_ENV.get_template("setup.py").render())

    def write_slug_route(self):
        os.makedirs(f"{self.appdir}/src/routes/[slug]", exist_ok=True)
        self.write_file(
            "src/routes/[slug]/+page.svelte",
            JINJA_ENV.get_template("page.slug.svelte.jinja").render(),
        )

    def write_svelte_config(self):
        self.write_file(
            "svelte.config.js",
            JINJA_ENV.get_template("svelte.config.js").render(),
        )

    def write_tailwind_config(self):
        self.write_file(
            "tailwind.config.cjs",
            JINJA_ENV.get_template("tailwind.config.cjs").render(),
        )


class CreateSvelteInstallerApp(App):
    """An installer app that is used to call `create-svelte` and create a new
    SvelteKit app.

    Rather than directly calling `create-svelte`, we use this installer
    app to make it easier to add setup steps programatically.
    """

    def render_package_json(self):
        return JINJA_ENV.get_template("installer/package.json").render()

    def render_installer_js(self):
        return JINJA_ENV.get_template("installer/installer.js").render(
            name=self.appname,
        )

    def create(self):
        """Create the installer app that will be used to run `create-
        svelte`."""
        super().create()
        self.write_file("package.json", self.render_package_json())
        self.write_file("installer.js", self.render_installer_js())

    def install(self):
        """Install the installer app.

        This will run `create-svelte`, which in turn will create the
        SvelteKit app.
        """
        # Install the `create-svelte` dependency.
        super().install()

        # Run the `create-svelte` app installer to create the SvelteKit app.
        self.pm_run("create")

        # Rename the app directory to `app`.
        os.rename(f"{self.appname}", "app")

    def delete(self):
        """Delete the installer app directory."""
        shutil.rmtree(self.appdir)


class SystemHelper(metaclass=Singleton):
    """Information about the user's system."""

    def __init__(self):
        # Cache properties.
        self._has_brew = None
        self._has_node = None
        self._has_npm = None

    @property
    def is_windows(self) -> bool:
        """Returns True if the system is Windows."""
        return sys.platform == "win32"

    @property
    def is_linux(self) -> bool:
        """Returns True if the system is Linux."""
        return sys.platform == "linux"

    @property
    def is_macos(self) -> bool:
        """Returns True if the system is MacOS."""
        return sys.platform == "darwin"

    @property
    def has_brew(self) -> bool:
        """Check if the user has `homebrew` installed."""
        if self._has_brew is None:
            self._has_brew = subprocess.run(["which", "brew"]).returncode == 0
        return self._has_brew

    @property
    def has_node(self) -> bool:
        """Check if the user has `node` installed."""
        if self._has_node is None:
            self._has_node = subprocess.run(["which", "node"]).returncode == 0
        return self._has_node

    @property
    def has_npm(self) -> bool:
        """Check if the user has `npm` installed."""
        if self._has_npm is None:
            self._has_npm = subprocess.run(["which", "npm"]).returncode == 0
        return self._has_npm

    def install_svelte(self, cwd: str, package_manager: PackageManager = "npm"):
        return subprocess.run(
            [package_manager, "i", "create-svelte@latest"],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def install_bun(self):
        return subprocess.run(
            ["curl https://bun.sh/install | sh"],
            check=True,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    def install_node(self):
        if self.has_node:
            return

        if self.is_macos:  # M1 or Intel
            if not self.has_brew:
                raise RuntimeError(
                    "Homebrew is recommended to install Meerkat on M1 Macs. "
                    "See these instructions: https://docs.brew.sh/Installation. "
                    "Alternatively, you can install Node manually."
                )
            rich.print(
                "[yellow]Installing Node with Homebrew. "
                "This may take a few minutes.[/yellow]"
            )
            return subprocess.run(["brew install node"], shell=True, check=True)
        elif self.is_linux:
            if self.has_npm:
                # Has npm, so has node
                # KG: do we need all of these?
                subprocess.run("npm install -g n", shell=True, check=True)
                subprocess.run("n latest", shell=True, check=True)
                subprocess.run("npm install -g npm", shell=True, check=True)
                subprocess.run("hash -d npm", shell=True, check=True)
                subprocess.run("nvm install node", shell=True, check=True)
            else:
                subprocess.run(
                    "curl -fsSL https://deb.nodesource.com/setup_16.x | bash -",
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                subprocess.run("apt-get install gcc g++ make", shell=True, check=True)
                subprocess.run("apt-get install -y nodejs", shell=True, check=True)
        elif self.is_windows:
            raise RuntimeError(
                "Windows is not supported yet. Please file an issue on GitHub."
            )

    def __repr__(self) -> str:
        return f"""\
{self.__class__.__name__}(
    is_windows={self.is_windows},
    is_linux={self.is_linux},
    is_macos={self.is_macos},
    has_brew={self.has_brew},
)\
"""


logger.debug(
    tabulate(
        [
            ["sys.argv", sys.argv],
            ["", ""],
            ["Environment Variables", ""],
            ["Meerkat Base Directory", MEERKAT_BASE_DIR],
            ["Meerkat Internal App Directory", MEERKAT_INTERNAL_APP_DIR],
            ["Meerkat App Directory", MEERKAT_APP_DIR],
            ["Meerkat NPM Package", MEERKAT_NPM_PACKAGE],
            ["Meerkat Templates Directory", MEERKAT_TEMPLATES_DIR],
            ["Meerkat Run Subprocess", MEERKAT_RUN_SUBPROCESS],
            ["Meerkat Run Script Path", MEERKAT_RUN_SCRIPT_PATH],
            ["Meerkat Run Process", MEERKAT_RUN_PROCESS],
            ["Meerkat Run Reload Count", MEERKAT_RUN_RELOAD_COUNT],
            ["", ""],
            ["Path Helper", ""],
            ["rundir", PathHelper().rundir],
            ["appdir", PathHelper().appdir],
            ["is_user_app", PathHelper().is_user_app],
            ["scriptpath", PathHelper().scriptpath],
            ["scriptpath_abs", PathHelper().scriptpath_abs],
            ["scriptdir_abs", PathHelper().scriptdir_abs],
            ["scriptname", PathHelper().scriptname],
        ]
    )
)
