"""Startup script for interactive Meerkat.

Code is heavily borrowed from Gradio.
"""

import atexit
import fnmatch
import os
import pathlib
import re
import socket
import subprocess
import time
from typing import List, Literal, Tuple

import rich
from uvicorn import Config

from meerkat.constants import APP_DIR, BASE_DIR
from meerkat.interactive.api import MeerkatAPI
from meerkat.interactive.server import (
    API_PORT,
    FRONTEND_PORT,
    LOCALHOST_NAME,
    TRY_NUM_PORTS,
    Server,
)
from meerkat.interactive.svelte import SvelteWriter
from meerkat.interactive.tunneling import setup_tunnel
from meerkat.state import APIInfo, FrontendInfo, state


def file_find_replace(directory, find, replace, pattern):
    for path, _, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)


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


def get_first_available_port(initial: int, final: int) -> int:
    """Gets the first open port in a specified range of port numbers. Taken
    from https://github.com/gradio-app/gradio/blob/main/gradio/networking.py.

    More reading:
    https://stackoverflow.com/questions/19196105/how-to-check-if-a-network-port-is-open

    Args:
        initial: the initial value in the range of port numbers
        final: final (exclusive) value in the range of port numbers,
            should be greater than `initial`
    Returns:
        port: the first open port in the range
    """
    # rich.print(f"Trying to find an open port in ({initial}, {final}). ", end="")
    for port in range(initial, final):
        try:
            s = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )  # create a socket object
            result = s.bind((LOCALHOST_NAME, port))  # Bind to the port  # noqa: F841
            s.close()
            # rich.print(f"Found open port: {port}")
            return port
        except OSError:
            pass

    raise OSError(
        "All ports from {} to {} are in use. Please close a port.".format(
            initial, final - 1
        )
    )


def snake_case_to_camel_case(snake_case: str) -> str:
    """Converts a snake case string to camel case.

    Args:
        snake_case (str): the snake case string to convert.

    Returns:
        str: the camel case string.
    """
    substrings = snake_case.split("_")
    return substrings[0] + "".join(x.title() for x in substrings[1:])


def to_py_module_name(script: str) -> str:
    """Converts a script name to a Python module name.

    Args:
        script (str): the script name to convert.

    Returns:
        str: the Python module name.
    """
    # Make sure script is in module format.
    if script.endswith(".py"):
        # Strip the .py extension.
        script = script[:-3]
        # Replace all / with .
        script = script.replace("/", ".")

    return script


def run_script(
    script: str,
    server_name: str = LOCALHOST_NAME,
    port: int = API_PORT,
    dev: bool = True,
    target: str = "interface",
    shareable: bool = False,
    subdomain: str = "app",
    frontend_url: str = None,
    apiurl: str = None,
    debug: bool = False,
) -> APIInfo:
    """Run a script with uvicorn.

    Args:
        script (str): the script to run.
        server_name (str, optional): the name of the server to run the script on. Defaults
            to "localhost".
        port (int, optional): the port to run the script on. Defaults to the default API
            port in Meerkat, which is 5000.
        dev (bool, optional): whether to run the script in development mode. Defaults to
            True.
        target (str, optional): the target `Interface` instance to run. Defaults to
            "interface".
    """
    # Make sure script is in module format.
    script = to_py_module_name(script)

    # Run the script with uvicorn. This will start the FastAPI server and serve the
    # backend.
    env = os.environ.copy()
    if frontend_url is not None:
        env["MEERKAT_FRONTEND_URL"] = frontend_url
    if apiurl is not None:
        env["MEERKAT_API_URL"] = apiurl
    if debug:
        env["MEERKAT_LOGGING_LEVEL"] = "DEBUG"
    env["MEERKAT_RUN"] = str(1)

    process = subprocess.Popen(
        [
            "uvicorn",
            f"{script}:{target}",
            "--port",
            str(port),
            "--host",
            server_name,
            "--log-level",
            "warning",
            "--factory",
        ]
        + (["--reload"] if dev else []),
        env=env,
        stderr=subprocess.STDOUT,
    )

    # If shareable, start the tunnel
    if shareable:
        server_name = setup_tunnel(port, subdomain=f"{subdomain}server")

    return APIInfo(
        api=MeerkatAPI,
        port=port,
        name=server_name,
        shared=shareable,
        process=process,
    )


def run_api_server(
    server_name: str = LOCALHOST_NAME,
    port: int = API_PORT,
    dev: bool = True,
    shareable: bool = False,
    subdomain: str = "app",
) -> APIInfo:
    # Move to the base directory at meerkat/
    currdir = os.getcwd()
    os.chdir(BASE_DIR)

    # Start the FastAPI server
    # Note: it isn't possible to support live reloading
    # via uvicorn with this method
    server = Server(
        Config(
            "meerkat.interactive.api.main:app",
            port=port,
            host=server_name,
            # log_level="info" if dev else "warning",
            log_level="warning",
        )
    )
    server.run_in_thread()
    os.chdir(currdir)

    # If shareable, start the tunnel
    if shareable:
        server_name = setup_tunnel(port, subdomain=f"{subdomain}server")

    return APIInfo(
        api=MeerkatAPI,
        server=server,
        port=port,
        name=server_name,
        shared=shareable,
    )


def run_frontend_dev(
    port: int,
    package_manager: Literal["npm", "bun"] = "npm",
    env: dict = {},
) -> subprocess.Popen:
    # Run the npm server in dev mode
    process = subprocess.Popen(
        [
            package_manager,
            "run",
            "dev",
            "--",
            "--port",
            str(port),
            "--strictPort",
            "true",
            "--logLevel",
            "info",
        ],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Make a regex for
    #   `Local:   http://127.0.0.1:8000/\n` and
    #   `Local:   http://localhost:8000/\n`
    regex_1 = re.compile(r"http://" + "127.0.0.1" + r":(\d+)/\n")
    regex_2 = re.compile(r"http://" + "localhost" + r":(\d+)/\n")

    # Need to check if it started successfully
    start_time = time.time()
    while process.poll() is None:
        out = process.stdout.readline().decode("utf-8")
        match_1 = regex_1.search(out)
        match_2 = regex_2.search(out)
        if match_1 or match_2:
            break

        if time.time() - start_time > 10:
            raise TimeoutError(
                """Could not start frontend dev server.

Here are the stderr logs (if they are empty, this is likely an 
issue with how we recognize if the server started successfully, please 
file an issue on GitHub):
"""
                + process.stderr.read().decode("utf-8")
            )
    return process


def get_subclasses_recursive(cls: type) -> List[type]:
    """Recursively find all subclasses of a class.

    Args:
        cls (type): the class to find subclasses of.

    Returns:
        List[type]: a list of all subclasses of cls.
    """
    subclasses = []
    for subclass in cls.__subclasses__():
        subclasses.append(subclass)
        subclasses.extend(get_subclasses_recursive(subclass))
    return subclasses


def wrap_all_components(exclude_meerkat: bool = False):
    from meerkat.interactive import Component

    # Recursively find all subclasses of Component
    subclasses = get_subclasses_recursive(Component)
    exclude = set(["AutoComponent", "Component"])
    for subclass in subclasses:
        if subclass.__name__ in exclude:
            continue

        if subclass.namespace == "@meerkat-ml/meerkat" and exclude_meerkat:
            continue

        # Use subclass.__name__ as the component name, instead of
        # subclass.component_name, because the latter is not guaranteed to be unique.
        component_name = subclass.__name__

        # Make a file for the component, inside a subdirectory for the namespace
        # e.g. src/lib/wrappers/__meerkat/Component.svelte
        os.makedirs(f"{APP_DIR}/src/lib/wrappers/__{subclass.namespace}", exist_ok=True)
        with open(
            f"{APP_DIR}/src/lib/wrappers/__{subclass.namespace}/{component_name}.svelte",
            "w",
        ) as f:
            f.write(subclass._to_svelte_wrapper())


def run_frontend_prod(
    port: int,
    api_url: str,
    libpath: pathlib.Path,
    package_manager: Literal["npm", "bun"] = "npm",
    env: dict = {},
    skip_build: bool = False,
) -> subprocess.Popen:
    # # Location of the build folder
    # buildpath = libpath / "build"

    if not skip_build:
        build_process = subprocess.Popen(
            [
                package_manager,
                "run",
                "build",
                # "--",
                # "--watch",
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Print a progress bar with rich, show the time elapsed
        start_time = time.time()
        while build_process.poll() is None:
            output = build_process.stdout.readline().decode("utf-8").strip()
            # Pad output to 100 characters
            output = output.ljust(100)
            if "node_modules/" in output:
                continue
            # Remove any symbols that would mess up the progress bar
            rich.print(
                f"Building... {time.time() - start_time:.2f}s | {output}", end="\r"
            )
            if 'Wrote site to "build"' in output:
                rich.print(
                    f"Build completed in {time.time() - start_time:.2f}s." + " " * 120
                )
                break

        rich.print("")

    # Run the statically built app with preview mode
    env.update({"VITE_API_URL_PLACEHOLDER": api_url})
    process = subprocess.Popen(
        [
            package_manager,
            "run",
            "preview",
            "--",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Alternately run the statically built app with a simple python server
    # os.chdir(buildpath)
    # process = subprocess.Popen(
    #     [
    #         "python",
    #         "-m",
    #         "http.server",
    #         str(port),
    #     ],
    #     env=env,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    # )
    # os.chdir(libpath)

    return process


def run_frontend(
    package_manager: Literal["npm", "bun"] = "npm",
    port: int = FRONTEND_PORT,
    dev: bool = True,
    shareable: bool = False,
    subdomain: str = "app",
    apiurl: str = None,
    appdir: str = APP_DIR,
) -> FrontendInfo:
    """Run the frontend server.

    Args:
        package_manager (Literal["npm", "bun"], optional): The package manager to use. Defaults to "npm".
        port (int, optional): The port to run the frontend server on. Defaults to FRONTEND_PORT.
        dev (bool, optional): Whether to run the frontend in development mode. Defaults to True.
        shareable (bool, optional): Whether to create a shareable link. Defaults to False.
        subdomain (str, optional): The subdomain to use for the shareable link. Defaults to "app".
        apiurl (str, optional): The URL of the API server. Defaults to None.
        appdir (str, optional): The directory of the frontend app. Defaults to APP_DIR.

    Returns:
        FrontendInfo: A FrontendInfo object containing the port and process of the frontend server.
    """
    currdir = os.getcwd()

    # Search for the first available port in the range
    # [port, port + TRY_NUM_PORTS)
    port = get_first_available_port(int(port), int(port) + TRY_NUM_PORTS)

    # Enter the "app/" directory
    libpath = pathlib.Path(appdir)
    os.chdir(libpath)

    # Update the VITE_API_URL environment variable
    env = os.environ.copy()
    env.update({"VITE_API_URL": apiurl})

    # Start the frontend server
    if dev:
        process = run_frontend_dev(port, package_manager, env)
    else:
        process = run_frontend_prod(port, apiurl, libpath, package_manager, env)

    if shareable:
        domain = setup_tunnel(port, subdomain=subdomain)

    os.chdir(currdir)

    return FrontendInfo(
        package_manager=package_manager,
        port=port,
        name="localhost" if not shareable else domain,
        shared=shareable,
        process=process,
    )


def start(
    package_manager: Literal["npm", "bun"] = "npm",
    shareable: bool = False,
    subdomain: str = "app",
    api_server_name: str = LOCALHOST_NAME,
    api_port: int = API_PORT,
    frontend_port: int = FRONTEND_PORT,
    dev: bool = True,
) -> Tuple[APIInfo, FrontendInfo]:
    """Start a Meerkat interactive server.

    Args:
        package_manager (str): the frontend package_manager to use. Defaults to "npm".
        shareable (bool): whether to share the interface at a publicly accesible link.
            This feature works by establishing a reverse SSH tunnel to a Meerkat server.
            Do not use this feature with private data. In order to use this feature, you
            will need an SSH key for the server. If you already have one, add it to the
            file at f"{config.system.ssh_identity_file}, or set the option
            `mk.config.system.ssh_identity_file` to the file where they are stored. If
            you don't yet have a key, you can request access by emailing
            eyuboglu@stanford.edu. Remember to ensure after downloading it that the
            identity file is read/write only by the user (e.g. with
            `chmod 600 path/to/id_file`). See `subdomain` arg for controlling the
            domain name of the shared link. Defaults to False.
        subdomain (str): the subdomain to use for the shared link. For example, if
            `subdomain="myinterface"`, then the shareable link will have the domain
            `myinterface.meerkat.wiki`. Defaults to None, in which case a random
            subdomain will be generated.
        api_server_name (str): the name of the API server. Defaults to "localhost".
        api_port (int): the port to use for the Meerkat API server. Defaults to 5000.
        frontend_port (int): the port to use for the Meerkat Vite server. Defaults to 8000.
        dev (bool): whether to run in development mode. Defaults to True.

    Returns:
        Tuple[APIInfo, FrontendInfo]: A tuple containing the APIInfo and FrontendInfo objects.
    """
    in_mk_run_subprocess = int(os.environ.get("MEERKAT_RUN", 0))
    if in_mk_run_subprocess:
        rich.print(
            "Cannot call `start` from a script run with `mk run`. "
            "Ignoring and continuing..."
        )
        return

    from meerkat.interactive.svelte import svelte_writer

    svelte_writer.init_run()

    # Run the API server
    api_info = run_api_server(api_server_name, api_port, dev, shareable, subdomain)

    # Run the frontend server
    frontend_info = run_frontend(
        package_manager,
        frontend_port,
        dev,
        shareable,
        subdomain,
        api_info.url,
        svelte_writer.appdir,
    )

    # Store in global state
    state.frontend_info = frontend_info
    state.api_info = api_info

    return api_info, frontend_info


@atexit.register
def cleanup():
    """Clean up Meerkat processes and files when exiting."""
    # Shut down servers
    in_mk_run_subprocess = int(os.environ.get("MEERKAT_RUN", 0))
    if in_mk_run_subprocess:
        return

    if state.frontend_info or state.api_info:
        # Keep message inside if statement to avoid printing when not needed
        # e.g. when running `mk run --help`
        rich.print(
            "\n:electric_plug: Cleaning up [violet]Meerkat[/violet].\n" ":wave: Bye!",
        )

    if state.frontend_info is not None:
        if state.frontend_info.process:
            state.frontend_info.process.terminate()
            state.frontend_info.process.wait()

    if state.api_info is not None:
        if state.api_info.server:
            state.api_info.server.close()
        if state.api_info.process:
            state.api_info.process.terminate()
            state.api_info.process.wait()

    svelte_writer = SvelteWriter()
    # svelte_writer.cleanup_run()


def output_startup_message(url: str, docs_url: str = None):
    meerkat_header = "[bold violet]\[Meerkat][/bold violet]"  # noqa: W605

    rich.print("")
    rich.print(
        f"{meerkat_header} [bold green]➜[/bold green] Open interface at "
        f"[turqoise]{url}[/turqoise]"
    )
    if docs_url:
        rich.print(
            f"{meerkat_header} [bold green]➜[/bold green] Open API docs at "
            f"[turqoise]{docs_url}[/turqoise]"
        )
    rich.print(
        f"{meerkat_header} [bold green]➜[/bold green] Interact with Meerkat "
        "programatically with the console below. Use [yellow]quit()[/yellow] to end "
        "session."
    )
    rich.print("")
