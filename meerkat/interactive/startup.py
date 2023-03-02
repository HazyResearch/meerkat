"""Startup script for interactive Meerkat.

Some code and design patters are borrowed from Gradio and Pynecone.
"""

import atexit
import fnmatch
import logging
import os
import pathlib
import re
import socket
import subprocess
import time
from typing import Literal, Tuple

import requests
import rich
from uvicorn import Config

from meerkat.constants import (
    MEERKAT_APP_DIR,
    MEERKAT_BASE_DIR,
    MEERKAT_INTERNAL_APP_BUILD_DIR,
    MEERKAT_RUN_ID,
    MEERKAT_RUN_PROCESS,
    MEERKAT_RUN_SUBPROCESS,
    PathHelper,
    is_notebook,
    write_file,
)
from meerkat.interactive.api import MeerkatAPI
from meerkat.interactive.server import (
    API_PORT,
    FRONTEND_PORT,
    LOCALHOST_NAME,
    TRY_NUM_PORTS,
    Server,
)
from meerkat.interactive.tunneling import setup_tunnel
from meerkat.state import APIInfo, FrontendInfo, state
from meerkat.version import __version__ as mk_version

logger = logging.getLogger(__name__)


def file_find_replace(directory, find, replace, pattern):
    for path, _, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, pattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            # s = s.replace(find, replace)
            s = re.sub(find, replace, s)
            with open(filepath, "w") as f:
                f.write(s)


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
    target: str = "page",
    shareable: bool = False,
    subdomain: str = "app",
    frontend_url: str = None,
    apiurl: str = None,
    debug: bool = False,
) -> APIInfo:
    """Run a script with uvicorn.

    Args:
        script (str): the script to run.
        server_name (str, optional): the name of the server to run the
            script on. Defaults to "localhost".
        port (int, optional): the port to run the script on. Defaults to
            the default API port in Meerkat, which is 5000.
        dev (bool, optional): whether to run the script in development
            mode. Defaults to True.
        target (str, optional): the target `Page` instance to run. Defaults to
            "page".
    """
    # Make sure script is in module format.
    script = os.path.abspath(script)  # to_py_module_name(script)

    # Run the script with uvicorn. This will start the FastAPI server
    # and serve the backend.
    env = os.environ.copy()
    if frontend_url is not None:
        env["MEERKAT_FRONTEND_URL"] = frontend_url
    if apiurl is not None:
        env["MEERKAT_API_URL"] = apiurl
    if debug:
        env["MEERKAT_LOGGING_LEVEL"] = "DEBUG"
    env["MEERKAT_RUN_SUBPROCESS"] = str(1)
    env["MEERKAT_RUN_SCRIPT_PATH"] = script
    env["MEERKAT_RUN_ID"] = MEERKAT_RUN_ID

    # Create a file that will be used to count the number of times the script has been
    # reloaded. This is used by `SvelteWriter` to decide when to write the Svelte
    # wrappers (see that class for more details).
    write_file(f"{PathHelper().appdir}/.{MEERKAT_RUN_ID}.reload", str(1))
    process = subprocess.Popen(
        [
            "uvicorn",
            f"{os.path.basename(script).rsplit('.')[0]}:{target}",
            "--port",
            str(port),
            "--host",
            server_name,
            "--log-level",
            "warning",
            "--factory",
        ]
        + (["--reload"] if dev else [])
        + (["--reload-dir", os.path.dirname(script)] if dev else [])
        + (["--app-dir", os.path.dirname(script)]),
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
    os.chdir(MEERKAT_BASE_DIR)

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
    regex_1 = re.compile(r"http://" + "127.0.0.1" + r":(\d+)/")
    regex_2 = re.compile(r"http://" + "localhost" + r":(\d+)/")

    def escape_ansi(line):
        ansi_escape = re.compile(r"(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]")
        return ansi_escape.sub("", line)

    # Need to check if it started successfully
    start_time = time.time()
    while process.poll() is None:
        out = process.stdout.readline().decode("utf-8")
        out = escape_ansi(out)

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


def run_frontend_build(
    package_manager: Literal["npm", "bun"] = "npm",
    env: dict = {},
):
    env.update({"VITE_API_URL_PLACEHOLDER": "http://meerkat.dummy"})
    build_process = subprocess.Popen(
        [
            package_manager,
            "run",
            "build",
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
        if "node_modules/" in output or "unused" in output:
            continue
        # Remove any symbols that would mess up the progress bar
        stmt = f"Building (may take up to a minute)... {time.time() - start_time:.2f}s"
        if is_notebook():
            print(stmt, end="\r", flush=True)
        else:
            rich.print(stmt, end="\r", flush=True)

        # Put a sleep to make the progress smoother
        time.sleep(0.1)

        if 'Wrote site to "build"' in output:
            rich.print(
                f"Build completed in {time.time() - start_time:.2f}s." + " " * 140
            )
            break


def download_frontend_build(version: str = None):
    """Download the frontend build to the meerkat internal app folder.

    This function should be used when users who downloaded the meerkat
    repo do not have npm/bun to build.


    Args:
        version: The meerkat version associated with the build.
            Defaults to the current meerkat package version.
    """
    from meerkat.datasets.utils import download_url, extract_tar_file

    if version is None:
        version = mk_version

    # Download the build from huggingface.
    # This is a tarfile that contains the build folder.
    # We extract it to the meerkat internal app folder.
    url = f"https://huggingface.co/datasets/meerkat-ml/component-static-builds/resolve/main/static-build-{version}.tar.gz"  # noqa: E501

    # Check if the url exists.
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(
            f"Could not find a build for version {version}. "
            "Please check the the url for a list of versions: "
            "https://huggingface.co/datasets/meerkat-ml/component-static-builds"
        )

    build_file = download_url(
        url, dataset_dir=os.path.join(pathlib.Path.home(), ".meerkat", "build")
    )
    extract_tar_file(build_file, download_dir=MEERKAT_INTERNAL_APP_BUILD_DIR)


def run_frontend_prod(
    port: int,
    api_url: str,
    libpath: pathlib.Path,
    package_manager: Literal["npm", "bun"] = "npm",
    env: dict = {},
    skip_build: bool = False,
) -> subprocess.Popen:

    if not skip_build:
        run_frontend_build(package_manager, env)
    else:
        logger.debug("Skipping build step.")
        assert (libpath / "build").exists(), "libpath must exist if skip_build is True."

    # File find replacement for the VITE_API_URL_PLACEHOLDER
    # Replace VITE_API_URL||"http://some.url.here:port" with
    # VITE_API_URL||"http://localhost:8000"
    # using a regex
    file_find_replace(
        libpath / "build",
        r"(VITE_API_URL\|\|\".*?\")",
        f'VITE_API_URL||"{api_url}"',
        "*js",
    )
    file_find_replace(
        libpath / ".svelte-kit/output/client/_app/",
        r"(VITE_API_URL\|\|\".*?\")",
        f'VITE_API_URL||"{api_url}"',
        "*js",
    )

    # Run the statically built app with preview mode
    # This doesn't serve the build directory, but instead serves the
    # .svelte-kit/output/client/_app directory.
    # This works with [slug] routes, but it requires a Cmd + Shift + R to
    # hard refresh the page when the API server port changes. We need to
    # investigate this further, so we can use this to serve the [slug] routes.
    # process = subprocess.Popen(
    #     [
    #         package_manager,
    #         "run",
    #         "preview",
    #         "--",
    #         "--port",
    #         str(port),
    #     ],
    #     env=env,
    #     stdout=subprocess.PIPE,
    #     stderr=subprocess.STDOUT,
    # )

    # Alternately we run the statically built app with a simple python server
    # Note: this does not seem to work with [slug] routes, so we
    # should use the preview mode instead (gives a 404 error).
    # We can use this if we explicitly write routes for each
    # page. We are using this to serve pages using /?id=page_id for now.
    os.chdir(libpath / "build")
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "http.server",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    os.chdir(libpath)

    return process


def run_frontend(
    package_manager: Literal["npm", "bun"] = "npm",
    port: int = FRONTEND_PORT,
    dev: bool = True,
    shareable: bool = False,
    subdomain: str = "app",
    apiurl: str = None,
    appdir: str = MEERKAT_APP_DIR,
    skip_build: bool = False,
) -> FrontendInfo:
    """Run the frontend server.

    Args:
        package_manager (Literal["npm", "bun"], optional):
            The package manager to use. Defaults to "npm".
        port (int, optional): The port to run the frontend server on.
            Defaults to FRONTEND_PORT.
        dev (bool, optional): Whether to run the frontend in development mode.
            Defaults to True.
        shareable (bool, optional): Whether to create a shareable link.
            Defaults to False.
        subdomain (str, optional): The subdomain to use for the shareable link.
            Defaults to "app".
        apiurl (str, optional): The URL of the API server.
            Defaults to None.
        appdir (str, optional): The directory of the frontend app.
            Defaults to APP_DIR.
        skip_build (bool, optional): Whether to skip the build step in production.
            Defaults to False.

    Returns:
        FrontendInfo: A FrontendInfo object containing the port
            and process of the frontend server.
    """
    currdir = os.getcwd()

    # Search for the first available port in the range
    # [port, port + TRY_NUM_PORTS)
    port = get_first_available_port(int(port), int(port) + TRY_NUM_PORTS)

    # Enter the "app/" directory
    libpath = pathlib.Path(appdir)
    os.chdir(libpath)

    env = os.environ.copy()

    # Start the frontend server
    if dev:
        # Update the VITE_API_URL environment variable
        env.update({"VITE_API_URL": apiurl})
        process = run_frontend_dev(port, package_manager, env)
    else:
        # Download the build folder from HF
        if not os.path.exists(MEERKAT_INTERNAL_APP_BUILD_DIR):
            logger.debug(f"Downloading frontend build for meerkat v{mk_version}")
            download_frontend_build(version=mk_version)

        # Read the timestamp of the most recent file change, for the last build
        if os.path.exists(".buildprint"):
            with open(".buildprint", "r") as f:
                last_buildprint = int(f.read())
        else:
            last_buildprint = 0

        # Get the timestamp of the most recent file change
        most_recent_file = max(
            (
                # Exclude the build and package folders
                os.path.join(root, f)
                for root, _, files in os.walk(os.getcwd())
                for f in files
                if ("build" not in root)
                and ("/package/" not in root)
                and ("node_modules" not in root)
                and (".svelte-kit" not in root)
                and ("src/lib/wrappers" not in root)
                and (f != "ComponentContext.svelte")
                and (f != ".buildprint")
            ),
            key=os.path.getctime,
        )
        buildprint = int(os.path.getctime(most_recent_file))
        logger.debug("Most recent file change:", most_recent_file, buildprint)

        # Record the timestamp of the most recent file change, for the current build
        with open(".buildprint", "w") as f:
            f.write(str(buildprint))

        # If the most recent file change is the same as the last build,
        # skip the build step
        skip_build = skip_build or (last_buildprint == buildprint)
        if not (libpath / "build").exists():
            skip_build = False
        process = run_frontend_prod(
            port, apiurl, libpath, package_manager, env, skip_build
        )

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
    dev: bool = False,
    skip_build: bool = False,
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
        frontend_port (int): the port to use for the Meerkat Vite server.
            Defaults to 8000.
        dev (bool): whether to run in development mode. Defaults to True.

    Returns:
        Tuple[APIInfo, FrontendInfo]: A tuple containing the APIInfo and
            FrontendInfo objects.
    """
    if MEERKAT_RUN_SUBPROCESS:
        rich.print(
            "Calling `start` from a script run with `mk run` has no effect. "
            "Continuing..."
        )
        return

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
        PathHelper().appdir,
        skip_build=skip_build,
    )

    # Store in global state
    state.frontend_info = frontend_info
    state.api_info = api_info

    return api_info, frontend_info


# TODO: @atexist.register doesn't work for notebooks, find a way to make it work
# there.
@atexit.register
def cleanup():
    """Clean up Meerkat processes and files when exiting."""
    if MEERKAT_RUN_SUBPROCESS:
        # Don't clean up anything if running from the `mk run` subprocess.
        return

    if MEERKAT_RUN_PROCESS:
        try:
            # Remove the reload counter file if running from the `mk run` process.
            os.remove(f"{PathHelper().appdir}/.{MEERKAT_RUN_ID}.reload")
        except FileNotFoundError:
            pass

    if state.frontend_info or state.api_info:
        # Keep message inside if statement to avoid printing when not needed
        # e.g. when running `mk run --help`
        rich.print(
            "\n:electric_plug: Cleaning up [violet]Meerkat[/violet].\n" ":wave: Bye!",
        )

    # Shut down servers
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

    # from meerkat.interactive.svelte import SvelteWriter
    # TODO: this was causing issues earlier, but it seems to be working now.
    # Investigate further and we can remove this comment.
    # SvelteWriter().cleanup()
