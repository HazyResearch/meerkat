"""Startup script for interactive Meerkat.

Code is heavily borrowed from Gradio.
"""

import os
import pathlib
import re
import socket
import subprocess
import time
from contextlib import closing
from tempfile import mkstemp, mktemp

import requests
from uvicorn import Config

from meerkat.interactive.api import MeerkatAPI
from meerkat.interactive.server import (
    INITIAL_PORT_VALUE,
    LOCALHOST_NAME,
    MEERKAT_API_SERVER,
    TRY_NUM_PORTS,
    Server,
)
from meerkat.interactive.tunneling import create_tunnel
from meerkat.state import NetworkInfo, state


def is_notebook() -> bool:
    """Check if the current environment is a notebook.

    Taken from
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook.
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

    New solution: https://stackoverflow.com/questions/19196105/how-to-check-if-a-network-port-is-open
    Args:
        initial: the initial value in the range of port numbers
        final: final (exclusive) value in the range of port numbers,
            should be greater than `initial`
    Returns:
        port: the first open port in the range
    """
    for port in range(initial, final):
        try:
            s = socket.socket()  # create a socket object
            s.bind((LOCALHOST_NAME, port))  # Bind to the port
            s.close()
            return port
        except OSError:
            pass

    raise OSError(
        "All ports from {} to {} are in use. Please close a port.".format(
            initial, final - 1
        )
    )


def interactive_mode(
    api_server_name: str = None,
    api_port: int = None,
    npm_port: int = None,
):
    """Start Meerkat interactive mode in a Jupyter notebook."""

    api_server_name = api_server_name or LOCALHOST_NAME

    # if port is not specified, search for first available port
    if api_port is None:
        api_port = get_first_available_port(
            INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS
        )
    else:
        api_port = get_first_available_port(api_port, api_port + 1)

    # Start the FastAPI server
    api_server = Server(
        Config(MeerkatAPI, port=api_port, host=api_server_name, log_level="warning")
    )
    api_server.run_in_thread()

    # Start the npm server
    if npm_port is None:
        npm_port = get_first_available_port(
            INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS
        )
    else:
        npm_port = get_first_available_port(npm_port, npm_port + 1)

    # Enter the "app/" directory
    libpath = pathlib.Path(__file__).parent.resolve() / "app"
    currdir = os.getcwd()
    os.chdir(libpath)

    network_info = NetworkInfo(
        api=MeerkatAPI,
        api_server=api_server,
        api_server_name=api_server_name,
        api_server_port=api_port,
        npm_server_port=npm_port,
    )

    # npm run dev -- --port {npm_port}
    current_env = os.environ.copy()
    current_env.update({"VITE_API_URL": network_info.api_server_url})

    # open a temporary file to write the output of the npm process
    out_file, out_path = mkstemp(suffix=".out")
    err_file, err_path = mkstemp(suffix=".err")

    npm_process = subprocess.Popen(
        [
            "npm",
            "run",
            "dev",
            "--",
            "--port",
            str(npm_port),
            "--logLevel",
            "info",
        ],
        env=current_env,
        stdout=out_file,
        stderr=err_file,
    )
    network_info.npm_process = npm_process
    network_info.npm_out_path = out_path
    network_info.npm_err_path = err_path

    time.sleep(1)

    # this is a hack to address the issue that the vite skips over a port that we
    # deem to be open per `get_first_available_port`
    # TODO: remove this once we figure out how to properly check for unavailable ports
    # in a way that is compatible with vite's port selection logic
    network_info.npm_server_port = int(
        re.search("Local:   http://localhost:(.*)/", network_info.npm_server_out).group(
            1
        )
    )

    # Back to the original directory
    os.chdir(currdir)

    # Store in global state
    state.network_info = network_info

    return network_info


def setup_tunnel(local_server_port: int, endpoint: str) -> str:
    response = requests.get(
        endpoint + "/v1/tunnel-request" if endpoint is not None else MEERKAT_API_SERVER
    )
    if response and response.status_code == 200:
        try:
            payload = response.json()[0]
            return create_tunnel(payload, LOCALHOST_NAME, local_server_port)
        except Exception as e:
            raise RuntimeError(str(e))
    else:
        raise RuntimeError("Could not get share link from Meerkat API Server.")


def output_startup_message(url: str):
    import rich

    meerkat_header = "[bold violet]\[Meerkat][/bold violet]"

    rich.print("")
    rich.print(
        f"{meerkat_header} [bold green]➜[/bold green] [bold] Open interface at: "
        f"[/bold] [turqoise] {url} [/turqoise]"
    )
    rich.print(
        f"{meerkat_header} [bold green]➜[/bold green] [bold] Interact with Meerkat "
        " programatically with the console below. Use [yellow]quit()[/yellow] to end "
        "session. [/bold]"
    )
    rich.print("")
