"""Startup script for interactive Meerkat.

Code is heavily borrowed from Gradio.
"""

import os
import pathlib
import re
import socket
import subprocess
import time
from tempfile import mkstemp

from uvicorn import Config

from meerkat.interactive.api import MeerkatAPI
from meerkat.interactive.server import (
    INITIAL_PORT_VALUE,
    LOCALHOST_NAME,
    TRY_NUM_PORTS,
    Server,
)
from meerkat.interactive.tunneling import setup_tunnel
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


def start(
    shareable: bool = False,
    subdomain: str = None,
    api_server_name: str = None,
    api_port: int = None,
    npm_port: int = None,
):
    """Start a Meerkat interactive server.
    
    Args:
        shareable (bool): whether to share the interface at a publicly accesible link.
            This feature works by establishing a reverse SSH tunnel to a Meerkat server. 
            Do not use this feature with private data. In order to use this feature, you
            will need an SSH key for the server. If you already have one, add it to the 
            file at f"{config.system.ssh_identity_file}, or set the option 
            `mk.config.system.ssh_identity_file` to the file where they are stored. If 
            you don't yet have a key, you can request access by emailing 
            eyuboglu@stanford.edu. See `subdomain` arg for controlling the domain name 
            of the shared link. Defaults to False.
        subdomain (str): the subdomain to use for the shared link. For example, if
            `subdomain="myinterface"`, then the shareable link will have the domain
            `myinterface.meerkat.wiki`. Defaults to None, in which case a random 
            subdomain will be generated.
        api_port (int): the port to use for the Meerkat API server. Defaults to None,
            in which case a random port will be used.
        npm_port (int): the port to use for the Meerkat Vite server. Defaults to None,
            in which case a random port will be used.
    """
    if subdomain is None:
        subdomain = "app"

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

    if shareable:
        domain = setup_tunnel(network_info.api_server_port, subdomain=f"{subdomain}server")
        network_info.shareable_api_server_name = domain
        

    # npm run dev -- --port {npm_port}
    current_env = os.environ.copy()
    if shareable:
        current_env.update({"VITE_API_URL": network_info.shareable_api_server_url})
    else:
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

    MAX_WAIT = 10
    for i in range(MAX_WAIT):
        time.sleep(0.5)

        # this is a hack to address the issue that the vite skips over a port that we
        # deem to be open per `get_first_available_port`
        # TODO: remove this once we figure out how to properly check for unavailable ports
        # in a way that is compatible with vite's port selection logic
        match = re.search(
            "Local:   http://127.0.0.1:(.*)/", network_info.npm_server_out
        ) or re.search(
            "Local:   http://localhost:(.*)/", network_info.npm_server_out
        )
        if match is not None:
            break

    if match is None:
        raise ValueError(
            f"Failed to start dev server: out={network_info.npm_server_out} err={network_info.npm_server_err}"
        )
    network_info.npm_server_port = int(match.group(1))

    if shareable:
        domain = setup_tunnel(network_info.npm_server_port, subdomain=subdomain)
        network_info.shareable_npm_server_name = domain

    # Back to the original directory
    os.chdir(currdir)

    # Store in global state
    state.network_info = network_info

    # Print a message
    print(
        f"Meerkat interactive mode started! API on {network_info.api_server_url}, \
        and GUI server on {network_info.npm_server_url}."
    )
    return network_info


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
