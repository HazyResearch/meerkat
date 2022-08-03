"""
Startup script for interactive Meerkat.
Code is heavily borrowed from Gradio.
"""
import os
import pathlib
import socket
import subprocess
import time

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
    """
    Check if the current environment is a notebook.
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
    """
    Gets the first open port in a specified range of port numbers.
    Taken from https://github.com/gradio-app/gradio/blob/main/gradio/networking.py.

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
    """
    Start Meerkat interactive mode in a Jupyter notebook.
    """
    if not is_notebook():
        raise RuntimeError("This function can only be run in a notebook.")

    api_server_name = api_server_name or LOCALHOST_NAME

    # if port is not specified, search for first available port
    if api_port is None:
        api_port = get_first_available_port(
            INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS
        )
    else:
        api_port = get_first_available_port(api_port, api_port + 1)

    # url_host_name = "localhost" if api_server_name == "0.0.0.0" else api_server_name
    # path_to_local_server = "http://{}:{}/".format(url_host_name, api_port)

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

    # npm run dev -- --port {npm_port}
    npm_process = subprocess.Popen(["npm", "run", "dev", "--", "--port", str(npm_port)])
    time.sleep(1)

    # Back to the original directory
    os.chdir(currdir)

    # Store in global state
    network_info = NetworkInfo(
        api=MeerkatAPI,
        api_server=api_server,
        api_server_name=api_server_name,
        api_server_port=api_port,
        npm_server_port=npm_port,
        npm_process=npm_process,
    )
    state.network_info = network_info

    from IPython.display import IFrame

    # This register_fn must be invoked in the notebook
    # TODO: figure out why requests.get and urllib.request.urlopen are not working
    # (maybe related to localStorage only being accessible from the notebook context?)
    register_fn = lambda: IFrame(
        f"{network_info.npm_server_url}/network/register?api={network_info.api_server_url}",  # noqa
        width=1,
        height=1,
    )

    return network_info, register_fn


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
