"""
Startup script for interactive Meerkat.
Code is heavily borrowed from Gradio.
"""
import os
import pathlib
import socket
import subprocess
import threading
import time
from types import SimpleNamespace

import requests
import uvicorn
from uvicorn import Config

from meerkat.interactive.api import MeerkatAPI
from meerkat.interactive.tunneling import create_tunnel

# By default, the local server will try to open on localhost, port 7860.
# If that is not available, then it will try 7861, 7862, ... 7959.
INITIAL_PORT_VALUE = int(os.getenv("MK_SERVER_PORT", "7860"))
TRY_NUM_PORTS = int(os.getenv("MK_NUM_PORTS", "100"))
LOCALHOST_NAME = os.getenv("MK_SERVER_NAME", "127.0.0.1")
MEERKAT_API_SERVER = "https://api.meerkat.app/v1/tunnel-request"


class Server(uvicorn.Server):
    """
    Taken from
    https://stackoverflow.com/questions/61577643/python-how-to-use-fastapi-and-uvicorn-run-without-blocking-the-thread
    and Gradio.
    """

    def install_signal_handlers(self):
        pass

    def run_in_thread(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        while not self.started:
            time.sleep(1e-3)

    def close(self):
        self.should_exit = True
        self.thread.join()


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
    server_name: str = None,
    server_port: int = None,
):
    """
    Start Meerkat interactive mode in a Jupyter notebook.
    """
    if not is_notebook():
        raise RuntimeError("This function can only be run in a notebook.")

    server_name = server_name or LOCALHOST_NAME
    url_host_name = "localhost" if server_name == "0.0.0.0" else server_name

    # if port is not specified, search for first available port
    if server_port is None:
        port = get_first_available_port(
            INITIAL_PORT_VALUE, INITIAL_PORT_VALUE + TRY_NUM_PORTS
        )
    else:
        port = get_first_available_port(server_port, server_port + 1)

    path_to_local_server = "http://{}:{}/".format(url_host_name, port)

    # Start the FastAPI server
    apiserver = Server(
        Config(MeerkatAPI, port=port, host=server_name, log_level="warning")
    )
    apiserver.run_in_thread()

    # Start the npm server
    libpath = pathlib.Path(__file__).parent.resolve() / "app"
    currdir = os.getcwd()
    os.chdir(libpath)
    npm_process = subprocess.Popen(["npm", "run", "dev"])
    os.chdir(currdir)

    return SimpleNamespace(
        port=port,
        path_to_local_server=path_to_local_server,
        api=MeerkatAPI,
        server=apiserver,
        npm_process=npm_process,
    )


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
