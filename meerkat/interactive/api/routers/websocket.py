import atexit
import functools
import json
from typing import List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import rich

from meerkat.constants import (
    MEERKAT_APP_TARGET,
    MEERKAT_HOSTNAME,
    MEERKAT_MULTIUSER,
    MEERKAT_RUN_SCRIPT_PATH,
    MEERKAT_APP_PORT,
)
from meerkat.state import APIInfo


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.apis = {}
        self.last_port = None
        self.run_script = None
        self.multiuser = False

    def set_run_script(self):
        if self.run_script:
            return

        from meerkat.interactive.startup import run_script
        from meerkat.state import state

        if MEERKAT_RUN_SCRIPT_PATH:
            # Create a partial-ed run_script function
            # so that it can spin up new uvicorn servers when a new websocket
            # connection is made.
            self.run_script = functools.partial(
                run_script,
                MEERKAT_RUN_SCRIPT_PATH,
                server_name=MEERKAT_HOSTNAME,
                dev=False,
                target=MEERKAT_APP_TARGET,
                frontend_url=state.frontend_info.url,
                debug=False,
            )

    def set_multiuser(self):
        if not self.multiuser:
            self.multiuser = MEERKAT_MULTIUSER

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        await self.create_app(client_id)

    async def disconnect(self, websocket: WebSocket, client_id: str):
        self.active_connections.remove(websocket)
        await self.shutdown_app(client_id)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def proxy(self, message: str, websocket: WebSocket):
        assert "request" in message, "Message must contain a request to proxy."
        request = message["request"]
        import requests

        print(request)
        response = requests.request(
            method=request["method"],
            url=request["url"],
        )
        print(response)
        await websocket.send_json(response.json())

    async def create_app(self, client_id: str):
        if not self.multiuser:
            return

        from meerkat.state import state

        if self.last_port is None:
            self.last_port = MEERKAT_APP_PORT

        self.last_port += 1

        this_api_info = APIInfo(state.api_info.api, self.last_port)

        print(self.run_script)
        assert self.run_script is not None, "Must set run_script before creating app."
        print(f"Creating app with {self.last_port}, {this_api_info.url}, {client_id}")
        self.apis[client_id] = self.run_script(
            port=self.last_port,
            apiurl=this_api_info.url,
        )

    async def shutdown_app(self, client_id: str):
        if not self.multiuser:
            return
        
        rich.print(f"🧹 Cleaning up worker for client {client_id}.")
        api_info = self.apis.pop(client_id)
        if api_info.server:
            api_info.server.close()
        if api_info.process:
            api_info.process.terminate()
            api_info.process.wait()


manager = ConnectionManager()


@atexit.register
def cleanup():
    if len(manager.apis):
        rich.print(f"🧹 Cleaning up {len(manager.apis)} worker processes.")
    for api_info in manager.apis.values():
        if api_info.server:
            api_info.server.close()
        if api_info.process:
            api_info.process.terminate()
            api_info.process.wait()


router = APIRouter(
    prefix="/ws",
    tags=["websocket"],
    responses={404: {"description": "Not found"}},
)


@router.websocket("/{client_id}/")
async def client_connection(websocket: WebSocket, client_id: str):
    # Set up the manager.
    manager.set_run_script()
    manager.set_multiuser()

    await manager.connect(websocket, client_id)

    # Check if multiple clients are allowed
    if not manager.multiuser:
        # If not, send the client a message refusing the use of
        # websockets. The client will use standard HTTP requests.
        await manager.send_personal_message("no", websocket)
        # Disconnect the user
        await websocket.close()
        await manager.disconnect(websocket, client_id)
        return
    else:
        await manager.send_personal_message("yes", websocket)

    # Otherwise, allow the use of multiple clients. The manager
    # will maintain all connections and act as a proxy between
    # the client and the servers (one per client).
    try:
        while True:
            # All messages will be sent as JSON strings.
            # If a message contains a "request" key, it will be
            # interpreted as a request to proxy to the server.
            data = await websocket.receive_text()
            print(f"Received from {client_id}:", data)
            if "request" in data:
                await manager.proxy(json.loads(data), websocket)
            else:
                raise ValueError("Message must contain a request to proxy.")
                # await manager.send_personal_message(f"You wrote: {data}", websocket)
                # await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        rich.print("The client disconnected.")
        await manager.disconnect(websocket, client_id)
        # await manager.broadcast(f"Client #{client_id} left the chat")
