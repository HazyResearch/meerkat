from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi import HTTPException


from meerkat.interactive.state import interfaces

from .routers import datapanel

app = FastAPI()

app.include_router(datapanel.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test/")
def get_config():
    return "hello"



def get_interface(interface_id: int):
    print(type(interface_id))
    if interface_id not in interfaces:
        raise HTTPException(
            status_code=404, detail="No interface with id {}".format(interface_id)
        )
    return interfaces[interface_id]


@app.get("/config/")
def get_config(id: int):
    interface = get_interface(id)
    return interface.config
