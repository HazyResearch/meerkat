from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from meerkat.interactive.state import interfaces

from .routers import datapanel
from .routers import interface

app = FastAPI()

app.include_router(interface.router)
app.include_router(datapanel.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
