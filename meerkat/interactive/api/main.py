from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import datapanel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datapanel.router)
