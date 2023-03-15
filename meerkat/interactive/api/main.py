from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import meerkat.interactive.api.routers.websocket as websocket

app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(websocket.router)