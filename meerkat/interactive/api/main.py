from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .routers import box, datapanel, interface, llm, sliceby

app = FastAPI()

app.include_router(interface.router)
app.include_router(datapanel.router)
app.include_router(sliceby.router)
app.include_router(llm.router)
app.include_router(box.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
