from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import dataframe, endpoint, interface, llm, ops, ref, sliceby, store

app = FastAPI(debug=True)

app.include_router(interface.router)
app.include_router(dataframe.router)
app.include_router(sliceby.router)
app.include_router(llm.router)
app.include_router(ref.router)
app.include_router(ops.router)
app.include_router(store.router)
app.include_router(endpoint.router)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
