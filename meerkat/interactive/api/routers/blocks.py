import functools

from fastapi import APIRouter, Body

router = APIRouter(
    prefix="/df",
    tags=["df"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)
