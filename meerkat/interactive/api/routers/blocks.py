import functools

from fastapi import APIRouter, Body

router = APIRouter(
    prefix="/dp",
    tags=["dp"],
    responses={404: {"description": "Not found"}},
)

EmbeddedBody = functools.partial(Body, embed=True)
