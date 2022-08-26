from dataclasses import dataclass
from typing import Any, Dict, Union

from fastapi import APIRouter, HTTPException

from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.state import state


class Interface(IdentifiableMixin):
    # TODO (all): I think this should probably be a subclassable thing that people
    # implement. e.g. TableInterface

    identifiable_group: str = "interfaces"



def get_interface(interface_id: int):
    try:
        interface = state.identifiables.interfaces[interface_id]
    except KeyError:
        raise HTTPException(
            status_code=404, detail="No interface with id {}".format(interface_id)
        )
    return interface


router = APIRouter(
    prefix="/interface",
    tags=["interface"],
    responses={404: {"description": "Not found"}},
)


@router.get("/{interface_id}/config/")
def get_config(interface_id: str):
    interface = get_interface(interface_id)
    return interface.config
