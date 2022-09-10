from fastapi import APIRouter, HTTPException


def get_interface(interface_id: int):
    try:
        from meerkat.state import state

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
