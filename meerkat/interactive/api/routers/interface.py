from meerkat.interactive import Interface
from meerkat.interactive.endpoint import endpoint


@endpoint(prefix="/interface", route="/{interface}/config/", method="GET")
def config(interface: Interface):
    return interface.config
