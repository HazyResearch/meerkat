# TODO: Think about whether we can move to
# from typing_extensions import Protocol
# in order to implement the EventInterface based type hints.


class EventInterface:
    """Defines the interface for an event.

    Subclass this to define the interface for a new event type.
    The class will specify the keyword arguments returned by an event from the
    frontend to any endpoint that has subscribed to it.

    All endpoints that are expected to receive an event of this type should
    ensure they have a signature that matches the keyword arguments defined
    in this class.
    """

    pass
