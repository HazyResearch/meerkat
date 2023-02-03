from typing import Any


class EventInterface:
    """
    Defines the interface for an event.

    Subclass this to define the interface for a new event type.
    The class will specify the keyword arguments returned by an event from the
    frontend to any endpoint that has subscribed to it.

    All endpoints that are expected to receive an event of this type should
    ensure they have a signature that matches the keyword arguments defined
    in this class.
    """

    pass


class OnClickInterface(EventInterface):
    """
    Interface for any on_click event.

    This event is triggered when a user clicks on a component.
    """

    pass


class OnChangeInterface(EventInterface):
    """
    Interface for any on_change event.

    This event is triggered when a user changes the value of a component
    e.g. radio button, checkbox, text input, etc.
    """

    value: Any
