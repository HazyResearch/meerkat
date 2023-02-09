import asyncio
import json
import logging

from sse_starlette.sse import EventSourceResponse

from meerkat.interactive.endpoint import endpoint

logger = logging.getLogger(__name__)

STREAM_DELAY = 0.150  # second
RETRY_TIMEOUT = 15000  # milisecond


async def progress_generator():
    from meerkat.state import state

    progress_queue = state.progress_queue

    while True:
        # Checks for any progress and return to client if any
        progress = progress_queue.clear()
        if progress:
            for item in progress:
                if isinstance(item, list):
                    # Start events just send the total length
                    # of the progressbar
                    logger.debug(f"Sending start event with {item}.")
                    yield {
                        "event": "start",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(item),
                    }
                elif item is None:
                    logger.debug("Sending end event.")
                    yield {
                        "event": "end",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": "",
                    }
                elif isinstance(item, str):
                    logger.debug(f"Sending endpoint event with {item}.")
                    yield {
                        "event": "endpoint",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(item),
                    }
                else:
                    # Progress events send the current progress
                    # and the operation name
                    logger.debug(f"Sending progress event with {item}.")
                    yield {
                        "event": "progress",
                        "id": "message_id",
                        "retry": RETRY_TIMEOUT,
                        "data": json.dumps(item),
                    }

        await asyncio.sleep(STREAM_DELAY)


@endpoint(prefix="/subscribe", route="/progress/", method="GET")
def progress():
    return EventSourceResponse(progress_generator())
