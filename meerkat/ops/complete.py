import os
import re

import meerkat as mk
import meerkat.tools.docs as docs
from meerkat.ops.map import _SHARED_DOCS_


@docs.doc(source=_SHARED_DOCS_)
def complete(
    df: mk.DataFrame,
    prompt: str,
    engine: str,
    batch_size: int = 1,
    use_ray: bool = False,
    num_blocks: int = 100,
    blocks_per_window: int = 10,
    pbar: bool = False,
    client_connection: str = None,
    cache_connection: str = "~/.manifest/cache.sqlite",
) -> mk.ScalarColumn:
    """Apply a generative language model to each row in a DataFrame.

    Args:
        df (DataFrame): The :class:`DataFrame` to which the
            function will be applied.
        prompt (str):
        engine (str):
        ${batch_size}
        ${materialize}
        ${use_ray}
        ${num_blocks}
        ${blocks_per_window}
        ${pbar}
        client_connection: The connection string for the client.
            This is typically the key (e.g. OPENAI).
            If it is not provided, it will be inferred from the engine.
        cache_connection: The sqlite connection string for the cache.

    Returns:
        Union[Column]: A :class:`DeferredColumn` or a
            :class:`DataFrame` containing :class:`DeferredColumn` representing the
            deferred map.
    """
    from manifest import Manifest

    input_engine = engine
    client_name, engine = engine.split("/")
    if client_connection is None:
        if client_name == "openai":
            client_connection = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                f"Cannot infer client connection from engine {input_engine}."
            )

    cache_connection = os.path.abspath(os.path.expanduser(cache_connection))
    os.makedirs(os.path.dirname(cache_connection), exist_ok=True)

    manifest = Manifest(
        client_name=client_name,
        client_connection=client_connection,
        engine=engine,
        temperature=0,
        max_tokens=1,
        cache_name="sqlite",
        cache_connection=cache_connection,
    )

    def _run_manifest(rows: mk.DataFrame):
        out = manifest.run([prompt.format(**row) for row in rows.iterrows()])
        return out

    keys = re.findall(r"{(.*?)}", prompt)
    output = mk.map(
        df[keys],
        function=_run_manifest,
        inputs="row",
        is_batched_fn=True,
        batch_size=batch_size,
        pbar=pbar,
        use_ray=use_ray,
        num_blocks=num_blocks,
        blocks_per_window=blocks_per_window,
    )

    return output
