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

    Returns:
        Union[Column]: A :class:`DeferredColumn` or a
            :class:`DataFrame` containing :class:`DeferredColumn` representing the
            deferred map.
    """
    from manifest import Manifest

    client_name, engine = engine.split("/")
    manifest = Manifest(
        client_name=client_name,
        client_connection=open("/Users/sabrieyuboglu/.meerkat/keys/.openai").read(),
        engine=engine,
        temperature=0,
        max_tokens=1,
        cache_name="sqlite",
        cache_connection="/Users/sabrieyuboglu/.manifest/cache.sqlite",
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
