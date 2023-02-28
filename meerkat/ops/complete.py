import re
import meerkat as mk


def complete(
    df: mk.DataFrame,
    prompt: str,
    client_name: str,
    engine: str,
) -> mk.ScalarColumn:
    from manifest import Manifest
    # TODO: check if prompt ends with {.*} using a regex and extract the content
    keys = re.findall(r'{(.*?)}', prompt)

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

    output = mk.map(
        df[keys],
        function=_run_manifest,
        inputs="row",
        is_batched_fn=True,
        batch_size=4,
        pbar=True,
        materialize=False
    )

    return output
