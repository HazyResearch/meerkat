"""Upload dataframes that are used often (e.g. demos, documentation, etc.).

- Use a machine with a GPU if possible.
- This script should be run sparingly, as it will upload a lot of data and can be computationally expensive.
"""

from huggingface_hub.repository import Repository
import os
import pathlib
import meerkat as mk
import tarfile
import subprocess

_PATH = pathlib.Path(os.path.abspath(os.path.expanduser("~/.meerkat/hf/dataframes-gen")))
_HF_PATH = pathlib.Path(os.path.abspath(os.path.expanduser("~/.meerkat/hf/dataframes")))

def imagenette_clip():
    df = mk.get("imagenette", version="160px")

    # Embed the image.
    df: mk.DataFrame = mk.embed(df, input="img", out_col="img_clip")
    return df[["img_id", "img_clip"]]


def prepare_dataframes():
    """Prepare dataframes for upload."""
    dfs = {
        "imagenette_clip": imagenette_clip,
    }

    # DataFrame -> tar
    for dirname in os.listdir(_PATH):
        path = _PATH / dirname
        subprocess.run(["tar", "-czf", str(_HF_PATH / f"{dirname}.tar.gz"), str(path)])

def tar():
    # this does not work. We have to change directories first.
    # DataFrame -> tar
    for dirname in os.listdir(_PATH):
        path = _PATH / dirname
        subprocess.run(["tar", "-czf", str(_HF_PATH / f"{dirname}.tar.gz"), str(path)])

def upload():
    path = str(_HF_PATH)
    repo = Repository(
        local_dir=path,
        clone_from="arjundd/meerkat-dataframes",
        repo_type="dataset",
    )

    repo.git_pull()

    repo.push_to_hub()

# prepare_dataframes()
# tar()
upload()