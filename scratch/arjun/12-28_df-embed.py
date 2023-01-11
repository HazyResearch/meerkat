"""Generate dataframes with embeddings for the different image datasets."""

import os
import pathlib
import subprocess
import shutil

from huggingface_hub.repository import Repository

import meerkat as mk
from meerkat.datasets.imagenette import imagenette

mk.config.datasets.root_dir = "/home/common/data"

ENCODERS = {
    "bit": [None],  # ["BiT-M-R50x1",  "BiT-M-R101x3", "Bit-M-R152x4"],
    "clip": [
        None
    ],  # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
}

SAVE_PATH = "/home/arjun/temp/meerkat-dataframes/embeddings"
HF_PATH = "/home/arjun/hf/meerkat-dataframes"

def _embed(df, input: str, encoder: str, variant: str = None):
    if variant is None:
        df = mk.embed(
            df, input=input, out_col=f"{input}_{encoder}", encoder=encoder
        )
    else:
        df = mk.embed(
            df,
            input=input,
            out_col=f"{input}_{encoder}_{variant}",
            encoder=encoder,
            variant=variant,
        )
    return df

def embed_imagenette():
    versions = imagenette.VERSIONS
    for version in versions:
        print("==" * 20)
        print("version:", version)
        print("==" * 20)
        df: mk.DataFrame = mk.get("imagenette", version=version)
        for encoder in ENCODERS:
            for variant in ENCODERS[encoder]:
                df = _embed(df, input="img", encoder=encoder, variant=variant)

        df.write(f"{SAVE_PATH}/imagenette_{version}.mk")


def embed_imagenet(overwrite=False):
    path = "{SAVE_PATH}/imagenet.mk"
    if os.path.exists(path) and not overwrite:
        df = mk.DataFrame.read(path)
    else:
        df: mk.DataFrame = mk.get("imagenet")

    for encoder in ENCODERS:
        for variant in ENCODERS[encoder]:
            df = _embed(df, input="img", encoder=encoder, variant=variant)
            # Save every loop because it's expensive to compute.
            df.write(path)


def prepare_dataframes():
    dirpath = SAVE_PATH
    os.chdir(os.path.join(dirpath))
    for dirname in os.listdir(dirpath):
        df = mk.DataFrame.read(os.path.join(dirpath, dirname))
        out_filename = f"{dirname}.tar.gz"
        subprocess.run(["tar", "-cvzf", out_filename, dirname])
        os.makedirs(f"{HF_PATH}/embeddings", exist_ok=True)
        shutil.move(f"{dirpath}/{out_filename}", f"{HF_PATH}/embeddings/{out_filename}")


def upload():
    path = str(HF_PATH)
    repo = Repository(
        local_dir=path,
        clone_from="arjundd/meerkat-dataframes",
        repo_type="dataset",
    )

    repo.git_pull()

    repo.push_to_hub()


# embed_imagenette()
# embed_imagenet()
prepare_dataframes()
upload()