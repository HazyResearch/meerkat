#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev
import shutil
import subprocess
import sys
from distutils.util import convert_path
from shutil import rmtree

from setuptools import Command, find_packages, setup

main_ns = {}
ver_path = convert_path("meerkat/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)


# Package meta-data.
NAME = "meerkat-ml"
DESCRIPTION = (
    "Meerkat is building new data abstractions to make " "machine learning easier."
)
URL = "https://github.com/robustness-gym/meerkat"
EMAIL = "kgoel@cs.stanford.edu"
AUTHOR = "The Meerkat Team"
REQUIRES_PYTHON = ">=3.7.0"
VERSION = main_ns["__version__"]

# What packages are required for this module to be executed?
REQUIRED = [
    "dataclasses>=0.6",
    "pandas",
    "dill>=0.3.3",
    "numpy>=1.18.0",
    "cytoolz",
    "ujson",
    "torch>=1.7.0",
    "scikit-learn",
    "tqdm>=4.49.0",
    "datasets>=1.4.1",
    "pyarrow>=11.0.0",
    "PyYAML>=5.4.1",
    "omegaconf>=2.0.5",
    "semver>=2.13.0",
    "multiprocess>=0.70.11",
    "Cython>=0.29.21",
    "progressbar>=2.5",
    "fvcore",
    "ipywidgets>=7.0.0",
    "IPython",
    "fastapi",
    "uvicorn",
    "rich",
    "cryptography",
    "fastapi",
    "wrapt",
    "typer",
    "jinja2",
    "nbformat",
    "sse-starlette",
    "tabulate",
    "pyparsing",
]

# Read in docs/requirements.txt
with open("docs/requirements.txt") as f:
    DOCS_REQUIREMENTS = f.read().splitlines()

# What packages are optional?
EXTRAS = {
    "dev": [
        "black==22.12.0",
        "isort>=5.12.0",
        "flake8>=3.8.4",
        "docformatter>=1.4",
        "pytest-cov>=2.10.1",
        "recommonmark>=0.7.1",
        "parameterized",
        "pre-commit>=2.9.3",
        "twine",
        "httpx",
        "ray",
    ]
    + DOCS_REQUIREMENTS,
    "embeddings-mac": [
        "faiss-cpu",
        "umap-learn[plot]",
    ],
    "embeddings-linux": [
        "faiss-gpu",
        "umap-learn[plot]",
    ],
    "text": [
        "transformers",
        "spacy>=3.0.0",
    ],
    "vision": ["torchvision>=0.9.0", "opencv-python", "Pillow"],
    "audio": ["torchaudio"],
    "medimg": [
        "dosma>=0.0.13",
        "kaggle",
        "google-cloud-storage",
        "google-cloud-bigquery[bqstorage,pandas]",
    ],
}
EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from huggingface_hub.repository import Repository

        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
            rmtree(os.path.join(here, "build"))
        except OSError:
            pass

        # Build static components
        self.status("Building static components…")
        env = os.environ.copy()
        env.update({"VITE_API_URL_PLACEHOLDER": "http://meerkat.dummy"})
        if os.path.exists("./meerkat/interactive/app/build"):
            shutil.rmtree("./meerkat/interactive/app/build")
        build_process = subprocess.run(
            "npm run build",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            cwd="./meerkat/interactive/app",
        )
        if build_process.returncode != 0:
            print(build_process.stdout.decode("utf-8"))
            sys.exit(1)

        # Package static components to a tar file and push to huggingface.
        # This requires having write access to meerkat-ml.
        # TODO: Consider making this a github action.
        self.status("Packaging static component build...")
        components_build_targz = shutil.make_archive(
            base_name=f"static-build-{VERSION}",
            format="gztar",
            root_dir="./meerkat/interactive/app/build",
        )

        self.status("Uploading static build to huggingface...")
        local_repo_dir = os.path.abspath(
            os.path.expanduser("~/.meerkat/hf/component-static-builds")
        )
        repo = Repository(
            local_dir=local_repo_dir,
            clone_from="meerkat-ml/component-static-builds",
            repo_type="dataset",
        )
        shutil.move(
            components_build_targz,
            os.path.join(local_repo_dir, os.path.basename(components_build_targz)),
        )
        repo.git_pull()
        repo.push_to_hub(commit_message=f"{VERSION}: new component builds")

        # Build the source and wheel.
        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*", "demo"])
    + ["meerkat-demo"],
    package_dir={"meerkat-demo": "demo"},
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],
    entry_points={
        "console_scripts": ["mk=meerkat.cli.main:cli"],
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)
