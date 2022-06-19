import os
import subprocess

GIT_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset.git"

def _clone(dst):
    os.makedirs(dst, exist_ok=True)

    # clone url using subprocess
    subprocess.run(["git", "clone", GIT_URL, dst])


def download_fsdd(download_dir: str, ):
    _clone(download_dir)

    
    filenames = os.listdir(os.path.join(download_dir, "recordings")
