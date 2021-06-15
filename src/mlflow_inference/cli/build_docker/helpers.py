import errno
import os
import shutil
import sys
import tempfile
from subprocess import PIPE, Popen, STDOUT
from typing import Any, Dict, List

from mlflow.pyfunc import ENV
from mlflow_inference.cli._helpers import _download_model_artifact, _model_config


def download_manylinux_package(package_src: str, dest: str):
    proc = Popen(
        [
            "pip",
            "download",
            "--platform=manylinux1_x86_64",
            "-d",
            dest,
            "--no-deps",
            package_src,
        ],
        cwd=os.getcwd(),
        stdout=PIPE,
        stderr=STDOUT,
        universal_newlines=True,
    )

    for x in iter(proc.stdout.readline, ""):
        print(x, end="", file=sys.stdout)


def build_docker_image(image_name: str, model_uri: str):

    folder = tempfile.TemporaryDirectory()

    cfg = _model_config(model_uri)
    _download_model_artifact(model_uri, output_path=folder.name)

    extra_packages_folder = os.path.join(folder.name, "extra_libs")
    extra_packages = {
        "mlflow-inference": "git+ssh://git@github.com/recognai/mlflow-inference.git",
    }

    # Download extra packages
    for repo in extra_packages.values():
        download_manylinux_package(repo, dest=extra_packages_folder)

    # Creates dockerfile from template
    with open(os.path.join(folder.name, "Dockerfile"), "w") as dockerfile:
        dockerfile_content = generate_dockerfile(
            model_cfg=cfg,
            extra_packages=[package for package in extra_packages],
        )
        dockerfile.write(dockerfile_content)

    proc = Popen(
        [
            "docker",
            "build",
            "-t",
            image_name,
            ".",
        ],
        cwd=folder.name,
        stdout=PIPE,
        stderr=STDOUT,
        universal_newlines=True,
    )

    for x in iter(proc.stdout.readline, ""):
        print(x, end="", file=sys.stdout)


def generate_dockerfile(model_cfg: Dict[str, Any], extra_packages: List[str]):
    template = """
FROM continuumio/miniconda3:latest

SHELL ["/bin/bash", "-c"]

RUN apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
    && apt-get -y autoremove \
    && rm -rf /var/lib/apt/lists/*

# Set up the program in the image
WORKDIR /opt/ml_model

COPY . /opt/ml_model
    """

    if ENV in model_cfg:
        template += (
            "\nRUN conda env update -n base -f /opt/ml_model/{conda_env}".format(
                conda_env=model_cfg[ENV]
            )
        )
    if extra_packages:
        template += "\nRUN pip install -U {extra_packages} --find-links /opt/ml_model/extra_libs/".format(
            extra_packages=" ".join(extra_packages)
        )

    template += """
EXPOSE 8008

CMD MODEL_URI=/opt/ml_model \
 gunicorn -k uvicorn.workers.UvicornWorker \
 --timeout=60 -b 0.0.0.0:8008 -w 1 ${GUNICORN_CMD_ARGS} \
 -- mlflow_inference.wsgi:app
    
    """
    return template
