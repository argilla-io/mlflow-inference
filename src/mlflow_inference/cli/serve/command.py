import logging
import os
import subprocess

import click
from mlflow_inference.cli._helpers import _download_model_artifact, _model_config

_logger = logging.getLogger(__name__)


@click.command("serve", help="Launches the mlflow model as a http service")
@click.option(
    "-m",
    "--model-uri",
    "model_uri",
    type=str,
    help="The mlflow model uri",
)
@click.option(
    "-p",
    "--port",
    default=8008,
    type=int,
    show_default=True,
    help="Port used for api listening",
)
@click.option(
    "-w",
    "--workers",
    default=1,
    type=int,
    show_default=True,
)
def serve(model_uri: str, port: int, workers: int) -> None:
    if os.name == "nt":
        raise RuntimeError("Windows platform is not supported!!!")

    _model_config(model_uri)
    command = (
        "gunicorn -k uvicorn.workers.UvicornWorker"
        " --timeout=60 -b 0.0.0.0:{port} -w {nworkers} ${{GUNICORN_CMD_ARGS}}"
        " -- mlflow_inference.wsgi:app"
    ).format(port=port, nworkers=workers)

    local_uri = _download_model_artifact(model_uri)

    command_env = os.environ.copy()
    command_env["MODEL_URI"] = local_uri
    _logger.info("=== Running command '%s'", command)
    subprocess.Popen(["bash", "-c", command], env=command_env).wait()
