import click

from .helpers import build_docker_image


@click.command("build-docker", help="Creates a docker image for mlflow model")
@click.option(
    "-n",
    "--name",
    type=str,
    show_default=True,
    help="The created image name",
)
@click.option(
    "-m",
    "--model-uri",
    "model_uri",
    required=True,
    type=str,
    help="The mlflow model uri",
)
def build_docker(name: str, model_uri: str) -> None:
    """
    Build a docker image for a configured pipeline with an optional conda environment

    Parameters
    ----------
    name:
        The docker image name
    model_uri:
        The mlflow model uri
    -------

    """
    name = name or ":".join(model_uri.split("/")[-2:])
    build_docker_image(name, model_uri=model_uri)
