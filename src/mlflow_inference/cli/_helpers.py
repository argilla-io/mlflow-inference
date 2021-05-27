import tempfile
from typing import Any, Dict, Optional

import yaml
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc import ENV
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import append_to_uri_path


def _download_model_artifact(model_uri, output_path: Optional[str] = None):
    local_path = _download_artifact_from_uri(model_uri, output_path=output_path)
    return path_to_local_file_uri(local_path)


def _load_model_info(model_uri: str, output_path: Optional[str] = None) -> Model:
    tmp_folder = tempfile.TemporaryDirectory()
    output_path = output_path or tmp_folder.name
    local_path = _download_model_artifact_file(
        model_uri, MLMODEL_FILE_NAME, output_path
    )
    return Model.load(local_path)


def _download_model_artifact_file(model_uri: str, file: str, output_path: str):
    if ModelsArtifactRepository.is_models_uri(model_uri):
        underlying_model_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    else:
        underlying_model_uri = model_uri
    local_path = _download_artifact_from_uri(
        append_to_uri_path(underlying_model_uri, file),
        output_path=output_path,
    )
    return local_path


def _model_config(model_uri: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    model = _load_model_info(model_uri, output_path)
    config = model.flavors.get(pyfunc.FLAVOR_NAME)
    if not config:
        raise RuntimeError("Only pyfunc flavours are supported")
    # Append conda environment definition if found
    if ENV in config:
        conda_env_file = _download_model_artifact_file(
            model_uri, file=config[ENV], output_path=output_path
        )
        with open(conda_env_file) as conda_file:
            config[config[ENV]] = yaml.safe_load(conda_file)
    return config
