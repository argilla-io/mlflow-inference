import tempfile
from typing import Any, Dict, Optional

from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import path_to_local_file_uri
from mlflow.utils.uri import append_to_uri_path


def _download_model_artifact(model_uri, output_path: Optional[str] = None):
    local_path = _download_artifact_from_uri(model_uri, output_path=output_path)
    return path_to_local_file_uri(local_path)


def _model_config(model_uri: str) -> Dict[str, Any]:
    folder = tempfile.TemporaryDirectory()
    if ModelsArtifactRepository.is_models_uri(model_uri):
        underlying_model_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    else:
        underlying_model_uri = model_uri
    local_path = _download_artifact_from_uri(
        append_to_uri_path(underlying_model_uri, MLMODEL_FILE_NAME),
        output_path=folder.name,
    )
    model = Model.load(local_path)
    cfg = model.flavors.get(pyfunc.FLAVOR_NAME)
    if not cfg:
        raise RuntimeError("Only pyfunc flavours are supported")
    return cfg
