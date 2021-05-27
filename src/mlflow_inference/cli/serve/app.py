import logging
import tempfile
from typing import Any, List

import pandas
import yaml
from fastapi import FastAPI, HTTPException
from mlflow.pyfunc import ENV, load_model
from mlflow_inference.cli._helpers import _download_model_artifact_file, _model_config
from mlflow_inference.settings import ml_settings
from rubrix.client import asgi

_logger = logging.getLogger(__name__)


def init(model_uri: str) -> FastAPI:
    tmp_folder = tempfile.TemporaryDirectory()
    try:
        model = load_model(model_uri)
        config = _model_config(model_uri, output_path=tmp_folder.name)
        if ENV in config:
            conda_env_file = _download_model_artifact_file(
                model_uri, file=config[ENV], output_path=tmp_folder.name
            )
            with open(conda_env_file) as conda_file:
                config[config[ENV]] = yaml.safe_load(conda_file)
        load_message = None
    except Exception as ex:
        model = None
        load_message = ex

    app = FastAPI(redoc_url=None, openapi_url="/api/spec.json", docs_url="/api/doc")

    def status():
        if model is None:
            raise HTTPException(
                status_code=404, detail=f"Model not initialized. {load_message}"
            )
        return {"ok": True}

    def model_config():
        return config

    def predict(data: List[Any]) -> List[Any]:
        df = pandas.DataFrame(data)
        response = model.predict(df)
        return response.to_dict(orient="records")

    # Configure routes
    app.get("/_status")(status)
    app.get("/_config")(model_config)
    app.post("/predict")(predict)

    if ml_settings.rubrix_dataset:
        ml_settings.rubrix_task = (
            ml_settings.rubrix_task.lower().strip() or "text-classification"
        )
        _logger.info(
            f"Model predictions for task '{ml_settings.rubrix_task}' will be registered in rubrix. "
            f"Using dataset name '{ml_settings.rubrix_dataset}'"
        )

        if ml_settings.rubrix_task == "text-classification":
            records_mapper = asgi.text_classification_mapper
        elif ml_settings.rubrix_task == "token-classification":
            records_mapper = asgi.token_classification_mapper
        else:
            raise ValueError(
                f"Task {ml_settings.rubrix_task} not supported."
                " Use an task type from ['text-classification', 'token-classification']"
            )

        app.add_middleware(
            asgi.RubrixLogHTTPMiddleware,
            api_endpoint="/predict",
            dataset=ml_settings.rubrix_dataset,
            records_mapper=records_mapper,
        )

    return app
