import logging
from typing import Any, List

import pandas
from fastapi import FastAPI, HTTPException
from mlflow.pyfunc import PyFuncModel
from mlflow_inference.settings import ml_settings
from rubrix.client import asgi


_logger = logging.getLogger(__name__)


def init(model: PyFuncModel) -> FastAPI:

    app = FastAPI(redoc_url=None, openapi_url="/api/spec.json", docs_url="/api/doc")

    def status():
        if model is None:
            raise HTTPException(status_code=404, detail="Model not initialized")
        return {"ok": True}

    def model_config():
        return model.metadata.to_dict()

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
