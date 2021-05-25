from typing import Any, List

import pandas
from fastapi import FastAPI, HTTPException
from mlflow.pyfunc import PyFuncModel


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
    app.post("/invocations")(predict)

    return app
