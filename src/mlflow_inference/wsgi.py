from mlflow.pyfunc import load_model

from mlflow_inference.cli.serve.app import init
from mlflow_inference.settings import ml_settings

app = init(load_model(ml_settings.model_uri))
