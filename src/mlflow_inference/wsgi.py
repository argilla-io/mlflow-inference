from mlflow_inference.cli.serve.app import init
from mlflow_inference.settings import ml_settings

app = init(ml_settings.model_uri)


