from mlflow.models.container import _install_pyfunc_deps
from pydantic import BaseSettings


class MlSettings(BaseSettings):
    """Environment settings for ml serve"""

    model_uri: str



ml_settings = MlSettings()

_install_pyfunc_deps