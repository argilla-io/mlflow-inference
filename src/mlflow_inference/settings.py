from typing import Optional

from pydantic import BaseSettings


class MlSettings(BaseSettings):
    """Environment settings for ml serve"""

    model_uri: str
    rubrix_dataset: Optional[str] = None
    rubrix_task: str = "text-classification"


ml_settings = MlSettings()
