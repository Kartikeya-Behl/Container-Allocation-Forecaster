"""
src/models/__init__.py
======================
Import all model modules to populate the MODELS registry.
"""
from models.base import MODELS, Forecaster  # noqa: F401
import models.naive  # noqa: F401
import models.ma  # noqa: F401
import models.ets  # noqa: F401
import models.arima  # noqa: F401
