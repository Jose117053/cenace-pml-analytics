"""
Módulo de extracción de datos del Servicio Web de Precios Marginales Locales (SW-PML) de CENACE.
"""
from src.extract.windows import generate_date_windows
from src.extract.client import CenaceClient, CenaceAPIError
from src.extract.parse import parse_raw_jsons_to_dataframe

__all__ = [
    "generate_date_windows",
    "CenaceClient",
    "CenaceAPIError",
    "parse_raw_jsons_to_dataframe",
]
