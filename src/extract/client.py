import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Union
import requests

logger = logging.getLogger(__name__)


class CenaceAPIError(Exception):
    """Excepción para errores de la API de CENACE."""
    pass


class CenaceClient:
    def __init__(
        self,
        base_url: str = "https://ws01.cenace.gob.mx:8082/SWPML/SIM",
        system: str = "SIN",
        raw_data_dir: Union[str, Path] = "data/raw",
        timeout_seconds: int = 35,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        rate_limit_delay_seconds: float = 1.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.system = system.upper()
        self.raw_data_dir = Path(raw_data_dir)
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rate_limit_delay = rate_limit_delay_seconds
        self.session = requests.Session()

    def _build_url(self, process: str, nodes: List[str], start: date, end: date) -> str:
        nodes_str = ",".join(n.strip().upper() for n in nodes)
        return (
            f"{self.base_url}/{self.system}/{process.upper()}/{nodes_str}/"
            f"{start.year:04d}/{start.month:02d}/{start.day:02d}/"
            f"{end.year:04d}/{end.month:02d}/{end.day:02d}/JSON"
        )

    def _get_cache_path(self, process: str, nodes: List[str], start: date, end: date) -> Path:
        proc_dir = self.raw_data_dir / process.lower()
        proc_dir.mkdir(parents=True, exist_ok=True)
        sorted_nodes = sorted(n.strip().upper() for n in nodes)
        if len(sorted_nodes) == 1:
            node_tag = sorted_nodes[0]
        else:
            nodes_hash = hashlib.md5(",".join(sorted_nodes).encode("utf-8")).hexdigest()[:8]
            node_tag = f"{len(sorted_nodes)}n_{nodes_hash}"
        return proc_dir / f"{self.system.lower()}_{process.lower()}_{node_tag}_{start.isoformat()}_{end.isoformat()}.json"

    def fetch_window(
        self,
        process: str,
        nodes: List[str],
        start: date,
        end: date,
        force_reload: bool = False,
    ) -> Dict:
        """Descarga o recupera de caché local un bloque de fechas."""
        cache_path = self._get_cache_path(process, nodes, start, end)

        if not force_reload and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data.get("status") == "OK" and "Resultados" in cached_data:
                    logger.debug(f"Caché hit: {cache_path.name}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error leyendo caché {cache_path}: {e}")

        url = self._build_url(process, nodes, start, end)
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                logger.info(f"[{process}] Petición {start} a {end} (Intento {attempt}/{self.max_retries})")

                response = self.session.get(url, timeout=self.timeout)
                code = response.status_code

                if code == 200:
                    payload = response.json()
                    status = payload.get("status")
                    if status == "OK":
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False)
                        return payload
                    elif status == "NO_DATA":
                        logger.warning(f"CENACE reportó NO_DATA para {start} a {end}")
                        return payload
                    else:
                        raise CenaceAPIError(f"Estado no exitoso en respuesta: {status}")

                elif code == 204:
                    logger.warning(f"HTTP 204 No Content para periodo {start} a {end}")
                    return {"status": "NO_CONTENT", "Resultados": []}

                elif code == 401:
                    # Error definitivo de autenticación: detener sin reintentar
                    raise CenaceAPIError(f"HTTP 401 No Autorizado al consultar {url}")

                elif code == 404:
                    # Recurso no encontrado: detener sin reintentar
                    raise CenaceAPIError(f"HTTP 404 No Encontrado al consultar {url}")

                else:
                    logger.warning(f"HTTP {code} recibido en intento {attempt}.")

            except requests.RequestException as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(f"Falla de conexión ({exc}). Reintentando en {wait_time:.1f}s...")
                    time.sleep(wait_time)

        raise CenaceAPIError(
            f"Falla tras {self.max_retries} intentos para periodo {start} a {end}: {last_exception}"
        )

    def fetch_all(
        self,
        process: str,
        nodes: List[str],
        windows: List[Tuple[date, date]],
        force_reload: bool = False,
    ) -> List[Path]:
        """Descarga todas las ventanas y devuelve las rutas de archivos guardados en disco."""
        saved_paths = []
        total = len(windows)
        logger.info(f"Descargando {total} ventanas para proceso {process}...")

        for idx, (start, end) in enumerate(windows, 1):
            logger.info(f"Progreso [{process}]: {idx}/{total} ({start} a {end})")
            payload = self.fetch_window(process, nodes, start, end, force_reload=force_reload)
            cache_path = self._get_cache_path(process, nodes, start, end)

            if cache_path.exists():
                saved_paths.append(cache_path)
            else:
                status = payload.get("status") if isinstance(payload, dict) else "SIN_ARCHIVO"
                logger.warning(f"Ventana [{start} a {end}] sin archivo guardado (status: {status}).")

        return saved_paths
