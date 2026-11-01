import hashlib
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple
import requests

logger = logging.getLogger(__name__)


class CenaceAPIError(Exception):
    """Exception raised for errors during CENACE API interaction."""
    pass


class CenaceClient:
    def __init__(
        self,
        base_url: str = "https://ws01.cenace.gob.mx:8082/SWPML/SIM",
        system: str = "SIN",
        raw_data_dir: str | Path = "data/raw",
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
        """Download or retrieve from local cache a date window."""
        cache_path = self._get_cache_path(process, nodes, start, end)

        if not force_reload and cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                if cached_data.get("status") == "OK" and "Resultados" in cached_data:
                    logger.debug(f"Cache hit: {cache_path.name}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error reading cache {cache_path}: {e}")

        url = self._build_url(process, nodes, start, end)
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.rate_limit_delay > 0:
                    time.sleep(self.rate_limit_delay)
                logger.info(f"[{process}] Request {start} to {end} (Attempt {attempt}/{self.max_retries})")

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
                        logger.warning(f"CENACE returned NO_DATA for {start} to {end}")
                        return payload
                    else:
                        raise CenaceAPIError(f"Unsuccessful status in response: {status}")

                elif code == 204:
                    logger.warning(f"HTTP 204 No Content for period {start} to {end}")
                    return {"status": "NO_CONTENT", "Resultados": []}

                elif code == 401:
                    # Non-recoverable authorization error: fail fast without retry
                    raise CenaceAPIError(f"HTTP 401 Unauthorized for URL: {url}")

                elif code == 404:
                    # Resource not found: fail fast without retry
                    raise CenaceAPIError(f"HTTP 404 Not Found for URL: {url}")

                else:
                    logger.warning(f"HTTP {code} received on attempt {attempt}.")

            except requests.RequestException as exc:
                last_exception = exc
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(f"Connection failure ({exc}). Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)

        raise CenaceAPIError(
            f"Failed after {self.max_retries} attempts for period {start} to {end}: {last_exception}"
        )

    def fetch_all(
        self,
        process: str,
        nodes: List[str],
        windows: List[Tuple[date, date]],
        force_reload: bool = False,
    ) -> List[Path]:
        """Download all windows and return the saved cache file paths."""
        saved_paths = []
        total = len(windows)
        logger.info(f"Downloading {total} windows for process {process}...")

        for idx, (start, end) in enumerate(windows, 1):
            logger.info(f"Progress [{process}]: {idx}/{total} ({start} to {end})")
            payload = self.fetch_window(process, nodes, start, end, force_reload=force_reload)
            cache_path = self._get_cache_path(process, nodes, start, end)

            if payload.get("status") == "OK" and cache_path.exists():
                saved_paths.append(cache_path)
            else:
                status = payload.get("status") if isinstance(payload, dict) else "NO_FILE"
                logger.warning(f"Window [{start} to {end}] has no saved file (status: {status}).")

        return saved_paths
