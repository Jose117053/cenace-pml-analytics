"""
Execution pipeline to extract and process CENACE PML data for the Monclova load zone.
"""

import sys
from datetime import date
from pathlib import Path
import yaml

# Ensure project root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.extract.windows import generate_date_windows
from src.extract.client import CenaceClient
from src.extract.parse import parse_raw_jsons_to_dataframe


def run_extraction(config_path: str = "config.yaml", force: bool = False) -> Path:
    """
    Orchestrates the extraction pipeline:
    1. Load configuration from YAML.
    2. Generate date windows of up to 7 days.
    3. Fetch raw JSONs from CENACE via CenaceClient.
    4. Parse raw JSONs into structured DataFrame.
    5. Save consolidated dataset as Parquet.
    """
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    nodes = config["nodes"]
    start_date = date.fromisoformat(config["date_range"]["start"])
    end_date = date.fromisoformat(config["date_range"]["end"])

    api_cfg = config["api"]
    raw_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
    interim_dir = PROJECT_ROOT / config["paths"]["interim_data_dir"]
    interim_dir.mkdir(parents=True, exist_ok=True)

    # Date windowing
    windows = generate_date_windows(start_date, end_date, max_days=api_cfg["max_days_per_request"])

    # Client setup
    client = CenaceClient(
        base_url=api_cfg["base_url"],
        system=config["system"],
        raw_data_dir=raw_dir,
        timeout_seconds=api_cfg["timeout_seconds"],
        max_retries=api_cfg["max_retries"],
        backoff_factor=api_cfg["backoff_factor"],
        rate_limit_delay_seconds=api_cfg["rate_limit_delay_seconds"],
    )

    output_parquet = None
    for proc in config["processes"]:
        client.fetch_all(process=proc, nodes=nodes, windows=windows, force_reload=force)
        df_proc = parse_raw_jsons_to_dataframe(raw_dir=raw_dir, process=proc)

        output_parquet = interim_dir / f"pml_{proc.lower()}.parquet"
        df_proc.to_parquet(output_parquet, index=False)
        print(f"Extraction complete: {len(df_proc):,} records saved to {output_parquet}")

    return output_parquet


if __name__ == "__main__":
    run_extraction()
