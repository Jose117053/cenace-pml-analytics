import json
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


def parse_raw_jsons_to_dataframe(raw_dir: str | Path, process: str) -> pd.DataFrame:
    """
    Read all downloaded JSON files for a given market process,
    extract hourly records, validate data types, and build a sorted DataFrame.

    Returned fields:
    - clv_nodo (str): Node identifier
    - timestamp (datetime64[ns]): Hour 1 mapped to 00:00:00 interval start
    - fecha (str YYYY-MM-DD): Operating date
    - hora (int 1..24): Operating hour
    - pml (float): Total Local Marginal Price ($/MWh)
    - pml_ene (float): Energy component ($/MWh)
    - pml_per (float): Losses component ($/MWh)
    - pml_cng (float): Congestion component ($/MWh)
    - proceso (str): Market process (MDA or MTR)
    """
    proc_dir = Path(raw_dir) / process.lower()
    json_files = sorted(proc_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {proc_dir}")

    records = []
    logger.info(f"Processing {len(json_files)} JSON files for [{process}]...")

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading {jf.name}: {e}")
            continue

        if data.get("status") != "OK":
            continue

        proc = data.get("proceso", process.upper())
        resultados = data.get("Resultados", [])

        for res in resultados:
            clv_nodo = res.get("clv_nodo")
            valores = res.get("Valores", [])
            for v in valores:
                try:
                    fecha_str = v["fecha"]
                    hora_int = int(v["hora"])
                    pml_val = float(v["pml"])
                    ene_val = float(v["pml_ene"])
                    per_val = float(v["pml_per"])
                    cng_val = float(v["pml_cng"])

                    records.append({
                        "clv_nodo": clv_nodo,
                        "fecha": fecha_str,
                        "hora": hora_int,
                        "pml": pml_val,
                        "pml_ene": ene_val,
                        "pml_per": per_val,
                        "pml_cng": cng_val,
                        "proceso": proc,
                    })
                except (KeyError, ValueError, TypeError) as conv_err:
                    raise ValueError(
                        f"Incomplete or invalid record in file {jf.name}, node {clv_nodo}: {v} ({conv_err})"
                    ) from conv_err

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid records retrieved for {process}")

    # Build unified timestamp (hour 1 -> 00:00:00, hour 24 -> 23:00:00)
    df["timestamp"] = pd.to_datetime(df["fecha"]) + pd.to_timedelta(df["hora"] - 1, unit="h")

    # Deduplicate and sort chronologically per node
    initial_len = len(df)
    df = df.drop_duplicates(subset=["clv_nodo", "proceso", "timestamp"])
    if len(df) < initial_len:
        logger.info(f"Removed {initial_len - len(df)} duplicate records due to overlap.")

    df = df.sort_values(by=["clv_nodo", "timestamp"]).reset_index(drop=True)

    # Reorder columns
    ordered_cols = [
        "clv_nodo",
        "timestamp",
        "fecha",
        "hora",
        "pml",
        "pml_ene",
        "pml_per",
        "pml_cng",
        "proceso",
    ]
    df = df[ordered_cols]
    return df
