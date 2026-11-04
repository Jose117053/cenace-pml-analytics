# CENACE PML Analytics & Forecasting — Monclova Load Zone

A data science pipeline for **Day-Ahead Market (MDA) electricity price forecasting** across 9 nodes in Mexico's Monclova Load Zone (January 2024 – June 2025).

The project covers data extraction, exploratory analysis, forecasting, model evaluation, and retrospective anomaly detection.

## 1. Quick Start — Docker

**Requirement:** Docker Desktop (Windows/macOS) or Docker Engine (Linux).

Run the following commands from the project directory using PowerShell, CMD, or a Linux/macOS terminal:

```bash
docker build -t cenace-analytics .
docker run --rm -p 127.0.0.1:8888:8888 cenace-analytics
```

Open the JupyterLab URL displayed in the terminal.

Docker provides a consistent Python 3.10 environment with the required dependencies.

**Note:** Changes made inside the container are not preserved after it is removed unless persistent storage is configured.

## 2. Local Setup (Without Docker)

**Requirement:** Python 3.10.

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

Install dependencies and launch JupyterLab:

```bash
pip install -r requirements.txt
jupyter lab
```

## 3. Notebooks

Notebooks are located in `notebooks/`.

| Notebook | Description |
|---|---|
| `01_test_extraction.ipynb` | CENACE API integration, extraction inspection, and data validation |
| `02_exploratory_data_analysis.ipynb` | Training-data exploration, temporal patterns, and relationships across nodes |
| `03_sarimax_forecasting.ipynb` | Seasonal forecasting with SARIMAX |
| `04_xgboost_forecasting.ipynb` | Global multi-node XGBoost forecasting |
| `05_model_comparison_and_test.ipynb` | Model comparison and out-of-sample evaluation |
| `06_anomaly_detection.ipynb` | Retrospective anomaly screening using forecast residuals and Isolation Forest |

## 4. Project Scope

- **Forecasting:** Predict the next day's 24 hourly MDA prices for each node using information available at forecast issuance. Compare daily and weekly persistence baselines, SARIMAX, and XGBoost.
- **Evaluation:** Report MAE, RMSE, and rMAE relative to daily persistence across all 9 nodes. Use 2024 for training, January–March 2025 for validation, and April–June 2025 for final testing.
- **Anomaly detection:** Retrospective screening using forecast residuals, Isolation Forest, and consensus flags for analyst review. These results do not constitute an out-of-sample evaluation of anomaly detection.
- **Productionization:** The presentation outlines proposed scheduling, monitoring, retraining, and integration into analyst workflows.

## 5. Data and Presentation

- **Source:** CENACE's public SW-PML service.
- **System and market:** Sistema Interconectado Nacional (SIN), Mercado del Día en Adelanto (MDA).
- **Coverage:** January 1, 2024 – June 30, 2025.
- **Dataset:** `data/interim/pml_mda.parquet` contains 118,152 hourly node-price records across the 9 nodes.
- **Units:** Prices and price components are expressed in MXN/MWh.
- **Presentation:** See `docs/presentation.pdf` for architecture, results, and the proposed productionization plan.

### Nodes

`06AEO-115`, `06AHM-400`, `06FRO-115`, `06GRA-115`, `06MON-115`, `06MON-230`, `06RGS-115`, `06TKS-115`, and `06XOC-115`.

### Regenerating the Dataset


```bash
python scripts/run_extraction.py
```

The pipeline downloads raw JSON responses, converts them into a structured hourly table, and saves the consolidated Parquet dataset.

The included Parquet file allows the analysis notebooks to run without downloading the historical data again.