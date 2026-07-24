# Dataplor Cleaning Pipeline

A configurable, country-aware pipeline for cleaning point-of-interest (POI) datasets exported from **Dataplor**. It takes a raw CSV of businesses and runs it through six sequential filters that remove commercial chains, spatial duplicates, mislabeled records, closed/night-only venues, and over-crowded properties — leaving you with a clean list of *target* retail channels (small shops, groceries, restaurants, etc.).

Every step is auditable: you can trace exactly where a given business was dropped, keep a copy of every removed record, and get an interactive HTML report of the whole run.

---

## Table of contents

- [What it does](#what-it-does)
- [Requirements — files you must provide](#requirements--files-you-must-provide)
- [Installation (uv)](#installation-uv)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Example code](#example-code)
- [Outputs](#outputs)
- [Example cases](#example-cases)
- [Project structure](#project-structure)

---

## What it does

The pipeline runs six modules in order. Each one receives the surviving rows from the previous step:

| # | Module | What it removes | Key parameters |
|---|--------|-----------------|----------------|
| 1 | **Normalization** | Strips accents, upper-cases and trims `name` into `name_normalized`, then drops exact duplicates by `(name, lat, lon)`. | — |
| 2 | **Chain Filter** | Known commercial chains (Walmart, OXXO, Pemex…) via country-specific regex, plus names appearing more than `max_appearances` times. | `max_appearances=4` |
| 3 | **BallTree Dedup** | Records with the *same* normalized name within `radius_meters` of each other (keeps one). | `radius_meters=50` |
| 4 | **Hours & Fuzzy** | Mislabeled names (global keyword list), venues open too few hours or night-only, and fuzzy-similar duplicates that are spatially close. | `min_hours=20`, `night_start=19`, `fuzzy_thresh=80`, `dist_m=100` |
| 5 | **Final Regex** | A final keyword blacklist (with a `KEEP_WORDS` whitelist so legit shops survive). | — |
| 6 | **Crowded Property** | Points genuinely co-located in the same `parent_location` when `threshold`+ pile up within `radius_m`. | `threshold=4`, `radius_m=100` |

You choose which modules run and in what order (see [Example code](#example-code)).

---

## Requirements — files you must provide

> ⚠️ **The pipeline will not run without these.** It reads real files from disk at startup and raises an error if they are missing.

### 1. Input data — `data/raw/dataplor_cleaned.csv`

A Dataplor CSV export. Path is set in `config.yaml` (`paths.raw_data`). At minimum the following columns are used by one or more modules:

| Column | Used by | Notes |
|--------|---------|-------|
| `name` | all | Business name (required). |
| `latitude`, `longitude` | modules 3, 4, 6 | Decimal degrees. Rows without coordinates are skipped by the spatial steps. |
| `dataplor_id` | module 3 | Unique row id, used to pick which duplicate to keep. |
| `parent_location` | module 6 | Property/building identifier for the crowding filter. |
| `monday_hours` … `sunday_hours` | module 4 | Format `"HH:MM-HH:MM"` (e.g. `09:00-18:00`). `"Not available"`, empty, or `00:00-00:00` count as no hours. |
| `identified_as_chain` | module 2 | Optional. If present and `"True"`, the row is dropped as a source-flagged chain. |

Extra columns are preserved and carried through to the output untouched.

> A real 87k-row sample lives at [`data/raw/dataplor_cleaned.csv`](data/raw/dataplor_cleaned.csv) if you want to see the exact schema.

### 2. Country pattern file — `config/patterns_<country>.py`

The pipeline is **country-aware**. Whatever you set as `active_country` in `config.yaml` (e.g. `mx`), it dynamically imports `config/patterns_mx.py`. That file **must** define:

- `CHAIN_REGEX` — **required.** A list of regex patterns for chains to remove.
- `PATTERNS_MX` (optional) — extra patterns; silently ignored if absent.

Two are shipped out of the box: [`config/patterns_mx.py`](config/patterns_mx.py) (Mexico) and [`config/patterns_cr.py`](config/patterns_cr.py) (Costa Rica). To support a new country, copy one of these to `config/patterns_<code>.py`, adapt the chains, and set `active_country` to that code. Missing the file → the pipeline raises a clear `ValueError` at startup.

### 3. Keyword lists (already provided)

- [`config/global_exclude.py`](config/global_exclude.py) — `GLOBAL_EXCLUDE_KEYWORDS` (module 4) and `FINAL_KEYWORD_EXCLUDE_KEYWORDS` (module 5).
- [`config/keep_words.py`](config/keep_words.py) — `KEEP_WORDS`, the whitelist that protects legit shops from the module 5 blacklist.

---

## Installation (uv)

This project uses [**uv**](https://docs.astral.sh/uv/) for dependency and environment management. It targets **Python ≥ 3.14** (see `.python-version`).

**1. Install uv** (if you don't have it):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**2. Sync the environment** — this creates `.venv` and installs everything from `uv.lock`:

```bash
uv sync
```

That's it. `uv sync` reads `pyproject.toml` + `uv.lock` and installs pinned versions of pandas, polars, scikit-learn, rapidfuzz, unidecode, pydantic, pyyaml, and friends.

---

## Quick start

From the project root (`data_cleaning/`):

```bash
uv run main.py
```

This runs all six modules on `data/raw/dataplor_cleaned.csv` for the country set in `config.yaml`, and writes the cleaned file to `data/processed/dataplor_final_pipeline_output.csv`.

You'll see per-module progress in the console:

```
==================================================
🚀 STARTING PIPELINE FOR COUNTRY: MX
==================================================
📁 Input file: .../data/raw/dataplor_cleaned.csv
📁 Output file: .../data/processed/dataplor_final_pipeline_output.csv

... - INFO - Initial records: 87197
... - INFO - Executing Module 2: Chain Filter...
... - INFO - [Module 2] Removed 8123 rows. Dataset reduced by 9.32%. ...
...
... - INFO - Done in 1.24 mins. Final file saved to ...
```

---

## Configuration

### `config.yaml`

```yaml
pipeline_settings:
  # Which country pattern file to load: cr, mx, co, ...
  active_country: "mx"

paths:
  data_dir: "data"
  raw_data: "data/raw/dataplor_cleaned.csv"          # input
  output_data: "data/processed/dataplor_final_pipeline_output.csv"  # output
```

### Runtime switches — bottom of `main.py`

```python
if __name__ == "__main__":
    modules_to_run = [1, 2, 3, 4, 5, 6]        # pick & order the steps

    save_intermediate_csvs = True               # write data/processed/0X_*.csv after each step
    save_dropped_csvs      = True               # write data/dropped/0X_*.csv (the removed rows)
    track_these_elements   = ["CLUB DE NUTRICION HERBALIFE"]   # audit trail for specific names

    run_pipeline(
        modules_to_run=modules_to_run,
        save_tracking=save_intermediate_csvs,
        save_drops=save_dropped_csvs,
        items_to_track=track_these_elements,
    )
```

**Tuning a module's thresholds** is done in the `PIPELINE_STEPS` dict in `main.py`. For example, to make the spatial dedup radius tighter:

```python
3: {
    "name": "Module 3: BallTree Dedup",
    "file": "03_geo_balltree_deduped.csv",
    "func": lambda df, items: balltree_spatial_deduplication(
        df, radius_meters=25, items_to_track=items   # 50 → 25 meters
    ),
},
```

---

## Example code

### Run only a subset of steps

Skip the chain filter and crowding filter, run just normalize → spatial dedup → hours/fuzzy:

```python
from main import run_pipeline

run_pipeline(
    modules_to_run=[1, 3, 4],
    save_tracking=True,
    save_drops=True,
    items_to_track=None,        # no audit
)
```

### Fast mode (no auditing, no intermediate files)

The modules have a dedicated fast path when `items_to_track` is empty/None:

```python
from main import run_pipeline

run_pipeline(
    modules_to_run=[1, 2, 3, 4, 5, 6],
    save_tracking=False,   # don't dump 0X_*.csv after each step
    save_drops=False,      # don't keep removed rows
    items_to_track=None,   # skip audit bookkeeping (faster)
)
```

### Call a single module directly

Every module has the same signature `(df, ...) -> (df, items_to_track, report)` and works on a **Polars** DataFrame:

```python
import pandas as pd
import polars as pl
from src.chain_removal import filter_chains_and_duplicates

df = pl.from_pandas(pd.read_csv("data/raw/dataplor_cleaned.csv", low_memory=False))
# normalize first so 'name_normalized' exists
from src.normalize import normalize_names
df = normalize_names(df)

clean_df, _, report = filter_chains_and_duplicates(
    df, max_appearances=4, items_to_track=["OXXO"]
)
print(report)              # audit text: did "OXXO" survive or get dropped?
clean_df.write_csv("chains_removed.csv")
```

---

## Outputs

After a full run you get:

| Location | Contents |
|----------|----------|
| `data/processed/dataplor_final_pipeline_output.csv` | **The final cleaned dataset.** |
| `data/processed/0X_*.csv` | Snapshot of survivors after each step (if `save_tracking=True`). |
| `data/dropped/0X_*.csv` | The rows each step **removed** (if `save_drops=True`) — great for QA. |
| `data/audit/audit_report_<timestamp>.txt` | Per-name trace of where tracked items survived or were dropped (if `items_to_track` set). |
| `reports/pipeline_report_<timestamp>.html` | Interactive bar chart of records removed per step + top-5 dropped names. |

**Sample audit report** (from `track_these_elements=["CLUB DE NUTRICION HERBALIFE"]`):

```
🔍 Tracked text: 'CLUB DE NUTRICION HERBALIFE'
    ✅ SURVIVED: Passed this filter successfully (15 records).
    ❌ DROPPED: Removed at this step due to: [2_Hours_Filter] (37 records dropped).
```

This tells you 37 Herbalife records were cut by the **hours filter** in module 4 — instantly explaining an unexpected drop without digging through raw data.

---

## Example cases

**Case 1 — "Why did my count drop by 9%?"**
Open the newest `reports/pipeline_report_*.html`. The bar chart shows each step's removals; the cards list the top-5 most-removed names. If "Module 2: Chain Filter" is the tallest bar with `OXXO`, `WALMART` on top, that's expected chain removal.

**Case 2 — "Is a specific brand being wrongly filtered?"**
Set `track_these_elements = ["MI TIENDITA"]` and re-run. The audit file will say either `✅ SURVIVED` at every step, or pinpoint the exact module and reason (e.g. `[1_Global_Keywords]`) that removed it. If it's a false positive, add the word to `KEEP_WORDS` in `config/keep_words.py`.

**Case 3 — "I need to recover removed rows."**
Every step's casualties are in `data/dropped/0X_*.csv`. To see everything the hours/fuzzy step cut, open `data/dropped/04_hours_fuzzy_filtered.csv`.

**Case 4 — "Run the pipeline for Costa Rica instead of Mexico."**
Change `active_country: "cr"` in `config.yaml` (the CR pattern file already exists) and re-run `uv run main.py`. No code changes needed.

---

## Project structure

```
data_cleaning/
├── main.py                 # entry point + pipeline orchestration
├── config.yaml             # active country & file paths
├── pyproject.toml          # dependencies (uv)
├── uv.lock                 # pinned versions
├── config/
│   ├── settings.py         # loads config.yaml, dynamically imports country patterns
│   ├── patterns_mx.py      # Mexico chain regex  (CHAIN_REGEX)
│   ├── patterns_cr.py      # Costa Rica chain regex
│   ├── global_exclude.py   # keyword blacklists (modules 4 & 5)
│   └── keep_words.py        # whitelist protecting legit shops
├── src/
│   ├── normalize.py        # module 1
│   ├── chain_removal.py    # module 2
│   ├── geo_dedup.py        # module 3 (BallTree / haversine)
│   ├── adv_filters.py      # module 4 (hours + fuzzy)
│   ├── final_regex.py      # module 5
│   ├── crowded_property.py # module 6
│   ├── drop_tracker.py     # saves removed rows
│   ├── report.py           # HTML report + step metrics
│   └── utils.py            # dedup, logging, audit helpers
└── data/
    ├── raw/                # INPUT lives here
    ├── processed/          # survivors + final output
    ├── dropped/            # removed rows per step
    └── audit/              # per-name trace reports
```
