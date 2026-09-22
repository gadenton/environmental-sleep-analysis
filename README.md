# Environmental Sleep Analysis (CO₂ & Sleep Quality)

An end-to-end data pipeline and statistical modeling toolkit to evaluate how indoor bedroom carbon dioxide ($\text{CO}_2$) levels impact objective sleep architecture and biometrics.

By aligning overnight wearable health data (from Google Health Takeout) with high-resolution environmental sensor logs (from Home Assistant), this project quantifies the relationship between poor bedroom ventilation and sleep quality.

---

## Features

- **Automated Data Alignment & Ingestion**:
  - Merges Google Health Takeout data (`sleep_score.csv`, `sleep-*.json`, SpO₂ summaries, Heart Rate Variability, and nightly skin temperature).
  - Matches overnight sleep windows with Home Assistant time-series $\text{CO}_2$ exports.
  - Handles timezone adjustments (e.g., Mountain Time / IANA / Windows timezones) and UTC conversion.
  - Filters out nights with insufficient sensor coverage (configurable threshold, default $\ge 70\%$).

- **Advanced Environmental Metrics**:
  - **Mean & Peak Exposure**: Nightly average and maximum $\text{CO}_2$ levels ($\text{ppm}$).
  - **Threshold Dwell Times**: Total hours and percentage of sleep spent above $1{,}000\text{ ppm}$.
  - **Dynamic Ventilation Kinetics**: Net $\text{CO}_2$ accumulation and accumulation rate ($\text{ppm}/\text{hour}$).

- **Covariate Regression & Confounder Control** (`sleep_co2_regression.py`):
  - Controls for erratic schedules (e.g., parenting young children, irregular shift work) where fluctuating sleep durations or late bedtimes confound raw bivariate correlations.
  - Continuous bedtime encoding across midnight (e.g., $23{:}00 \rightarrow 23.0$, $01{:}00 \rightarrow 25.0$) to prevent circular boundary distortions.
  - Multi-predictor OLS regression with statistical significance testing ($p$-values, $t$-statistics, $R^2$, and standardized coefficients).
  - Exports detailed GitHub Markdown regression summary reports.

- **Cross-Platform Implementation**:
  - Python 3.13+ implementation with fast vectorization via Pandas, NumPy, and SciPy.
  - Legacy .NET / C# implementation (`ProcessSleepCo2.cs`) for standalone Windows execution.

---

## Project Structure

```text
├── ProcessSleepCo2.py         # Main ETL pipeline: merges Google Health & Home Assistant data
├── sleep_co2_regression.py     # Multiple regression analysis & report generator
├── ProcessSleepCo2.cs         # Original C# implementation
├── pyproject.toml             # Python packaging and dependency specifications
└── README.md                  # Documentation
```

---

## Prerequisites & Installation

### Requirements

- Python $\ge 3.13$
- [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`

### Setup with `uv`

```bash
# Clone the repository
git clone https://github.com/gadenton/environmental-sleep-analysis.git
cd environmental-sleep-analysis

# Create virtual environment and install dependencies
uv sync
```

---

## Data Preparation

### 1. Google Takeout (Google Health / Fitbit)

Request and download a Google Takeout export for Google Health. Extract the archive, which typically includes:

- `Takeout/Google Health/Sleep Score/sleep_score.csv`
- `Takeout/Google Health/Global Export Data/sleep-<year>-<month>.json`
- `Takeout/Google Health/Oxygen Saturation (SpO2)/`
- `Takeout/Google Health/Heart Rate Variability/`
- `Takeout/Google Health/Temperature/`

### 2. Home Assistant $\text{CO}_2$ Export

Export historical sensor data for your bedroom $\text{CO}_2$ monitor (e.g., Ikea ALPSTUGA) from Home Assistant History to a CSV file (e.g., `~/Downloads/CO2.csv`). The CSV should include:

- `last_changed` or `last_updated` (ISO timestamp)
- `state` ($\text{CO}_2$ concentration in ppm)

---

## Usage

### Step 1: Process and Merge Sleep & $\text{CO}_2$ Data

Run `ProcessSleepCo2.py` to match each night's sleep window to sensor readings and generate `sleep_co2_merged.csv`:

```bash
uv run python ProcessSleepCo2.py \
  --sleep-score "path/to/Takeout/Google Health/Sleep Score/sleep_score.csv" \
  --sleep-json-dir "path/to/Takeout/Google Health/Global Export Data" \
  --co2 "path/to/CO2.csv" \
  --spo2-dir "path/to/Takeout/Google Health/Oxygen Saturation (SpO2)" \
  --hrv-dir "path/to/Takeout/Google Health/Heart Rate Variability" \
  --temp-dir "path/to/Takeout/Google Health/Temperature" \
  --timezone "America/Denver" \
  --coverage 0.70 \
  --output sleep_co2_merged.csv
```

#### Key Options

| Flag | Description | Default |
| ------ | ------------- | --------- |
| `-s`, `--sleep-score` | Path to `sleep_score.csv` | Auto-detects in `~/Downloads` |
| `-j`, `--sleep-json-dir` | Directory containing `sleep-*.json` | Auto-detects in `~/Downloads` |
| `-c`, `--co2` | Path to Home Assistant `CO2.csv` | `~/Downloads/CO2.csv` |
| `-tz`, `--timezone` | Timezone for Google Health local records | `America/Denver` |
| `-cov`, `--coverage` | Minimum valid $\text{CO}_2$ sample coverage ($0.0 - 1.0$) | `0.70` |
| `--skip-regression` | Skip running the built-in regression summary | `False` |

---

### Step 2: Statistical Regression Analysis

Run `sleep_co2_regression.py` on the merged dataset to quantify the effect of $\text{CO}_2$ after controlling for sleep duration, bedtime, and wakefulness:

```bash
# Basic analysis controlling for sleep duration
uv run python sleep_co2_regression.py -i sleep_co2_merged.csv

# Analyze all CO2 metrics controlling for duration, bedtime, and awakenings
uv run python sleep_co2_regression.py \
  -i sleep_co2_merged.csv \
  -c all \
  --covariates all \
  --export-md regression_report.md
```

#### Covariates Explained

- **`--covariates duration`**: Controls for total sleep duration (`SleepDurationHours`). Essential because absolute minutes of Deep and REM sleep are strongly collinear with total time slept.
- **`--covariates all`**: Controls for:
  - `SleepDurationHours`: Total time asleep.
  - `BedtimeHour`: Linear continuous bedtime hour (accounting for circadian disruption).
  - `MinutesAwake`: Disruptions and nighttime awakenings.

---

## Metrics Analyzed

### Sleep Architecture & Biometrics

- **Sleep Quality**: Overall Sleep Score, Revitalization Score, Sleep Efficiency (%), Restlessness Index
- **Sleep Stages**: Deep Sleep (min & %), REM Sleep (min & %), Light Sleep (min)
- **Autonomic & Cardiovascular**: HRV RMSSD (ms), NREM Heart Rate (bpm), Resting Heart Rate (bpm)
- **Physiological**: Average SpO₂ (%), SpO₂ drops, Nightly Skin Temperature deviation (°C)

### Environmental Exposure ($\text{CO}_2$)

- `AvgCo2Ppm`: Mean overnight concentration.
- `MaxCo2Ppm`: Peak overnight concentration.
- `HoursAbove1000Ppm` / `PctSleepAbove1000Ppm`: Exposure time exceeding indoor air quality thresholds ($1{,}000\text{ ppm}$).
- `Co2RisePpm` & `Co2RiseRatePerHour`: Rate and total volume of carbon dioxide buildup in the bedroom.

---

## License

This project is licensed under the [MIT License](LICENSE).
