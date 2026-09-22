# Environmental Sleep Analysis (CO₂, Bedroom Temperature & Sleep Quality)

An end-to-end data pipeline and statistical modeling toolkit to evaluate how indoor bedroom carbon dioxide ($\text{CO}_2$) and ambient bedroom temperature impact objective sleep architecture and biometrics.

By aligning overnight wearable health data (from Google Health Takeout) with high-resolution environmental sensor logs (from Home Assistant), this project quantifies the relationship between bedroom air quality, ambient temperature, and sleep quality.

---

## Features

- **Automated Data Alignment & Ingestion**:
  - Merges Google Health Takeout data (`sleep_score.csv`, `sleep-*.json`, SpO₂ summaries, Heart Rate Variability, and nightly skin temperature).
  - Matches overnight sleep windows with Home Assistant time-series $\text{CO}_2$ (`CO2.csv`) and ambient bedroom temperature (`bedroom_temp.csv`) exports.
  - Computes time-weighted trapezoidal Riemann sum averages over exact sleep intervals.
  - Handles timezone adjustments (e.g., Mountain Time / IANA / Windows timezones) and UTC conversion.
  - Filters out nights with insufficient sensor coverage (configurable threshold, default $\ge 70\%$).

- **Advanced Environmental Metrics**:
  - **Mean & Peak Exposure**: Nightly average and maximum $\text{CO}_2$ levels ($\text{ppm}$) and room temperatures ($^\circ\text{F}$).
  - **Temperature Dynamics**: Minimum room temperature, peak temperature, and overnight thermal drift ($\Delta^\circ\text{F}$).
  - **Threshold Dwell Times**: Total hours and percentage of sleep spent above $1{,}000\text{ ppm}$.
  - **Dynamic Ventilation Kinetics**: Net $\text{CO}_2$ accumulation and accumulation rate ($\text{ppm}/\text{hour}$).

- **Covariate Regression & Confounder Control** (`sleep_co2_regression.py`):
  - Controls for erratic schedules (e.g., parenting young children, irregular shift work) where fluctuating sleep durations or late bedtimes confound raw bivariate correlations.
  - Continuous bedtime encoding across midnight (e.g., $23{:}00 \rightarrow 23.0$, $01{:}00 \rightarrow 25.0$) to prevent circular boundary distortions.
  - Multi-predictor OLS regression with statistical significance testing ($p$-values, $t$-statistics, $R^2$, and standardized coefficients).
  - Flags masking effects (uncovered by covariate controls) and pseudo-correlations (spurious raw correlations removed by sleep duration adjustments).

- **Bedroom Temperature Optimization Analysis** (`--optimal-temp`):
  - **Quadratic Curve Fitting**: Fits non-linear parabolic models ($\text{Metric} = \beta_0 + \beta_1 T + \beta_2 T^2 + \beta_3 \cdot \text{Duration}$) to solve for exact mathematical vertices ($T^* = -\frac{\beta_1}{2\beta_2}$) for peak sleep score, deep sleep, and minimal restlessness.
  - **Empirical Bracket Binning**: Groups nights into $3^\circ\text{F}$ temperature brackets ($<66^\circ\text{F}$ through $\ge 78^\circ\text{F}$) to analyze biometrics without parametric assumptions.
  - **Multi-Objective Composite Sleep Index**: Normalizes biometrics into weighted $Z$-scores (HRV, RHR, Deep Sleep, Restlessness, Sleep Score, Sleep Efficiency) to identify overall physiological recovery.
  - **Seasonal Confounder Detection**: Identifies and accounts for calendar date correlation ($r = +0.767$) with room temperature (e.g., infant sleep maturation across winter to summer).
  - Exports comprehensive GitHub Markdown reports with mathematical formulations (`--export-md`).

- **Cross-Platform Implementation**:
  - Python 3.13+ implementation with modern `uv` project mode, fast vectorization via Pandas, NumPy, and SciPy.
  - C# implementation (`ProcessSleepCo2.cs`) with dynamic user-relative path discovery (`~`) for standalone .NET execution.

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

### 2. Home Assistant Environmental Sensor Exports

Export historical sensor logs from Home Assistant History into CSV files:

- **$\text{CO}_2$ Monitor** (e.g. Ikea ALPSTUGA): `~/Downloads/CO2.csv` (timestamp and $\text{CO}_2$ concentration in ppm).
- **Bedroom Temperature Sensor**: `~/Downloads/bedroom_temp.csv` (timestamp and room temperature in °F or °C).

The CSV files should include:
- `last_changed` or `last_updated` (ISO timestamp)
- `state` (numerical sensor reading)

---

## Usage

### Step 1: Process and Merge Sleep, Temperature & $\text{CO}_2$ Data

Run `ProcessSleepCo2.py` to match each night's sleep window to sensor readings, compute time-weighted trapezoidal Riemann sum averages, and generate `sleep_co2_merged.csv`:

```bash
uv run python ProcessSleepCo2.py \
  --sleep-score "path/to/Takeout/Google Health/Sleep Score/sleep_score.csv" \
  --sleep-json-dir "path/to/Takeout/Google Health/Global Export Data" \
  --co2 "path/to/CO2.csv" \
  --room-temp "path/to/bedroom_temp.csv" \
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
| `-rt`, `--room-temp` | Path to Home Assistant `bedroom_temp.csv` | `~/Downloads/bedroom_temp.csv` |
| `-tz`, `--timezone` | Timezone for Google Health local records | `America/Denver` |
| `-cov`, `--coverage` | Minimum valid $\text{CO}_2$ sample coverage ($0.0 - 1.0$) | `0.70` |
| `--skip-regression` | Skip running the built-in regression summary | `False` |

---

### Step 2: Statistical CO₂ Regression Analysis

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

### Step 3: Optimal Bedroom Temperature Analysis

Run `sleep_co2_regression.py` with `--optimal-temp` to identify non-linear thermal optima using quadratic curve fitting, empirical $3^\circ\text{F}$ brackets, and multi-objective $Z$-score aggregation:

```bash
# Run quadratic curve fitting and empirical temperature brackets
uv run python sleep_co2_regression.py --optimal-temp

# Run temperature optimization and export full GitHub Markdown report
uv run python sleep_co2_regression.py --optimal-temp --export-md regression_report.md
```

---

## Metrics Analyzed

### Sleep Architecture & Biometrics

- **Sleep Quality**: Overall Sleep Score, Revitalization Score, Sleep Efficiency (%), Restlessness Index
- **Sleep Stages**: Deep Sleep (min & %), REM Sleep (min & %), Light Sleep (min)
- **Autonomic & Cardiovascular**: HRV RMSSD (ms), NREM Heart Rate (bpm), Resting Heart Rate (bpm)
- **Physiological**: Average SpO₂ (%), SpO₂ drops, Nightly Skin Temperature deviation (°C)
- **Composite Quality**: `CompositeSleepIndex` (weighted multi-objective $Z$-score of recovery metrics)

### Environmental Exposure: $\text{CO}_2$

- `AvgCo2Ppm`: Mean overnight concentration.
- `MaxCo2Ppm`: Peak overnight concentration.
- `HoursAbove1000Ppm` / `PctSleepAbove1000Ppm`: Exposure time exceeding indoor air quality thresholds ($1{,}000\text{ ppm}$).
- `Co2RisePpm` & `Co2RiseRatePerHour`: Rate and total volume of carbon dioxide buildup in the bedroom.

### Environmental Exposure: Bedroom Temperature

- `AvgRoomTempF`: Time-weighted average bedroom temperature ($^\circ\text{F}$) during sleep window.
- `MinRoomTempF`: Minimum overnight bedroom temperature ($^\circ\text{F}$).
- `MaxRoomTempF`: Peak overnight bedroom temperature ($^\circ\text{F}$).
- `RoomTempDeltaF`: Total overnight temperature fluctuation ($\text{Max} - \text{Min}$ in $^\circ\text{F}$).

---

## License

This project is licensed under the [MIT License](LICENSE).
