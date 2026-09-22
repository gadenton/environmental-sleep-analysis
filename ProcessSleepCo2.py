#!/usr/bin/env python3
"""
Align and analyze Fitbit/Google Health sleep scores and biometrics with Home Assistant CO2 sensor data.
Direct Python port of ProcessSleepCo2.cs.
"""

import os
import sys
import glob
import json
import csv
import math
import bisect
import argparse
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_EVEN
import zoneinfo
from tzlocal import windows_tz

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# =========================================================================================
# Helper Functions: Timezone & Numerical Rounding
# =========================================================================================

def resolve_timezone(tz_id: str):
    """
    Resolves timezone string to a zoneinfo.ZoneInfo without touching the Windows registry.
    Supports standard IANA names (e.g. 'America/Denver') and Windows timezone keys via tzlocal.
    """
    try:
        return zoneinfo.ZoneInfo(tz_id)
    except Exception:
        pass

    iana = windows_tz.win_tz.get(tz_id)
    if iana:
        try:
            return zoneinfo.ZoneInfo(iana)
        except Exception:
            pass

    return zoneinfo.ZoneInfo("America/Denver")


def round_half_even(val: float, digits: int) -> float:
    """
    Performs Banker's Rounding (MidpointRounding.ToEven) matching .NET's Math.Round.
    """
    if val is None:
        return None
    d = Decimal(str(val))
    q = Decimal("10") ** -digits
    return float(d.quantize(q, rounding=ROUND_HALF_EVEN))


def try_parse_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# =========================================================================================
# Data Models
# =========================================================================================

class Co2Point:
    __slots__ = ("timestamp", "ppm")
    def __init__(self, timestamp: datetime, ppm: float):
        self.timestamp = timestamp
        self.ppm = ppm


class SleepSessionInfo:
    __slots__ = (
        "log_id", "date_of_sleep", "start_local", "end_local",
        "start_utc", "end_utc", "duration_minutes", "minutes_asleep",
        "minutes_awake", "efficiency", "deep_minutes", "rem_minutes",
        "light_minutes", "wake_minutes"
    )
    def __init__(self, log_id, date_of_sleep, start_local, end_local,
                 start_utc, end_utc, duration_minutes, minutes_asleep,
                 minutes_awake, efficiency, deep_minutes, rem_minutes,
                 light_minutes, wake_minutes):
        self.log_id = log_id
        self.date_of_sleep = date_of_sleep
        self.start_local = start_local
        self.end_local = end_local
        self.start_utc = start_utc
        self.end_utc = end_utc
        self.duration_minutes = duration_minutes
        self.minutes_asleep = minutes_asleep
        self.minutes_awake = minutes_awake
        self.efficiency = efficiency
        self.deep_minutes = deep_minutes
        self.rem_minutes = rem_minutes
        self.light_minutes = light_minutes
        self.wake_minutes = wake_minutes


class SleepRecord:
    __slots__ = (
        "log_id", "session", "overall_score", "composition_score",
        "revitalization_score", "duration_score", "deep_sleep_minutes",
        "resting_heart_rate", "restlessness"
    )
    def __init__(self, log_id, session, overall_score, composition_score,
                 revitalization_score, duration_score, deep_sleep_minutes,
                 resting_heart_rate, restlessness):
        self.log_id = log_id
        self.session = session
        self.overall_score = overall_score
        self.composition_score = composition_score
        self.revitalization_score = revitalization_score
        self.duration_score = duration_score
        self.deep_sleep_minutes = deep_sleep_minutes
        self.resting_heart_rate = resting_heart_rate
        self.restlessness = restlessness


# =========================================================================================
# Data Loading Helpers for SpO2, HRV, Temperature
# =========================================================================================

def load_spo2(directory: str):
    mapping = {}
    if not os.path.exists(directory):
        return mapping

    for file in glob.glob(os.path.join(directory, "Daily SpO2 - *.csv")):
        try:
            with open(file, mode="r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp") or ""
                    date_str = ts[:10] if len(ts) >= 10 else ts
                    avg = try_parse_float(row.get("average_value"))
                    min_val = try_parse_float(row.get("lower_bound"))
                    if avg is not None and min_val is not None:
                        mapping[date_str] = (round_half_even(avg, 1), round_half_even(min_val, 1))
        except Exception:
            pass
    return mapping


def load_hrv(directory: str):
    mapping = {}
    if not os.path.exists(directory):
        return mapping

    for file in glob.glob(os.path.join(directory, "Daily Heart Rate Variability Summary - *.csv")):
        try:
            base_name = os.path.splitext(os.path.basename(file))[0]
            date_str = base_name.replace("Daily Heart Rate Variability Summary - ", "").strip()
            with open(file, mode="r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rmssd = try_parse_float(row.get("rmssd"))
                    nrem = try_parse_float(row.get("nremhr"))
                    ent = try_parse_float(row.get("entropy"))
                    if rmssd is not None and nrem is not None and ent is not None:
                        mapping[date_str] = (round_half_even(rmssd, 1), round_half_even(nrem, 1), round_half_even(ent, 3))
                    break
        except Exception:
            pass
    return mapping


def load_temperature(directory: str):
    mapping = {}
    if not os.path.exists(directory):
        return mapping

    for file in glob.glob(os.path.join(directory, "Computed Temperature - *.csv")):
        try:
            with open(file, mode="r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sleep_end = row.get("sleep_end") or ""
                    date_str = sleep_end[:10] if len(sleep_end) >= 10 else sleep_end
                    temp = try_parse_float(row.get("nightly_temperature"))
                    if temp is not None:
                        mapping[date_str] = round_half_even(temp, 2)
        except Exception:
            pass
    return mapping


# =========================================================================================
# General Math Helpers
# =========================================================================================

def interpolate_co2(pts, target: datetime) -> float:
    if not pts:
        return 500.0
    if target <= pts[0].timestamp:
        return pts[0].ppm
    if target >= pts[-1].timestamp:
        return pts[-1].ppm

    for i in range(len(pts) - 1):
        if pts[i].timestamp <= target <= pts[i + 1].timestamp:
            t1 = pts[i].timestamp
            t2 = pts[i + 1].timestamp
            if t2 == t1:
                return pts[i].ppm
            factor = (target - t1).total_seconds() / (t2 - t1).total_seconds()
            return pts[i].ppm + factor * (pts[i + 1].ppm - pts[i].ppm)
    return pts[-1].ppm


def pearson(x, y):
    n = len(x)
    if n != len(y) or n == 0:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    num = 0.0
    d1 = 0.0
    d2 = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        dy = yi - my
        num += dx * dy
        d1 += dx * dx
        d2 += dy * dy
    den = math.sqrt(d1 * d2)
    return num / den if den != 0.0 else 0.0


# =========================================================================================
# Statistical Analysis & Correlation Printing
# =========================================================================================

def print_analysis_summary(rows):
    banner_line = "=" * 141
    print(banner_line)
    print("                                                   COMPREHENSIVE MULTI-SENSOR ANALYSIS")
    print(banner_line)

    avg_co2_list = [r["AvgCo2Ppm"] for r in rows]
    max_co2_list = [r["MaxCo2Ppm"] for r in rows]
    scores = [r["OverallScore"] for r in rows]

    avg_co2_mean = sum(avg_co2_list) / len(avg_co2_list)
    max_co2_mean = sum(max_co2_list) / len(max_co2_list)
    scores_mean = sum(scores) / len(scores)

    print(f"Dataset Overview ({len(rows)} Qualifying Nights):")
    print(f"  Average Nightly CO2 : {avg_co2_mean:.1f} ppm (Min: {min(avg_co2_list):.1f}, Max: {max(avg_co2_list):.1f})")
    print(f"  Peak Nightly CO2    : {max_co2_mean:.1f} ppm (Min: {min(max_co2_list):.1f}, Max: {max(max_co2_list):.1f})")

    has_room_temp = any(r.get("AvgRoomTempF") is not None for r in rows)
    if has_room_temp:
        temps = [r["AvgRoomTempF"] for r in rows if r.get("AvgRoomTempF") is not None]
        print(f"  Bedroom Temperature : {sum(temps)/len(temps):.1f}°F (Min: {min(temps):.1f}°F, Max: {max(temps):.1f}°F)")

    print(f"  Overall Sleep Score : {scores_mean:.1f} (Min: {min(scores):.0f}, Max: {max(scores):.0f})")

    vent_count = sum(1 for r in rows if r["RoomVentilationState"] == "Ventilated")
    sealed_count = sum(1 for r in rows if r["RoomVentilationState"] == "Sealed")
    mod_count = len(rows) - vent_count - sealed_count
    print(f"  Room Dynamics       : {vent_count} Ventilated (<750 ppm), {mod_count} Moderate, {sealed_count} Sealed (>1000 ppm)\n")

    sep_width = 158 if has_room_temp else 141
    sep_line = "-" * sep_width
    print(sep_line)
    if has_room_temp:
        print("| Biometric / Sleep Metric       | vs Avg CO2 (r) | vs Max CO2 (r) | vs Hrs>1k (r) | vs Room Temp (r) | vs Sleep Score (r) | N Valid |  Metric Mean |")
    else:
        print("| Biometric / Sleep Metric       | vs Avg CO2 (r) | vs Max CO2 (r) | vs Hrs>1k (r) | vs %>1k (r) | vs Sleep Score (r) | N Valid |  Metric Mean |")
    print(sep_line)

    print_stat_row("Overall Sleep Score", rows, lambda r: r["OverallScore"], has_room_temp)
    print_stat_row("HRV Recovery (RMSSD ms)", rows, lambda r: r["HrvRmssd"], has_room_temp)
    print_stat_row("NREM Heart Rate (BPM)", rows, lambda r: r["HrvNremHeartRate"], has_room_temp)
    print_stat_row("Resting Heart Rate (BPM)", rows, lambda r: r["RestingHeartRate"], has_room_temp)
    print_stat_row("Nightly Skin Temp (°C)", rows, lambda r: r["NightlySkinTempCelsius"], has_room_temp)
    print_stat_row("SpO2 Average (%)", rows, lambda r: r["SpO2Avg"], has_room_temp)
    print_stat_row("SpO2 Min Drop (%)", rows, lambda r: r["SpO2Min"], has_room_temp)
    print_stat_row("Deep Sleep (Minutes)", rows, lambda r: r["DeepSleepMinutes"], has_room_temp)
    print_stat_row("REM Sleep (Minutes)", rows, lambda r: r["RemSleepMinutes"], has_room_temp)
    print_stat_row("Restlessness Score", rows, lambda r: r["Restlessness"], has_room_temp)
    print_stat_row("Sleep Efficiency (%)", rows, lambda r: float(r["SleepEfficiency"]) if r["SleepEfficiency"] is not None else None, has_room_temp)
    print_stat_row("Total Sleep Duration (Hours)", rows, lambda r: r["SleepDurationHours"], has_room_temp)
    print_stat_row("CO2 Rise Rate (ppm/hr)", rows, lambda r: r["Co2RiseRatePerHour"], has_room_temp)
    if has_room_temp:
        print_stat_row("Bedroom Temp (°F)", rows, lambda r: r.get("AvgRoomTempF"), has_room_temp)
    print(sep_line + "\n")


def print_stat_row(name, rows, metric_selector, has_room_temp: bool = False):
    valid = [r for r in rows if metric_selector(r) is not None]
    if len(valid) < 3:
        return

    xs = [metric_selector(r) for r in valid]
    co2_avg = [r["AvgCo2Ppm"] for r in valid]
    co2_max = [r["MaxCo2Ppm"] for r in valid]
    hrs_above = [r["HoursAbove1000Ppm"] for r in valid]
    pct_above = [r["PctSleepAbove1000Ppm"] for r in valid]
    scores = [r["OverallScore"] for r in valid]

    r_co2_avg = pearson(xs, co2_avg)
    r_co2_max = pearson(xs, co2_max)
    r_hrs = pearson(xs, hrs_above)
    r_score = pearson(xs, scores)
    mean_val = sum(xs) / len(xs)

    if has_room_temp:
        room_temps = [r["AvgRoomTempF"] for r in valid if r.get("AvgRoomTempF") is not None]
        if len(room_temps) == len(valid):
            r_rt = pearson(xs, room_temps)
            rt_col = f"{r_rt:16.4f}"
        else:
            rt_col = f"{'N/A':>16}"
        print(f"| {name:<30} | {r_co2_avg:14.4f} | {r_co2_max:14.4f} | {r_hrs:13.4f} | {rt_col} | {r_score:18.4f} | {len(valid):7d} | {mean_val:12.2f} |")
    else:
        r_pct = pearson(xs, pct_above)
        print(f"| {name:<30} | {r_co2_avg:14.4f} | {r_co2_max:14.4f} | {r_hrs:13.4f} | {r_pct:11.4f} | {r_score:18.4f} | {len(valid):7d} | {mean_val:12.2f} |")


# =========================================================================================
# Pipeline Implementation
# =========================================================================================

def run_pipeline(
    sleep_score_path: str,
    sleep_json_dir: str,
    co2_path: str,
    spo2_dir: str,
    hrv_dir: str,
    temp_dir: str,
    output_path: str,
    coverage_threshold: float,
    tz_id: str,
    bedroom_temp_path: str = None,
    skip_regression: bool = False
):
    sleep_score_path = os.path.expanduser(sleep_score_path)
    sleep_json_dir = os.path.expanduser(sleep_json_dir)
    co2_path = os.path.expanduser(co2_path)
    spo2_dir = os.path.expanduser(spo2_dir)
    hrv_dir = os.path.expanduser(hrv_dir)
    temp_dir = os.path.expanduser(temp_dir)
    output_path = os.path.expanduser(output_path)
    bedroom_temp_path = os.path.expanduser(bedroom_temp_path) if bedroom_temp_path else None

    print("====================================================================")
    print("     Sleep Score, Biometrics & CO2 Multi-Sensor Pre-processor")
    print("====================================================================")
    print(f"Sleep Score CSV : {sleep_score_path}")
    print(f"Sleep JSON Dir  : {sleep_json_dir}")
    print(f"CO2 CSV         : {co2_path}")
    if bedroom_temp_path:
        print(f"Bedroom Temp CSV: {bedroom_temp_path}")
    print(f"SpO2 Dir        : {spo2_dir}")
    print(f"HRV Dir         : {hrv_dir}")
    print(f"Temperature Dir : {temp_dir}")
    print(f"Coverage Cutoff : {int(coverage_threshold * 100)}%")
    print(f"Output File     : {os.path.abspath(output_path)}")

    tz = resolve_timezone(tz_id)
    print(f"Active Timezone : {tz.key}\n")

    # 1. Index Sleep Sessions from Google Takeout JSON files
    sleep_map = {}

    if os.path.exists(sleep_json_dir):
        json_files = glob.glob(os.path.join(sleep_json_dir, "sleep-*.json"))
        json_files.sort()
        print(f"[1/6] Scanning {len(json_files)} sleep export JSON files...")

        for file in json_files:
            try:
                with open(file, "r", encoding="utf-8") as fs:
                    doc = json.load(fs)
                    for elem in doc:
                        if "logId" not in elem:
                            continue
                        log_id = int(elem["logId"])

                        start_str = elem["startTime"]
                        end_str = elem["endTime"]
                        date_of_sleep = elem["dateOfSleep"]
                        dur_minutes = (elem.get("duration") / 60000.0) if "duration" in elem and elem["duration"] is not None else 0.0
                        min_asleep = elem.get("minutesAsleep") or 0
                        min_awake = elem.get("minutesAwake") or 0
                        efficiency = elem.get("efficiency") or 0

                        deep_min, rem_min, light_min, wake_min = None, None, None, None
                        levels = elem.get("levels")
                        if levels and "summary" in levels:
                            summary = levels["summary"]
                            if "deep" in summary and "minutes" in summary["deep"]:
                                deep_min = summary["deep"]["minutes"]
                            if "rem" in summary and "minutes" in summary["rem"]:
                                rem_min = summary["rem"]["minutes"]
                            if "light" in summary and "minutes" in summary["light"]:
                                light_min = summary["light"]["minutes"]
                            if "wake" in summary and "minutes" in summary["wake"]:
                                wake_min = summary["wake"]["minutes"]

                        start_dt = datetime.fromisoformat(start_str.replace("Z", ""))
                        end_dt = datetime.fromisoformat(end_str.replace("Z", ""))

                        start_utc = start_dt.replace(tzinfo=tz).astimezone(timezone.utc)
                        end_utc = end_dt.replace(tzinfo=tz).astimezone(timezone.utc)

                        sleep_map[log_id] = SleepSessionInfo(
                            log_id,
                            date_of_sleep,
                            start_dt,
                            end_dt,
                            start_utc,
                            end_utc,
                            dur_minutes,
                            min_asleep,
                            min_awake,
                            efficiency,
                            deep_min,
                            rem_min,
                            light_min,
                            wake_min
                        )
            except Exception as ex:
                print(f"  Warning reading {os.path.basename(file)}: {ex}")

        print(f"      Indexed {len(sleep_map)} unique sleep sessions from JSON.")
    else:
        print(f"[1/6] Sleep JSON dir not found: {sleep_json_dir}. Will rely solely on sleep_score.csv timestamps.")

    # 2. Read sleep_score.csv
    print("[2/6] Reading sleep score CSV...")
    sleep_records = []

    with open(sleep_score_path, mode="r", encoding="utf-8-sig", newline="") as reader:
        csv_reader = csv.DictReader(reader)
        for row in csv_reader:
            log_id = int(row["sleep_log_entry_id"])
            score_str = row.get("overall_score")
            overall_score = try_parse_float(score_str)
            if overall_score is None:
                continue

            comp = try_parse_float(row.get("composition_score"))
            rev = try_parse_float(row.get("revitalization_score"))
            dur_score = try_parse_float(row.get("duration_score"))
            deep_min_csv = try_parse_float(row.get("deep_sleep_in_minutes"))
            resting_hr = try_parse_float(row.get("resting_heart_rate"))
            restless = try_parse_float(row.get("restlessness"))

            if log_id in sleep_map:
                session = sleep_map[log_id]
            else:
                ts_str = row["timestamp"]
                end_local = datetime.fromisoformat(ts_str.replace("Z", ""))
                start_local = end_local - timedelta(hours=8)
                start_utc = start_local.replace(tzinfo=tz).astimezone(timezone.utc)
                end_utc = end_local.replace(tzinfo=tz).astimezone(timezone.utc)
                session = SleepSessionInfo(
                    log_id,
                    end_local.strftime("%Y-%m-%d"),
                    start_local,
                    end_local,
                    start_utc,
                    end_utc,
                    480.0,
                    int(480 * 0.85),
                    int(480 * 0.15),
                    85,
                    int(deep_min_csv) if deep_min_csv is not None else None,
                    None, None, None
                )

            sleep_records.append(SleepRecord(
                log_id,
                session,
                overall_score,
                comp,
                rev,
                dur_score,
                deep_min_csv if deep_min_csv is not None else (float(session.deep_minutes) if session.deep_minutes is not None else None),
                resting_hr,
                restless
            ))

    print(f"      Loaded {len(sleep_records)} sleep score entries.")

    # 3. Read Additional Biometrics: SpO2, HRV, Temperature
    print("[3/6] Reading additional biometrics (SpO2, HRV, Skin Temp)...")
    spo2_map = load_spo2(spo2_dir)
    hrv_map = load_hrv(hrv_dir)
    temp_map = load_temperature(temp_dir)
    print(f"      Loaded {len(spo2_map)} SpO2 daily entries, {len(hrv_map)} HRV entries, {len(temp_map)} nightly skin temp entries.")

    # 4. Read Home Assistant CO2 Data
    print("[4/6] Reading Home Assistant CO2 data...")
    co2_readings = []

    with open(co2_path, mode="r", encoding="utf-8-sig", newline="") as reader:
        csv_reader = csv.DictReader(reader)
        for row in csv_reader:
            ts_str = row.get("last_changed") or ""
            val_str = row.get("state") or ""
            val = try_parse_float(val_str)
            if ts_str and val is not None:
                try:
                    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    co2_readings.append(Co2Point(dt, val))
                except ValueError:
                    pass

    co2_readings.sort(key=lambda a: a.timestamp)
    co2_timestamps = [c.timestamp for c in co2_readings]
    print(f"      Loaded {len(co2_readings)} CO2 readings from {co2_readings[0].timestamp.strftime('%Y-%m-%d')} to {co2_readings[-1].timestamp.strftime('%Y-%m-%d')} UTC.")

    # 4b. Read Home Assistant Bedroom Temperature Data
    room_temp_readings = []
    room_temp_timestamps = []
    if bedroom_temp_path and os.path.exists(bedroom_temp_path):
        print(f"[4b] Reading Home Assistant Bedroom Temperature data from {bedroom_temp_path}...")
        with open(bedroom_temp_path, mode="r", encoding="utf-8-sig", newline="") as reader:
            csv_reader = csv.DictReader(reader)
            for row in csv_reader:
                ts_str = row.get("last_changed") or ""
                val_str = row.get("state") or ""
                val = try_parse_float(val_str)
                if ts_str and val is not None:
                    try:
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        room_temp_readings.append(Co2Point(dt, val))
                    except ValueError:
                        pass
        room_temp_readings.sort(key=lambda a: a.timestamp)
        room_temp_timestamps = [c.timestamp for c in room_temp_readings]
        if room_temp_readings:
            print(f"      Loaded {len(room_temp_readings)} bedroom temperature readings from {room_temp_readings[0].timestamp.strftime('%Y-%m-%d')} to {room_temp_readings[-1].timestamp.strftime('%Y-%m-%d')} UTC.")

    # 5. Align Sleep Windows with CO2, Calculate Coverage and Time-Weighted CO2
    print(f"[5/6] Aligning sleep intervals, computing time-weighted CO2 & dynamics, and filtering by >= {int(coverage_threshold * 100)}% coverage...")

    qualifying_records = []
    excluded_before_co2 = 0
    excluded_low_coverage = 0
    max_continuous_gap = timedelta(minutes=80)

    for rec in sleep_records:
        s = rec.session

        if s.end_utc < co2_readings[0].timestamp or s.start_utc > co2_readings[-1].timestamp:
            excluded_before_co2 += 1
            continue

        sleep_duration = s.end_utc - s.start_utc
        if sleep_duration.total_seconds() < 3600:
            continue

        target_first = s.start_utc - timedelta(hours=2)
        first_idx = bisect.bisect_left(co2_timestamps, target_first)
        if first_idx < 0:
            first_idx = 0

        window_readings = []
        target_last = s.end_utc + timedelta(hours=2)
        for i in range(first_idx, len(co2_readings)):
            pt = co2_readings[i]
            if pt.timestamp > target_last:
                break
            window_readings.append(pt)

        if not window_readings:
            excluded_low_coverage += 1
            continue

        # Calculate Coverage
        covered_seconds = 0.0
        for i in range(len(window_readings) - 1):
            t1 = window_readings[i].timestamp
            t2 = window_readings[i + 1].timestamp
            gap = t2 - t1

            if gap <= max_continuous_gap:
                seg_start = s.start_utc if t1 < s.start_utc else t1
                seg_end = s.end_utc if t2 > s.end_utc else t2
                if seg_end > seg_start:
                    covered_seconds += (seg_end - seg_start).total_seconds()

        coverage_pct = min(1.0, covered_seconds / sleep_duration.total_seconds())
        if coverage_pct < coverage_threshold:
            excluded_low_coverage += 1
            continue

        # Calculate Time-Weighted Average, Peak, and Dynamics
        points_in_sleep = [c for c in window_readings if s.start_utc <= c.timestamp <= s.end_utc]
        if not points_in_sleep:
            excluded_low_coverage += 1
            continue

        max_co2 = max(p.ppm for p in points_in_sleep)
        start_co2 = interpolate_co2(window_readings, s.start_utc)
        end_co2 = interpolate_co2(window_readings, s.end_utc)

        profile = [(s.start_utc, start_co2)]
        for pt in points_in_sleep:
            if s.start_utc < pt.timestamp < s.end_utc:
                profile.append((pt.timestamp, pt.ppm))
        profile.append((s.end_utc, end_co2))

        total_ppm_seconds = 0.0
        integrated_seconds = 0.0
        seconds_above_1000 = 0.0

        for i in range(len(profile) - 1):
            t1, v1 = profile[i]
            t2, v2 = profile[i + 1]
            dt = (t2 - t1).total_seconds()

            if dt > 0 and (t2 - t1) <= max_continuous_gap:
                total_ppm_seconds += 0.5 * (v1 + v2) * dt
                integrated_seconds += dt

                if v1 >= 1000 and v2 >= 1000:
                    seconds_above_1000 += dt
                elif v1 < 1000 and v2 > 1000:
                    frac = (v2 - 1000) / (v2 - v1)
                    seconds_above_1000 += dt * frac
                elif v1 > 1000 and v2 < 1000:
                    frac = (v1 - 1000) / (v1 - v2)
                    seconds_above_1000 += dt * frac

        avg_co2 = (total_ppm_seconds / integrated_seconds) if integrated_seconds > 0 else (sum(p.ppm for p in points_in_sleep) / len(points_in_sleep))
        hours_above_1000 = round_half_even(seconds_above_1000 / 3600.0, 2)
        pct_above_1000 = round_half_even((seconds_above_1000 / sleep_duration.total_seconds()) * 100.0, 1) if sleep_duration.total_seconds() > 0 else 0.0

        # Room ventilation dynamics:
        co2_rise = max(0.0, max_co2 - start_co2)
        hours_in_bed = sleep_duration.total_seconds() / 3600.0
        rise_rate = (co2_rise / hours_in_bed) if hours_in_bed > 0 else 0.0
        room_state = "Ventilated" if (max_co2 < 750 and co2_rise < 250) else ("Sealed" if max_co2 > 1000 else "Moderate")

        # Biometric matches by DateOfSleep:
        has_spo2 = s.date_of_sleep in spo2_map
        spo2_val = spo2_map.get(s.date_of_sleep)
        has_hrv = s.date_of_sleep in hrv_map
        hrv_val = hrv_map.get(s.date_of_sleep)
        has_temp = s.date_of_sleep in temp_map
        temp_val = temp_map.get(s.date_of_sleep)

        # Bedroom temperature integration during sleep
        avg_room_temp, min_room_temp, max_room_temp, delta_room_temp = None, None, None, None
        if room_temp_readings:
            first_t_idx = bisect.bisect_left(room_temp_timestamps, s.start_utc - timedelta(hours=2))
            t_window = []
            for i in range(max(0, first_t_idx), len(room_temp_readings)):
                pt = room_temp_readings[i]
                if pt.timestamp > s.end_utc + timedelta(hours=2):
                    break
                t_window.append(pt)

            t_in_sleep = [p for p in t_window if s.start_utc <= p.timestamp <= s.end_utc]
            if t_in_sleep:
                start_t = interpolate_co2(t_window, s.start_utc)
                end_t = interpolate_co2(t_window, s.end_utc)
                t_profile = [(s.start_utc, start_t)]
                for pt in t_in_sleep:
                    if s.start_utc < pt.timestamp < s.end_utc:
                        t_profile.append((pt.timestamp, pt.ppm))
                t_profile.append((s.end_utc, end_t))

                total_t_sec = 0.0
                int_t_sec = 0.0
                for i in range(len(t_profile) - 1):
                    t1, v1 = t_profile[i]
                    t2, v2 = t_profile[i + 1]
                    dt = (t2 - t1).total_seconds()
                    if dt > 0 and dt <= max_continuous_gap.total_seconds():
                        total_t_sec += 0.5 * (v1 + v2) * dt
                        int_t_sec += dt

                avg_room_temp = (total_t_sec / int_t_sec) if int_t_sec > 0 else (sum(p.ppm for p in t_in_sleep) / len(t_in_sleep))
                min_room_temp = min(p.ppm for p in t_in_sleep)
                max_room_temp = max(p.ppm for p in t_in_sleep)
                delta_room_temp = max_room_temp - min_room_temp
            elif 0 <= first_t_idx < len(room_temp_readings):
                val = room_temp_readings[first_t_idx].ppm
                avg_room_temp, min_room_temp, max_room_temp, delta_room_temp = val, val, val, 0.0

        qualifying_records.append({
            "DateOfSleep": s.date_of_sleep,
            "DayOfWeek": s.start_local.strftime("%A"),
            "StartTimeLocal": s.start_local.strftime("%Y-%m-%d %H:%M:%S"),
            "EndTimeLocal": s.end_local.strftime("%Y-%m-%d %H:%M:%S"),
            "StartTimeUtc": s.start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "EndTimeUtc": s.end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "SleepDurationHours": round_half_even(s.duration_minutes / 60.0, 2),
            "MinutesAsleep": s.minutes_asleep,
            "MinutesAwake": s.minutes_awake,
            "SleepEfficiency": s.efficiency,
            "OverallScore": rec.overall_score,
            "CompositionScore": rec.composition_score,
            "RevitalizationScore": rec.revitalization_score,
            "DurationScore": rec.duration_score,
            "DeepSleepMinutes": rec.deep_sleep_minutes,
            "RemSleepMinutes": s.rem_minutes,
            "LightSleepMinutes": s.light_minutes,
            "WakeSleepMinutes": s.wake_minutes,
            "RestingHeartRate": rec.resting_heart_rate,
            "Restlessness": rec.restlessness,
            "SpO2Avg": spo2_val[0] if has_spo2 else None,
            "SpO2Min": spo2_val[1] if has_spo2 else None,
            "HrvRmssd": hrv_val[0] if has_hrv else None,
            "HrvNremHeartRate": hrv_val[1] if has_hrv else None,
            "HrvEntropy": hrv_val[2] if has_hrv else None,
            "NightlySkinTempCelsius": temp_val if has_temp else None,
            "AvgRoomTempF": round_half_even(avg_room_temp, 1) if avg_room_temp is not None else None,
            "MinRoomTempF": round_half_even(min_room_temp, 1) if min_room_temp is not None else None,
            "MaxRoomTempF": round_half_even(max_room_temp, 1) if max_room_temp is not None else None,
            "RoomTempDeltaF": round_half_even(delta_room_temp, 1) if delta_room_temp is not None else None,
            "Co2StartPpm": round_half_even(start_co2, 1),
            "AvgCo2Ppm": round_half_even(avg_co2, 1),
            "MaxCo2Ppm": round_half_even(max_co2, 1),
            "HoursAbove1000Ppm": hours_above_1000,
            "PctSleepAbove1000Ppm": pct_above_1000,
            "Co2RisePpm": round_half_even(co2_rise, 1),
            "Co2RiseRatePerHour": round_half_even(rise_rate, 1),
            "RoomVentilationState": room_state,
            "Co2CoveragePct": round_half_even(coverage_pct * 100.0, 1),
            "Co2SampleCount": len(points_in_sleep)
        })

    # Deduplicate multiple logs on the same date by keeping the primary/longest session
    grouped = {}
    for r in qualifying_records:
        dos = r["DateOfSleep"]
        if dos not in grouped or r["SleepDurationHours"] > grouped[dos]["SleepDurationHours"]:
            grouped[dos] = r

    qualifying_records = [grouped[k] for k in sorted(grouped.keys())]

    # Identify and remove any columns that have no values at all across all qualifying records
    all_properties = [
        "DateOfSleep", "DayOfWeek", "StartTimeLocal", "EndTimeLocal",
        "StartTimeUtc", "EndTimeUtc", "SleepDurationHours", "MinutesAsleep",
        "MinutesAwake", "SleepEfficiency", "OverallScore", "CompositionScore",
        "RevitalizationScore", "DurationScore", "DeepSleepMinutes", "RemSleepMinutes",
        "LightSleepMinutes", "WakeSleepMinutes", "RestingHeartRate", "Restlessness",
        "SpO2Avg", "SpO2Min", "HrvRmssd", "HrvNremHeartRate", "HrvEntropy",
        "NightlySkinTempCelsius", "AvgRoomTempF", "MinRoomTempF", "MaxRoomTempF",
        "RoomTempDeltaF", "Co2StartPpm", "AvgCo2Ppm", "MaxCo2Ppm",
        "HoursAbove1000Ppm", "PctSleepAbove1000Ppm", "Co2RisePpm",
        "Co2RiseRatePerHour", "RoomVentilationState", "Co2CoveragePct",
        "Co2SampleCount"
    ]

    active_props = [
        p for p in all_properties
        if any(r[p] is not None and (not isinstance(r[p], str) or r[p].strip() != "") for r in qualifying_records)
    ]
    removed_props = [p for p in all_properties if p not in active_props]

    def format_csv_field(val):
        if val is None:
            return ""
        if isinstance(val, float):
            return str(int(val)) if val.is_integer() else str(val)
        return str(val)

    # Write Output CSV with only active columns
    with open(output_path, "w", newline="", encoding="utf-8") as writer_file:
        writer = csv.writer(writer_file)
        writer.writerow(active_props)
        for record in qualifying_records:
            writer.writerow([format_csv_field(record[p]) for p in active_props])

    print(f"\nSuccessfully generated: {output_path}")
    if removed_props:
        print(f"  - Removed Empty Columns:      {', '.join(removed_props)}")
    print(f"  - Active Columns Retained:    {len(active_props)} of {len(all_properties)}")
    print(f"  - Total Sleep Logs:           {len(sleep_records)}")
    print(f"  - Prior to CO2 Logging:       {excluded_before_co2}")
    print(f"  - Excluded (< {int(coverage_threshold * 100)}% coverage): {excluded_low_coverage}")
    print(f"  - Final Qualifying Nights:    {len(qualifying_records)} unique dates\n")

    # 6. Comprehensive Statistical Analysis
    if len(qualifying_records) > 2:
        print_analysis_summary(qualifying_records)

        if not skip_regression:
            try:
                from sleep_co2_regression import analyze_sleep_co2_regression
                print("\n" + "=" * 141)
                print("                     MULTIPLE REGRESSION ANALYSIS (ADJUSTING FOR ERRATIC SLEEP TIME AS COVARIATE)")
                print("=" * 141)
                reg_metrics = ["AvgCo2Ppm", "MaxCo2Ppm"]
                if any(r.get("AvgRoomTempF") is not None for r in qualifying_records):
                    reg_metrics.append("AvgRoomTempF")
                analyze_sleep_co2_regression(
                    qualifying_records,
                    co2_metrics=reg_metrics,
                    covariates=["duration"],
                    verbose=True
                )
            except Exception as ex:
                print(f"\n[Notice] Regression analysis could not be displayed: {ex}")


# =========================================================================================
# Command-Line Interface Definition
# =========================================================================================

def get_default_takeout_dir() -> str:
    user_home = os.path.expanduser("~")
    downloads = os.path.join(user_home, "Downloads")
    matches = sorted(glob.glob(os.path.join(downloads, "takeout-*", "Takeout", "Google Health")), reverse=True)
    if matches:
        return matches[0]
    return os.path.join(downloads, "Takeout", "Google Health")


def get_default_co2_path() -> str:
    return os.path.join(os.path.expanduser("~"), "Downloads", "CO2.csv")


def get_default_bedroom_temp_path() -> str:
    return os.path.join(os.path.expanduser("~"), "Downloads", "bedroom_temp.csv")


def main():
    takeout_default = get_default_takeout_dir()
    co2_default = get_default_co2_path()
    bedroom_temp_default = get_default_bedroom_temp_path()

    parser = argparse.ArgumentParser(
        description="Align and analyze Fitbit/Google Health sleep scores and biometrics with Home Assistant CO2 sensor data."
    )
    parser.add_argument(
        "-s", "--sleep-score",
        dest="sleep_score",
        default=os.path.join(takeout_default, "Sleep Score", "sleep_score.csv"),
        help="Path to sleep_score.csv"
    )
    parser.add_argument(
        "-j", "--sleep-json-dir",
        dest="sleep_json_dir",
        default=os.path.join(takeout_default, "Global Export Data"),
        help="Directory containing sleep-*.json export files from Google Health Takeout"
    )
    parser.add_argument(
        "-c", "--co2",
        dest="co2",
        default=co2_default,
        help="Path to Home Assistant CO2.csv"
    )
    parser.add_argument(
        "-bt", "--bedroom-temp",
        dest="bedroom_temp",
        default=bedroom_temp_default,
        help="Path to Home Assistant bedroom_temp.csv"
    )
    parser.add_argument(
        "-spo2", "--spo2-dir",
        dest="spo2_dir",
        default=os.path.join(takeout_default, "Oxygen Saturation (SpO2)"),
        help="Directory containing Daily SpO2 - *.csv files from Google Health Takeout"
    )
    parser.add_argument(
        "-hrv", "--hrv-dir",
        dest="hrv_dir",
        default=os.path.join(takeout_default, "Heart Rate Variability"),
        help="Directory containing Daily Heart Rate Variability Summary - *.csv files"
    )
    parser.add_argument(
        "-temp", "--temp-dir",
        dest="temp_dir",
        default=os.path.join(takeout_default, "Temperature"),
        help="Directory containing Computed Temperature - *.csv files"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output",
        default="sleep_co2_merged.csv",
        help="Output path for the merged CSV dataset"
    )
    parser.add_argument(
        "-cov", "--coverage",
        dest="coverage",
        type=float,
        default=0.70,
        help="Minimum CO2 coverage threshold required during sleep (0.0 - 1.0)"
    )
    parser.add_argument(
        "-tz", "--timezone",
        dest="timezone",
        default="America/Denver",
        help="Timezone ID for Fitbit local timestamps (e.g. America/Denver or Mountain Standard Time)"
    )
    parser.add_argument(
        "--skip-regression",
        dest="skip_regression",
        action="store_true",
        help="Skip covariate regression analysis and only display bivariate correlations"
    )

    args = parser.parse_args()

    run_pipeline(
        sleep_score_path=args.sleep_score,
        sleep_json_dir=args.sleep_json_dir,
        co2_path=args.co2,
        spo2_dir=args.spo2_dir,
        hrv_dir=args.hrv_dir,
        temp_dir=args.temp_dir,
        output_path=args.output,
        coverage_threshold=args.coverage,
        tz_id=args.timezone,
        bedroom_temp_path=args.bedroom_temp,
        skip_regression=args.skip_regression
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
