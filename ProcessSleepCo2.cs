#:package CsvHelper@33.0.1
#:package System.CommandLine@2.0.12

using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Globalization;
using System.Collections.Generic;
using System.CommandLine;
using CsvHelper;

// =========================================================================================
// Command-Line Interface Definition via System.CommandLine
// =========================================================================================

static string GetDefaultTakeoutDir()
{
    string downloads = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads");
    if (Directory.Exists(downloads))
    {
        var matches = Directory.GetDirectories(downloads, "takeout-*")
            .Select(d => Path.Combine(d, "Takeout", "Google Health"))
            .Where(Directory.Exists)
            .OrderByDescending(d => d)
            .ToList();
        if (matches.Count > 0) return matches[0];
    }
    return Path.Combine(downloads, "Takeout", "Google Health");
}

static string GetDefaultCo2Path() =>
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "CO2.csv");

static string GetDefaultRoomTempPath() =>
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Downloads", "bedroom_temp.csv");

static string ExpandPath(string path)
{
    if (string.IsNullOrWhiteSpace(path)) return path;
    if (path.StartsWith("~"))
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        path = Path.Combine(home, path.TrimStart('~', '/', '\\'));
    }
    return path;
}

var sleepScoreOption = new Option<string>("--sleep-score", "-s")
{
    Description = "Path to sleep_score.csv",
    DefaultValueFactory = _ => Path.Combine(GetDefaultTakeoutDir(), "Sleep Score", "sleep_score.csv")
};

var sleepJsonDirOption = new Option<string>("--sleep-json-dir", "-j")
{
    Description = "Directory containing sleep-*.json export files from Google Health Takeout",
    DefaultValueFactory = _ => Path.Combine(GetDefaultTakeoutDir(), "Global Export Data")
};

var co2Option = new Option<string>("--co2", "-c")
{
    Description = "Path to Home Assistant CO2.csv",
    DefaultValueFactory = _ => GetDefaultCo2Path()
};

var roomTempOption = new Option<string>("--bedroom-temp", "-bt")
{
    Description = "Path to Home Assistant bedroom_temp.csv",
    DefaultValueFactory = _ => GetDefaultRoomTempPath()
};

var spo2DirOption = new Option<string>("--spo2-dir", "-spo2")
{
    Description = "Directory containing Daily SpO2 - *.csv files from Google Health Takeout",
    DefaultValueFactory = _ => Path.Combine(GetDefaultTakeoutDir(), "Oxygen Saturation (SpO2)")
};

var hrvDirOption = new Option<string>("--hrv-dir", "-hrv")
{
    Description = "Directory containing Daily Heart Rate Variability Summary - *.csv files",
    DefaultValueFactory = _ => Path.Combine(GetDefaultTakeoutDir(), "Heart Rate Variability")
};

var tempDirOption = new Option<string>("--temp-dir", "-temp")
{
    Description = "Directory containing Computed Temperature - *.csv files",
    DefaultValueFactory = _ => Path.Combine(GetDefaultTakeoutDir(), "Temperature")
};

var outputOption = new Option<string>("--output", "-o")
{
    Description = "Output path for the merged CSV dataset",
    DefaultValueFactory = _ => "sleep_co2_merged.csv"
};

var coverageOption = new Option<double>("--coverage", "-cov")
{
    Description = "Minimum CO2 coverage threshold required during sleep (0.0 - 1.0)",
    DefaultValueFactory = _ => 0.70
};

var timezoneOption = new Option<string>("--timezone", "-tz")
{
    Description = "Timezone ID for Fitbit local timestamps (e.g. America/Denver or Mountain Standard Time)",
    DefaultValueFactory = _ => "America/Denver"
};

var rootCommand = new RootCommand("Align and analyze Fitbit/Google Health sleep scores and biometrics with Home Assistant CO2 sensor data.")
{
    sleepScoreOption,
    sleepJsonDirOption,
    co2Option,
    roomTempOption,
    spo2DirOption,
    hrvDirOption,
    tempDirOption,
    outputOption,
    coverageOption,
    timezoneOption
};

rootCommand.SetAction(parseResult =>
{
    string sleepScorePath = ExpandPath(parseResult.GetValue(sleepScoreOption)!);
    string sleepJsonDir = ExpandPath(parseResult.GetValue(sleepJsonDirOption)!);
    string co2Path = ExpandPath(parseResult.GetValue(co2Option)!);
    string roomTempPath = ExpandPath(parseResult.GetValue(roomTempOption)!);
    string spo2Dir = ExpandPath(parseResult.GetValue(spo2DirOption)!);
    string hrvDir = ExpandPath(parseResult.GetValue(hrvDirOption)!);
    string tempDir = ExpandPath(parseResult.GetValue(tempDirOption)!);
    string outputPath = ExpandPath(parseResult.GetValue(outputOption)!);
    double coverageCutoff = parseResult.GetValue(coverageOption);
    string timezoneId = parseResult.GetValue(timezoneOption)!;

    RunPipeline(sleepScorePath, sleepJsonDir, co2Path, roomTempPath, spo2Dir, hrvDir, tempDir, outputPath, coverageCutoff, timezoneId);
    return 0;
});

ParseResult result = rootCommand.Parse(args);
return result.Invoke();

// =========================================================================================
// Pipeline Implementation
// =========================================================================================
static void RunPipeline(
    string sleepScorePath,
    string sleepJsonDir,
    string co2Path,
    string roomTempPath,
    string spo2Dir,
    string hrvDir,
    string tempDir,
    string outputPath,
    double coverageThreshold,
    string tzId)
{
    Console.WriteLine("====================================================================");
    Console.WriteLine("     Sleep Score, Biometrics & CO2 Multi-Sensor Pre-processor");
    Console.WriteLine("====================================================================");
    Console.WriteLine($"Sleep Score CSV : {sleepScorePath}");
    Console.WriteLine($"Sleep JSON Dir  : {sleepJsonDir}");
    Console.WriteLine($"CO2 CSV         : {co2Path}");
    if (!string.IsNullOrEmpty(roomTempPath))
        Console.WriteLine($"Room Temp CSV   : {roomTempPath}");
    Console.WriteLine($"SpO2 Dir        : {spo2Dir}");
    Console.WriteLine($"HRV Dir         : {hrvDir}");
    Console.WriteLine($"Temperature Dir : {tempDir}");
    Console.WriteLine($"Coverage Cutoff : {coverageThreshold:P0}");
    Console.WriteLine($"Output File     : {Path.GetFullPath(outputPath)}");

    TimeZoneInfo tz;
    try
    {
        tz = TimeZoneInfo.FindSystemTimeZoneById(tzId);
    }
    catch
    {
        tz = TimeZoneInfo.FindSystemTimeZoneById("Mountain Standard Time");
    }
    Console.WriteLine($"Active Timezone : {tz.DisplayName}\n");

    // 1. Index Sleep Sessions from Google Takeout JSON files
    var sleepMap = new Dictionary<long, SleepSessionInfo>();

    if (Directory.Exists(sleepJsonDir))
    {
        var jsonFiles = Directory.GetFiles(sleepJsonDir, "sleep-*.json");
        Console.WriteLine($"[1/6] Scanning {jsonFiles.Length} sleep export JSON files...");

        foreach (var file in jsonFiles)
        {
            try
            {
                using var fs = File.OpenRead(file);
                using var doc = JsonDocument.Parse(fs);
                foreach (var elem in doc.RootElement.EnumerateArray())
                {
                    if (!elem.TryGetProperty("logId", out var logIdProp)) continue;
                    long logId = logIdProp.GetInt64();

                    string startStr = elem.GetProperty("startTime").GetString()!;
                    string endStr = elem.GetProperty("endTime").GetString()!;
                    string dateOfSleep = elem.GetProperty("dateOfSleep").GetString()!;
                    double durMinutes = elem.TryGetProperty("duration", out var durProp) ? durProp.GetDouble() / 60000.0 : 0;
                    int minAsleep = elem.TryGetProperty("minutesAsleep", out var maProp) ? maProp.GetInt32() : 0;
                    int minAwake = elem.TryGetProperty("minutesAwake", out var mwProp) ? mwProp.GetInt32() : 0;
                    int efficiency = elem.TryGetProperty("efficiency", out var effProp) ? effProp.GetInt32() : 0;

                    int? deepMin = null, remMin = null, lightMin = null, wakeMin = null;
                    if (elem.TryGetProperty("levels", out var levelsElem) && levelsElem.TryGetProperty("summary", out var summaryElem))
                    {
                        if (summaryElem.TryGetProperty("deep", out var dProp) && dProp.TryGetProperty("minutes", out var dm)) deepMin = dm.GetInt32();
                        if (summaryElem.TryGetProperty("rem", out var rProp) && rProp.TryGetProperty("minutes", out var rm)) remMin = rm.GetInt32();
                        if (summaryElem.TryGetProperty("light", out var lProp) && lProp.TryGetProperty("minutes", out var lm)) lightMin = lm.GetInt32();
                        if (summaryElem.TryGetProperty("wake", out var wProp) && wProp.TryGetProperty("minutes", out var wm)) wakeMin = wm.GetInt32();
                    }

                    DateTime startDt = DateTime.Parse(startStr);
                    DateTime endDt = DateTime.Parse(endStr);

                    var startUtc = TimeZoneInfo.ConvertTimeToUtc(startDt, tz);
                    var endUtc = TimeZoneInfo.ConvertTimeToUtc(endDt, tz);

                    sleepMap[logId] = new SleepSessionInfo(
                        logId,
                        dateOfSleep,
                        startDt,
                        endDt,
                        startUtc,
                        endUtc,
                        durMinutes,
                        minAsleep,
                        minAwake,
                        efficiency,
                        deepMin,
                        remMin,
                        lightMin,
                        wakeMin
                    );
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Warning reading {Path.GetFileName(file)}: {ex.Message}");
            }
        }
        Console.WriteLine($"      Indexed {sleepMap.Count} unique sleep sessions from JSON.");
    }
    else
    {
        Console.WriteLine($"[1/6] Sleep JSON dir not found: {sleepJsonDir}. Will rely solely on sleep_score.csv timestamps.");
    }

    // 2. Read sleep_score.csv
    Console.WriteLine($"[2/6] Reading sleep score CSV...");
    var sleepRecords = new List<SleepRecord>();

    using (var reader = new StreamReader(sleepScorePath))
    using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
    {
        csv.Read();
        csv.ReadHeader();
        while (csv.Read())
        {
            long logId = csv.GetField<long>("sleep_log_entry_id");
            string? scoreStr = csv.GetField("overall_score");
            if (!double.TryParse(scoreStr, NumberStyles.Any, CultureInfo.InvariantCulture, out double overallScore))
                continue;

            double? comp = double.TryParse(csv.GetField("composition_score"), NumberStyles.Any, CultureInfo.InvariantCulture, out var cs) ? cs : null;
            double? rev = double.TryParse(csv.GetField("revitalization_score"), NumberStyles.Any, CultureInfo.InvariantCulture, out var rs) ? rs : null;
            double? durScore = double.TryParse(csv.GetField("duration_score"), NumberStyles.Any, CultureInfo.InvariantCulture, out var ds) ? ds : null;
            double? deepMinCsv = double.TryParse(csv.GetField("deep_sleep_in_minutes"), NumberStyles.Any, CultureInfo.InvariantCulture, out var dsm) ? dsm : null;
            double? restingHr = double.TryParse(csv.GetField("resting_heart_rate"), NumberStyles.Any, CultureInfo.InvariantCulture, out var rhr) ? rhr : null;
            double? restless = double.TryParse(csv.GetField("restlessness"), NumberStyles.Any, CultureInfo.InvariantCulture, out var rest) ? rest : null;

            SleepSessionInfo? session = null;
            if (sleepMap.TryGetValue(logId, out var sInfo))
            {
                session = sInfo;
            }
            else
            {
                string tsStr = csv.GetField("timestamp")!;
                DateTime endLocal = DateTime.Parse(tsStr.Replace("Z", ""));
                DateTime startLocal = endLocal.AddHours(-8);
                var startUtc = TimeZoneInfo.ConvertTimeToUtc(startLocal, tz);
                var endUtc = TimeZoneInfo.ConvertTimeToUtc(endLocal, tz);
                session = new SleepSessionInfo(
                    logId,
                    endLocal.ToString("yyyy-MM-dd"),
                    startLocal,
                    endLocal,
                    startUtc,
                    endUtc,
                    480,
                    (int)(480 * 0.85),
                    (int)(480 * 0.15),
                    85,
                    (int?)deepMinCsv,
                    null, null, null
                );
            }

            sleepRecords.Add(new SleepRecord(
                logId,
                session,
                overallScore,
                comp,
                rev,
                durScore,
                deepMinCsv ?? session.DeepMinutes,
                restingHr,
                restless
            ));
        }
    }
    Console.WriteLine($"      Loaded {sleepRecords.Count} sleep score entries.");

    // 3. Read Additional Biometrics: SpO2, HRV, Temperature
    Console.WriteLine($"[3/6] Reading additional biometrics (SpO2, HRV, Skin Temp)...");
    var spo2Map = LoadSpO2(spo2Dir);
    var hrvMap = LoadHrv(hrvDir);
    var tempMap = LoadTemperature(tempDir);
    Console.WriteLine($"      Loaded {spo2Map.Count} SpO2 daily entries, {hrvMap.Count} HRV entries, {tempMap.Count} nightly skin temp entries.");

    // 4. Read Home Assistant CO2 Data
    Console.WriteLine($"[4/6] Reading Home Assistant CO2 data...");
    var co2Readings = new List<Co2Point>();

    using (var reader = new StreamReader(co2Path))
    using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
    {
        csv.Read();
        csv.ReadHeader();
        while (csv.Read())
        {
            string tsStr = csv.GetField("last_changed")!;
            string valStr = csv.GetField("state")!;
            if (DateTimeOffset.TryParse(tsStr, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var dt) &&
                double.TryParse(valStr, NumberStyles.Any, CultureInfo.InvariantCulture, out double val))
            {
                co2Readings.Add(new Co2Point(dt, val));
            }
        }
    }

    co2Readings.Sort((a, b) => a.Timestamp.CompareTo(b.Timestamp));
    Console.WriteLine($"      Loaded {co2Readings.Count} CO2 readings from {co2Readings.First().Timestamp:yyyy-MM-dd} to {co2Readings.Last().Timestamp:yyyy-MM-dd} UTC.");

    // 5. Align Sleep Windows with CO2, Calculate Coverage and Time-Weighted CO2
    Console.WriteLine($"[5/6] Aligning sleep intervals, computing time-weighted CO2 & dynamics, and filtering by >= {coverageThreshold:P0} coverage...");

    var qualifyingRecords = new List<MergedOutputRow>();
    int excludedBeforeCo2 = 0;
    int excludedLowCoverage = 0;

    var maxContinuousGap = TimeSpan.FromMinutes(80);

    foreach (var rec in sleepRecords)
    {
        var s = rec.Session;

        if (s.EndUtc < co2Readings.First().Timestamp || s.StartUtc > co2Readings.Last().Timestamp)
        {
            excludedBeforeCo2++;
            continue;
        }

        var sleepDuration = s.EndUtc - s.StartUtc;
        if (sleepDuration.TotalMinutes < 60) continue;

        int firstIdx = co2Readings.FindIndex(c => c.Timestamp >= s.StartUtc - TimeSpan.FromHours(2));
        if (firstIdx < 0) firstIdx = 0;

        var windowReadings = new List<Co2Point>();
        for (int i = firstIdx; i < co2Readings.Count; i++)
        {
            var pt = co2Readings[i];
            if (pt.Timestamp > s.EndUtc + TimeSpan.FromHours(2)) break;
            windowReadings.Add(pt);
        }

        if (windowReadings.Count == 0)
        {
            excludedLowCoverage++;
            continue;
        }

        // Calculate Coverage
        double coveredSeconds = 0;
        for (int i = 0; i < windowReadings.Count - 1; i++)
        {
            var t1 = windowReadings[i].Timestamp;
            var t2 = windowReadings[i + 1].Timestamp;
            var gap = t2 - t1;

            if (gap <= maxContinuousGap)
            {
                var segStart = t1 < s.StartUtc ? s.StartUtc : t1;
                var segEnd = t2 > s.EndUtc ? s.EndUtc : t2;
                if (segEnd > segStart)
                {
                    coveredSeconds += (segEnd - segStart).TotalSeconds;
                }
            }
        }

        double coveragePct = Math.Min(1.0, coveredSeconds / sleepDuration.TotalSeconds);
        if (coveragePct < coverageThreshold)
        {
            excludedLowCoverage++;
            continue;
        }

        // Calculate Time-Weighted Average, Peak, and Dynamics
        var pointsInSleep = windowReadings.Where(c => c.Timestamp >= s.StartUtc && c.Timestamp <= s.EndUtc).ToList();
        if (pointsInSleep.Count == 0)
        {
            excludedLowCoverage++;
            continue;
        }

        double maxCo2 = pointsInSleep.Max(p => p.Ppm);
        double startCo2 = InterpolateCo2(windowReadings, s.StartUtc);
        double endCo2 = InterpolateCo2(windowReadings, s.EndUtc);

        var profile = new List<(DateTimeOffset T, double V)> { (s.StartUtc, startCo2) };
        foreach (var pt in pointsInSleep)
        {
            if (pt.Timestamp > s.StartUtc && pt.Timestamp < s.EndUtc)
            {
                profile.Add((pt.Timestamp, pt.Ppm));
            }
        }
        profile.Add((s.EndUtc, endCo2));

        double totalPpmSeconds = 0;
        double integratedSeconds = 0;
        double secondsAbove1000 = 0;

        for (int i = 0; i < profile.Count - 1; i++)
        {
            var t1 = profile[i].T;
            var t2 = profile[i + 1].T;
            var v1 = profile[i].V;
            var v2 = profile[i + 1].V;
            var dt = (t2 - t1).TotalSeconds;

            if (dt > 0 && (t2 - t1) <= maxContinuousGap)
            {
                totalPpmSeconds += 0.5 * (v1 + v2) * dt;
                integratedSeconds += dt;

                if (v1 >= 1000 && v2 >= 1000)
                {
                    secondsAbove1000 += dt;
                }
                else if (v1 < 1000 && v2 > 1000)
                {
                    double frac = (v2 - 1000) / (v2 - v1);
                    secondsAbove1000 += dt * frac;
                }
                else if (v1 > 1000 && v2 < 1000)
                {
                    double frac = (v1 - 1000) / (v1 - v2);
                    secondsAbove1000 += dt * frac;
                }
            }
        }

        double avgCo2 = integratedSeconds > 0 ? (totalPpmSeconds / integratedSeconds) : pointsInSleep.Average(p => p.Ppm);
        double hoursAbove1000 = Math.Round(secondsAbove1000 / 3600.0, 2);
        double pctAbove1000 = sleepDuration.TotalSeconds > 0 ? Math.Round((secondsAbove1000 / sleepDuration.TotalSeconds) * 100.0, 1) : 0;

        // Room ventilation dynamics:
        double co2Rise = Math.Max(0, maxCo2 - startCo2);
        double hoursInBed = sleepDuration.TotalHours;
        double riseRate = hoursInBed > 0 ? (co2Rise / hoursInBed) : 0;
        string roomState = (maxCo2 < 750 && co2Rise < 250) ? "Ventilated" : (maxCo2 > 1000 ? "Sealed" : "Moderate");

        // Biometric matches by DateOfSleep:
        bool hasSpo2 = spo2Map.TryGetValue(s.DateOfSleep, out var spo2Val);
        bool hasHrv = hrvMap.TryGetValue(s.DateOfSleep, out var hrvVal);
        bool hasTemp = tempMap.TryGetValue(s.DateOfSleep, out var tempVal);

        qualifyingRecords.Add(new MergedOutputRow(
            s.DateOfSleep,
            s.StartLocal.DayOfWeek.ToString(),
            s.StartLocal.ToString("yyyy-MM-dd HH:mm:ss"),
            s.EndLocal.ToString("yyyy-MM-dd HH:mm:ss"),
            s.StartUtc.ToString("yyyy-MM-ddTHH:mm:ssZ"),
            s.EndUtc.ToString("yyyy-MM-ddTHH:mm:ssZ"),
            Math.Round(s.DurationMinutes / 60.0, 2),
            s.MinutesAsleep,
            s.MinutesAwake,
            s.Efficiency,
            rec.OverallScore,
            rec.CompositionScore,
            rec.RevitalizationScore,
            rec.DurationScore,
            rec.DeepSleepMinutes,
            s.RemMinutes,
            s.LightMinutes,
            s.WakeMinutes,
            rec.RestingHeartRate,
            rec.Restlessness,
            hasSpo2 ? spo2Val.Avg : null,
            hasSpo2 ? spo2Val.Min : null,
            hasHrv ? hrvVal.Rmssd : null,
            hasHrv ? hrvVal.NremHr : null,
            hasHrv ? hrvVal.Entropy : null,
            hasTemp ? tempVal : null,
            Math.Round(startCo2, 1),
            Math.Round(avgCo2, 1),
            Math.Round(maxCo2, 1),
            hoursAbove1000,
            pctAbove1000,
            Math.Round(co2Rise, 1),
            Math.Round(riseRate, 1),
            roomState,
            Math.Round(coveragePct * 100.0, 1),
            pointsInSleep.Count
        ));
    }

    // Deduplicate multiple logs on the same date by keeping the primary/longest session
    qualifyingRecords = qualifyingRecords
        .GroupBy(r => r.DateOfSleep)
        .Select(g => g.OrderByDescending(r => r.SleepDurationHours).First())
        .OrderBy(r => r.DateOfSleep, StringComparer.Ordinal)
        .ToList();

    // Identify and remove any columns that have no values at all across all qualifying records
    var properties = typeof(MergedOutputRow).GetProperties();
    var activeProps = properties.Where(p => qualifyingRecords.Any(r =>
    {
        var val = p.GetValue(r);
        return val != null && (!(val is string str) || !string.IsNullOrWhiteSpace(str));
    })).ToList();

    var removedProps = properties.Except(activeProps).Select(p => p.Name).ToList();

    // Write Output CSV with only active columns
    using (var writer = new StreamWriter(outputPath))
    using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
    {
        foreach (var prop in activeProps)
        {
            csv.WriteField(prop.Name);
        }
        csv.NextRecord();

        foreach (var record in qualifyingRecords)
        {
            foreach (var prop in activeProps)
            {
                var val = prop.GetValue(record);
                csv.WriteField(val);
            }
            csv.NextRecord();
        }
    }

    Console.WriteLine($"\nSuccessfully generated: {outputPath}");
    if (removedProps.Count > 0)
    {
        Console.WriteLine($"  - Removed Empty Columns:      {string.Join(", ", removedProps)}");
    }
    Console.WriteLine($"  - Active Columns Retained:    {activeProps.Count} of {properties.Length}");
    Console.WriteLine($"  - Total Sleep Logs:           {sleepRecords.Count}");
    Console.WriteLine($"  - Prior to CO2 Logging:       {excludedBeforeCo2}");
    Console.WriteLine($"  - Excluded (< {coverageThreshold:P0} coverage): {excludedLowCoverage}");
    Console.WriteLine($"  - Final Qualifying Nights:    {qualifyingRecords.Count} unique dates\n");

    // 6. Comprehensive Statistical Analysis
    if (qualifyingRecords.Count > 2)
    {
        PrintAnalysisSummary(qualifyingRecords);
    }
}

// =========================================================================================
// Data Loading Helpers for SpO2, HRV, Temperature
// =========================================================================================
static Dictionary<string, (double Avg, double Min)> LoadSpO2(string dir)
{
    var map = new Dictionary<string, (double Avg, double Min)>();
    if (!Directory.Exists(dir)) return map;

    foreach (var file in Directory.GetFiles(dir, "Daily SpO2 - *.csv"))
    {
        try
        {
            using var reader = new StreamReader(file);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            csv.Read();
            csv.ReadHeader();
            while (csv.Read())
            {
                string ts = csv.GetField("timestamp")!;
                string dateStr = ts.Length >= 10 ? ts.Substring(0, 10) : ts;
                if (double.TryParse(csv.GetField("average_value"), NumberStyles.Any, CultureInfo.InvariantCulture, out double avg) &&
                    double.TryParse(csv.GetField("lower_bound"), NumberStyles.Any, CultureInfo.InvariantCulture, out double min))
                {
                    map[dateStr] = (Math.Round(avg, 1), Math.Round(min, 1));
                }
            }
        }
        catch { }
    }
    return map;
}

static Dictionary<string, (double Rmssd, double NremHr, double Entropy)> LoadHrv(string dir)
{
    var map = new Dictionary<string, (double Rmssd, double NremHr, double Entropy)>();
    if (!Directory.Exists(dir)) return map;

    foreach (var file in Directory.GetFiles(dir, "Daily Heart Rate Variability Summary - *.csv"))
    {
        try
        {
            string dateStr = Path.GetFileNameWithoutExtension(file).Replace("Daily Heart Rate Variability Summary - ", "").Trim();
            using var reader = new StreamReader(file);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            csv.Read();
            csv.ReadHeader();
            if (csv.Read())
            {
                if (double.TryParse(csv.GetField("rmssd"), NumberStyles.Any, CultureInfo.InvariantCulture, out double rmssd) &&
                    double.TryParse(csv.GetField("nremhr"), NumberStyles.Any, CultureInfo.InvariantCulture, out double nrem) &&
                    double.TryParse(csv.GetField("entropy"), NumberStyles.Any, CultureInfo.InvariantCulture, out double ent))
                {
                    map[dateStr] = (Math.Round(rmssd, 1), Math.Round(nrem, 1), Math.Round(ent, 3));
                }
            }
        }
        catch { }
    }
    return map;
}

static Dictionary<string, double> LoadTemperature(string dir)
{
    var map = new Dictionary<string, double>();
    if (!Directory.Exists(dir)) return map;

    foreach (var file in Directory.GetFiles(dir, "Computed Temperature - *.csv"))
    {
        try
        {
            using var reader = new StreamReader(file);
            using var csv = new CsvReader(reader, CultureInfo.InvariantCulture);
            csv.Read();
            csv.ReadHeader();
            while (csv.Read())
            {
                string sleepEnd = csv.GetField("sleep_end")!;
                string dateStr = sleepEnd.Length >= 10 ? sleepEnd.Substring(0, 10) : sleepEnd;
                if (double.TryParse(csv.GetField("nightly_temperature"), NumberStyles.Any, CultureInfo.InvariantCulture, out double temp))
                {
                    map[dateStr] = Math.Round(temp, 2);
                }
            }
        }
        catch { }
    }
    return map;
}

// =========================================================================================
// Statistical Analysis & Correlation Printing
// =========================================================================================
static void PrintAnalysisSummary(List<MergedOutputRow> rows)
{
    string bannerLine = new string('=', 141);
    Console.WriteLine(bannerLine);
    Console.WriteLine("                                                   COMPREHENSIVE MULTI-SENSOR ANALYSIS");
    Console.WriteLine(bannerLine);

    var avgCo2List = rows.Select(r => r.AvgCo2Ppm).ToList();
    var maxCo2List = rows.Select(r => r.MaxCo2Ppm).ToList();
    var scores = rows.Select(r => r.OverallScore).ToList();

    Console.WriteLine($"Dataset Overview ({rows.Count} Qualifying Nights):");
    Console.WriteLine($"  Average Nightly CO2 : {avgCo2List.Average():F1} ppm (Min: {avgCo2List.Min():F1}, Max: {avgCo2List.Max():F1})");
    Console.WriteLine($"  Peak Nightly CO2    : {maxCo2List.Average():F1} ppm (Min: {maxCo2List.Min():F1}, Max: {maxCo2List.Max():F1})");
    Console.WriteLine($"  Overall Sleep Score : {scores.Average():F1} (Min: {scores.Min():F0}, Max: {scores.Max():F0})");

    var ventCount = rows.Count(r => r.RoomVentilationState == "Ventilated");
    var sealedCount = rows.Count(r => r.RoomVentilationState == "Sealed");
    var modCount = rows.Count - ventCount - sealedCount;
    Console.WriteLine($"  Room Dynamics       : {ventCount} Ventilated (<750 ppm), {modCount} Moderate, {sealedCount} Sealed (>1000 ppm)\n");

    string sepLine = new string('-', 141);
    Console.WriteLine(sepLine);
    Console.WriteLine("| Biometric / Sleep Metric       | vs Avg CO2 (r) | vs Max CO2 (r) | vs Hrs>1k (r) | vs %>1k (r) | vs Sleep Score (r) | N Valid |  Metric Mean |");
    Console.WriteLine(sepLine);

    PrintStatRow("Overall Sleep Score", rows, r => r.OverallScore);
    PrintStatRow("HRV Recovery (RMSSD ms)", rows, r => r.HrvRmssd);
    PrintStatRow("NREM Heart Rate (BPM)", rows, r => r.HrvNremHeartRate);
    PrintStatRow("Resting Heart Rate (BPM)", rows, r => r.RestingHeartRate);
    PrintStatRow("Nightly Skin Temp (°C)", rows, r => r.NightlySkinTempCelsius);
    PrintStatRow("SpO2 Average (%)", rows, r => r.SpO2Avg);
    PrintStatRow("SpO2 Min Drop (%)", rows, r => r.SpO2Min);
    PrintStatRow("Deep Sleep (Minutes)", rows, r => r.DeepSleepMinutes);
    PrintStatRow("REM Sleep (Minutes)", rows, r => r.RemSleepMinutes);
    PrintStatRow("Restlessness Score", rows, r => r.Restlessness);
    PrintStatRow("Sleep Efficiency (%)", rows, r => (double?)r.SleepEfficiency);
    PrintStatRow("Total Sleep Duration (Hours)", rows, r => r.SleepDurationHours);
    PrintStatRow("CO2 Rise Rate (ppm/hr)", rows, r => r.Co2RiseRatePerHour);
    Console.WriteLine(sepLine + "\n");
}

static void PrintStatRow(string name, List<MergedOutputRow> rows, Func<MergedOutputRow, double?> metricSelector)
{
    var valid = rows.Where(r => metricSelector(r).HasValue).ToList();
    if (valid.Count < 3) return;

    var xs = valid.Select(r => metricSelector(r)!.Value).ToList();
    var co2Avg = valid.Select(r => r.AvgCo2Ppm).ToList();
    var co2Max = valid.Select(r => r.MaxCo2Ppm).ToList();
    var hrsAbove = valid.Select(r => r.HoursAbove1000Ppm).ToList();
    var pctAbove = valid.Select(r => r.PctSleepAbove1000Ppm).ToList();
    var scores = valid.Select(r => r.OverallScore).ToList();

    double rCo2Avg = Pearson(xs, co2Avg);
    double rCo2Max = Pearson(xs, co2Max);
    double rHrs = Pearson(xs, hrsAbove);
    double rPct = Pearson(xs, pctAbove);
    double rScore = Pearson(xs, scores);

    Console.WriteLine($"| {name,-30} | {rCo2Avg,14:F4} | {rCo2Max,14:F4} | {rHrs,13:F4} | {rPct,11:F4} | {rScore,18:F4} | {valid.Count,7} | {xs.Average(),12:F2} |");
}

// =========================================================================================
// General Math Helpers
// =========================================================================================
static double InterpolateCo2(List<Co2Point> pts, DateTimeOffset target)
{
    if (pts.Count == 0) return 500;
    if (target <= pts.First().Timestamp) return pts.First().Ppm;
    if (target >= pts.Last().Timestamp) return pts.Last().Ppm;

    for (int i = 0; i < pts.Count - 1; i++)
    {
        if (pts[i].Timestamp <= target && pts[i + 1].Timestamp >= target)
        {
            var t1 = pts[i].Timestamp;
            var t2 = pts[i + 1].Timestamp;
            if (t2 == t1) return pts[i].Ppm;
            double factor = (target - t1).TotalSeconds / (t2 - t1).TotalSeconds;
            return pts[i].Ppm + factor * (pts[i + 1].Ppm - pts[i].Ppm);
        }
    }
    return pts.Last().Ppm;
}

static double Pearson(List<double> x, List<double> y)
{
    if (x.Count != y.Count || x.Count == 0) return 0;
    double mx = x.Average();
    double my = y.Average();
    double num = 0, d1 = 0, d2 = 0;
    for (int i = 0; i < x.Count; i++)
    {
        double dx = x[i] - mx;
        double dy = y[i] - my;
        num += dx * dy;
        d1 += dx * dx;
        d2 += dy * dy;
    }
    double den = Math.Sqrt(d1 * d2);
    return den != 0 ? num / den : 0;
}

// =========================================================================================
// Data Record Definitions
// =========================================================================================
record Co2Point(DateTimeOffset Timestamp, double Ppm);

record SleepSessionInfo(
    long LogId,
    string DateOfSleep,
    DateTime StartLocal,
    DateTime EndLocal,
    DateTimeOffset StartUtc,
    DateTimeOffset EndUtc,
    double DurationMinutes,
    int MinutesAsleep,
    int MinutesAwake,
    int Efficiency,
    int? DeepMinutes,
    int? RemMinutes,
    int? LightMinutes,
    int? WakeMinutes
);

record SleepRecord(
    long LogId,
    SleepSessionInfo Session,
    double OverallScore,
    double? CompositionScore,
    double? RevitalizationScore,
    double? DurationScore,
    double? DeepSleepMinutes,
    double? RestingHeartRate,
    double? Restlessness
);

record MergedOutputRow(
    string DateOfSleep,
    string DayOfWeek,
    string StartTimeLocal,
    string EndTimeLocal,
    string StartTimeUtc,
    string EndTimeUtc,
    double SleepDurationHours,
    int MinutesAsleep,
    int MinutesAwake,
    int SleepEfficiency,
    double OverallScore,
    double? CompositionScore,
    double? RevitalizationScore,
    double? DurationScore,
    double? DeepSleepMinutes,
    int? RemSleepMinutes,
    int? LightSleepMinutes,
    int? WakeSleepMinutes,
    double? RestingHeartRate,
    double? Restlessness,
    double? SpO2Avg,
    double? SpO2Min,
    double? HrvRmssd,
    double? HrvNremHeartRate,
    double? HrvEntropy,
    double? NightlySkinTempCelsius,
    double Co2StartPpm,
    double AvgCo2Ppm,
    double MaxCo2Ppm,
    double HoursAbove1000Ppm,
    double PctSleepAbove1000Ppm,
    double Co2RisePpm,
    double Co2RiseRatePerHour,
    string RoomVentilationState,
    double Co2CoveragePct,
    int Co2SampleCount
);
