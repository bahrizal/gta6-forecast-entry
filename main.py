#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

Reads one case file (schema makefun/twitch-ccu-case/v1) from the input
directory and writes five quantiles (5%, 25%, 50%, 75%, 95%) of the Twitch
category's concurrent viewers for each of hours 2..24 of the release day,
from everything recorded up to the end of hour 1.

Method, fitted at run time from the case's own comparables (no network):
  A rolling-midnight console launch has a stable day shape once it is
  expressed as log-ratios to the first hour's print: a dip in hours 2-4,
  a rise when the US unlocks (hours 5-9), a midday lull, then a climb to
  an evening peak. How high that peak runs depends strongly on how big
  the first hour already is: the bigger the first hour, the smaller the
  evening multiple (across the comparables the evening log-ratio falls
  almost one-for-one with the log of the first-hour print). So for each
  hour we regress the log-ratio on the log first-hour print across past
  rolling-midnight launches, and we take the quantiles from the residual
  spread (a robust MAD sigma, smoothed over neighbouring hours).

  Training launches are the comparables that actually look like this
  launch: rolling midnight-local unlock, no early-access contamination,
  no simultaneous PC unlock, and a meaningful first hour. Launches that
  unlocked at a single global time (so their first UTC hour is still
  pre-release) or sold early access are left out: their first hour is
  not a post-unlock hour and their ratios are meaningless. If too few
  clean launches remain we widen the set (rolling, then all), and any
  hour we cannot fit falls back to the house baseline shape.

Standard library only, deterministic, no randomness, no network.
"""

import json
import math
import os
from datetime import datetime

# Standard normal quantiles at the five levels the case asks for.
Z_VALUES = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Launches whose standard release was preceded by paid early access: their
# first hour already holds early-access players, so their day shape is not
# a clean launch shape. Matched by name substring.
EARLY_ACCESS_NAMES = (
    "Hogwarts Legacy",
    "Starfield",
    "Diablo IV",
    "Assassin's Creed Shadows",
    "DOOM: The Dark Ages",
    "Death Stranding 2",
    "Forza Horizon 5",
    "Forza Horizon 6",
)

# Launches with a simultaneous PC unlock at midnight GMT on top of the
# rolling console unlock: their first hour mixes both, so they are not
# clean rolling analogues either.
MIXED_UNLOCK_NAMES = ("Cyberpunk 2077",)

# A first hour below this many viewers is pre-release noise, not a launch.
MIN_FIRST_HOUR = 1000.0

# Smallest spread we ever quote (log units) and smallest training set we
# trust for a per-hour regression.
MIN_SIGMA = 0.08
MIN_FIT_POINTS = 5


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")


def release_day_curve(comparable):
    """Hourly viewers {hour: viewers} for a comparable's release UTC day."""
    try:
        day_start = parse_time(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    curve = {}
    for point in comparable.get("hourly") or []:
        try:
            when = parse_time(point["from"])
            viewers = point["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        delta_seconds = (when - day_start).total_seconds()
        if 0 <= delta_seconds < 24 * 3600:
            curve[int(delta_seconds // 3600) + 1] = float(viewers)
    return curve


def is_rolling_midnight(comparable):
    """True when the launch unlocked at rolling midnight local time."""
    unlock = (comparable.get("unlock") or "").lower()
    return "rolling" in unlock or "midnight local" in unlock


def is_early_access(comparable):
    name = comparable.get("name") or ""
    return any(part in name for part in EARLY_ACCESS_NAMES)


def is_mixed_unlock(comparable):
    name = comparable.get("name") or ""
    return any(part in name for part in MIXED_UNLOCK_NAMES)


def build_training(case):
    """Usable past launches as (log first-hour print, {hour: log-ratio}).

    A launch is usable when its release day has a first-hour print and at
    least 20 of the 24 hours. The clean set is rolling-midnight launches
    without early access or a mixed unlock; we widen it only if it is too
    small to fit on.
    """
    usable = []
    for comparable in case.get("comparables") or []:
        curve = release_day_curve(comparable)
        if 1 not in curve or len(curve) < 20:
            continue
        first = curve[1]
        if first < MIN_FIRST_HOUR:
            continue
        ratios = {}
        for hour in range(2, 25):
            if hour in curve:
                ratios[hour] = math.log(curve[hour] / first)
        usable.append(
            {
                "log_first": math.log(first),
                "ratios": ratios,
                "rolling": is_rolling_midnight(comparable),
                "early": is_early_access(comparable),
                "mixed": is_mixed_unlock(comparable),
            }
        )
    clean = [c for c in usable if c["rolling"] and not c["early"] and not c["mixed"]]
    if len(clean) >= 8:
        return clean
    rolling = [c for c in usable if c["rolling"]]
    if len(rolling) >= 8:
        return rolling
    return usable


def ols_fit(xs, ys):
    """Least-squares fit y = a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return mean_y, 0.0, [y - mean_y for y in ys]
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = mean_y - b * mean_x
    residuals = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, residuals


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def mad_sigma(residuals):
    """Robust spread of residuals: 1.4826 * median absolute deviation."""
    if not residuals:
        return MIN_SIGMA
    centre = median(residuals)
    spread = 1.4826 * median([abs(r - centre) for r in residuals])
    return max(spread, MIN_SIGMA)


def baseline_row(first_print, hour):
    """House baseline shape: flat at the first hour, widening spread."""
    spread = 0.15 * math.sqrt(hour - 1)
    return [round(first_print * math.exp(z * spread), 2) for z in Z_VALUES]


def forecast_case(case):
    """Five quantiles per hour for one case; returns the output object."""
    known = case.get("known") or []
    first_print = 1.0
    if known:
        try:
            first_print = float(known[0]["viewers"])
        except (KeyError, TypeError, ValueError):
            first_print = 1.0
    if not (first_print > 0.0) or not math.isfinite(first_print):
        first_print = 1.0

    hours = []
    forecast_rows = case.get("forecast") or []
    if forecast_rows:
        hours = [int(row["hour"]) for row in forecast_rows]
    else:
        hours = list(range(2, 25))
    hours = sorted(set(h for h in hours if 2 <= h <= 24)) or list(range(2, 25))

    training = build_training(case)
    log_first = math.log(first_print)

    # Per-hour regression of the log-ratio on the log first-hour print.
    fitted = {}
    raw_sigma = {}
    for hour in hours:
        points = [(c["log_first"], c["ratios"][hour]) for c in training if hour in c["ratios"]]
        if len(points) >= MIN_FIT_POINTS:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            a, b, residuals = ols_fit(xs, ys)
            fitted[hour] = a + b * log_first
            raw_sigma[hour] = mad_sigma(residuals)
        elif len(points) >= 3:
            values = [p[1] for p in points]
            fitted[hour] = median(values)
            raw_sigma[hour] = mad_sigma([v - fitted[hour] for v in values])
        else:
            fitted[hour] = None
            raw_sigma[hour] = None

    # Smooth the spread over neighbouring hours: with only a dozen
    # training launches the per-hour MAD is noisy on its own.
    sigma = {}
    for hour in hours:
        neighbours = [raw_sigma[h] for h in (hour - 1, hour, hour + 1)
                      if h in raw_sigma and raw_sigma[h] is not None]
        sigma[hour] = sum(neighbours) / len(neighbours) if neighbours else None

    rows = []
    for hour in hours:
        if fitted[hour] is None:
            rows.append({"hour": hour, "q": baseline_row(first_print, hour)})
            continue
        centre = fitted[hour]
        spread = sigma[hour] if sigma[hour] is not None else 0.30
        quantiles = []
        previous = 0.0
        for z in Z_VALUES:
            value = first_print * math.exp(centre + z * spread)
            value = min(max(value, 1.0), 1_000_000_000.0)
            value = round(value, 2)
            if value < previous:
                value = previous
            quantiles.append(value)
            previous = value
        rows.append({"hour": hour, "q": quantiles})
    return {"hours": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir) if name.endswith(".in"))
    for name in names:
        case_id = name[: -len(".in")]
        out_path = os.path.join(output_dir, case_id + ".out")
        try:
            with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
                case = json.load(handle)
            result = forecast_case(case)
        except Exception:
            # Never leave a case without a valid forecast: fall back to a
            # flat forecast at the first hour we could read, else 1 viewer.
            first_print = 1.0
            try:
                with open(os.path.join(input_dir, name), "r", encoding="utf-8") as handle:
                    case = json.load(handle)
                first_print = float((case.get("known") or [{}])[0].get("viewers") or 1.0)
            except Exception:
                first_print = 1.0
            if not (first_print > 0.0) or not math.isfinite(first_print):
                first_print = 1.0
            result = {"hours": [{"hour": h, "q": baseline_row(first_print, h)}
                                for h in range(2, 25)]}
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
