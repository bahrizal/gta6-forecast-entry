# GTA VI launch-day Twitch viewers — hour-by-hour forecaster

Forecasts how many people watch Grand Theft Auto VI live on Twitch in
each of hours 2–24 of its release day (2026-11-19 UTC), as five quantiles
(5%, 25%, 50%, 75%, 95%) per hour, from everything recorded up to the
end of hour 1.

## Method

`main.py` (Python 3.13, standard library only, no network) fits the
forecast at run time from the case's own comparables:

- Each past launch is reduced to log-ratios of its release-day hourly
  viewers to its first hour's print. For a rolling-midnight console
  launch that shape is stable: a dip in hours 2–4, a rise when the US
  unlocks (hours 5–9), a midday lull, then a climb to an evening peak.
- How high the evening peak runs depends strongly on how big the first
  hour already is, so for each hour the log-ratio is regressed on the
  log first-hour print across past launches. Quantiles come from the
  residual spread (robust MAD sigma, smoothed over neighbouring hours).
- Training launches are the comparables that actually look like this
  launch: rolling midnight-local unlock, no early-access contamination,
  no simultaneous PC unlock, and a meaningful first hour. Launches that
  unlocked at one global time (their first UTC hour is still
  pre-release) or sold early access are left out. The set widens
  (rolling, then all) only if too few clean launches remain, and any
  hour that cannot be fitted falls back to the house baseline shape.

## Practice score (local, same pinball rule as the grader)

Mean pinball loss on the four practice launches, scored against
SullyGnome's hourly averages for those release days:

| case | loss |
| ---- | ---- |
| Marvel's Wolverine (p00) | 0.181 |
| Red Dead Redemption 2 (p01) | 0.110 |
| God of War Ragnarök (p02) | 0.118 |
| Marvel's Spider-Man 2 (p03) | 0.233 |
| **mean** | **0.160** |

House baseline on the same practice: **0.387**. The entry beats it by
about 59%.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` file in the input directory and writes one
`<id>.out` per case. Deterministic: two runs write identical bytes.
