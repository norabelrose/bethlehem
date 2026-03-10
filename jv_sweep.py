#!/usr/bin/env python3
"""
jv_sweep.py

Sweep the full DE422 ephemeris (~3000 BC – 3000 AD) for every
Jupiter–Venus conjunction whose closest approach as seen from a chosen
city is < 1 arcminute and which is visible from that city.

Usage:
  python jv_sweep.py                    # default: Jerusalem
  python jv_sweep.py babylon
  python jv_sweep.py --list             # show available cities
  python jv_sweep.py --lat 41.9 --lon 12.5 --name Rome

Visibility criteria (both must hold):
  1. Solar elongation of Jupiter > 11° at closest approach
     (Jupiter's arcus visionis — same threshold as heliacal risings)
  2. The < 1′ window overlaps with a moment when the Sun is below −6°
     (astronomical twilight or darker) AND Jupiter is above the horizon

Algorithm — three phases:
  Phase 1  Daily geocentric scan — finds candidate windows (sep < 1.5°)
  Phase 2  1-minute topocentric scan within each candidate window
  Phase 3  Bisection for precise entry/exit of the < 1′ interval,
           visibility check, then print

Results are printed immediately as each qualifying event is found.
"""

import argparse
import sys
import numpy as np
from skyfield.api import load, wgs84, N, E
from skyfield.framelib import ecliptic_frame

# ── City database ─────────────────────────────────────────────────────────────
# (lat °N, lon °E, display name)
CITIES = {
    "jerusalem":  ( 31.7683,  35.2137, "Jerusalem"),
    "babylon":    ( 32.5427,  44.4215, "Babylon"),
    "alexandria": ( 31.2001,  29.9187, "Alexandria"),
    "antioch":    ( 36.2021,  36.1601, "Antioch"),
    "athens":     ( 37.9838,  23.7275, "Athens"),
    "rome":       ( 41.9028,  12.4964, "Rome"),
    "carthage":   ( 36.8527,  10.3233, "Carthage"),
    "nineveh":    ( 36.3590,  43.1527, "Nineveh"),
    "memphis":    ( 29.8511,  31.2521, "Memphis (Egypt)"),
    "ur":         ( 30.9625,  46.1035, "Ur"),
    "persepolis": ( 29.9350,  52.8905, "Persepolis"),
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Sweep DE422 for Jupiter–Venus conjunctions < 1′ visible from an ancient city.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Available cities:\n" + "\n".join(
        f"  {k:<12}  {v[2]}  ({v[0]:.4f}°N, {v[1]:.4f}°E)"
        for k, v in CITIES.items()
    ),
)
parser.add_argument(
    "location",
    nargs="?",
    default="jerusalem",
    metavar="CITY",
    help="Observation city name (default: jerusalem)",
)
parser.add_argument("--lat",  type=float, metavar="DEG",
                    help="Custom latitude in degrees N")
parser.add_argument("--lon",  type=float, metavar="DEG",
                    help="Custom longitude in degrees E")
parser.add_argument("--name", type=str,   metavar="NAME", default="Custom site",
                    help="Display name for a custom location")

args = parser.parse_args()

if args.lat is not None or args.lon is not None:
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon must both be supplied together")
    obs_lat  = args.lat
    obs_lon  = args.lon
    obs_name = args.name
else:
    key = args.location.lower()
    if key not in CITIES:
        parser.error(
            f"Unknown city '{args.location}'. "
            f"Run with --list to see available cities."
        )
    obs_lat, obs_lon, obs_name = CITIES[key]

# ── Load ephemeris ────────────────────────────────────────────────────────────
print(f"Observer   : {obs_name}  ({obs_lat:.4f}°N, {obs_lon:.4f}°E)")
print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
eph = load("de422.bsp")
ts  = load.timescale()
print("Loaded.\n", flush=True)

sun   = eph["sun"]
earth = eph["earth"]
jup   = eph["jupiter barycenter"]
ven   = eph["venus barycenter"]

obs_site = wgs84.latlon(obs_lat * N, obs_lon * E)

# ── Formatting helpers ────────────────────────────────────────────────────────
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

def era(y: int) -> str:
    return f"{-y+1} BC" if y <= 0 else f"{y} AD"

def fmt(t, hhmm: bool = False) -> str:
    y, mo, d, H, Mi, S = t.tt_calendar()
    s = f"{int(d):2d} {MONTHS[mo-1]} {era(y)}"
    if hhmm:
        s += f"  {H + Mi/60 + S/3600:05.2f}h TT"
    return s

def lst_str(t) -> str:
    """Local mean solar time at the observer site (HH:MM)."""
    h = ((t.ut1 + 0.5) % 1.0 * 24.0 + obs_lon / 15.0) % 24.0
    return f"{int(h):02d}:{int((h % 1) * 60):02d}"

# ── Vectorised position helpers ───────────────────────────────────────────────
def geo_sep(jd_arr: np.ndarray) -> np.ndarray:
    """Geocentric Jupiter–Venus angular separation (°). Used for coarse scan."""
    t  = ts.tt_jd(jd_arr)
    aJ = earth.at(t).observe(jup).apparent()
    aV = earth.at(t).observe(ven).apparent()
    return aJ.separation_from(aV).degrees

def site_sep(jd_arr: np.ndarray) -> np.ndarray:
    """Topocentric Jupiter–Venus angular separation (°) from obs_site."""
    t   = ts.tt_jd(jd_arr)
    obs = earth + obs_site
    aJ  = obs.at(t).observe(jup).apparent()
    aV  = obs.at(t).observe(ven).apparent()
    return aJ.separation_from(aV).degrees

def site_sun_alt(jd_arr: np.ndarray) -> np.ndarray:
    """Altitude of Sun from obs_site (°)."""
    t   = ts.tt_jd(jd_arr)
    aS  = (earth + obs_site).at(t).observe(sun).apparent()
    alt, _, _ = aS.altaz()
    return alt.degrees

def site_jup_alt(jd_arr: np.ndarray) -> np.ndarray:
    """Altitude of Jupiter from obs_site (°)."""
    t   = ts.tt_jd(jd_arr)
    aJ  = (earth + obs_site).at(t).observe(jup).apparent()
    alt, _, _ = aJ.altaz()
    return alt.degrees

# ── Bisection for threshold crossing ─────────────────────────────────────────
ARCMIN = 1.0 / 60.0   # 1 arcminute in degrees

def bisect_sep(jd_lo: float, jd_hi: float,
               entering: bool, threshold: float = ARCMIN) -> float:
    """Bisect to ~6-second precision the moment site_sep crosses threshold.
    entering=True  → sep going from above to below threshold.
    entering=False → sep going from below to above threshold.
    Returns JD (TT) of crossing.
    """
    lo, hi = jd_lo, jd_hi
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s   = site_sep(np.array([mid]))[0]
        if (s > threshold) == entering:
            lo = mid
        else:
            hi = mid
        if (hi - lo) * 86400 < 6:    # 6-second precision
            break
    return (lo + hi) / 2.0

# ── Scan parameters ───────────────────────────────────────────────────────────
JD_START = ts.tt(-3000, 11, 14).tt  # inside DE422 lower bound
JD_END   = ts.tt( 3000,  1,  1).tt  # inside DE422 upper bound

THRESH_COARSE = 1.5     # deg — geocentric daily trigger
THRESH_FINE   = ARCMIN  # deg — 1 arcminute
AV_JUP        = 11.0    # deg — Jupiter arcus visionis
NIGHT_ALT     = -6.0    # deg — Sun must reach this to count as night

total_days = int(JD_END - JD_START)
print(f"Scan range : {era(-3000)} to {era(3000)}  (~{total_days:,} days)")
print(f"Target     : closest approach < 1′ as seen from {obs_name}")
print(f"Visibility : elongation > {AV_JUP}°  AND  night overlap (sun < {NIGHT_ALT}°,"
      f" Jupiter > 0°)\n", flush=True)

# ── Phase 1: Daily geocentric scan ───────────────────────────────────────────
print("Phase 1: Daily geocentric scan…", flush=True)

CHUNK    = 50_000
jd_all   = np.arange(JD_START, JD_END, 1.0)
n        = len(jd_all)
all_seps = np.empty(n)

for ci in range(0, n, CHUNK):
    end = min(ci + CHUNK, n)
    all_seps[ci:end] = geo_sep(jd_all[ci:end])
    yr = ts.tt_jd(jd_all[end - 1]).tt_calendar()[0]
    print(f"  … {end:>7,}/{n:,} days  ({era(yr)})", flush=True)

# Vectorised window detection
padded = np.concatenate([[False], all_seps < THRESH_COARSE, [False]])
diff   = np.diff(padded.astype(np.int8))
starts = np.where(diff ==  1)[0]
ends   = np.where(diff == -1)[0]

# 2-day buffer so the fine scan doesn't clip the true minimum
windows = [
    (jd_all[max(0, s - 2)], jd_all[min(n - 1, e + 1)])
    for s, e in zip(starts, ends)
]
print(f"\nFound {len(windows)} candidate windows (geocentric sep < {THRESH_COARSE}°).\n",
      flush=True)

# ── Phases 2 + 3: Refine each window, print qualifying events ────────────────
DIVIDER = "═" * 72
print("Phases 2/3: 1-minute refinement + visibility check…")
print("(Events appear below as they are found)\n", flush=True)
print(DIVIDER)

found = 0

for jd_lo, jd_hi in windows:
    span = jd_hi - jd_lo

    # ── 1-minute topocentric scan ──────────────────────────────────────────
    n_pts = max(int(span * 1440), 200)
    jd_f  = np.linspace(jd_lo, jd_hi, n_pts)
    s_f   = site_sep(jd_f)
    mi    = int(np.argmin(s_f))

    if s_f[mi] >= THRESH_FINE:
        continue   # minimum doesn't reach 1′

    # Find first/last indices where sep < 1′ in the 1-minute array
    below = s_f < THRESH_FINE
    i_en  = int(np.argmax(below))
    i_ex  = len(below) - 1 - int(np.argmax(below[::-1]))

    # Bisect precise entry and exit
    jd_en = (bisect_sep(jd_f[i_en - 1], jd_f[i_en], entering=True)
             if i_en > 0 else jd_f[0])
    jd_ex = (bisect_sep(jd_f[i_ex], jd_f[i_ex + 1], entering=False)
             if i_ex < n_pts - 1 else jd_f[-1])

    dur_min = (jd_ex - jd_en) * 1440.0
    jd_c    = jd_f[mi]
    sep_c   = s_f[mi]

    # ── Elongation check (visibility criterion 1) ─────────────────────────
    t1  = ts.tt_jd(np.array([jd_c]))
    obs = earth + obs_site
    aJ  = obs.at(t1).observe(jup).apparent()
    aS  = obs.at(t1).observe(sun).apparent()
    elong = float(aJ.separation_from(aS).degrees[0])

    if elong < AV_JUP:
        continue

    # ── Night-overlap check (visibility criterion 2) ──────────────────────
    # Require at least one moment during the < 1′ window where
    # sun < −6° AND Jupiter > 0° simultaneously.
    n_samp  = max(int(dur_min / 30) + 2, 4)
    jd_samp = np.linspace(jd_en, jd_ex, n_samp)
    sun_alt = site_sun_alt(jd_samp)
    jup_alt = site_jup_alt(jd_samp)

    if not np.any((sun_alt < NIGHT_ALT) & (jup_alt > 0.0)):
        continue

    # ── Full detail at closest approach ───────────────────────────────────
    aV = obs.at(t1).observe(ven).apparent()

    lat_j, lon_j, _ = aJ.frame_latlon(ecliptic_frame)
    lat_v, lon_v, _ = aV.frame_latlon(ecliptic_frame)
    lat_s, lon_s, _ = aS.frame_latlon(ecliptic_frame)

    alt_j, az_j, _  = aJ.altaz(temperature_C=20, pressure_mbar=1013)
    alt_v, az_v, _  = aV.altaz(temperature_C=20, pressure_mbar=1013)
    alt_sun_c        = float(aS.altaz()[0].degrees[0])

    t_obj     = ts.tt_jd(jd_c)
    delta_lon = (float(lon_j.degrees[0]) - float(lon_s.degrees[0])) % 360.0
    skypos    = "morning (W of Sun)" if delta_lon > 180 else "evening (E of Sun)"
    sky_cond  = ("night" if alt_sun_c < NIGHT_ALT
                 else f"daytime (sun alt {alt_sun_c:+.1f}°)")

    t_en_obj = ts.tt_jd(jd_en)
    t_ex_obj = ts.tt_jd(jd_ex)

    found += 1
    print()
    print(f"  EVENT #{found}  ──  {fmt(t_obj)}")
    print(f"    Closest approach : {fmt(t_obj, hhmm=True)}")
    print(f"    Local solar time : {lst_str(t_obj)}")
    print(f"    Separation       : {sep_c * 60:.3f}′  ({sep_c:.5f}°)")
    print(f"    Jupiter ecl lon  : {float(lon_j.degrees[0]):.3f}°"
          f"   lat: {float(lat_j.degrees[0]):+.3f}°")
    print(f"    Venus   ecl lon  : {float(lon_v.degrees[0]):.3f}°"
          f"   lat: {float(lat_v.degrees[0]):+.3f}°")
    print(f"    Solar elongation : {elong:.2f}°  ({skypos})")
    print(f"    Altitude at closest approach ({obs_name}):")
    print(f"      Jupiter : alt {float(alt_j.degrees[0]):+6.2f}°"
          f"   az {float(az_j.degrees[0]):6.2f}°")
    print(f"      Venus   : alt {float(alt_v.degrees[0]):+6.2f}°"
          f"   az {float(az_v.degrees[0]):6.2f}°")
    print(f"    Sky at closest approach  : {sky_cond}")
    print(f"    Within 1′ of separation:")
    print(f"      Enter  : {fmt(t_en_obj, hhmm=True)}"
          f"  (local {lst_str(t_en_obj)})")
    print(f"      Leave  : {fmt(t_ex_obj, hhmm=True)}"
          f"  (local {lst_str(t_ex_obj)})")
    print(f"      Duration : {dur_min:.1f} min")
    print(DIVIDER, flush=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(f"TOTAL Jupiter–Venus conjunctions < 1′ visible from {obs_name} : {found}")
print(f"  Range      : {era(-3000)} to {era(3000)}"
      f"  (~{total_days // 365} years, DE422 ephemeris)")
print(f"  Visibility : elongation > {AV_JUP}°,"
      f" night overlap with sun < {NIGHT_ALT}° and Jupiter above horizon")
print()
