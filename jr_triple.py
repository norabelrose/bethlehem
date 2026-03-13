#!/usr/bin/env python3
"""
jr_triple.py

Search the DE422 ephemeris (~3000 BC – 3000 AD) for Jupiter–Regulus
triple conjunctions.

A triple conjunction arises when Jupiter's retrograde loop carries it
within a configurable threshold of Regulus on three separate passes
(D→R→D: direct → retrograde → direct):
    Hit 1 – approaching Regulus in direct (eastward) motion
    Hit 2 – passing Regulus again during retrograde (westward) motion
    Hit 3 – departing Regulus in direct motion

Note: R→D→R (spanning two separate retrograde loops) is geometrically
impossible at any reasonable threshold. Consecutive retrograde arcs are
~20–30° apart in longitude, so a fixed star cannot lie within a few
degrees of both arcs simultaneously.

Algorithm:
  Phase 1  Daily geocentric scan — ecliptic longitude (for stations)
           and Jupiter–Regulus separation
  Phase 2  Detect Jupiter stations from sign changes in daily velocity
  Phase 3  Build ordered list of alternating Direct/Retrograde windows
  Phase 4  For each consecutive D,R,D triple of windows:
             – quick check using daily data (all 3 must be < threshold)
             – refine each hit with a 1-minute scan
             – report

Usage:
  python jr_triple.py                        # Jerusalem, 1° threshold
  python jr_triple.py babylon
  python jr_triple.py --threshold 0.5        # degrees
  python jr_triple.py --start -10 --end 10  # year range (astronomical)
  python jr_triple.py --lat 41.9 --lon 12.5 --name Rome
  python jr_triple.py --list
"""

import argparse
import sys
import numpy as np
from skyfield.api import load, Star, wgs84, N, E
from skyfield.framelib import ecliptic_J2000_frame as ecliptic_frame

# ── City database ──────────────────────────────────────────────────────────────
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

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Search DE422 for Jupiter–Regulus triple conjunctions.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="Available cities:\n" + "\n".join(
        f"  {k:<12}  {v[2]}  ({v[0]:.4f}°N, {v[1]:.4f}°E)"
        for k, v in CITIES.items()
    ),
)
parser.add_argument(
    "location", nargs="?", default="jerusalem", metavar="CITY",
    help="Observation city (default: jerusalem)",
)
parser.add_argument(
    "--threshold", type=float, default=1.0, metavar="DEG",
    help="Hit threshold in degrees (default: 1.0)",
)
parser.add_argument(
    "--start", type=int, default=-3000, metavar="YEAR",
    help="Start year, astronomical (default: -3000 = 3001 BC)",
)
parser.add_argument(
    "--end", type=int, default=3000, metavar="YEAR",
    help="End year, astronomical (default: 3000 AD)",
)
parser.add_argument("--lat",  type=float, metavar="DEG",
                    help="Custom observer latitude (°N)")
parser.add_argument("--lon",  type=float, metavar="DEG",
                    help="Custom observer longitude (°E)")
parser.add_argument("--name", type=str,   metavar="NAME", default="Custom site",
                    help="Display name for a custom location")
parser.add_argument("--list", action="store_true",
                    help="List available cities and exit")

args = parser.parse_args()

if args.list:
    print("Available cities:")
    for k, (lat, lon, name) in CITIES.items():
        print(f"  {k:<12}  {name}  ({lat:.4f}°N, {lon:.4f}°E)")
    sys.exit(0)

if args.lat is not None or args.lon is not None:
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon must both be provided together")
    obs_lat, obs_lon, obs_name = args.lat, args.lon, args.name
else:
    key = args.location.lower()
    if key not in CITIES:
        parser.error(f"Unknown city '{args.location}'. Use --list to see options.")
    obs_lat, obs_lon, obs_name = CITIES[key]

THRESHOLD = args.threshold   # degrees

# ── Load ephemeris ─────────────────────────────────────────────────────────────
print(f"Observer   : {obs_name}  ({obs_lat:.4f}°N, {obs_lon:.4f}°E)")
print(f"Threshold  : {THRESHOLD}°  ({THRESHOLD * 60:.1f}′)")
print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
eph = load("de422.bsp")
ts  = load.timescale()
print("Loaded.\n", flush=True)

earth   = eph["earth"]
sun     = eph["sun"]
jup     = eph["jupiter barycenter"]
regulus = Star(ra_hours=(10, 8, 22.311), dec_degrees=(11, 58, 1.95))

obs_site = wgs84.latlon(obs_lat * N, obs_lon * E)

# ── Formatting helpers ─────────────────────────────────────────────────────────
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

def sep_str(deg: float) -> str:
    """Format separation as arcminutes if < 1°, else degrees."""
    if deg < 1.0:
        return f"{deg * 60:.2f}′"
    return f"{deg:.3f}°"

# ── Vectorised position functions ──────────────────────────────────────────────
def geo_jup_lon(jd_arr: np.ndarray) -> np.ndarray:
    """Geocentric ecliptic longitude of Jupiter (°)."""
    t = ts.tt_jd(jd_arr)
    a = earth.at(t).observe(jup).apparent()
    _, lon, _ = a.frame_latlon(ecliptic_frame)
    return lon.degrees

def geo_sep(jd_arr: np.ndarray) -> np.ndarray:
    """Geocentric Jupiter–Regulus angular separation (°)."""
    t  = ts.tt_jd(jd_arr)
    aJ = earth.at(t).observe(jup).apparent()
    aR = earth.at(t).observe(regulus).apparent()
    return aJ.separation_from(aR).degrees

def geo_elong(jd_val: float) -> float:
    """Solar elongation of Jupiter (°) at a single JD."""
    t  = ts.tt_jd(np.array([jd_val]))
    aJ = earth.at(t).observe(jup).apparent()
    aS = earth.at(t).observe(sun).apparent()
    return float(aJ.separation_from(aS).degrees[0])

def geo_jup_ecl(jd_val: float):
    """Geocentric ecliptic lon/lat of Jupiter at a single JD. Returns (lon°, lat°)."""
    t = ts.tt_jd(np.array([jd_val]))
    a = earth.at(t).observe(jup).apparent()
    lat, lon, _ = a.frame_latlon(ecliptic_frame)
    return float(lon.degrees[0]), float(lat.degrees[0])

# ── Zodiac helpers ─────────────────────────────────────────────────────────────
# Fagan-Bradley ayanamsa: offset from J2000.0 ecliptic to Babylonian sidereal.
# Both frames are inertial/fixed, so this constant offset is valid for all dates.
# We use ecliptic_J2000_frame throughout, so coordinates are always J2000.0.
# Cross-check: Regulus at 149.8° J2000.0 → 125.1° Babylonian = 5° Leo ✓
#              Spica  at 203.9° J2000.0 → 179.1° Babylonian = 29° Virgo ✓
BABYLONIAN_AYANAMSA = 24.74  # degrees

ZODIAC_SIGNS = [
    (  0,  30, "Aries"),       ( 30,  60, "Taurus"),
    ( 60,  90, "Gemini"),      ( 90, 120, "Cancer"),
    (120, 150, "Leo"),         (150, 180, "Virgo"),
    (180, 210, "Libra"),       (210, 240, "Scorpius"),
    (240, 270, "Sagittarius"), (270, 300, "Capricorn"),
    (300, 330, "Aquarius"),    (330, 360, "Pisces"),
]

def zodiac_sign(j2000_lon_deg: float) -> tuple:
    """Return (sign_name, babylonian_lon) for a J2000.0 ecliptic longitude."""
    bab_lon = (j2000_lon_deg - BABYLONIAN_AYANAMSA) % 360
    for lo, hi, name in ZODIAC_SIGNS:
        if lo <= bab_lon < hi:
            return name, bab_lon
    return "Pisces", bab_lon

def get_sun_lon_vec(jd_arr: np.ndarray) -> np.ndarray:
    """Geocentric ecliptic longitude of the Sun (vectorised)."""
    t = ts.tt_jd(jd_arr)
    a = earth.at(t).observe(sun).apparent()
    _, lon, _ = a.frame_latlon(ecliptic_frame)
    return lon.degrees

def geo_elong_vec(jd_arr: np.ndarray) -> np.ndarray:
    """Solar elongation of Jupiter (vectorised)."""
    t  = ts.tt_jd(jd_arr)
    aJ = earth.at(t).observe(jup).apparent()
    aS = earth.at(t).observe(sun).apparent()
    return aJ.separation_from(aS).degrees


def find_heliacal_rising(jd_ref: float) -> tuple:
    """
    Find the most recent heliacal rising of Jupiter strictly before jd_ref.

    Strategy:
      1. Scan backward from jd_ref to find the most recent superior
         conjunction (local minimum of elongation with elong < 10°).
         Backward scan guarantees we find the immediately preceding
         conjunction, not one from an earlier synodic period.
      2. From that conjunction, scan forward day by day; return the first
         day on which Jupiter is in the morning sky (west of Sun) with
         geocentric elongation ≥ HELIACAL_THRESHOLD.

    Returns (jd_rising, jup_lon_j2000_degrees) or (None, None).
    """
    HELIACAL_THRESHOLD = 12.0   # degrees

    jd_start = max(jd_ref - 800, JD_START)
    jd_scan  = np.arange(jd_start, jd_ref, 1.0)
    if len(jd_scan) < 3:
        return None, None

    elong   = geo_elong_vec(jd_scan)
    jup_lon = geo_jup_lon(jd_scan)
    sun_lon = get_sun_lon_vec(jd_scan)
    morning = ((jup_lon - sun_lon) % 360) > 180   # Jupiter west of Sun

    # Backward scan: find the last local elongation minimum = last conjunction
    last_conj_idx = None
    for i in range(len(jd_scan) - 2, 0, -1):
        if elong[i] < elong[i - 1] and elong[i] < elong[i + 1] and elong[i] < 10.0:
            last_conj_idx = i
            break

    if last_conj_idx is None:
        return None, None

    # Forward scan: first morning day with enough elongation
    for i in range(last_conj_idx, len(jd_scan)):
        if morning[i] and elong[i] >= HELIACAL_THRESHOLD:
            return float(jd_scan[i]), float(jup_lon[i])

    return None, None

# ── Closest-approach refinement ────────────────────────────────────────────────
def closest_approach(jd_lo: float, jd_hi: float) -> tuple:
    """
    Find the geocentric closest Jupiter–Regulus approach in [jd_lo, jd_hi].
    Returns (jd_min, sep_min_degrees) with ~1-minute precision.
    """
    span = jd_hi - jd_lo

    # 1-hour scan across the window
    n_hr = max(int(span * 24) + 2, 48)
    jd_h = np.linspace(jd_lo, jd_hi, n_hr)
    sep_h = geo_sep(jd_h)
    mi = int(np.argmin(sep_h))

    # 1-minute scan ±1 hour around the hourly minimum
    lo2 = jd_h[max(mi - 1, 0)]
    hi2 = jd_h[min(mi + 1, n_hr - 1)]
    n_min = max(int((hi2 - lo2) * 1440) + 2, 120)
    jd_m = np.linspace(lo2, hi2, n_min)
    sep_m = geo_sep(jd_m)
    mi2 = int(np.argmin(sep_m))

    return float(jd_m[mi2]), float(sep_m[mi2])

# ── Scan bounds ────────────────────────────────────────────────────────────────
DE422_LO = ts.tt(-3000, 11, 14).tt
DE422_HI = ts.tt( 3000,  1,  1).tt
JD_START = max(ts.tt(args.start, 1, 1).tt, DE422_LO)
JD_END   = min(ts.tt(args.end,   1, 1).tt, DE422_HI)
total_days = int(JD_END - JD_START)

print(f"Scan range : {era(args.start)} to {era(args.end)}  (~{total_days:,} days)")

# ── Phase 1: Daily geocentric scan ─────────────────────────────────────────────
print("Phase 1: Daily geocentric scan (longitude + Regulus separation)…", flush=True)

CHUNK   = 50_000
jd_all  = np.arange(JD_START, JD_END, 1.0)
n       = len(jd_all)
lon_arr = np.empty(n)
sep_arr = np.empty(n)

for ci in range(0, n, CHUNK):
    end = min(ci + CHUNK, n)
    lon_arr[ci:end] = geo_jup_lon(jd_all[ci:end])
    sep_arr[ci:end] = geo_sep(jd_all[ci:end])
    yr = ts.tt_jd(jd_all[end - 1]).tt_calendar()[0]
    print(f"  … {end:>7,}/{n:,} days  ({era(yr)})", flush=True)

print(f"  Done.\n", flush=True)

# ── Phase 2: Find Jupiter stations ─────────────────────────────────────────────
print("Phase 2: Finding Jupiter stations…", flush=True)

# Unwrap longitude to handle the 360°→0° crossing
lon_uw = np.unwrap(lon_arr, period=360.0)

# Daily velocity (°/day); length = n-1
vel = np.diff(lon_uw)

# Replace exact zeros with the previous non-zero sign
sign_vel = np.sign(vel)
for i in range(1, len(sign_vel)):
    if sign_vel[i] == 0:
        sign_vel[i] = sign_vel[i - 1]

# Station indices: where sign_vel changes sign
# sign_vel[i] → sign_vel[i+1] change means station near day index i+1
sc = np.where(np.diff(sign_vel) != 0)[0]

stations = []   # list of (jd_tt, type)  type ∈ {'R', 'D'}
for i in sc:
    jd_stat = jd_all[i + 1]
    if sign_vel[i] > 0 and sign_vel[i + 1] < 0:
        stations.append((jd_stat, 'R'))   # direct→retrograde: 1st station
    elif sign_vel[i] < 0 and sign_vel[i + 1] > 0:
        stations.append((jd_stat, 'D'))   # retrograde→direct: 2nd station

print(f"  Found {len(stations)} stations.", flush=True)

# ── Phase 3: Build ordered motion windows ─────────────────────────────────────
# Each window = (motion, jd_lo, jd_hi)  motion ∈ {'D', 'R'}
# Consecutive windows alternate D and R.
# The station at the boundary between windows[k] and windows[k+1] has
# type = motion of windows[k+1] (the motion it initiates).

windows = []   # (motion, jd_lo, jd_hi)

if stations:
    # Window before the first station
    pre_motion = 'D' if stations[0][1] == 'R' else 'R'
    windows.append((pre_motion, JD_START, stations[0][0]))

    # Windows between consecutive stations
    for i in range(len(stations) - 1):
        # stations[i] initiates the motion in this interval
        motion = stations[i][1]   # 'R' → retrograde; 'D' → direct
        windows.append((motion, stations[i][0], stations[i + 1][0]))

    # Window after the last station
    post_motion = stations[-1][1]
    windows.append((post_motion, stations[-1][0], JD_END))
else:
    windows.append(('D', JD_START, JD_END))

print(f"  Built {len(windows)} motion windows.\n", flush=True)

# ── Helper: minimum daily separation in a window ──────────────────────────────
def window_min_sep(jd_lo: float, jd_hi: float):
    """
    Return (jd_of_daily_min, min_sep_deg) from the pre-computed sep_arr
    for the time window [jd_lo, jd_hi].  Returns (None, inf) if empty.
    """
    i_lo = max(0, int(round((jd_lo - JD_START))))
    i_hi = min(n, int(round((jd_hi - JD_START))) + 1)
    if i_hi <= i_lo:
        return None, np.inf
    seg = sep_arr[i_lo:i_hi]
    mi  = int(np.argmin(seg))
    return jd_all[i_lo + mi], float(seg[mi])

# ── Phase 4: Detect triple conjunctions ───────────────────────────────────────
print("Phase 4: Searching for triple conjunctions…", flush=True)
print("(Events appear below as they are found)\n", flush=True)

DIVIDER = "═" * 72

found = 0
heliacal_signs = []   # zodiac sign of each preceding heliacal rising

for k in range(len(windows) - 2):
    m1, lo1, hi1 = windows[k]
    m2, lo2, hi2 = windows[k + 1]
    m3, lo3, hi3 = windows[k + 2]

    # Only D→R→D
    if not (m1 == 'D' and m2 == 'R' and m3 == 'D'):
        continue

    # Quick check with daily data — all three windows must have a hit
    _, sep1_daily = window_min_sep(lo1, hi1)
    if sep1_daily >= THRESHOLD:
        continue
    _, sep2_daily = window_min_sep(lo2, hi2)
    if sep2_daily >= THRESHOLD:
        continue
    _, sep3_daily = window_min_sep(lo3, hi3)
    if sep3_daily >= THRESHOLD:
        continue

    # Refine each hit to ~1-minute precision
    jd1, s1 = closest_approach(lo1, hi1)
    jd2, s2 = closest_approach(lo2, hi2)
    jd3, s3 = closest_approach(lo3, hi3)

    # Confirm all three still qualify after refinement
    if s1 >= THRESHOLD or s2 >= THRESHOLD or s3 >= THRESHOLD:
        continue

    # ── Assemble event details ────────────────────────────────────────────────
    t1 = ts.tt_jd(jd1)
    t2 = ts.tt_jd(jd2)
    t3 = ts.tt_jd(jd3)

    e1 = geo_elong(jd1)
    e2 = geo_elong(jd2)
    e3 = geo_elong(jd3)

    lon1, lat1 = geo_jup_ecl(jd1)
    lon2, lat2 = geo_jup_ecl(jd2)
    lon3, lat3 = geo_jup_ecl(jd3)

    # Station times at the two window boundaries
    t_s1 = ts.tt_jd(hi1)   # hi1 == lo2: 1st station (direct → retrograde)
    t_s2 = ts.tt_jd(hi2)   # hi2 == lo3: 2nd station (retrograde → direct)

    found += 1
    print(DIVIDER)
    print(f"  TRIPLE CONJUNCTION #{found}")
    print()
    print(f"    Hit 1  (    direct):  {fmt(t1, hhmm=True)}"
          f"   sep {sep_str(s1):>9}   elong {e1:.1f}°")
    print(f"    Hit 2  (retrograde):  {fmt(t2, hhmm=True)}"
          f"   sep {sep_str(s2):>9}   elong {e2:.1f}°")
    print(f"    Hit 3  (    direct):  {fmt(t3, hhmm=True)}"
          f"   sep {sep_str(s3):>9}   elong {e3:.1f}°")
    print()
    print(f"    Stations:")
    print(f"      Between hits 1–2:  {fmt(t_s1)}   (1st station: direct→retrograde)")
    print(f"      Between hits 2–3:  {fmt(t_s2)}   (2nd station: retrograde→direct)")
    print()
    print(f"    Jupiter ecliptic position at each hit:")
    print(f"      Hit 1:  lon {lon1:.2f}°   lat {lat1:+.3f}°")
    print(f"      Hit 2:  lon {lon2:.2f}°   lat {lat2:+.3f}°")
    print(f"      Hit 3:  lon {lon3:.2f}°   lat {lat3:+.3f}°")
    print()
    jd_hr, hr_lon = find_heliacal_rising(jd1)
    if jd_hr is not None:
        t_hr = ts.tt_jd(jd_hr)
        sign, bab_lon = zodiac_sign(hr_lon)
        heliacal_signs.append(sign)
        print(f"    Preceding Jupiter heliacal rising:")
        print(f"      Date  :  {fmt(t_hr)}")
        print(f"      Zodiac:  {sign}  ({bab_lon:.1f}° Babylonian sidereal)")
    else:
        print(f"    Preceding Jupiter heliacal rising:  not found in search window")
    print(DIVIDER, flush=True)

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print(f"TOTAL Jupiter–Regulus triple conjunctions"
      f" (threshold {THRESHOLD}°, {era(args.start)}–{era(args.end)}): {found}")
print(f"  Ephemeris : DE422  (~{total_days // 365} years scanned)")

if heliacal_signs:
    print()
    print("Zodiac sign of preceding Jupiter heliacal rising:")
    from collections import Counter
    counts = Counter(heliacal_signs)
    total_hr = len(heliacal_signs)
    sign_order = [name for _, _, name in ZODIAC_SIGNS]
    for name in sign_order:
        n = counts.get(name, 0)
        if n == 0:
            continue
        bar = "█" * n
        print(f"  {name:<13}  {n:>5}  ({100 * n / total_hr:5.1f}%)  {bar}")
    print(f"  {'(not found)':<13}  {found - total_hr:>5}  ({100 * (found - total_hr) / found:5.1f}%)")

print()
