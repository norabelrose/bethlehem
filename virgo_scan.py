#!/usr/bin/env python3
"""
virgo_scan.py

Scans a configurable date range for dates when the Sun is "in" the
constellation Virgo and the Moon is near the "feet" of Virgo,
as seen from a chosen location.  The combination is associated with the sign
described in Revelation 12:1.

Coordinate system: ICRS (J2000.0) RA/Dec — star-fixed, so constellation
boundaries are stable regardless of epoch.  IAU Virgo boundaries are
defined at B1875.0 but are accurate to within ~1° for our purposes.

Astronomical year -2 = 3 BC.
"""

import argparse
from skyfield.api import load, wgs84, N, E
from skyfield.framelib import ecliptic_frame

# ── CLI ───────────────────────────────────────────────────────────────────────
CITIES = {
    "jerusalem": (31.7683, 35.2137, "Jerusalem"),
    "babylon": (32.5427, 44.4215, "Babylon"),
    "alexandria": (31.2001, 29.9187, "Alexandria"),
    "antioch": (36.2021, 36.1601, "Antioch"),
    "athens": (37.9838, 23.7275, "Athens"),
    "rome": (41.9028, 12.4964, "Rome"),
    "fatima": (39.6284, -8.6718, "Fátima"),
}

parser = argparse.ArgumentParser(
    description="Scan a date range for Sun-in-Virgo + Moon-at-feet-of-Virgo.",
    epilog="Available cities: " + ", ".join(CITIES),
)
parser.add_argument(
    "location",
    nargs="?",
    default="jerusalem",
    metavar="CITY",
    help="Observation city (default: jerusalem)",
)
parser.add_argument("--lat", type=float, metavar="DEG", help="Custom latitude °N")
parser.add_argument("--lon", type=float, metavar="DEG", help="Custom longitude °E")
parser.add_argument("--name", type=str, metavar="NAME", default="Custom site")
parser.add_argument(
    "--start", type=int, nargs=2, metavar=("YEAR", "MONTH"), default=[-2, 7],
    help="Start year and month (astronomical year, default: -2 7 = Jul 3 BC)",
)
parser.add_argument(
    "--end", type=int, nargs=2, metavar=("YEAR", "MONTH"), default=[-2, 12],
    help="End year and month (astronomical year, default: -2 12 = Dec 3 BC)",
)

args = parser.parse_args()

if args.lat is not None or args.lon is not None:
    if args.lat is None or args.lon is None:
        parser.error("--lat and --lon must both be supplied together")
    obs_lat, obs_lon, obs_name = args.lat, args.lon, args.name
else:
    key = args.location.lower()
    if key not in CITIES:
        parser.error(f"Unknown city '{args.location}'. Available: {', '.join(CITIES)}")
    obs_lat, obs_lon, obs_name = CITIES[key]

start_year, start_month = args.start
end_year, end_month = args.end
if (start_year, start_month) > (end_year, end_month):
    parser.error("--start must not be later than --end")

# ── Constellation boundary approximations (ICRS/J2000, degrees) ───────────────
#
# Virgo (IAU approximate rectangle):
#   RA  11h 36m – 15h 12m  →  174.0° – 228.0°
#   Dec  −22°  –  +14°
#
# "Feet of Virgo" — the lower/southern portion of the stick figure.
# In traditional star-atlas drawings, Virgo's feet point south-west and
# lie in roughly the lower third of the constellation figure, around:
#   RA  13h 00m – 15h 00m  →  195° – 225°
#   Dec  −20°  –   0°
# The moon being "at/under her feet" is satisfied when it is in this
# region, or just south of it (Dec just below −20°).

VIRGO_RA_MIN = 174.0  # 11h 36m
VIRGO_RA_MAX = 228.0  # 15h 12m
VIRGO_DEC_MIN = -22.0
VIRGO_DEC_MAX = 14.0

# Feet sub-region (southern portion of Virgo figure)
FEET_RA_MIN = 195.0  # 13h 00m
FEET_RA_MAX = 225.0  # 15h 00m
FEET_DEC_MIN = -20.0
FEET_DEC_MAX = 0.0


def in_virgo(ra: float, dec: float) -> bool:
    return VIRGO_RA_MIN <= ra <= VIRGO_RA_MAX and VIRGO_DEC_MIN <= dec <= VIRGO_DEC_MAX


def in_feet(ra: float, dec: float) -> bool:
    return FEET_RA_MIN <= ra <= FEET_RA_MAX and FEET_DEC_MIN <= dec <= FEET_DEC_MAX


def phase_name(deg: float) -> str:
    """Approximate lunar phase label from elongation angle."""
    d = deg % 360
    if d < 22 or d > 338:
        return "new"
    if 22 <= d < 68:
        return "waxing crescent"
    if 68 <= d < 112:
        return "first quarter"
    if 112 <= d < 158:
        return "waxing gibbous"
    if 158 <= d < 202:
        return "full"
    if 202 <= d < 248:
        return "waning gibbous"
    if 248 <= d < 292:
        return "last quarter"
    return "waning crescent"


# ── Load ephemeris ────────────────────────────────────────────────────────────
print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
eph = load("de422.bsp")
ts = load.timescale()
print("Loaded.\n")

earth = eph["earth"]
sun_b = eph["sun"]
moon_b = eph["moon"]

obs = earth + wgs84.latlon(obs_lat * N, obs_lon * E)

# ── Date helpers ──────────────────────────────────────────────────────────────
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def era(y: int) -> str:
    return f"{-y+1} BC" if y <= 0 else f"{y} AD"


def fmt(t) -> str:
    y, mo, d, *_ = t.tt_calendar()
    return f"{int(d):2d} {MONTHS[mo-1]} {era(y)}"


# Julian calendar offset relative to proleptic Gregorian in 3 BC.
# Going forward from 3 BC to 1582 AD, 12 century years (not divisible by 400)
# each give Julian an extra leap day, so Julian falls 12 days behind Gregorian
# by 1582.  But in 1582 it was only 10 days behind → in 3 BC Julian was
# 10 − 12 = −2 days behind, i.e. 2 days AHEAD of Gregorian.
# Proleptic Julian date = Proleptic Gregorian date + 2 days.
JULIAN_OFFSET = 2  # days to add to Gregorian to get proleptic Julian


def fmt_julian(t) -> str:
    """Return the proleptic Julian calendar date corresponding to t."""
    y, mo, d, H, Mi, S = t.tt_calendar()
    # Advance by JULIAN_OFFSET days using JD arithmetic
    jd_julian = t.tt + JULIAN_OFFSET
    t2 = ts.tt_jd(jd_julian)
    y2, mo2, d2, *_ = t2.tt_calendar()
    return f"{int(d2):2d} {MONTHS[mo2-1]} {era(y2)} (Julian)"


# ── Build list of days to scan ────────────────────────────────────────────────
def _days_in_month(year: int, month: int) -> int:
    is_leap = (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)
    return [0, 31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month]


def _era(y: int) -> str:
    return f"{-y+1} BC" if y <= 0 else f"{y} AD"


def _month_label(year: int, month: int) -> str:
    return f"1 {MONTHS[month-1]} {_era(year)}"


def _last_day_label(year: int, month: int) -> str:
    d = _days_in_month(year, month)
    return f"{d} {MONTHS[month-1]} {_era(year)}"


scan_days = []
yr, mo = start_year, start_month
while (yr, mo) <= (end_year, end_month):
    for da in range(1, _days_in_month(yr, mo) + 1):
        scan_days.append((yr, mo, da))
    mo += 1
    if mo > 12:
        mo, yr = 1, yr + 1

print(f"Observer  : {obs_name}  ({obs_lat:.4f}°N, {obs_lon:.4f}°E)")
print(
    f"Scan range: {_month_label(start_year, start_month)} – "
    f"{_last_day_label(end_year, end_month)}  ({len(scan_days)} days, noon TT each day)"
)
print()
print(
    f"Virgo boundary (ICRS J2000): RA {VIRGO_RA_MIN:.0f}°–{VIRGO_RA_MAX:.0f}°"
    f"  Dec {VIRGO_DEC_MIN:+.0f}°–{VIRGO_DEC_MAX:+.0f}°"
)
print(
    f"Feet sub-region           : RA {FEET_RA_MIN:.0f}°–{FEET_RA_MAX:.0f}°"
    f"  Dec {FEET_DEC_MIN:+.0f}°–{FEET_DEC_MAX:+.0f}°"
)
print()

# ── Column headers ────────────────────────────────────────────────────────────
H1 = (
    f"{'Date (Gregorian)':<15}  {'(Julian)':>15}  "
    f"{'Sun RA°':>7}  {'Sun Dec°':>8}  {'In Virgo':>8}    "
    f"{'Moon RA°':>8}  {'Moon Dec°':>9}  {'Phase':>7}  {'Elng°':>5}  {'Feet?':>5}  "
    f"{'Moon ecl lon°':>13}"
)
print(H1)
print("─" * len(H1))

# ── Day loop ──────────────────────────────────────────────────────────────────
matches = []

for yr, mo, da in scan_days:
    t = ts.tt(yr, mo, da, 12)  # noon TT

    # Sun — ICRS RA/Dec
    sun_app = obs.at(t).observe(sun_b).apparent()
    sun_ra, sun_dec, _ = sun_app.radec()  # ICRS (J2000), star-fixed
    sun_ra_d = sun_ra.hours * 15.0
    sun_dec_d = sun_dec.degrees
    _, sun_ecl_lon, _ = sun_app.frame_latlon(ecliptic_frame)
    sun_lon_d = sun_ecl_lon.degrees

    # Moon — ICRS RA/Dec
    moon_app = obs.at(t).observe(moon_b).apparent()
    moon_ra, moon_dec, _ = moon_app.radec()
    moon_ra_d = moon_ra.hours * 15.0
    moon_dec_d = moon_dec.degrees
    _, moon_ecl_lon, _ = moon_app.frame_latlon(ecliptic_frame)
    moon_lon_d = moon_ecl_lon.degrees

    # Lunar phase: elongation of moon from sun (ecliptic longitude difference)
    elong_d = (moon_lon_d - sun_lon_d) % 360.0
    ph = phase_name(elong_d)

    sun_in_v = in_virgo(sun_ra_d, sun_dec_d)
    moon_feet = in_feet(moon_ra_d, moon_dec_d)
    flag = "  ◄◄◄ MATCH" if (sun_in_v and moon_feet) else ""

    # Julian date label (same JD, calendar shifted by JULIAN_OFFSET)
    jd_julian = t.tt + JULIAN_OFFSET
    t_jul = ts.tt_jd(jd_julian)
    yj, moj, dj, *_ = t_jul.tt_calendar()
    jul_label = f"{int(dj):2d} {MONTHS[moj-1]} (Jul)"

    print(
        f"{fmt(t):<15}  {jul_label:<15}  "
        f"{sun_ra_d:7.2f}  {sun_dec_d:+8.2f}  {str(sun_in_v):>8}    "
        f"{moon_ra_d:8.2f}  {moon_dec_d:+9.2f}  {ph:>15}  {elong_d:5.1f}  {str(moon_feet):>5}  "
        f"{moon_lon_d:13.2f}{flag}"
    )

    if sun_in_v and moon_feet:
        matches.append(
            (
                fmt(t),
                fmt_julian(t),
                sun_ra_d,
                sun_dec_d,
                moon_ra_d,
                moon_dec_d,
                elong_d,
                ph,
                moon_lon_d,
            )
        )

# ── Summary ───────────────────────────────────────────────────────────────────
DIV = "═" * 100
print()
print(DIV)
print(f"\nSun-in-Virgo ∧ Moon-in-feet-of-Virgo : {len(matches)} day(s) found\n")
if matches:
    for date, julian_date, sra, sdec, mra, mdec, el, ph, mlon in matches:
        print(f"  {date}  =  {julian_date}")
        print(f"    Sun  : RA {sra:.2f}°  Dec {sdec:+.2f}°")
        print(f"    Moon : RA {mra:.2f}°  Dec {mdec:+.2f}°  ecl lon {mlon:.2f}°")
        print(f"    Phase: {ph}  (elongation {el:.1f}°)")
        print()

# ── Also report all Sun-in-Virgo days, for reference ─────────────────────────
print(DIV)
print("\nAll Sun-in-Virgo days in the scan period:\n")

in_virgo_run = []
prev_virgo = False
for yr, mo, da in scan_days:
    t = ts.tt(yr, mo, da, 12)
    sun_app = obs.at(t).observe(sun_b).apparent()
    sun_ra, sun_dec, _ = sun_app.radec()
    sun_ra_d = sun_ra.hours * 15.0
    sun_dec_d = sun_dec.degrees
    cur_virgo = in_virgo(sun_ra_d, sun_dec_d)
    if cur_virgo and not prev_virgo:
        in_virgo_run.append([fmt(t), None])
    if not cur_virgo and prev_virgo:
        in_virgo_run[-1][1] = fmt(ts.tt(yr, mo, da - 1 if da > 1 else 1, 12))
    prev_virgo = cur_virgo

if in_virgo_run and in_virgo_run[-1][1] is None:
    in_virgo_run[-1][1] = fmt(ts.tt(end_year, end_month, _days_in_month(end_year, end_month), 12))

for start, end in in_virgo_run:
    print(f"  {start}  –  {end}")
print()
