#!/usr/bin/env python3
"""
hebrew_calendar.py

Reconstructs the observation-based Hebrew lunisolar calendar
for the period covering 1 Jan 3 BC – 1 Jan 1 BC (proleptic Gregorian),
with extended context to cover complete Jewish years.

Method
------
Lunar conjunctions (new moons) are found with Skyfield / JPL DE422.
The first visible crescent after each conjunction is determined using the
Yallop (1997) criterion, computed for Jerusalem (31.77°N, 35.22°E).

The Yallop q-factor is evaluated at the moment the sun reaches –5° altitude
(best naked-eye crescent-watching window) each candidate evening:

    q = ARCV – (11.8371 – 6.3226√W + 0.7319W³ – 0.1018W⁴)

where ARCV = moon altitude – sun altitude (degrees) and W = crescent width
(arcminutes) = SD × (1 – cos ARCL), with SD = moon topocentric semi-diameter
in arcminutes and ARCL = topocentric elongation.

  Category  q range         Ancient naked-eye visibility
  ───────── ──────────────  ────────────────────────────
  A         q > 0.216       Easily visible
  B         0.216 ≥ q > –0.014  Visible under good conditions
  C        –0.014 ≥ q > –0.160  Uncertain — experienced observer might see it
  D        –0.160 ≥ q > –0.232  Not visible naked-eye
  E         q ≤ –0.232     Definitely not visible

Month-start rules applied here:
  A, B → month begins on the evening of that date (firm)
  C    → month begins that evening or the next (flagged ±1 day)
  D, E → crescent not seen; try next evening

No Hebrew month is shorter than 29 days or longer than 30 days.

Month naming
------------
1 Nisan is defined as the first month whose full moon (≈ day 14-15 of the
month, ≈ 14 days after new moon) falls on or after the spring equinox.
If the candidate Nisan's full moon falls before the equinox, an intercalary
Adar II is inserted and the following month becomes Nisan.

The Jewish civil year begins with Tishri (month 7 from Nisan).
Jewish AM years follow the convention AM = astronomical_year + 3760.

Date convention
---------------
Dates reported are the proleptic Gregorian evening on which the crescent was
first sighted (or would have been sighted). The Hebrew day began at that
sunset. In common Western notation this is sometimes written as the *next*
Gregorian calendar day; this script uses the sunset evening date.

Error bars
----------
±1 day is shown for category-C months (borderline visibility).
The ΔT uncertainty for this era (~3.5 ± 1 h) rarely shifts crescent
sightings across a day boundary, so no separate ΔT error bar is given.
Intercalation uncertainty (whether the Sanhedrin actually intercalated in a
given year) is noted where the decision was close.
"""

import numpy as np
from skyfield.api import load, wgs84, N, E
from skyfield import almanac

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
eph   = load("de422.bsp")
ts    = load.timescale()
sun   = eph["sun"]
moon  = eph["moon"]
earth = eph["earth"]

jerusalem = wgs84.latlon(31.7683 * N, 35.2137 * E)
observer  = earth + jerusalem

MONTHS_GR = ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]

HEBREW_MONTHS = [
    "Nisan","Iyar","Sivan","Tammuz","Av","Elul",
    "Tishri","Cheshvan","Kislev","Tevet","Shevat","Adar",
]
# Leap-year sequence (from Nisan):
# Nisan Iyar Sivan Tammuz Av Elul Tishri Cheshvan Kislev Tevet Shevat Adar-I Adar-II

# ─────────────────────────────────────────────────────────────────────────────
# Date formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def era(y: int) -> str:
    return f"{-y+1} BC" if y <= 0 else f"{y} AD"

def fmt_date(t) -> str:
    y, mo, d = t.tt_calendar()[:3]
    return f"{int(d):2d} {MONTHS_GR[mo-1]} {era(y)}"

def fmt_datetime(t) -> str:
    y, mo, d, H, Mi, S = t.tt_calendar()
    fh = H + Mi/60 + S/3600
    return f"{int(d):2d} {MONTHS_GR[mo-1]} {era(y)} {fh:05.2f}h TT"

# ─────────────────────────────────────────────────────────────────────────────
# Astronomical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _body_alts(body, jd_start, jd_end, n=120):
    """Vectorised altitude array for body over [jd_start, jd_end]."""
    jds  = np.linspace(jd_start, jd_end, n)
    t_arr = ts.tt_jd(jds)
    alts = observer.at(t_arr).observe(body).apparent().altaz()[0].degrees
    return jds, alts

def _crossing_time(body, target_alt, jd_start, jd_end, rising=False):
    """
    Find the first crossing of target_alt in [jd_start, jd_end] via
    vectorised scan + bisection refinement.
    rising=True → ascending; False → descending.  Returns Skyfield Time or None.
    """
    jds, alts = _body_alts(body, jd_start, jd_end)
    shifted = alts - target_alt

    for i in range(len(shifted) - 1):
        a0, a1 = shifted[i], shifted[i + 1]
        cross = (a0 < 0 and a1 >= 0) if rising else (a0 >= 0 and a1 < 0)
        if not cross:
            continue
        lo, hi = jds[i], jds[i + 1]
        for _ in range(30):
            mid = (lo + hi) / 2
            t_mid = ts.tt_jd(mid)
            a_mid = observer.at(t_mid).observe(body).apparent().altaz()[0].degrees
            if ((a_mid - target_alt) * a0 > 0):
                lo = mid
            else:
                hi = mid
        return ts.tt_jd((lo + hi) / 2)
    return None

# NOTE on JD convention: floor(JD) is *noon* TT of that Julian Day.
# Jerusalem sunset (ΔT≈3.5h, so UT≈TT-3.5h) occurs at roughly:
#   Sep: ~15:30 UT → ~19:00 TT → floor(JD) + 0.29 days after noon
#   Dec: ~14:30 UT → ~18:00 TT → floor(JD) + 0.25 days after noon
# So we search from floor(JD)+0.10 to floor(JD)+0.45 to bracket all seasons.

def find_sunset(date_jd: float):
    """Sunset TT on the evening of the Julian Day containing date_jd."""
    d0 = np.floor(date_jd)
    return _crossing_time(sun, 0.0, d0 + 0.10, d0 + 0.48, rising=False)

def find_sun_at_minus5(date_jd: float):
    """When sun descends through –5° on the evening of date_jd."""
    d0 = np.floor(date_jd)
    return _crossing_time(sun, -5.0, d0 + 0.12, d0 + 0.52, rising=False)

def find_moonset(date_jd: float):
    """First moonset after noon TT on date_jd."""
    d0 = np.floor(date_jd)
    return _crossing_time(moon, 0.0, d0 + 0.15, d0 + 0.80, rising=False)

def spring_equinox(astro_year: int):
    """Return the spring equinox time for an astronomical year."""
    t0 = ts.tt(astro_year, 2, 15)
    t1 = ts.tt(astro_year, 5, 15)
    f = almanac.seasons(eph)
    times, events = almanac.find_discrete(t0, t1, f)
    # Spring equinox = event 0
    eq_times = times[events == 0]
    return eq_times[0] if len(eq_times) > 0 else None

# ─────────────────────────────────────────────────────────────────────────────
# Yallop crescent-visibility criterion
# ─────────────────────────────────────────────────────────────────────────────

MOON_RADIUS_KM = 1737.4
AU_KM          = 149_597_870.7

def yallop(t_obs):
    """
    Yallop q-factor at observation time t_obs from Jerusalem.
    Returns dict with q, category, arcl, arcv, W, moon_alt.
    """
    a_moon = observer.at(t_obs).observe(moon).apparent()
    a_sun  = observer.at(t_obs).observe(sun).apparent()

    alt_m, _, dist_m = a_moon.altaz()
    alt_s, _,  _     = a_sun.altaz()

    arcl = a_moon.separation_from(a_sun).degrees
    arcv = alt_m.degrees - alt_s.degrees

    sd_arcmin = np.degrees(np.arctan(MOON_RADIUS_KM / (dist_m.au * AU_KM))) * 60
    W = sd_arcmin * (1.0 - np.cos(np.radians(arcl)))   # crescent width, arcmin

    q = arcv - (11.8371 - 6.3226 * np.sqrt(W) + 0.7319 * W**3 - 0.1018 * W**4)

    if   q >  0.216: cat = "A"
    elif q > -0.014: cat = "B"
    elif q > -0.160: cat = "C"
    elif q > -0.232: cat = "D"
    else:            cat = "E"

    return {"q": q, "cat": cat, "arcl": arcl, "arcv": arcv,
            "W": W, "moon_alt": alt_m.degrees, "sun_alt": alt_s.degrees}

# ─────────────────────────────────────────────────────────────────────────────
# First-crescent finder
# ─────────────────────────────────────────────────────────────────────────────

def first_crescent(nm_t):
    """
    Given a new-moon Skyfield Time nm_t, find the first evening when the
    crescent is visible from Jerusalem.

    Returns dict:
      evening_jd  – TT JD of the observing evening (floor of calendar date)
      evening_t   – Skyfield Time of the –5° sun moment
      day_offset  – days after conjunction (1, 2, or 3)
      yallop      – Yallop result dict for that evening
      moon_age_h  – hours since conjunction
      lag_min     – moonset lag after sunset (minutes)
      uncertain   – True if category C (±1 day)
      note        – human-readable note
    """
    nm_jd = nm_t.tt
    best   = None
    second = None   # next day, for uncertainty

    for offset in range(1, 4):
        cand_jd = np.floor(nm_jd) + offset

        # Preferred observation time: sun at –5° (best crescent window)
        t_obs = find_sun_at_minus5(cand_jd)
        if t_obs is None:
            t_obs = find_sunset(cand_jd)
        if t_obs is None:
            continue

        moon_age_h = (t_obs.tt - nm_jd) * 24.0

        # Danjon limit: moon rarely visible if age < 13.5 h
        if moon_age_h < 13.5:
            continue

        v = yallop(t_obs)

        # Moon must be above horizon
        if v["moon_alt"] < 0:
            continue

        # Lag time (moonset – sunset)
        t_ss = find_sunset(cand_jd)
        t_ms = find_moonset(cand_jd)
        lag_min = (t_ms.tt - t_ss.tt) * 1440 if (t_ms is not None and t_ss is not None) else 0.0

        rec = {
            "evening_jd": cand_jd,
            "evening_t": t_obs,
            "day_offset": offset,
            "yallop": v,
            "moon_age_h": moon_age_h,
            "lag_min": lag_min,
            "uncertain": False,
            "note": "",
        }

        if v["cat"] in ("A", "B"):
            return rec                     # definite visibility
        elif v["cat"] == "C":
            if best is None:
                best = rec
                best["uncertain"] = True
                best["note"] = "±1 day (borderline)"
            elif second is None:
                second = rec
            # Continue to check if next day is firmer (A/B)
        else:
            # D or E: not visible; continue to next evening
            if best is None:
                best = rec   # save in case nothing better turns up
                best["uncertain"] = True
                best["note"] = "visibility doubtful – possible ±1 day"

    # If best is C, check whether second (next day) is A/B → if so, second is
    # the "lower-uncertainty" anchor but first day remains possible
    if best and best["yallop"]["cat"] == "C":
        best["note"] = "±1 day (borderline Yallop C)"
    return best

# ─────────────────────────────────────────────────────────────────────────────
# Find all new moons in extended period
# ─────────────────────────────────────────────────────────────────────────────

print("Finding new moons (Sep 5 BC → Mar 1 BC)…", flush=True)
t_scan_start = ts.tt(-4, 9, 1)   # Sep 5 BC  (ensures spring 4 BC months exist)
t_scan_end   = ts.tt( 0, 4, 1)   # Apr 1 BC

phase_times, phase_idx = almanac.find_discrete(
    t_scan_start, t_scan_end, almanac.moon_phases(eph)
)
new_moons = phase_times[phase_idx == 0]
print(f"  Found {len(new_moons)} new moons.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Compute first-crescent date for every new moon
# ─────────────────────────────────────────────────────────────────────────────

print("Computing first-crescent visibility for each month…", flush=True)
month_starts = []   # list of crescent-result dicts, in order

for nm in new_moons:
    fc = first_crescent(nm)
    if fc:
        month_starts.append(fc)
        y, mo, d = ts.tt_jd(fc["evening_jd"]).tt_calendar()[:3]
        print(f"  New moon {fmt_datetime(nm)}  →  "
              f"crescent {fmt_date(ts.tt_jd(fc['evening_jd']))}  "
              f"[{fc['yallop']['cat']}]  "
              f"{'(uncertain)' if fc['uncertain'] else ''}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Assign Hebrew month names
# ─────────────────────────────────────────────────────────────────────────────
# Strategy:
#   1. Compute spring equinoxes for each astronomical year in range.
#   2. For each equinox, find the first new moon such that its full moon
#      (roughly nm + 14.75 days) falls ON or AFTER the equinox.
#      That new moon's crescent evening is 1 Nisan.
#   3. Assign months forward from 1 Nisan.
#   4. If the interval from the previous Tishri to the next Nisan spans
#      8 new moons instead of 7, a leap year was observed.

print("Computing spring equinoxes…", flush=True)
equinoxes = {}
for astro_yr in range(-3, 1):    # 4 BC, 3 BC, 2 BC, 1 BC
    eq = spring_equinox(astro_yr)
    if eq is not None:
        equinoxes[astro_yr] = eq
        print(f"  Spring equinox {astro_yr:+d} ({era(astro_yr)}): {fmt_datetime(eq)}")
print()

# For each equinox, find corresponding 1 Nisan
nisan_starts = {}   # astro_year → index into month_starts[]

SYNODIC = 29.53059   # mean synodic month (days)
NISAN_WINDOW_DAYS = 40.0  # latest plausible Nisan start after spring equinox

for astro_yr, eq_t in equinoxes.items():
    eq_jd = eq_t.tt
    # Full moon ≈ nm_jd + 14.75 days
    # We want: nm_jd + 14.75 >= eq_jd  →  nm_jd >= eq_jd – 14.75
    candidates = []
    for idx, ms in enumerate(month_starts):
        # Constrain candidates to spring months near this equinox.
        # Without this bound, an autumn month (e.g., previous Tishri) also
        # satisfies fm_approx >= eq_jd and can be misidentified as Nisan.
        if ms["evening_jd"] > eq_jd + NISAN_WINDOW_DAYS:
            break
        fm_approx = ms["evening_jd"] + 14.75
        if fm_approx >= eq_jd:
            candidates.append((idx, fm_approx))
    if candidates:
        # Take the earliest (first month whose full moon is on or after equinox)
        idx0 = candidates[0][0]
        nisan_starts[astro_yr] = idx0
        print(f"  1 Nisan {astro_yr} ({era(astro_yr)}): "
              f"{fmt_date(ts.tt_jd(month_starts[idx0]['evening_jd']))}  "
              f"(full moon ~{fmt_date(ts.tt_jd(month_starts[idx0]['evening_jd']+14.75))}  "
              f"equinox {fmt_date(eq_t)})")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Build annotated calendar
# ─────────────────────────────────────────────────────────────────────────────
# Walk through month_starts[], assigning Hebrew names using Nisan anchors.

# Month sequence from Tishri (civil new year):
#   Tishri(7) Cheshvan(8) Kislev(9) Tevet(10) Shevat(11)
#   Adar(12) [Adar-II(13)] Nisan(1) Iyar(2) Sivan(3) Tammuz(4) Av(5) Elul(6)
#   → back to Tishri

SEQ_FROM_NISAN = [
    "Nisan","Iyar","Sivan","Tammuz","Av","Elul",
    "Tishri","Cheshvan","Kislev","Tevet","Shevat","Adar",
]

def am_year_for_tishri(tishri_jd):
    """Jewish AM year that begins with this Tishri."""
    t = ts.tt_jd(tishri_jd)
    y = t.tt_calendar()[0]
    # Tishri of astronomical year y begins AM year = y + 3761
    return y + 3761

# Determine the AM year for each month by finding which Tishri it belongs to
# after assigning names.

# First pass: assign names
calendar = []   # list of dicts

# We'll work through the sorted Nisan anchors
sorted_nisans = sorted(nisan_starts.items())   # [(astro_yr, idx), ...]

# For each Nisan anchor, label months forward and backward
# Build a mapping: month_index → name
name_map = {}

for (astro_yr, nisan_idx) in sorted_nisans:
    # Forward from Nisan: Nisan(0), Iyar(1), ..., Elul(5), Tishri(6), ...
    for i, hname in enumerate(SEQ_FROM_NISAN):
        mi = nisan_idx + i
        if 0 <= mi < len(month_starts):
            if mi not in name_map:
                name_map[mi] = {"name": hname, "nisan_yr": astro_yr}

# Check for leap-year Adar II:
# Between consecutive Nisan anchors, if there are 13 months, one is Adar II.
for k in range(len(sorted_nisans) - 1):
    yr0, idx0 = sorted_nisans[k]
    yr1, idx1 = sorted_nisans[k+1]
    n_months = idx1 - idx0
    if n_months == 13:
        # Leap year: insert Adar II between Adar (idx0+11) and Nisan (idx1)
        adar2_idx = idx0 + 12
        if 0 <= adar2_idx < len(month_starts):
            name_map[adar2_idx] = {"name": "Adar II", "nisan_yr": yr0}
            # Re-label Nisan onward (already done above, this just adds Adar II)
    elif n_months == 12:
        pass   # regular year
    else:
        # Unusual – note it
        print(f"  NOTE: {n_months} months between Nisan {yr0} and Nisan {yr1}")

# Second pass: assign AM years (Tishri starts the AM year)
# Find Tishri indices
tishri_indices = [mi for mi, v in name_map.items() if v["name"] == "Tishri"]

def am_year_for_month(mi):
    """Return AM year for month index mi."""
    # Find the most recent Tishri at or before mi
    prev_tishri = [t for t in tishri_indices if t <= mi]
    if prev_tishri:
        ti = max(prev_tishri)
        ev_jd = month_starts[ti]["evening_jd"]
        return am_year_for_tishri(ev_jd)
    else:
        # Before first Tishri in our range: use Nisan year
        nyr = name_map[mi]["nisan_yr"]
        return nyr + 3760   # Nisan of astro_yr nyr → AM nyr+3760 (approx)

# Build final calendar list
for mi in sorted(name_map.keys()):
    ms = month_starts[mi]
    ev_jd = ms["evening_jd"]
    ev_t  = ts.tt_jd(ev_jd)
    y, mo, d = ev_t.tt_calendar()[:3]
    hname = name_map[mi]["name"]
    am_yr = am_year_for_month(mi)

    calendar.append({
        "mi": mi,
        "hname": hname,
        "am_yr": am_yr,
        "evening_jd": ev_jd,
        "greg_d": int(d),
        "greg_mo": mo,
        "greg_yr": y,
        "greg_str": fmt_date(ev_t),
        "cat": ms["yallop"]["cat"],
        "q": ms["yallop"]["q"],
        "arcl": ms["yallop"]["arcl"],
        "arcv": ms["yallop"]["arcv"],
        "W": ms["yallop"]["W"],
        "moon_alt": ms["yallop"]["moon_alt"],
        "moon_age_h": ms["moon_age_h"],
        "lag_min": ms["lag_min"],
        "uncertain": ms["uncertain"],
        "note": ms["note"],
    })

# ─────────────────────────────────────────────────────────────────────────────
# Print calendar
# ─────────────────────────────────────────────────────────────────────────────

SEP = "=" * 90

print(SEP)
print("OBSERVATION-BASED HEBREW CALENDAR  ·  ~4 BC – 1 BC")
print("All dates proleptic Gregorian (evening of first crescent sighting).")
print("Jewish day begins at that sunset; Western date of same civil day is one day later.")
print(SEP)
print()

# Group by AM year
current_am = None
for row in calendar:
    # Section header for new Jewish year
    if row["am_yr"] != current_am:
        current_am = row["am_yr"]
        # Compute astronomical year from AM
        astro_yr = current_am - 3761   # Tishri of this AM year falls in astro_yr
        print()
        print(f"  ── Jewish Year AM {current_am}  "
              f"({era(astro_yr)} / {era(astro_yr+1)}) ──")
        print()
        print(f"  {'Month':<14} {'Evening of first crescent':>26}  "
              f"{'Cat':>3}  {'q':>6}  {'ARCL':>6}  {'ARCV':>6}  "
              f"{'W′':>5}  {'Age(h)':>7}  {'Lag′':>5}  Note")
        print("  " + "─"*86)

    # Error bar string
    eb = "±1d" if row["uncertain"] else "   "

    # Special day markers
    special = ""
    if row["hname"] == "Tishri":
        special = " ← Rosh Hashanah (civil new year)"
    elif row["hname"] == "Nisan":
        special = " ← 1 Nisan (religious new year)"
    elif row["hname"] == "Adar II":
        special = " ← intercalary month (leap year)"

    print(f"  1 {row['hname']:<12} {row['greg_str']:>26}  "
          f"{row['cat']:>3}  "
          f"{row['q']:>6.3f}  "
          f"{row['arcl']:>6.2f}°  "
          f"{row['arcv']:>6.2f}°  "
          f"{row['W']:>5.2f}  "
          f"{row['moon_age_h']:>7.1f}h  "
          f"{row['lag_min']:>5.0f}′  "
          f"{eb}{special}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary notes
# ─────────────────────────────────────────────────────────────────────────────

print()
print(SEP)
print("NOTES")
print(SEP)
print(f"""
Visibility criterion: Yallop (1997) NAO Technical Note 69.
  Evaluated at the moment the sun reaches –5° altitude (best crescent window).
  ARCL = topocentric moon–sun elongation.
  ARCV = moon altitude – sun altitude at that moment.
  W    = crescent width in arcminutes = SD × (1 – cos ARCL).
  q    = ARCV – (11.8371 – 6.3226√W + 0.7319W³ – 0.1018W⁴).

Intercalation rule used: 1 Nisan is the first month whose full moon
  (approximated as evening_jd + 14.75 days) falls on or after the spring
  equinox. If 13 months intervene between consecutive Nisan starts, an
  Adar II is inserted before Nisan.

Error bars:
  ±1 day   — Yallop category C (borderline crescent).  In practice the
              Sanhedrin would have received witness testimony; a ±1 day
              uncertainty means the crescent was either just detectable or
              just below the threshold.
  Firm     — Category A or B; crescent clearly visible or reliably so.

ΔT for this era: ~3.5 h (Morrison–Stephenson tables via Skyfield).
  This shifts all UTC times by ~3.5 h relative to TT but rarely moves a
  crescent sighting across an evening boundary.

Observer: Jerusalem (31.7683°N, 35.2137°E), sea level.
Ephemeris: JPL DE422.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Cross-reference: key events from star_of_bethlehem.py
# ─────────────────────────────────────────────────────────────────────────────

print(SEP)
print("KEY ASTRONOMICAL EVENTS AND THEIR HEBREW CALENDAR DATES")
print(SEP)
print()

events = [
    ("Jupiter heliacal rising",           ts.tt(-2, 7, 28)),
    ("1st Jupiter–Regulus conjunction",   ts.tt(-2, 9, 11)),
    ("2nd Jupiter–Regulus conjunction",   ts.tt(-1, 2, 16)),
    ("3rd Jupiter–Regulus conjunction",   ts.tt(-1, 5,  6)),
    ("Jupiter–Venus conjunction",         ts.tt(-1, 6, 15)),
    ("Jupiter heliacal rising",           ts.tt(-1, 8, 29)),
    ("Jupiter 1st station (retrograde)",  ts.tt(-1, 12, 25)),
]

def hebrew_date_for_jd(event_jd, calendar):
    """Find which Hebrew month/day an event falls in."""
    # Find the month whose evening_jd is <= event_jd and next month's > event_jd
    for i in range(len(calendar)-1):
        start = calendar[i]["evening_jd"]
        nxt   = calendar[i+1]["evening_jd"]
        if start <= event_jd < nxt:
            day_num = int(event_jd - start) + 1
            return calendar[i]["hname"], calendar[i]["am_yr"], day_num
    # Last month
    if calendar and event_jd >= calendar[-1]["evening_jd"]:
        day_num = int(event_jd - calendar[-1]["evening_jd"]) + 1
        return calendar[-1]["hname"], calendar[-1]["am_yr"], day_num
    return None, None, None

for label, t_event in events:
    ev_jd = t_event.tt
    hmo, am_yr, hday = hebrew_date_for_jd(ev_jd, calendar)
    greg_str = fmt_date(t_event)
    if hmo:
        print(f"  {label:<40} {greg_str}  →  "
              f"{hday} {hmo} AM {am_yr}")
    else:
        print(f"  {label:<40} {greg_str}  →  (outside computed range)")

print()
