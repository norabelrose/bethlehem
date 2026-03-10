#!/usr/bin/env python3
"""
jv_sweep.py

Sweep the full DE422 ephemeris (~3000 BC – 3000 AD) for every
Jupiter–Venus conjunction whose closest approach as seen from Jerusalem
is < 1 arcminute and which is visible from Jerusalem.

Visibility criteria (both must hold):
  1. Solar elongation of Jupiter > 11° at closest approach
     (Jupiter's arcus visionis — same threshold as heliacal risings)
  2. The < 1′ window overlaps with nighttime from Jerusalem
     (Sun below −6°, i.e. astronomical twilight or darker)

Algorithm — three phases:
  Phase 1  Daily geocentric scan — finds candidate windows (sep < 1.5°)
  Phase 2  1-minute topocentric (Jerusalem) scan within each window
  Phase 3  Bisection for precise entry/exit of < 1′ interval,
           elongation + night-overlap check, then print

Results are printed immediately as each qualifying event is found.
"""

import numpy as np
from skyfield.api import load, wgs84, N, E
from skyfield.framelib import ecliptic_frame

# ── Load ephemeris ────────────────────────────────────────────────────────────
print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
eph = load("de422.bsp")
ts  = load.timescale()
print("Loaded.\n", flush=True)

sun   = eph["sun"]
earth = eph["earth"]
jup   = eph["jupiter barycenter"]
ven   = eph["venus barycenter"]

jerusalem = wgs84.latlon(31.7683 * N, 35.2137 * E)
JER_LON   = 35.2137   # degrees east

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
    """Local mean solar time at Jerusalem (HH:MM)."""
    h = ((t.ut1 + 0.5) % 1.0 * 24.0 + JER_LON / 15.0) % 24.0
    return f"{int(h):02d}:{int((h % 1) * 60):02d}"

# ── Vectorised position helpers ───────────────────────────────────────────────
def geo_sep(jd_arr: np.ndarray) -> np.ndarray:
    """Geocentric Jupiter–Venus angular separation (°). Used for coarse scan."""
    t  = ts.tt_jd(jd_arr)
    aJ = earth.at(t).observe(jup).apparent()
    aV = earth.at(t).observe(ven).apparent()
    return aJ.separation_from(aV).degrees

def jer_sep(jd_arr: np.ndarray) -> np.ndarray:
    """Topocentric (Jerusalem) Jupiter–Venus angular separation (°)."""
    t   = ts.tt_jd(jd_arr)
    obs = earth + jerusalem
    aJ  = obs.at(t).observe(jup).apparent()
    aV  = obs.at(t).observe(ven).apparent()
    return aJ.separation_from(aV).degrees

def jer_sun_alt(jd_arr: np.ndarray) -> np.ndarray:
    """Altitude of Sun from Jerusalem (°)."""
    t   = ts.tt_jd(jd_arr)
    obs = earth + jerusalem
    aS  = obs.at(t).observe(sun).apparent()
    alt, _, _ = aS.altaz()
    return alt.degrees

def jer_jup_alt(jd_arr: np.ndarray) -> np.ndarray:
    """Altitude of Jupiter from Jerusalem (°)."""
    t   = ts.tt_jd(jd_arr)
    obs = earth + jerusalem
    aJ  = obs.at(t).observe(jup).apparent()
    alt, _, _ = aJ.altaz()
    return alt.degrees

# ── Bisection for threshold crossing ─────────────────────────────────────────
ARCMIN = 1.0 / 60.0   # 1 arcminute in degrees

def bisect_sep(jd_lo: float, jd_hi: float,
               entering: bool, threshold: float = ARCMIN) -> float:
    """Bisect to ~6-second precision the moment jer_sep crosses threshold.
    entering=True  → sep going from above to below threshold.
    entering=False → sep going from below to above threshold.
    Returns JD (TT) of crossing.
    """
    lo, hi = jd_lo, jd_hi
    for _ in range(50):
        mid = (lo + hi) / 2.0
        s   = jer_sep(np.array([mid]))[0]
        if (s > threshold) == entering:
            lo = mid
        else:
            hi = mid
        if (hi - lo) * 86400 < 6:    # 6-second precision
            break
    return (lo + hi) / 2.0

# ── Scan parameters ───────────────────────────────────────────────────────────
JD_START = ts.tt(-3000, 11, 14).tt # inside DE422 lower bound
JD_END   = ts.tt( 3000, 1, 1).tt   # inside DE422 upper bound

THRESH_COARSE = 1.5     # deg — geocentric daily trigger
THRESH_FINE   = ARCMIN  # deg — 1 arcminute
AV_JUP        = 11.0    # deg — Jupiter arcus visionis
NIGHT_ALT     = -6.0    # deg — Sun must reach this to count as night

total_days = int(JD_END - JD_START)
print(f"Scan range : {era(-3000)} to {era(3000)}  (~{total_days:,} days)")
print(f"Target     : closest approach < 1′ as seen from Jerusalem")
print(f"Visibility : elongation > {AV_JUP}°  AND  night overlap (sun < {NIGHT_ALT}°)\n",
      flush=True)

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
starts = np.where(diff ==  1)[0]   # first day below threshold
ends   = np.where(diff == -1)[0]   # first day back above threshold

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
    s_f   = jer_sep(jd_f)
    mi    = int(np.argmin(s_f))

    if s_f[mi] >= THRESH_FINE:
        continue   # minimum doesn't reach 1′

    # Find first/last indices where sep < 1′ in the 1-minute array
    below = s_f < THRESH_FINE
    i_en  = int(np.argmax(below))                             # first True
    i_ex  = len(below) - 1 - int(np.argmax(below[::-1]))     # last True

    # Bisect precise entry and exit (need one point above on each side)
    if i_en > 0:
        jd_en = bisect_sep(jd_f[i_en - 1], jd_f[i_en], entering=True)
    else:
        jd_en = jd_f[0]   # window clipped; buffer prevents this in practice

    if i_ex < n_pts - 1:
        jd_ex = bisect_sep(jd_f[i_ex], jd_f[i_ex + 1], entering=False)
    else:
        jd_ex = jd_f[-1]

    dur_min = (jd_ex - jd_en) * 1440.0
    jd_c    = jd_f[mi]    # JD of closest approach (1-minute precision)
    sep_c   = s_f[mi]

    # ── Elongation check (visibility criterion 1) ─────────────────────────
    t1  = ts.tt_jd(np.array([jd_c]))
    obs = earth + jerusalem
    aJ  = obs.at(t1).observe(jup).apparent()
    aS  = obs.at(t1).observe(sun).apparent()
    elong = float(aJ.separation_from(aS).degrees[0])

    if elong < AV_JUP:
        continue   # too close to sun

    # ── Night-overlap check (visibility criterion 2) ──────────────────────
    # Sample altitudes at 30-min intervals across the < 1′ window.
    # Require at least one moment where sun < −6° AND Jupiter > 0°.
    n_samp  = max(int(dur_min / 30) + 2, 4)
    jd_samp = np.linspace(jd_en, jd_ex, n_samp)
    sun_alt = jer_sun_alt(jd_samp)
    jup_alt = jer_jup_alt(jd_samp)

    if not np.any((sun_alt < NIGHT_ALT) & (jup_alt > 0.0)):
        continue   # no moment during the < 1′ window is dark AND above horizon

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
    print(f"    Altitude at closest approach (Jerusalem):")
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
print(f"TOTAL Jupiter–Venus conjunctions < 1′ visible from Jerusalem : {found}")
print(f"  Range      : {era(-3000)} to {era(3000)}"
      f"  (~{total_days // 365} years, DE422 ephemeris)")
print(f"  Visibility : elongation > {AV_JUP}°,"
      f" night overlap with sun < {NIGHT_ALT}°")
print()
