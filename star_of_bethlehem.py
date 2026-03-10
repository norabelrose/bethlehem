#!/usr/bin/env python3
"""
Ephemeris script: Positions of Jupiter, Venus, and Regulus
observed from Babylon and Jerusalem, 1 Jan 3 BC – 1 Jan 1 BC
(proleptic Gregorian; astronomical year numbering: 1 BC = year 0)

Events detected:
  1. Jupiter–Venus closest conjunction   (~June 17, 2 BC)
  2. Jupiter–Regulus conjunctions        (three passes)
  3. Jupiter heliacal rising
  4. Jupiter stationary points           (first & second stations)
  5. Jupiter longitude table (2 BC, fixed heliacal-rising observation time)

Uses JPL DE422 ephemeris via Skyfield (vectorised for speed).
"""

import numpy as np
from skyfield.api import load, Star, wgs84, N, E
from skyfield.framelib import ecliptic_frame
from skyfield import almanac

# ---------------------------------------------------------------------------
# 0. Ephemeris & bodies
# ---------------------------------------------------------------------------
print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
eph = load("de422.bsp")
ts  = load.timescale()
print("Loaded.\n")

sun   = eph["sun"]
earth = eph["earth"]
jup   = eph["jupiter barycenter"]
ven   = eph["venus barycenter"]

# Regulus (α Leonis) J2000.0  —  proper motion negligible over this span
regulus = Star(ra_hours=(10, 8, 22.311), dec_degrees=(11, 58, 1.95))

# ---------------------------------------------------------------------------
# 1. Observer sites
# ---------------------------------------------------------------------------
babylon   = wgs84.latlon(32.5427 * N, 44.4215 * E)   # ancient Babylon
jerusalem = wgs84.latlon(31.7683 * N, 35.2137 * E)   # Jerusalem

# ---------------------------------------------------------------------------
# 2. Date helpers
# ---------------------------------------------------------------------------
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

def era(y: int) -> str:
    return f"{-y+1} BC" if y <= 0 else f"{y} AD"

def fmt(t, hhmm=False) -> str:
    y, mo, d, H, Mi, S = t.tt_calendar()
    s = f"{int(d):2d} {MONTHS[mo-1]} {era(y)}"
    if hhmm:
        fh = H + Mi/60 + S/3600
        s += f"  {fh:05.2f}h TT"
    return s

# ---------------------------------------------------------------------------
# 3. Vectorised helpers — pass whole time-array, get numpy arrays back
# ---------------------------------------------------------------------------

def ecl_lon_arr(site, body, times):
    """Ecliptic longitude array (°) for body from earth+site, apparent."""
    astr = (earth + site).at(times).observe(body).apparent()
    _, lon, _ = astr.frame_latlon(ecliptic_frame)
    return lon.degrees          # shape (n,)

def ecl_lon_star_arr(site, star, times):
    astr = (earth + site).at(times).observe(star).apparent()
    _, lon, _ = astr.frame_latlon(ecliptic_frame)
    return lon.degrees

def ecl_lat_arr(site, body, times):
    astr = (earth + site).at(times).observe(body).apparent()
    lat, _, _ = astr.frame_latlon(ecliptic_frame)
    return lat.degrees

def sep_arr(site, bodyA, bodyB, times):
    """Angular separation array (°) between two bodies as seen from earth+site."""
    aA = (earth + site).at(times).observe(bodyA).apparent()
    aB = (earth + site).at(times).observe(bodyB).apparent()
    return aA.separation_from(aB).degrees

def sep_star_arr(site, body, star, times):
    """Angular separation array (°) between body and a Star."""
    aA = (earth + site).at(times).observe(body).apparent()
    aB = (earth + site).at(times).observe(star).apparent()
    return aA.separation_from(aB).degrees

def elong_arr(site, body, times):
    """Solar elongation array (°, 0–180)."""
    ab = (earth + site).at(times).observe(body).apparent()
    as_ = (earth + site).at(times).observe(sun).apparent()
    return ab.separation_from(as_).degrees

# ---------------------------------------------------------------------------
# 4. Build daily time grid   (1 Jan 3 BC → 1 Jan 1 BC, proleptic Gregorian)
#    Astronomical years: 3 BC = -2, 2 BC = -1, 1 BC = 0
# ---------------------------------------------------------------------------
t_start = ts.tt(-2, 1, 1)
t_end   = ts.tt( 0, 1, 1)
n_days  = int(round(t_end.tt - t_start.tt)) + 1
jd_daily = np.linspace(t_start.tt, t_end.tt, n_days)
times_d  = ts.tt_jd(jd_daily)

print(f"Daily scan: {n_days} epochs  "
      f"({fmt(t_start)} – {fmt(t_end)})\n", flush=True)

# ---------------------------------------------------------------------------
# 5. Vectorised daily computation
# ---------------------------------------------------------------------------
print("Computing ecliptic longitudes & separations (vectorised)…", flush=True)

jup_lon_jer  = ecl_lon_arr(jerusalem, jup, times_d)
jup_lon_bab  = ecl_lon_arr(babylon,   jup, times_d)
ven_lon_jer  = ecl_lon_arr(jerusalem, ven, times_d)
reg_lon_jer  = ecl_lon_star_arr(jerusalem, regulus, times_d)
sun_lon_jer  = ecl_lon_arr(jerusalem, sun, times_d)

jup_elong_jer = elong_arr(jerusalem, jup, times_d)

jv_sep_jer = sep_arr(jerusalem, jup, ven, times_d)
jv_sep_bab = sep_arr(babylon,   jup, ven, times_d)

jr_sep_jer = sep_star_arr(jerusalem, jup, regulus, times_d)
jr_sep_bab = sep_star_arr(babylon,   jup, regulus, times_d)

print("Done.\n")

# Morning/evening: (jup_lon - sun_lon) mod 360; >180 → morning (west of sun)
delta_lon = (jup_lon_jer - sun_lon_jer) % 360
morning_jer = delta_lon > 180       # boolean array


# ===========================================================================
# EVENTS
# ===========================================================================
SEP = "=" * 72

# ---------------------------------------------------------------------------
# Event 1: Jupiter–Venus closest conjunction
# ---------------------------------------------------------------------------
print(SEP)
print("EVENT 1: JUPITER–VENUS CLOSEST CONJUNCTION")
print(SEP)

min_idx = int(np.argmin(jv_sep_jer))

# Zoom ±4 days, 1-minute resolution
z0 = jd_daily[max(0, min_idx-4)]
z1 = jd_daily[min(n_days-1, min_idx+4)]
jd_z = np.linspace(z0, z1, 11520)       # 8 days × 1440 min/day
tz   = ts.tt_jd(jd_z)

zv_jer = sep_arr(jerusalem, jup, ven, tz)
zv_bab = sep_arr(babylon,   jup, ven, tz)

mi_jer = int(np.argmin(zv_jer))
mi_bab = int(np.argmin(zv_bab))

t_jv_jer = tz[mi_jer]
t_jv_bab = tz[mi_bab]

# Detailed position at conjunction
jup_lon_c = ecl_lon_arr(jerusalem, jup, ts.tt_jd(np.array([jd_z[mi_jer]])))[0]
ven_lon_c = ecl_lon_arr(jerusalem, ven, ts.tt_jd(np.array([jd_z[mi_jer]])))[0]
jup_lat_c = ecl_lat_arr(jerusalem, jup, ts.tt_jd(np.array([jd_z[mi_jer]])))[0]
ven_lat_c = ecl_lat_arr(jerusalem, ven, ts.tt_jd(np.array([jd_z[mi_jer]])))[0]
elong_c   = elong_arr(jerusalem, jup, ts.tt_jd(np.array([jd_z[mi_jer]])))[0]

def _lst_str(t, lon_deg):
    """Local mean solar time string (HH:MM) from a Skyfield Time and east longitude."""
    h = ((t.ut1 + 0.5) % 1.0 * 24.0 + lon_deg / 15.0) % 24.0
    return f"{int(h):02d}:{int((h % 1) * 60):02d}"

_app_jup_jer = (earth + jerusalem).at(t_jv_jer).observe(jup).apparent()
_app_ven_jer = (earth + jerusalem).at(t_jv_jer).observe(ven).apparent()
_alt_jup_jer, _az_jup_jer, _ = _app_jup_jer.altaz(temperature_C=20, pressure_mbar=1013)
_alt_ven_jer, _az_ven_jer, _ = _app_ven_jer.altaz(temperature_C=20, pressure_mbar=1013)

_app_jup_bab = (earth + babylon).at(t_jv_bab).observe(jup).apparent()
_app_ven_bab = (earth + babylon).at(t_jv_bab).observe(ven).apparent()
_alt_jup_bab, _az_jup_bab, _ = _app_jup_bab.altaz(temperature_C=20, pressure_mbar=1013)
_alt_ven_bab, _az_ven_bab, _ = _app_ven_bab.altaz(temperature_C=20, pressure_mbar=1013)

# 1-arcminute window: bisect the entry and exit threshold crossings
_1M   = 1.0 / 60.0   # 1 arcminute in degrees
_JV_WINDOW = 6 * _1M    # 6 arcminutes = 0.1° window for "close" conjunctions

def _bisect_sep_crossing(site, bodyA, bodyB, jd_lo, jd_hi, entering, threshold=_1M):
    """Bisect to ~6-second precision the moment sep crosses threshold.
    entering=True: sep goes from above to below threshold (entry).
    entering=False: sep goes from below to above threshold (exit).
    """
    lo, hi = jd_lo, jd_hi
    for _ in range(50):
        mid = (lo + hi) / 2
        s = sep_arr(site, bodyA, bodyB, ts.tt_jd(mid))
        if (s > threshold) == entering:
            lo = mid
        else:
            hi = mid
        if (hi - lo) * 1440 < 0.1:
            break
    return ts.tt_jd((lo + hi) / 2)

def _1m_window(site, zv):
    """Return (t_entry, t_exit, dur_min) for the <1' window, or None if none."""
    below = zv < _1M
    if not np.any(below):
        return None
    i_en = int(np.argmax(below))
    i_ex = len(below) - 1 - int(np.argmax(below[::-1]))
    t_en = _bisect_sep_crossing(site, jup, ven,
                                 jd_z[max(0, i_en - 1)], jd_z[i_en],
                                 entering=True)
    t_ex = _bisect_sep_crossing(site, jup, ven,
                                 jd_z[i_ex], jd_z[min(len(jd_z) - 1, i_ex + 1)],
                                 entering=False)
    return t_en, t_ex, (t_ex.tt - t_en.tt) * 1440

def _deg_window(site, jv_sep_daily):
    """Return (t_entry, t_exit, dur_days) for the <0.5° window using daily array."""
    below = jv_sep_daily < _JV_WINDOW
    if not np.any(below):
        return None
    i_en = int(np.argmax(below))
    i_ex = len(below) - 1 - int(np.argmax(below[::-1]))
    t_en = _bisect_sep_crossing(site, jup, ven,
                                 jd_daily[max(0, i_en - 1)], jd_daily[i_en],
                                 entering=True, threshold=_JV_WINDOW)
    t_ex = _bisect_sep_crossing(site, jup, ven,
                                 jd_daily[i_ex], jd_daily[min(n_days - 1, i_ex + 1)],
                                 entering=False, threshold=_JV_WINDOW)
    return t_en, t_ex, t_ex.tt - t_en.tt

_w_jer = _1m_window(jerusalem, zv_jer)
_w_bab = _1m_window(babylon,   zv_bab)
_d_jer = _deg_window(jerusalem, jv_sep_jer)
_d_bab = _deg_window(babylon,   jv_sep_bab)

print()
print("  From JERUSALEM")
print(f"    Closest approach : {fmt(t_jv_jer, hhmm=True)}")
print(f"    Local solar time : {_lst_str(t_jv_jer, 35.2137)}")
print(f"    Separation       : {zv_jer[mi_jer]*60:.3f}′  "
      f"({zv_jer[mi_jer]:.5f}°)")
print(f"    Jupiter ecl lon  : {jup_lon_c:.3f}°   lat: {jup_lat_c:+.3f}°")
print(f"    Venus   ecl lon  : {ven_lon_c:.3f}°   lat: {ven_lat_c:+.3f}°")
print(f"    Jupiter elong    : {elong_c:.2f}° from Sun")
print(f"    Altitude at conjunction:")
print(f"      Jupiter : alt {_alt_jup_jer.degrees:+6.2f}°   az {_az_jup_jer.degrees:6.2f}°")
print(f"      Venus   : alt {_alt_ven_jer.degrees:+6.2f}°   az {_az_ven_jer.degrees:6.2f}°")
if _d_jer:
    print(f"    Within {_JV_WINDOW}° of separation:")
    print(f"      Enter : {fmt(_d_jer[0], hhmm=True)}  (local {_lst_str(_d_jer[0], 35.2137)})")
    print(f"      Leave : {fmt(_d_jer[1], hhmm=True)}  (local {_lst_str(_d_jer[1], 35.2137)})")
    print(f"      Duration: {_d_jer[2]:.2f} days")
if _w_jer:
    print(f"    Within 1′ of separation:")
    print(f"      Enter : {fmt(_w_jer[0], hhmm=True)}  (local {_lst_str(_w_jer[0], 35.2137)})")
    print(f"      Leave : {fmt(_w_jer[1], hhmm=True)}  (local {_lst_str(_w_jer[1], 35.2137)})")
    print(f"      Duration: {_w_jer[2]:.1f} min")
print()
print("  From BABYLON")
print(f"    Closest approach : {fmt(t_jv_bab, hhmm=True)}")
print(f"    Local solar time : {_lst_str(t_jv_bab, 44.4215)}")
print(f"    Separation       : {zv_bab[mi_bab]*60:.3f}′  "
      f"({zv_bab[mi_bab]:.5f}°)")
print(f"    Altitude at conjunction:")
print(f"      Jupiter : alt {_alt_jup_bab.degrees:+6.2f}°   az {_az_jup_bab.degrees:6.2f}°")
print(f"      Venus   : alt {_alt_ven_bab.degrees:+6.2f}°   az {_az_ven_bab.degrees:6.2f}°")
if _d_bab:
    print(f"    Within {_JV_WINDOW}° of separation:")
    print(f"      Enter : {fmt(_d_bab[0], hhmm=True)}  (local {_lst_str(_d_bab[0], 44.4215)})")
    print(f"      Leave : {fmt(_d_bab[1], hhmm=True)}  (local {_lst_str(_d_bab[1], 44.4215)})")
    print(f"      Duration: {_d_bab[2]:.2f} days")
if _w_bab:
    print(f"    Within 1′ of separation:")
    print(f"      Enter : {fmt(_w_bab[0], hhmm=True)}  (local {_lst_str(_w_bab[0], 44.4215)})")
    print(f"      Leave : {fmt(_w_bab[1], hhmm=True)}  (local {_lst_str(_w_bab[1], 44.4215)})")
    print(f"      Duration: {_w_bab[2]:.1f} min")
print()

# ---------------------------------------------------------------------------
# Event 2: Jupiter–Regulus conjunctions
# ---------------------------------------------------------------------------
print(SEP)
print("EVENT 2: JUPITER–REGULUS CONJUNCTIONS")
print(SEP)

# Local minima in separation < 5°
jr_minima = [i for i in range(1, n_days-1)
             if jr_sep_jer[i] < jr_sep_jer[i-1]
             and jr_sep_jer[i] < jr_sep_jer[i+1]
             and jr_sep_jer[i] < 5.0]

for idx in jr_minima:
    z0 = jd_daily[max(0, idx-6)]
    z1 = jd_daily[min(n_days-1, idx+6)]
    jd_z2 = np.linspace(z0, z1, 17280)   # 12 days × 1440
    tz2 = ts.tt_jd(jd_z2)

    zr_jer = sep_star_arr(jerusalem, jup, regulus, tz2)
    zr_bab = sep_star_arr(babylon,   jup, regulus, tz2)

    mi2_jer = int(np.argmin(zr_jer))
    mi2_bab = int(np.argmin(zr_bab))

    # Ecliptic lat/lon detail
    jl  = ecl_lon_arr(jerusalem, jup, ts.tt_jd(np.array([jd_z2[mi2_jer]])))[0]
    jla = ecl_lat_arr(jerusalem, jup, ts.tt_jd(np.array([jd_z2[mi2_jer]])))[0]
    rl  = ecl_lon_star_arr(jerusalem, regulus, ts.tt_jd(np.array([jd_z2[mi2_jer]])))[0]

    # Is Jupiter currently in retrograde? Check 2-day velocity
    if idx > 2 and idx < n_days-2:
        dj = (jup_lon_jer[idx+2] - jup_lon_jer[idx-2] + 360) % 360
        if dj > 180: dj -= 360
        motion_str = "retrograde" if dj < 0 else "direct"
    else:
        motion_str = "—"

    ns = "N" if jla > 0 else "S"   # Regulus is near ecliptic; Jupiter lat sign tells it

    print()
    print(f"  Conjunction near {fmt(times_d[idx])}:")
    print(f"    From JERUSALEM : {fmt(tz2[mi2_jer], hhmm=True)}")
    print(f"      Separation   : {zr_jer[mi2_jer]*60:.2f}′  ({motion_str})")
    print(f"      Jupiter lon  : {jl:.3f}°   lat: {jla:+.3f}°  (Regulus lon: {rl:.3f}°)")
    print(f"    From BABYLON   : {fmt(tz2[mi2_bab], hhmm=True)}")
    print(f"      Separation   : {zr_bab[mi2_bab]*60:.2f}′")
print()

# ---------------------------------------------------------------------------
# Event 3: Jupiter stationary points
# ---------------------------------------------------------------------------
print(SEP)
print("EVENT 3: JUPITER STATIONARY POINTS")
print(SEP)

# Unwrap longitude to detect sign changes in daily motion
jup_lon_unw = np.degrees(np.unwrap(np.radians(jup_lon_jer)))
dlon_day    = np.diff(jup_lon_unw)     # deg/day

stations = []
for i in range(1, len(dlon_day)):
    if dlon_day[i-1] * dlon_day[i] < 0:
        kind = ("FIRST STATION → retrograde begins"
                if dlon_day[i] < 0
                else "SECOND STATION → direct motion resumes")
        stations.append((i, kind))

for idx, kind in stations:
    z0 = jd_daily[max(0, idx-5)]
    z1 = jd_daily[min(n_days-1, idx+5)]
    jd_z3 = np.linspace(z0, z1, 14400)    # 10 days × 1440
    tz3 = ts.tt_jd(jd_z3)

    z_lon3 = ecl_lon_arr(jerusalem, jup, tz3)
    z_lon3_unw = np.degrees(np.unwrap(np.radians(z_lon3)))
    z_dlon3 = np.diff(z_lon3_unw)

    # Find zero crossing
    stat_i = None
    for j in range(1, len(z_dlon3)):
        if z_dlon3[j-1] * z_dlon3[j] < 0:
            stat_i = j
            break
    if stat_i is None:
        stat_i = int(np.argmin(np.abs(z_dlon3)))

    t_st = tz3[stat_i]
    lon_st = z_lon3[stat_i]
    el_st  = elong_arr(jerusalem, jup, ts.tt_jd(np.array([jd_z3[stat_i]])))[0]
    dl_st  = (jup_lon_jer[idx] - sun_lon_jer[idx]) % 360
    sky    = "morning sky (W of Sun)" if dl_st > 180 else "evening sky (E of Sun)"

    print()
    print(f"  {kind}")
    print(f"    Date (Jerusalem) : {fmt(t_st, hhmm=True)}")
    print(f"    Jupiter ecl lon  : {lon_st:.3f}°")
    print(f"    Solar elongation : {el_st:.2f}°  ({sky})")
    print(f"    Daily lon motion at station: ~{z_dlon3[stat_i]*60:.3f}′/day")
print()

# ---------------------------------------------------------------------------
# Event 4: Jupiter heliacal rising(s)
# ---------------------------------------------------------------------------
print(SEP)
print("EVENT 4: JUPITER HELIACAL RISING")
print(SEP)

AV = 11.0   # arcus visionis for Jupiter (degrees)

# Detect rising: elongation crosses AV upward while planet is in morning sky
heliacal = []
for i in range(1, n_days):
    if (morning_jer[i]
            and jup_elong_jer[i-1] < AV
            and jup_elong_jer[i]   >= AV):
        heliacal.append(i)

if not heliacal:
    print("  No heliacal risings found with AV = 11°. "
          "(Possibly Jupiter didn't have a solar conjunction in range.)")
else:
    for idx in heliacal:
        z0 = jd_daily[max(0, idx-1)]
        z1 = jd_daily[min(n_days-1, idx+1)]
        jd_z4 = np.linspace(z0, z1, 2880)
        tz4 = ts.tt_jd(jd_z4)

        el4 = elong_arr(jerusalem, jup, tz4)
        dl4 = (ecl_lon_arr(jerusalem, jup, tz4)
               - ecl_lon_arr(jerusalem, sun, tz4)) % 360
        morning4 = dl4 > 180

        hr_i = None
        for j in range(len(tz4)):
            if morning4[j] and el4[j] >= AV:
                hr_i = j
                break

        t_hr = tz4[hr_i] if hr_i is not None else times_d[idx]
        el_hr = el4[hr_i] if hr_i is not None else jup_elong_jer[idx]
        lon_hr = ecl_lon_arr(jerusalem, jup, ts.tt_jd(
                    np.array([t_hr.tt])))[0]

        print()
        print(f"  Heliacal Rising — Jerusalem")
        print(f"    Date             : {fmt(t_hr, hhmm=True)}")
        print(f"    Ecliptic lon     : {lon_hr:.3f}°")
        print(f"    Solar elongation : {el_hr:.2f}°  (threshold {AV}°)")
print()

# ---------------------------------------------------------------------------
# Event 5: Jupiter altitude & azimuth — 2 BC, weekly at fixed 04:38 local time
# ---------------------------------------------------------------------------
print(SEP)
print("EVENT 5: JUPITER ALTITUDE & AZIMUTH — 2 BC, WEEKLY AT 04:38 LOCAL")
print("Observer : Jerusalem")
print("Time     : fixed at 04:38 Jerusalem local mean solar time")
print("           (= time of Jupiter's heliacal rise on 29 Aug 2 BC)")
print(SEP)

_COMPASS = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW']

def _compass_pt(az_deg):
    return _COMPASS[round(az_deg / 22.5) % 16]

_JER_LON = 35.2137   # degrees east

# Find the Jupiter rise on 29 Aug 2 BC to anchor the fixed time
_f_rise = almanac.risings_and_settings(eph, jup, jerusalem)
_all_t, _all_ev = almanac.find_discrete(ts.tt(-1, 8, 29), ts.tt(-1, 8, 30), _f_rise)

_aug29_rise_ut1 = None
for _t, _ev in zip(_all_t, _all_ev):
    if _ev == 1:
        _aug29_rise_ut1 = _t.ut1
        break

if _aug29_rise_ut1 is None:
    print("  ERROR: Could not locate Jupiter rise near 29 Aug 2 BC.\n")
else:
    # Local time sanity check
    _lf = (_aug29_rise_ut1 + 0.5 + _JER_LON / (15.0 * 24.0)) % 1.0
    _lh = _lf * 24.0
    print(f"  Anchored to rise on 29 Aug 2 BC at "
          f"{int(_lh):02d}:{int((_lh % 1)*60):02d} local\n")

    # Weekly UT1 times: same time-of-day, 7-day steps through end of 2 BC
    _end_ut1 = ts.tt(0, 1, 1).ut1
    _n_weeks = int((_end_ut1 - _aug29_rise_ut1) / 7) + 2
    _obs_ut1 = np.array([_aug29_rise_ut1 + n * 7 for n in range(_n_weeks)])
    _obs_ut1 = _obs_ut1[_obs_ut1 <= _end_ut1]
    _obs_times5 = ts.ut1_jd(_obs_ut1)

    # Altitude & azimuth from Jerusalem
    _app5 = (earth + jerusalem).at(_obs_times5).observe(jup).apparent()
    _alt5, _az5, _ = _app5.altaz(temperature_C=20, pressure_mbar=1013)
    _alts5 = _alt5.degrees
    _azs5  = _az5.degrees

    hdr5 = f"  {'Date':<20}  {'Altitude':>9}  {'Azimuth':>9}  Direction"
    print(hdr5)
    print("  " + "—" * (len(hdr5) - 2))

    for i in range(len(_obs_times5)):
        print(f"  {fmt(_obs_times5[i]):<20}  {_alts5[i]:8.2f}°  "
              f"{_azs5[i]:8.2f}°  {_compass_pt(_azs5[i])}")

print()

# ---------------------------------------------------------------------------
# Event 6: Jupiter alt/az — weekly, fixed interval before sunrise
#          Interval = (sunrise on 29 Aug 2 BC) − (04:38 local on that day)
# ---------------------------------------------------------------------------
print(SEP)
print("EVENT 6: JUPITER ALT/AZ — WEEKLY, FIXED INTERVAL BEFORE SUNRISE")
print("Observer : Jerusalem")
print("Anchor   : interval between 04:38 local and sunrise on 29 Aug 2 BC,")
print("           applied before each week's sunrise thereafter")
print(SEP)

# Find all sunrises from 29 Aug 2 BC through 1 Jan 1 BC
_f_sr6 = almanac.risings_and_settings(eph, sun, jerusalem)
_sr6_t, _sr6_ev = almanac.find_discrete(
    ts.tt(-1, 8, 28), ts.tt(0, 1, 2), _f_sr6
)
_sunrises6 = np.array([
    _t.ut1 for _t, _ev in zip(_sr6_t, _sr6_ev) if _ev == 1
])  # UT1 JDs of every sunrise

# Locate 29 Aug 2 BC sunrise
_s6_start = ts.tt(-1, 8, 29).ut1
_s6_end   = ts.tt(-1, 8, 30).ut1
_aug29_sr6 = next(
    (s for s in _sunrises6 if _s6_start <= s < _s6_end), None
)

if _aug29_sr6 is None:
    print("  ERROR: Could not find sunrise on 29 Aug 2 BC.\n")
else:
    def _local_h(jd_ut1):
        """UT1 JD → local mean solar time in hours (Jerusalem)."""
        return (((jd_ut1 + 0.5) % 1.0) * 24.0 + _JER_LON / 15.0) % 24.0

    # Sunrise and 04:38 in local hours on 29 Aug
    _sr6_local   = _local_h(_aug29_sr6)
    _anchor_local = 4 + 38 / 60.0          # 04:38
    _offset6_days = (_sr6_local - _anchor_local) / 24.0  # positive = before sunrise

    print(f"  Sunrise on 29 Aug 2 BC  : "
          f"{int(_sr6_local):02d}:{int((_sr6_local % 1)*60):02d} local")
    print(f"  Observation time anchor : 04:38 local")
    print(f"  Fixed interval          : {_offset6_days * 24 * 60:.1f} min before sunrise\n")

    # Build weekly observation times
    _w_end_ut1 = ts.tt(0, 1, 1).ut1
    _n_weeks6  = int((_w_end_ut1 - _s6_start) / 7) + 2

    _week_obs_ut1 = []
    _week_sr_ut1  = []
    for _w in range(_n_weeks6):
        _wjd = _s6_start + _w * 7
        if _wjd > _w_end_ut1:
            break
        # Nearest sunrise at or after the weekly mark
        _idx = int(np.searchsorted(_sunrises6, _wjd))
        if _idx >= len(_sunrises6):
            break
        _sr = _sunrises6[_idx]
        _week_sr_ut1.append(_sr)
        _week_obs_ut1.append(_sr - _offset6_days)

    _obs6_times = ts.ut1_jd(np.array(_week_obs_ut1))

    # Jupiter alt/az
    _app6 = (earth + jerusalem).at(_obs6_times).observe(jup).apparent()
    _alt6, _az6, _ = _app6.altaz(temperature_C=20, pressure_mbar=1013)
    _alts6 = _alt6.degrees
    _azs6  = _az6.degrees

    hdr6 = (f"  {'Date':<20}  {'Obs (local)':>11}  "
            f"{'Sunrise':>8}  {'Altitude':>9}  {'Azimuth':>9}  Direction")
    print(hdr6)
    print("  " + "—" * (len(hdr6) - 2))

    for i in range(len(_obs6_times)):
        _obs_lh = _local_h(_week_obs_ut1[i])
        _sr_lh  = _local_h(_week_sr_ut1[i])
        print(f"  {fmt(_obs6_times[i]):<20}  "
              f"{int(_obs_lh):02d}:{int((_obs_lh % 1)*60):02d}        "
              f"{int(_sr_lh):02d}:{int((_sr_lh % 1)*60):02d}   "
              f"{_alts6[i]:8.2f}°  "
              f"{_azs6[i]:8.2f}°  {_compass_pt(_azs6[i])}")

print()

# ---------------------------------------------------------------------------
# 7. Formatted position table (every 10 days)
# ---------------------------------------------------------------------------
print(SEP)
print("POSITION TABLE — every 10 days, Jerusalem observer")
print("Ecl. lon in degrees (J2000 ecliptic).  Sep columns in arcminutes.")
print(SEP)

hdr = (f"{'Date':<26}  {'J lon':>7}  {'V lon':>7}  {'R lon':>7}  "
       f"{'J elong':>8}  {'J–V':>7}  {'J–R':>7}  {'Motion':<8}")
print(hdr)
print("—" * len(hdr))

for i in range(0, n_days, 10):
    t = times_d[i]
    if i + 5 < n_days:
        dl = (jup_lon_jer[i+5] - jup_lon_jer[i] + 360) % 360
        if dl > 180: dl -= 360
        mot = "Retro" if dl < -0.02 else "Direct"
    else:
        mot = "—"

    print(f"  {fmt(t):<24}  "
          f"{jup_lon_jer[i]:7.2f}  "
          f"{ven_lon_jer[i]:7.2f}  "
          f"{reg_lon_jer[i]:7.2f}  "
          f"{jup_elong_jer[i]:8.2f}°  "
          f"{jv_sep_jer[i]*60:7.1f}′  "
          f"{jr_sep_jer[i]*60:7.1f}′  "
          f"{mot}")

# ---------------------------------------------------------------------------
# 7. Final notes
# ---------------------------------------------------------------------------
print()
print(SEP)
print("NOTES")
print(SEP)
print("""
  All dates proleptic Gregorian.  Astronomical year numbering:
    3 BC = year -2,  2 BC = year -1,  1 BC = year 0.

  Positions are geocentric apparent (includes aberration, refraction ignored).
  Ecliptic frame: J2000 mean ecliptic and equinox.
  Jupiter arcus visionis threshold: 11°.
  DE422 ephemeris, ΔT from Skyfield built-in tables.

  Observer coordinates:
    Babylon    32.54°N  44.42°E
    Jerusalem  31.77°N  35.21°E
""")
