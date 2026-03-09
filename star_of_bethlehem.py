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

Uses JPL DE422 ephemeris via Skyfield (vectorised for speed).
"""

import numpy as np
from skyfield.api import load, Star, wgs84, N, E
from skyfield.framelib import ecliptic_frame

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
print(f"  Daily minimum on {fmt(times_d[min_idx])}:  "
      f"{jv_sep_jer[min_idx]*60:.1f}′ (Jerusalem)")

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

print()
print("  From JERUSALEM")
print(f"    Closest approach : {fmt(t_jv_jer, hhmm=True)}")
print(f"    Separation       : {zv_jer[mi_jer]*60:.3f}′  "
      f"({zv_jer[mi_jer]:.5f}°)")
print(f"    Jupiter ecl lon  : {jup_lon_c:.3f}°   lat: {jup_lat_c:+.3f}°")
print(f"    Venus   ecl lon  : {ven_lon_c:.3f}°   lat: {ven_lat_c:+.3f}°")
print(f"    Jupiter elong    : {elong_c:.2f}° from Sun")
print()
print("  From BABYLON")
print(f"    Closest approach : {fmt(t_jv_bab, hhmm=True)}")
print(f"    Separation       : {zv_bab[mi_bab]*60:.3f}′  "
      f"({zv_bab[mi_bab]:.5f}°)")
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
# 6. Formatted position table (every 10 days)
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
