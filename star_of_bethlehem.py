#!/usr/bin/env python3
"""
Ephemeris script: Positions of Jupiter, Venus, and Regulus
observed from Babylon and Jerusalem, 1 Jan 3 BC – 1 Jan 1 BC
(proleptic Gregorian; astronomical year numbering: 1 BC = year 0)

Events detected:
  1. Jupiter–Venus closest conjunction   (~June 17, 2 BC)
  2. Jupiter–Regulus conjunctions        (three passes)
  3. Moon & Jupiter at Regulus           (around 2nd J-R conjunction)
  4. Jupiter stationary points           (first & second stations)
  5. Jupiter heliacal rising
  6. Jupiter alt/az — weekly, 60 min before sunrise
  7. Total lunar eclipse — 1 BC January
  8. Moon–Regulus closest visible approach after 3rd J-R conjunction

Uses JPL DE422 ephemeris via Skyfield (vectorised for speed).
"""

import argparse
from dataclasses import dataclass

import numpy as np
from skyfield import almanac
from skyfield.api import E, N, Star, load, wgs84
from skyfield.framelib import ecliptic_frame

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
SEP = "=" * 72
_1M = 1.0 / 60.0       # 1 arcminute in degrees
_JV_WINDOW = 6 * _1M   # 6 arcminutes threshold for extended window reporting
_JER_LON = 35.2137     # Jerusalem east longitude, degrees
_COMPASS = ["N","NNE","NE","ENE","E","ESE","SE","SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]


def era(y: int) -> str:
    return f"{-y+1} BC" if y <= 0 else f"{y} AD"


def fmt(t, hhmm=False) -> str:
    y, mo, d, H, Mi, S = t.tt_calendar()
    s = f"{int(d):2d} {MONTHS[mo-1]} {era(y)}"
    if hhmm:
        fh = H + Mi / 60 + S / 3600
        s += f"  {fh:05.2f}h TT"
    return s


def _lst_str(t, lon_deg):
    """Local mean solar time string (HH:MM) from a Skyfield Time and east longitude."""
    h = ((t.ut1 + 0.5) % 1.0 * 24.0 + lon_deg / 15.0) % 24.0
    return f"{int(h):02d}:{int((h % 1) * 60):02d}"


def _compass_pt(az_deg):
    return _COMPASS[round(az_deg / 22.5) % 16]


def _local_h(jd_ut1):
    """UT1 JD → local mean solar time in hours (Jerusalem)."""
    return (((jd_ut1 + 0.5) % 1.0) * 24.0 + _JER_LON / 15.0) % 24.0


# ---------------------------------------------------------------------------
# Shared context
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    ts: object
    eph: object
    earth: object
    sun: object
    jup: object
    ven: object
    moon: object
    regulus: object
    babylon: object
    jerusalem: object
    jd_daily: np.ndarray
    n_days: int
    times_d: object
    jup_lon_jer: np.ndarray
    sun_lon_jer: np.ndarray
    jup_elong_jer: np.ndarray
    morning_jer: np.ndarray


# ---------------------------------------------------------------------------
# Vectorised helpers
# ---------------------------------------------------------------------------

def ecl_lon_arr(ctx, site, body, times):
    """Ecliptic longitude array (°) for body from earth+site, apparent."""
    astr = (ctx.earth + site).at(times).observe(body).apparent()
    _, lon, _ = astr.frame_latlon(ecliptic_frame)
    return lon.degrees


def ecl_lon_star_arr(ctx, site, star, times):
    astr = (ctx.earth + site).at(times).observe(star).apparent()
    _, lon, _ = astr.frame_latlon(ecliptic_frame)
    return lon.degrees


def ecl_lat_arr(ctx, site, body, times):
    astr = (ctx.earth + site).at(times).observe(body).apparent()
    lat, _, _ = astr.frame_latlon(ecliptic_frame)
    return lat.degrees


def sep_arr(ctx, site, bodyA, bodyB, times):
    """Angular separation array (°) between two bodies as seen from earth+site."""
    aA = (ctx.earth + site).at(times).observe(bodyA).apparent()
    aB = (ctx.earth + site).at(times).observe(bodyB).apparent()
    return aA.separation_from(aB).degrees


def sep_star_arr(ctx, site, body, star, times):
    """Angular separation array (°) between body and a Star."""
    aA = (ctx.earth + site).at(times).observe(body).apparent()
    aB = (ctx.earth + site).at(times).observe(star).apparent()
    return aA.separation_from(aB).degrees


def elong_arr(ctx, site, body, times):
    """Solar elongation array (°, 0–180)."""
    ab = (ctx.earth + site).at(times).observe(body).apparent()
    as_ = (ctx.earth + site).at(times).observe(ctx.sun).apparent()
    return ab.separation_from(as_).degrees


def _bisect_sep(ctx, site, bodyA, bodyB, jd_lo, jd_hi, entering, threshold=_1M):
    """Bisect to ~6-second precision the moment sep crosses threshold."""
    lo, hi = jd_lo, jd_hi
    for _ in range(50):
        mid = (lo + hi) / 2
        s = sep_arr(ctx, site, bodyA, bodyB, ctx.ts.tt_jd(mid))
        if (s > threshold) == entering:
            lo = mid
        else:
            hi = mid
        if (hi - lo) * 1440 < 0.1:
            break
    return ctx.ts.tt_jd((lo + hi) / 2)


# ---------------------------------------------------------------------------
# Eclipse helpers (used only by event_7)
# ---------------------------------------------------------------------------

_RE = 6378.1
_RM = 1737.4
_RS = 696000.0
_K  = 1.0128


def _ecl_mags(ctx, t_arr):
    """Return (umbral_mag, penumbral_mag) numpy arrays for a Time array."""
    m_app = ctx.earth.at(t_arr).observe(ctx.moon).apparent()  # type: ignore[union-attr]
    s_app = ctx.earth.at(t_arr).observe(ctx.sun).apparent()   # type: ignore[union-attr]
    dM = m_app.distance().km
    dS = s_app.distance().km
    r_pen = (_RE + dM * (_RE + _RS) / dS) * _K
    r_umb = (_RE - dM * (_RS  - _RE) / dS) * _K
    rho_pen  = np.arcsin(r_pen  / dM)
    rho_umb  = np.arcsin(r_umb  / dM)
    rho_moon = np.arcsin(_RM    / dM)
    m_ra, m_dec, _ = m_app.radec()
    s_ra, s_dec, _ = s_app.radec()
    as_ra  = np.radians(((s_ra.hours + 12.0) % 24.0) * 15.0)
    as_dec = np.radians(-s_dec.degrees)
    m_ra_r = np.radians(m_ra.hours * 15.0)
    m_dec_r = np.radians(m_dec.degrees)
    dra = m_ra_r - as_ra
    a = (np.sin((m_dec_r - as_dec) / 2.0) ** 2
         + np.cos(m_dec_r) * np.cos(as_dec) * np.sin(dra / 2.0) ** 2)
    delta = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    umb = (rho_umb  + rho_moon - delta) / (2.0 * rho_moon)
    pen = (rho_pen  + rho_moon - delta) / (2.0 * rho_moon)
    return umb, pen


def _bisect_ecl(ctx, jd_lo, jd_hi, use_umb, threshold):
    """Bisect to ~1-second precision where the chosen magnitude crosses threshold."""
    def mag(jd):
        u, p = _ecl_mags(ctx, ctx.ts.tt_jd(np.array([jd])))
        return u[0] if use_umb else p[0]
    sign_lo = mag(jd_lo) < threshold
    for _ in range(50):
        jd_mid = (jd_lo + jd_hi) / 2.0
        if (mag(jd_mid) < threshold) == sign_lo:
            jd_lo = jd_mid
        else:
            jd_hi = jd_mid
        if (jd_hi - jd_lo) * 86400 < 1.0:
            break
    return ctx.ts.tt_jd((jd_lo + jd_hi) / 2.0)


# ---------------------------------------------------------------------------
# Event functions
# ---------------------------------------------------------------------------

def event_1_jv_conjunctions(ctx):
    print(SEP)
    print("EVENT 1: JUPITER–VENUS CONJUNCTIONS")
    print(SEP)

    jd_det = np.linspace(ctx.jd_daily[0], ctx.jd_daily[-1], ctx.n_days * 4)
    jv_det = sep_arr(ctx, ctx.jerusalem, ctx.jup, ctx.ven, ctx.ts.tt_jd(jd_det))

    raw = [
        i for i in range(1, len(jd_det) - 1)
        if jv_det[i] < jv_det[i - 1] and jv_det[i] < jv_det[i + 1]
        and jv_det[i] < 0.2
    ]
    det_jds, prev = [], -999.0
    for i in raw:
        if jd_det[i] - prev > 20:
            det_jds.append(jd_det[i])
            prev = jd_det[i]

    jv_minima = [int(round(jd - ctx.jd_daily[0])) for jd in det_jds]
    jv_minima = [i for i in jv_minima if 0 < i < ctx.n_days - 1]

    for conj_num, idx in enumerate(jv_minima):
        z0 = ctx.jd_daily[max(0, idx - 4)]
        z1 = ctx.jd_daily[min(ctx.n_days - 1, idx + 4)]
        jd_z = np.linspace(z0, z1, 11520)
        tz = ctx.ts.tt_jd(jd_z)

        zv_jer = sep_arr(ctx, ctx.jerusalem, ctx.jup, ctx.ven, tz)
        zv_bab = sep_arr(ctx, ctx.babylon,   ctx.jup, ctx.ven, tz)

        mi_jer = int(np.argmin(zv_jer))
        mi_bab = int(np.argmin(zv_bab))
        t_jv_jer = tz[mi_jer]
        t_jv_bab = tz[mi_bab]

        t1 = ctx.ts.tt_jd(np.array([jd_z[mi_jer]]))
        jup_lon_c = ecl_lon_arr(ctx, ctx.jerusalem, ctx.jup, t1)[0]
        ven_lon_c = ecl_lon_arr(ctx, ctx.jerusalem, ctx.ven, t1)[0]
        jup_lat_c = ecl_lat_arr(ctx, ctx.jerusalem, ctx.jup, t1)[0]
        ven_lat_c = ecl_lat_arr(ctx, ctx.jerusalem, ctx.ven, t1)[0]
        elong_c   = elong_arr(ctx,  ctx.jerusalem, ctx.jup, t1)[0]

        app_jup_jer = (ctx.earth + ctx.jerusalem).at(t_jv_jer).observe(ctx.jup).apparent()
        app_ven_jer = (ctx.earth + ctx.jerusalem).at(t_jv_jer).observe(ctx.ven).apparent()
        app_moon_jer = (ctx.earth + ctx.jerusalem).at(t_jv_jer).observe(ctx.moon).apparent()
        app_reg_jer = (ctx.earth + ctx.jerusalem).at(t_jv_jer).observe(ctx.regulus).apparent()
        alt_jup_jer, az_jup_jer, _ = app_jup_jer.altaz(temperature_C=20, pressure_mbar=1013)
        alt_ven_jer, az_ven_jer, _ = app_ven_jer.altaz(temperature_C=20, pressure_mbar=1013)
        alt_moon_jer, az_moon_jer, _ = app_moon_jer.altaz(temperature_C=20, pressure_mbar=1013)
        alt_reg_jer, az_reg_jer, _ = app_reg_jer.altaz(temperature_C=20, pressure_mbar=1013)

        app_jup_bab = (ctx.earth + ctx.babylon).at(t_jv_bab).observe(ctx.jup).apparent()
        app_ven_bab = (ctx.earth + ctx.babylon).at(t_jv_bab).observe(ctx.ven).apparent()
        app_moon_bab = (ctx.earth + ctx.babylon).at(t_jv_bab).observe(ctx.moon).apparent()
        app_reg_bab = (ctx.earth + ctx.babylon).at(t_jv_bab).observe(ctx.regulus).apparent()
        alt_jup_bab, az_jup_bab, _ = app_jup_bab.altaz(temperature_C=20, pressure_mbar=1013)
        alt_ven_bab, az_ven_bab, _ = app_ven_bab.altaz(temperature_C=20, pressure_mbar=1013)
        alt_moon_bab, az_moon_bab, _ = app_moon_bab.altaz(temperature_C=20, pressure_mbar=1013)
        alt_reg_bab, az_reg_bab, _ = app_reg_bab.altaz(temperature_C=20, pressure_mbar=1013)

        # Illumination fraction from elongation (geocentric, same for both sites)
        moon_elong = elong_arr(ctx, ctx.jerusalem, ctx.moon, t1)[0]
        moon_illum = (1 - np.cos(np.radians(moon_elong))) / 2 * 100

        def _window(site, zv, threshold):
            below = zv < threshold
            if not np.any(below):
                return None
            i_en = int(np.argmax(below))
            i_ex = len(below) - 1 - int(np.argmax(below[::-1]))
            t_en = _bisect_sep(ctx, site, ctx.jup, ctx.ven,
                               jd_z[max(0, i_en - 1)], jd_z[i_en],
                               entering=True, threshold=threshold)
            t_ex = _bisect_sep(ctx, site, ctx.jup, ctx.ven,
                               jd_z[i_ex], jd_z[min(len(jd_z) - 1, i_ex + 1)],
                               entering=False, threshold=threshold)
            return t_en, t_ex

        w1_jer = _window(ctx.jerusalem, zv_jer, 2 * _1M)
        w1_bab = _window(ctx.babylon,   zv_bab, 2 * _1M)
        wd_jer = _window(ctx.jerusalem, zv_jer, _JV_WINDOW)
        wd_bab = _window(ctx.babylon,   zv_bab, _JV_WINDOW)

        print()
        print(f"  Conjunction {conj_num + 1}  (near {fmt(ctx.times_d[idx])})")
        print("  From JERUSALEM")
        print(f"    Closest approach : {fmt(t_jv_jer, hhmm=True)}")
        print(f"    Local solar time : {_lst_str(t_jv_jer, 35.2137)}")
        print(f"    Separation       : {zv_jer[mi_jer]*60:.3f}′  ({zv_jer[mi_jer]:.5f}°)")
        print(f"    Jupiter ecl lon  : {jup_lon_c:.3f}°   lat: {jup_lat_c:+.3f}°")
        print(f"    Venus   ecl lon  : {ven_lon_c:.3f}°   lat: {ven_lat_c:+.3f}°")
        print(f"    Jupiter elong    : {elong_c:.2f}° from Sun")
        print("    Altitude at conjunction:")
        print(f"      Jupiter : alt {alt_jup_jer.degrees:+6.2f}°   az {az_jup_jer.degrees:6.2f}°")
        print(f"      Venus   : alt {alt_ven_jer.degrees:+6.2f}°   az {az_ven_jer.degrees:6.2f}°")
        print(f"      Regulus : alt {alt_reg_jer.degrees:+6.2f}°   az {az_reg_jer.degrees:6.2f}°")
        print(f"      Moon    : alt {alt_moon_jer.degrees:+6.2f}°   az {az_moon_jer.degrees:6.2f}°   illumination {moon_illum:.0f}%")
        if wd_jer:
            t_en, t_ex = wd_jer
            print(f"    Within {_JV_WINDOW*60:.0f}′ of separation:")
            print(f"      Enter : {fmt(t_en, hhmm=True)}  (local {_lst_str(t_en, 35.2137)})")
            print(f"      Leave : {fmt(t_ex, hhmm=True)}  (local {_lst_str(t_ex, 35.2137)})")
            print(f"      Duration: {t_ex.tt - t_en.tt:.2f} days")
        if w1_jer:
            t_en, t_ex = w1_jer
            print("    Within 2′ of separation:")
            print(f"      Enter : {fmt(t_en, hhmm=True)}  (local {_lst_str(t_en, 35.2137)})")
            print(f"      Leave : {fmt(t_ex, hhmm=True)}  (local {_lst_str(t_ex, 35.2137)})")
            print(f"      Duration: {(t_ex.tt - t_en.tt) * 1440:.1f} min")
        print("  From BABYLON")
        print(f"    Closest approach : {fmt(t_jv_bab, hhmm=True)}")
        print(f"    Local solar time : {_lst_str(t_jv_bab, 44.4215)}")
        print(f"    Separation       : {zv_bab[mi_bab]*60:.3f}′  ({zv_bab[mi_bab]:.5f}°)")
        print("    Altitude at conjunction:")
        print(f"      Jupiter : alt {alt_jup_bab.degrees:+6.2f}°   az {az_jup_bab.degrees:6.2f}°")
        print(f"      Venus   : alt {alt_ven_bab.degrees:+6.2f}°   az {az_ven_bab.degrees:6.2f}°")
        print(f"      Regulus : alt {alt_reg_bab.degrees:+6.2f}°   az {az_reg_bab.degrees:6.2f}°")
        print(f"      Moon    : alt {alt_moon_bab.degrees:+6.2f}°   az {az_moon_bab.degrees:6.2f}°   illumination {moon_illum:.0f}%")
        if wd_bab:
            t_en, t_ex = wd_bab
            print(f"    Within {_JV_WINDOW*60:.0f}′ of separation:")
            print(f"      Enter : {fmt(t_en, hhmm=True)}  (local {_lst_str(t_en, 44.4215)})")
            print(f"      Leave : {fmt(t_ex, hhmm=True)}  (local {_lst_str(t_ex, 44.4215)})")
            print(f"      Duration: {t_ex.tt - t_en.tt:.2f} days")
        if w1_bab:
            t_en, t_ex = w1_bab
            print("    Within 2′ of separation:")
            print(f"      Enter : {fmt(t_en, hhmm=True)}  (local {_lst_str(t_en, 44.4215)})")
            print(f"      Leave : {fmt(t_ex, hhmm=True)}  (local {_lst_str(t_ex, 44.4215)})")
            print(f"      Duration: {(t_ex.tt - t_en.tt) * 1440:.1f} min")
    print()


def event_2_jr_conjunctions(ctx) -> tuple:
    """Returns (t_jr2_jd, t_jr3_jd): TT JDs of the 2nd and 3rd J-R conjunctions."""
    print(SEP)
    print("EVENT 2: JUPITER–REGULUS CONJUNCTIONS")
    print(SEP)

    jr_sep_jer = sep_star_arr(ctx, ctx.jerusalem, ctx.jup, ctx.regulus, ctx.times_d)
    # jr_sep_bab = sep_star_arr(ctx, ctx.babylon,   ctx.jup, ctx.regulus, ctx.times_d)

    jr_minima = [
        i for i in range(1, ctx.n_days - 1)
        if jr_sep_jer[i] < jr_sep_jer[i - 1]
        and jr_sep_jer[i] < jr_sep_jer[i + 1]
        and jr_sep_jer[i] < 5.0
    ]

    t_jr2_jd = None
    t_jr3_jd = None
    for conj_num, idx in enumerate(jr_minima):
        z0 = ctx.jd_daily[max(0, idx - 6)]
        z1 = ctx.jd_daily[min(ctx.n_days - 1, idx + 6)]
        jd_z2 = np.linspace(z0, z1, 17280)
        tz2 = ctx.ts.tt_jd(jd_z2)

        zr_jer = sep_star_arr(ctx, ctx.jerusalem, ctx.jup, ctx.regulus, tz2)
        zr_bab = sep_star_arr(ctx, ctx.babylon,   ctx.jup, ctx.regulus, tz2)

        mi2_jer = int(np.argmin(zr_jer))
        mi2_bab = int(np.argmin(zr_bab))

        t1 = ctx.ts.tt_jd(np.array([jd_z2[mi2_jer]]))
        jl  = ecl_lon_arr(ctx,      ctx.jerusalem, ctx.jup,     t1)[0]
        jla = ecl_lat_arr(ctx,      ctx.jerusalem, ctx.jup,     t1)[0]
        rl  = ecl_lon_star_arr(ctx, ctx.jerusalem, ctx.regulus, t1)[0]

        if idx > 2 and idx < ctx.n_days - 2:
            dj = (ctx.jup_lon_jer[idx + 2] - ctx.jup_lon_jer[idx - 2] + 360) % 360
            if dj > 180:
                dj -= 360
            motion_str = "retrograde" if dj < 0 else "direct"
        else:
            motion_str = "—"

        print()
        print(f"  Conjunction near {fmt(ctx.times_d[idx])}:")
        print(f"    From JERUSALEM : {fmt(tz2[mi2_jer], hhmm=True)}  (local {_lst_str(tz2[mi2_jer], 35.2137)})")
        print(f"      Separation   : {zr_jer[mi2_jer]*60:.2f}′  ({motion_str})")
        print(f"      Jupiter lon  : {jl:.3f}°   lat: {jla:+.3f}°  (Regulus lon: {rl:.3f}°)")
        print(f"    From BABYLON   : {fmt(tz2[mi2_bab], hhmm=True)}  (local {_lst_str(tz2[mi2_bab], 44.4215)})")
        print(f"      Separation   : {zr_bab[mi2_bab]*60:.2f}′")

        if conj_num == 1:
            t_jr2_jd = tz2[mi2_jer].tt
        elif conj_num == 2:
            t_jr3_jd = tz2[mi2_jer].tt
    print()

    assert t_jr2_jd is not None, "No 2nd Jupiter–Regulus conjunction found in range"
    assert t_jr3_jd is not None, "No 3rd Jupiter–Regulus conjunction found in range"
    return t_jr2_jd, t_jr3_jd


def event_3_moon_at_regulus(ctx, t_jr2_jd: float):
    print(SEP)
    print("EVENT 3: MOON & JUPITER AT REGULUS — AROUND 2ND J-R CONJUNCTION")
    print("Best moon visibility from Babylon: 3 nights before + 1 night after")
    print(SEP)

    for night_offset in [3, 2, 1, 0]:
        pre_jd = np.linspace(t_jr2_jd - night_offset, t_jr2_jd - night_offset + 1.0, 1440)
        pre_t  = ctx.ts.tt_jd(pre_jd)

        sun_app = (ctx.earth + ctx.babylon).at(pre_t).observe(ctx.sun).apparent()
        sun_alt3, _, _ = sun_app.altaz()
        night_mask = sun_alt3.degrees < 0.0

        moon_app = (ctx.earth + ctx.babylon).at(pre_t).observe(ctx.moon).apparent()
        moon_alt3, _, _ = moon_app.altaz()

        moon_alt_night = np.where(night_mask, moon_alt3.degrees, -999.0)
        best = int(np.argmax(moon_alt_night))

        t_best = pre_t[best]
        t_best_arr = ctx.ts.tt_jd(np.array([pre_jd[best]]))
        mr_sep  = sep_star_arr(ctx, ctx.babylon, ctx.moon, ctx.regulus, t_best_arr)[0]
        jr_sep3 = sep_star_arr(ctx, ctx.babylon, ctx.jup,  ctx.regulus, t_best_arr)[0]
        jm_sep3 = sep_arr(ctx,      ctx.babylon, ctx.jup,  ctx.moon,    t_best_arr)[0]

        earth_at = ctx.earth.at(t_best)  # type: ignore[union-attr]
        moon_elong = (earth_at.observe(ctx.moon).apparent()
                      .separation_from(earth_at.observe(ctx.sun).apparent()).degrees)
        illumination = (1 - np.cos(np.radians(moon_elong))) / 2 * 100

        print()
        night_label = f"{night_offset} day(s) before" if night_offset > 0 else "night after"
        print(f"  Night {4 - night_offset} of 4  ({night_label} conjunction)")
        print(f"    Observation time : {fmt(t_best, hhmm=True)}  (local {_lst_str(t_best, 44.4215)})")
        print(f"    Moon altitude    : {moon_alt_night[best]:.2f}°")
        print(f"    Sun altitude     : {sun_alt3.degrees[best]:.2f}°")
        print(f"    Moon phase       : {illumination:.1f}% illuminated  (elongation {moon_elong:.1f}°)")
        print(f"    Moon–Regulus     : {mr_sep:.4f}°")
        print(f"    Jup–Regulus      : {jr_sep3:.4f}°")
        print(f"    Jup–Moon         : {jm_sep3:.4f}°")
    print()


def event_4_stationary_points(ctx):
    print(SEP)
    print("EVENT 4: JUPITER STATIONARY POINTS")
    print(SEP)

    jup_lon_unw = np.degrees(np.unwrap(np.radians(ctx.jup_lon_jer)))
    dlon_day = np.diff(jup_lon_unw)

    stations = []
    for i in range(1, len(dlon_day)):
        if dlon_day[i - 1] * dlon_day[i] < 0:
            kind = (
                "FIRST STATION → retrograde begins"
                if dlon_day[i] < 0
                else "SECOND STATION → direct motion resumes"
            )
            stations.append((i, kind))

    for idx, kind in stations:
        z0 = ctx.jd_daily[max(0, idx - 5)]
        z1 = ctx.jd_daily[min(ctx.n_days - 1, idx + 5)]
        jd_z3 = np.linspace(z0, z1, 14400)
        tz3 = ctx.ts.tt_jd(jd_z3)

        z_lon3     = ecl_lon_arr(ctx, ctx.jerusalem, ctx.jup, tz3)
        z_lon3_unw = np.degrees(np.unwrap(np.radians(z_lon3)))
        z_dlon3    = np.diff(z_lon3_unw)

        stat_i = None
        for j in range(1, len(z_dlon3)):
            if z_dlon3[j - 1] * z_dlon3[j] < 0:
                stat_i = j
                break
        if stat_i is None:
            stat_i = int(np.argmin(np.abs(z_dlon3)))

        t_st   = tz3[stat_i]
        lon_st = z_lon3[stat_i]
        el_st  = elong_arr(ctx, ctx.jerusalem, ctx.jup, ctx.ts.tt_jd(np.array([jd_z3[stat_i]])))[0]
        dl_st  = (ctx.jup_lon_jer[idx] - ctx.sun_lon_jer[idx]) % 360
        sky    = "morning sky (W of Sun)" if dl_st > 180 else "evening sky (E of Sun)"

        app_st = (ctx.earth + ctx.jerusalem).at(t_st).observe(ctx.jup).apparent()
        alt_st, az_st, _ = app_st.altaz(temperature_C=20, pressure_mbar=1013)

        print()
        print(f"  {kind}")
        print(f"    Date (Jerusalem) : {fmt(t_st, hhmm=True)}  (local {_lst_str(t_st, 35.2137)})")
        print(f"    Jupiter ecl lon  : {lon_st:.3f}°")
        print(f"    Solar elongation : {el_st:.2f}°  ({sky})")
        print(f"    Altitude         : {alt_st.degrees:+.2f}°")
        print(f"    Azimuth          : {az_st.degrees:.2f}°  ({_compass_pt(az_st.degrees)})")
        print(f"    Daily lon motion at station: ~{z_dlon3[stat_i]*60:.3f}′/day")

        # Rise/set times near the station
        t_cen = jd_z3[stat_i]
        t_s0  = ctx.ts.tt_jd(t_cen - 1.5)
        t_s1  = ctx.ts.tt_jd(t_cen + 1.5)

        def _nearest(body, want_rise: bool):
            """Return the rise (want_rise=True) or set nearest to t_cen."""
            f_rs = almanac.risings_and_settings(ctx.eph, body, ctx.jerusalem)
            evts, kinds = almanac.find_discrete(t_s0, t_s1, f_rs)
            flag = 1 if want_rise else 0
            cands = [evts[k] for k in range(len(evts)) if kinds[k] == flag]
            if not cands:
                return None
            return min(cands, key=lambda t: abs(t.tt - t_cen))

        t_jup_set = _nearest(ctx.jup,     False)
        t_reg_set = _nearest(ctx.regulus, False)
        t_moon_rise = _nearest(ctx.moon,  True)

        def _fmt_evt(t):
            if t is None:
                return "not found"
            return f"{fmt(t, hhmm=True)}  (local {_lst_str(t, _JER_LON)})"

        print(f"    Jupiter sets      : {_fmt_evt(t_jup_set)}")
        print(f"    Regulus sets      : {_fmt_evt(t_reg_set)}")
        print(f"    Moonrise          : {_fmt_evt(t_moon_rise)}")

    print()


def event_regulus_heliacal_rising(ctx, av_star: float = 13.5):
    print(SEP)
    print("EVENT: REGULUS HELIACAL RISING")
    print(SEP)

    reg_lon_jer  = ecl_lon_star_arr(ctx, ctx.jerusalem, ctx.regulus, ctx.times_d)
    reg_elong_jer = elong_arr(ctx, ctx.jerusalem, ctx.regulus, ctx.times_d)
    morning_reg  = ((reg_lon_jer - ctx.sun_lon_jer) % 360) > 180

    heliacal = [
        i for i in range(1, ctx.n_days)
        if morning_reg[i]
        and reg_elong_jer[i - 1] < av_star
        and reg_elong_jer[i] >= av_star
    ]

    if not heliacal:
        print(
            f"  No heliacal risings found with AV = {av_star}°. "
            "(Possibly Regulus didn't have a solar conjunction in range.)"
        )
    else:
        for idx in heliacal:
            f_reg = almanac.risings_and_settings(ctx.eph, ctx.regulus, ctx.jerusalem)
            t0 = ctx.ts.tt_jd(ctx.jd_daily[max(0, idx - 2)])
            t1 = ctx.ts.tt_jd(ctx.jd_daily[min(ctx.n_days - 1, idx + 3)])
            hr_t, hr_ev = almanac.find_discrete(t0, t1, f_reg)
            reg_rises = [hr_t[k] for k in range(len(hr_t)) if hr_ev[k] == 1]

            t_hr = None
            for rr in reg_rises:
                if elong_arr(ctx, ctx.jerusalem, ctx.regulus, ctx.ts.tt_jd(np.array([rr.tt])))[0] >= av_star:
                    t_hr = rr
                    break
            if t_hr is None:
                t_hr = reg_rises[0] if reg_rises else ctx.times_d[idx]

            el_hr  = elong_arr(ctx,      ctx.jerusalem, ctx.regulus, ctx.ts.tt_jd(np.array([t_hr.tt])))[0]
            lon_hr = ecl_lon_star_arr(ctx, ctx.jerusalem, ctx.regulus, ctx.ts.tt_jd(np.array([t_hr.tt])))[0]

            t_hr_arr = ctx.ts.tt_jd(np.array([t_hr.tt]))
            jup_app_hr = (ctx.earth + ctx.jerusalem).at(t_hr).observe(ctx.jup).apparent()
            jup_alt_hr, _, _ = jup_app_hr.altaz(temperature_C=20, pressure_mbar=1013)
            ven_app_hr = (ctx.earth + ctx.jerusalem).at(t_hr).observe(ctx.ven).apparent()
            ven_alt_hr, _, _ = ven_app_hr.altaz(temperature_C=20, pressure_mbar=1013)
            moon_app_hr = (ctx.earth + ctx.jerusalem).at(t_hr).observe(ctx.moon).apparent()
            moon_alt_hr, _, _ = moon_app_hr.altaz(temperature_C=20, pressure_mbar=1013)
            moon_elong_hr = elong_arr(ctx, ctx.jerusalem, ctx.moon, t_hr_arr)[0]
            moon_illum_hr = (1 - np.cos(np.radians(moon_elong_hr))) / 2 * 100

            print()
            print("  Heliacal Rising — Jerusalem")
            print(f"    Date             : {fmt(t_hr, hhmm=True)}  (local {_lst_str(t_hr, 35.2137)})")
            print(f"    Ecliptic lon     : {lon_hr:.3f}°")
            print(f"    Solar elongation : {el_hr:.2f}°  (threshold {av_star}°)")
            print(f"    Jupiter alt      : {jup_alt_hr.degrees:+.2f}°")
            print(f"    Venus alt        : {ven_alt_hr.degrees:+.2f}°")
            print(f"    Moon alt at rise : {moon_alt_hr.degrees:+.2f}°   illumination {moon_illum_hr:.0f}%")
    print()


def event_5_heliacal_rising(ctx, av: float):
    print(SEP)
    print("EVENT 5: JUPITER HELIACAL RISING")
    print(SEP)

    heliacal = [
        i for i in range(1, ctx.n_days)
        if ctx.morning_jer[i]
        and ctx.jup_elong_jer[i - 1] < av
        and ctx.jup_elong_jer[i] >= av
    ]

    if not heliacal:
        print(
            f"  No heliacal risings found with AV = {av}°. "
            "(Possibly Jupiter didn't have a solar conjunction in range.)"
        )
    else:
        for idx in heliacal:
            # Heliacal rising = first morning Jupiter crests the horizon while elongation
            # >= AV (sky dark enough to see it before the sun washes it out).
            f_jr = almanac.risings_and_settings(ctx.eph, ctx.jup, ctx.jerusalem)
            t0   = ctx.ts.tt_jd(ctx.jd_daily[max(0, idx - 2)])
            t1   = ctx.ts.tt_jd(ctx.jd_daily[min(ctx.n_days - 1, idx + 3)])
            hr_t, hr_ev = almanac.find_discrete(t0, t1, f_jr)
            jup_rises = [hr_t[k] for k in range(len(hr_t)) if hr_ev[k] == 1]

            t_hr = None
            for jr in jup_rises:
                if elong_arr(ctx, ctx.jerusalem, ctx.jup, ctx.ts.tt_jd(np.array([jr.tt])))[0] >= av:
                    t_hr = jr
                    break
            if t_hr is None:
                t_hr = jup_rises[0] if jup_rises else ctx.times_d[idx]

            el_hr  = elong_arr(ctx,  ctx.jerusalem, ctx.jup, ctx.ts.tt_jd(np.array([t_hr.tt])))[0]
            lon_hr = ecl_lon_arr(ctx, ctx.jerusalem, ctx.jup, ctx.ts.tt_jd(np.array([t_hr.tt])))[0]

            # Find when Jupiter crosses 5° altitude that morning (bisect on fine grid)
            ALT5 = 5.0
            jd_fine  = np.linspace(t_hr.tt, t_hr.tt + 4.0 / 24.0, 2880)
            alt_fine = (
                (ctx.earth + ctx.jerusalem).at(ctx.ts.tt_jd(jd_fine)).observe(ctx.jup).apparent()
                .altaz(temperature_C=20, pressure_mbar=1013)[0].degrees
            )
            t5 = None
            for j in range(len(alt_fine) - 1):
                if alt_fine[j] < ALT5 <= alt_fine[j + 1]:
                    lo5, hi5 = jd_fine[j], jd_fine[j + 1]
                    for _ in range(40):
                        mid5 = (lo5 + hi5) / 2
                        a5 = (
                            (ctx.earth + ctx.jerusalem).at(ctx.ts.tt_jd(mid5)).observe(ctx.jup).apparent()
                            .altaz(temperature_C=20, pressure_mbar=1013)[0].degrees
                        )
                        if a5 < ALT5:
                            lo5 = mid5
                        else:
                            hi5 = mid5
                    t5 = ctx.ts.tt_jd((lo5 + hi5) / 2)
                    break

            # Moon and Venus position at heliacal rising time
            t_hr_arr = ctx.ts.tt_jd(np.array([t_hr.tt]))
            moon_app_hr = (ctx.earth + ctx.jerusalem).at(t_hr).observe(ctx.moon).apparent()
            moon_alt_hr, _, _ = moon_app_hr.altaz(temperature_C=20, pressure_mbar=1013)
            moon_elong_hr = elong_arr(ctx, ctx.jerusalem, ctx.moon, t_hr_arr)[0]
            moon_illum_hr = (1 - np.cos(np.radians(moon_elong_hr))) / 2 * 100
            ven_app_hr = (ctx.earth + ctx.jerusalem).at(t_hr).observe(ctx.ven).apparent()
            ven_alt_hr, _, _ = ven_app_hr.altaz(temperature_C=20, pressure_mbar=1013)

            print()
            print("  Heliacal Rising — Jerusalem")
            print(f"    Date             : {fmt(t_hr, hhmm=True)}  (local {_lst_str(t_hr, 35.2137)})")
            print(f"    Ecliptic lon     : {lon_hr:.3f}°")
            print(f"    Solar elongation : {el_hr:.2f}°  (threshold {av}°)")
            if t5 is not None:
                print(f"    Above 5° alt     : local {_lst_str(t5, 35.2137)}")
            else:
                print(f"    Above 5° alt     : not reached within 4 h of rise")
            print(f"    Venus alt at rise: {ven_alt_hr.degrees:+.2f}°")
            print(f"    Moon alt at rise : {moon_alt_hr.degrees:+.2f}°   illumination {moon_illum_hr:.0f}%")
    print()


def event_6_weekly_altaz(ctx, av: float):
    print(SEP)
    print("EVENT 6: JUPITER ALT/AZ — WEEKLY, AT SUN ALTITUDE OF AUG 2 BC HELIACAL RISING")
    print("Observer : Jerusalem")
    print("Time     : Sun at same altitude as during the 29 Aug 2 BC heliacal rising")
    print(SEP)

    # --- Find the heliacal rising around 29 Aug 2 BC (year -1) ---
    heliacal_idxs = [
        i for i in range(1, ctx.n_days)
        if ctx.morning_jer[i]
        and ctx.jup_elong_jer[i - 1] < av
        and ctx.jup_elong_jer[i] >= av
    ]
    if not heliacal_idxs:
        print("  ERROR: No heliacal rising found.\n")
        return

    t_target_tt = ctx.ts.tt(-1, 8, 27).tt
    best_idx = min(heliacal_idxs, key=lambda i: abs(ctx.jd_daily[i] - t_target_tt))

    f_jr = almanac.risings_and_settings(ctx.eph, ctx.jup, ctx.jerusalem)
    t0_hr = ctx.ts.tt_jd(ctx.jd_daily[max(0, best_idx - 2)])
    t1_hr = ctx.ts.tt_jd(ctx.jd_daily[min(ctx.n_days - 1, best_idx + 3)])
    hr_t, hr_ev = almanac.find_discrete(t0_hr, t1_hr, f_jr)
    jup_rises = [hr_t[k] for k in range(len(hr_t)) if hr_ev[k] == 1]

    t_hr = None
    for jr in jup_rises:
        if elong_arr(ctx, ctx.jerusalem, ctx.jup, ctx.ts.tt_jd(np.array([jr.tt])))[0] >= av:
            t_hr = jr
            break
    if t_hr is None:
        t_hr = jup_rises[0] if jup_rises else ctx.times_d[best_idx]

    sun_app_hr = (ctx.earth + ctx.jerusalem).at(t_hr).observe(ctx.sun).apparent()
    target_sun_alt_obj, _, _ = sun_app_hr.altaz(temperature_C=20, pressure_mbar=1013)
    target_alt_deg = target_sun_alt_obj.degrees

    print(f"  Heliacal rising ref : {fmt(t_hr, hhmm=True)}")
    print(f"  Sun altitude ref    : {target_alt_deg:.3f}°")
    print()

    # --- Find sunrises over the weekly window ---
    f_sr = almanac.risings_and_settings(ctx.eph, ctx.sun, ctx.jerusalem)
    sr_t, sr_ev = almanac.find_discrete(ctx.ts.tt(-1, 8, 27), ctx.ts.tt(0, 1, 2), f_sr)
    sunrises = np.array([t.tt for t, ev in zip(sr_t, sr_ev) if ev == 1])

    if len(sunrises) == 0:
        print("  ERROR: Could not find sunrises.\n")
        return

    s_start   = ctx.ts.tt(-1, 8, 27).tt
    w_end_tt  = ctx.ts.tt(0, 1, 2).tt
    n_weeks   = int((w_end_tt - s_start) / 7) + 2

    def _sun_alt_deg(jd_tt):
        app = (ctx.earth + ctx.jerusalem).at(ctx.ts.tt_jd(jd_tt)).observe(ctx.sun).apparent()
        return app.altaz(temperature_C=20, pressure_mbar=1013)[0].degrees

    def _find_crossing(sr_tt):
        """Bisect to find TT JD when sun alt = target_alt_deg in the morning before sunrise."""
        lo = sr_tt - 4.0 / 24.0
        hi = sr_tt
        if _sun_alt_deg(lo) > target_alt_deg:
            return None
        for _ in range(50):
            mid = (lo + hi) / 2
            if _sun_alt_deg(mid) < target_alt_deg:
                lo = mid
            else:
                hi = mid
            if (hi - lo) * 1440 < 0.1:
                break
        return (lo + hi) / 2

    week_obs_tt, week_sr_tt = [], []
    for w in range(n_weeks):
        wjd = s_start + w * 7
        if wjd > w_end_tt:
            break
        i = int(np.searchsorted(sunrises, wjd))
        if i >= len(sunrises):
            break
        sr = sunrises[i]
        obs_tt = _find_crossing(sr)
        if obs_tt is None:
            continue
        week_sr_tt.append(sr)
        week_obs_tt.append(obs_tt)

    obs_times = ctx.ts.tt_jd(np.array(week_obs_tt))
    app6 = (ctx.earth + ctx.jerusalem).at(obs_times).observe(ctx.jup).apparent()
    alts, azs, _ = app6.altaz(temperature_C=20, pressure_mbar=1013)
    alts = alts.degrees
    azs  = azs.degrees

    obs_ut1_arr = np.array([ctx.ts.tt_jd(jd).ut1 for jd in week_obs_tt])
    sr_ut1_arr  = np.array([ctx.ts.tt_jd(jd).ut1 for jd in week_sr_tt])

    hdr = (
        f"  {'Date':<20}  {'Obs (local)':>11}  "
        f"{'Sunrise':>8}  {'Altitude':>9}  {'Azimuth':>9}  Direction"
    )
    print(hdr)
    print("  " + "—" * (len(hdr) - 2))
    for i in range(len(obs_times)):
        obs_lh = _local_h(obs_ut1_arr[i])
        sr_lh  = _local_h(sr_ut1_arr[i])
        print(
            f"  {fmt(obs_times[i]):<20}  "
            f"{int(obs_lh):02d}:{int((obs_lh % 1)*60):02d}        "
            f"{int(sr_lh):02d}:{int((sr_lh % 1)*60):02d}   "
            f"{alts[i]:8.2f}°  {azs[i]:8.2f}°  {_compass_pt(azs[i])}"
        )
    print()


def event_7_lunar_eclipse(ctx):
    from skyfield import eclipselib

    print(SEP)
    print("EVENT 7: TOTAL LUNAR ECLIPSE — 1 BC JANUARY")
    print(SEP)

    t0_ecl = ctx.ts.utc(0, 1, 1)
    t1_ecl = ctx.ts.utc(0, 1, 31)
    t_ecl, y_ecl, _ = eclipselib.lunar_eclipses(t0_ecl, t1_ecl, ctx.eph)

    if len(t_ecl) == 0:
        print("  No eclipse found in Jan 1 BC.\n")
        return

    idx_ecl   = int(np.argmin(np.abs(t_ecl.tt - t0_ecl.tt)))
    t_peak    = t_ecl[idx_ecl]
    kind_name = eclipselib.LUNAR_ECLIPSES[int(y_ecl[idx_ecl])]
    is_total  = (kind_name == "Total")

    u_pk, p_pk = _ecl_mags(ctx, ctx.ts.tt_jd(np.array([t_peak.tt])))

    jd_sc       = np.linspace(t_peak.tt - 5.0/24, t_peak.tt + 5.0/24, 600)
    u_sc, p_sc  = _ecl_mags(ctx, ctx.ts.tt_jd(jd_sc))

    def _bracket(mag_arr, thresh, rising):
        out = []
        for i in range(len(mag_arr) - 1):
            a, b = float(mag_arr[i]), float(mag_arr[i + 1])
            if rising and a < thresh <= b:
                out.append((jd_sc[i], jd_sc[i + 1]))
            elif not rising and a >= thresh > b:
                out.append((jd_sc[i], jd_sc[i + 1]))
        return out

    def _contact(mag_arr, use_umb, thresh, rising):
        bra = _bracket(mag_arr, thresh, rising)
        return _bisect_ecl(ctx, bra[0][0], bra[0][1], use_umb, thresh) if bra else None

    t_P1 = _contact(p_sc, False, 0.0, True)
    t_U1 = _contact(u_sc, True,  0.0, True)
    t_U2 = _contact(u_sc, True,  1.0, True)  if is_total else None
    t_U3 = _contact(u_sc, True,  1.0, False) if is_total else None
    t_U4 = _contact(u_sc, True,  0.0, False)
    t_P4 = _contact(p_sc, False, 0.0, False)

    def _altaz(t):
        obs = (ctx.earth + ctx.jerusalem).at(t).observe(ctx.moon).apparent()
        alt, az, _ = obs.altaz()
        return alt.degrees, az.degrees

    def _row(label, t):
        alt, az = _altaz(t)
        flag = "" if alt > 0 else "  [below horizon]"
        print(f"    {label:<22} : {fmt(t, hhmm=True)}  local {_lst_str(t, 35.2137)}"
              f"  alt {alt:+6.2f}°  az {az:6.2f}°{flag}")

    def _dur(ta, tb):
        if ta is None or tb is None:
            return "—"
        dm = round((tb.tt - ta.tt) * 1440)
        return f"{dm // 60}h {dm % 60:02d}m"

    alt_pk, az_pk = _altaz(t_peak)

    print()
    print(f"  Type             : {kind_name}")
    print(f"  Peak             : {fmt(t_peak, hhmm=True)}  local {_lst_str(t_peak, 35.2137)}")
    print(f"  Umbral magnitude : {u_pk[0]:.3f}")
    print(f"  Penumbral mag    : {p_pk[0]:.3f}")
    print(f"  Moon at maximum  : alt {alt_pk:+.2f}°  az {az_pk:.2f}°")
    print()
    print("  Contact times (Jerusalem)")
    if t_P1 is not None: _row("P1  penumbral ingress", t_P1)
    if t_U1 is not None: _row("U1  umbral ingress   ", t_U1)
    if t_U2 is not None: _row("U2  totality begins  ", t_U2)
    print(f"      Maximum          : {fmt(t_peak, hhmm=True)}  local {_lst_str(t_peak, 35.2137)}"
          f"  alt {alt_pk:+6.2f}°  az {az_pk:6.2f}°")
    if t_U3 is not None: _row("U3  totality ends    ", t_U3)
    if t_U4 is not None: _row("U4  umbral egress    ", t_U4)
    if t_P4 is not None: _row("P4  penumbral egress ", t_P4)
    print()
    print("  Durations")
    if is_total:
        print(f"    Totality  (U2–U3) : {_dur(t_U2, t_U3)}")
    print(    f"    Umbral    (U1–U4) : {_dur(t_U1, t_U4)}")
    print(    f"    Penumbral (P1–P4) : {_dur(t_P1, t_P4)}")
    print()


def event_moon_regulus_occultations(ctx):
    print(SEP)
    print("EVENT: MOON OCCULTATIONS OF REGULUS — 1 Jan 3 BC – 1 Jan 1 BC")
    print("Observer : Jerusalem  (visibility: Regulus above horizon, sun below)")
    print(SEP)

    # Hourly grid over the full period
    n_hours = int(round((ctx.jd_daily[-1] - ctx.jd_daily[0]) * 24)) + 1
    jd_hr = np.linspace(ctx.jd_daily[0], ctx.jd_daily[-1], n_hours)
    t_hr  = ctx.ts.tt_jd(jd_hr)

    sep_hr = sep_star_arr(ctx, ctx.jerusalem, ctx.moon, ctx.regulus, t_hr)

    # Moon's mean angular radius ~0.267°; use a generous threshold to catch all events
    THRESHOLD = 0.35

    # Find hourly intervals where separation dips below threshold
    below = sep_hr < THRESHOLD
    # Identify entry points into sub-threshold intervals
    crossings = np.where(~below[:-1] & below[1:])[0]  # hour index just before dip starts

    occultations = []
    site = ctx.earth + ctx.jerusalem

    for ci in crossings:
        # Widen search window: start 2 h before, end 4 h after first sub-threshold sample
        w0 = jd_hr[max(0, ci - 1)]
        w1 = jd_hr[min(n_hours - 1, ci + 5)]

        # 1-minute resolution zoom
        n_min = int(round((w1 - w0) * 24 * 60)) + 1
        jd_m = np.linspace(w0, w1, n_min)
        t_m  = ctx.ts.tt_jd(jd_m)
        sep_m = sep_star_arr(ctx, ctx.jerusalem, ctx.moon, ctx.regulus, t_m)

        # Moon's apparent angular radius at midpoint
        t_mid = ctx.ts.tt_jd(np.array([(w0 + w1) / 2]))
        moon_astrometric = site.at(t_mid).observe(ctx.moon).apparent()
        moon_dist_au = moon_astrometric.distance().au[0]
        MOON_RADIUS_KM = 1737.4
        AU_KM = 1.495978707e8
        moon_radius_deg = np.degrees(np.arcsin(MOON_RADIUS_KM / (moon_dist_au * AU_KM)))

        min_sep = sep_m.min()
        if min_sep >= moon_radius_deg:
            continue  # near-miss; no actual occultation

        # Find ingress and egress contacts (separation crosses moon_radius_deg)
        inside = sep_m < moon_radius_deg
        ingress_idx = int(np.argmax(inside))          # first True
        egress_idx  = int(len(inside) - 1 - np.argmax(inside[::-1]))  # last True

        t_ing = ctx.ts.tt_jd(jd_m[ingress_idx])
        t_egr = ctx.ts.tt_jd(jd_m[egress_idx])
        t_min_t = ctx.ts.tt_jd(jd_m[int(np.argmin(sep_m))])

        # Visibility at mid-occultation
        t_mid2 = ctx.ts.tt_jd(np.array([jd_m[int(np.argmin(sep_m))]]))
        reg_alt = site.at(t_mid2).observe(ctx.regulus).apparent() \
                  .altaz(temperature_C=10, pressure_mbar=1013)[0].degrees[0]
        sun_alt = site.at(t_mid2).observe(ctx.sun).apparent() \
                  .altaz(temperature_C=10, pressure_mbar=1013)[0].degrees[0]
        visible = reg_alt > 0 and sun_alt < 0

        occultations.append({
            "t_ing": t_ing, "t_egr": t_egr, "t_min": t_min_t,
            "min_sep": min_sep, "moon_r": moon_radius_deg,
            "reg_alt": reg_alt, "sun_alt": sun_alt, "visible": visible,
        })

    if not occultations:
        print("  No occultations found.\n")
        return

    print(f"  Found {len(occultations)} occultation(s).  "
          f"Visibility = Regulus above horizon & sun below.\n")
    print(f"  {'Ingress (TT)':>24}  {'local':>5}  {'Egress (TT)':>24}  {'local':>5}  "
          f"{'Min sep':>8}  {'Moon r':>7}  {'Reg alt':>8}  {'Sun alt':>8}  Visible?")
    print(f"  {'-'*24}  {'-'*5}  {'-'*24}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}  {'-'*8}")
    for oc in occultations:
        vis = "YES ***" if oc["visible"] else "no (daytime/below horizon)"
        print(f"  {fmt(oc['t_ing'], hhmm=True):>24}  {_lst_str(oc['t_ing'], _JER_LON):>5}  "
              f"{fmt(oc['t_egr'], hhmm=True):>24}  {_lst_str(oc['t_egr'], _JER_LON):>5}  "
              f"{oc['min_sep']*60:>6.2f}′  {oc['moon_r']*60:>5.2f}′  "
              f"{oc['reg_alt']:>+7.2f}°  {oc['sun_alt']:>+7.2f}°  {vis}")
    print()



def event_8_moon_regulus_after_jr3(ctx, t_jr3_jd: float):
    print(SEP)
    print("EVENT 8: MOON–REGULUS CLOSEST VISIBLE APPROACH AFTER 3RD J-R CONJUNCTION")
    print("Observer : Jerusalem  (visibility: moon & Regulus above horizon, sun below)")
    print(SEP)

    # Hourly grid from 3rd J-R conjunction to end of period
    n_hours = int(round((ctx.jd_daily[-1] - t_jr3_jd) * 24)) + 1
    jd_hr = np.linspace(t_jr3_jd, ctx.jd_daily[-1], n_hours)
    t_hr  = ctx.ts.tt_jd(jd_hr)

    site = ctx.earth + ctx.jerusalem

    sep_hr     = sep_star_arr(ctx, ctx.jerusalem, ctx.moon,    ctx.regulus, t_hr)
    moon_alt_hr = site.at(t_hr).observe(ctx.moon).apparent().altaz(temperature_C=10, pressure_mbar=1013)[0].degrees
    reg_alt_hr  = site.at(t_hr).observe(ctx.regulus).apparent().altaz(temperature_C=10, pressure_mbar=1013)[0].degrees
    sun_alt_hr  = site.at(t_hr).observe(ctx.sun).apparent().altaz(temperature_C=10, pressure_mbar=1013)[0].degrees

    visible_hr = (moon_alt_hr > 0) & (reg_alt_hr > 0) & (sun_alt_hr < 0)

    if not np.any(visible_hr):
        print("  No visible Moon–Regulus approach found after the 3rd J-R conjunction.\n")
        return

    sep_vis_hr = np.where(visible_hr, sep_hr, np.inf)
    best_hr = int(np.argmin(sep_vis_hr))

    # Zoom to 1-minute resolution around the hourly best
    z0 = jd_hr[max(0, best_hr - 1)]
    z1 = jd_hr[min(n_hours - 1, best_hr + 1)]
    n_min = int(round((z1 - z0) * 24 * 60)) + 1
    jd_m = np.linspace(z0, z1, n_min)
    t_m  = ctx.ts.tt_jd(jd_m)

    sep_m      = sep_star_arr(ctx, ctx.jerusalem, ctx.moon,    ctx.regulus, t_m)
    moon_alt_m  = site.at(t_m).observe(ctx.moon).apparent().altaz(temperature_C=10, pressure_mbar=1013)[0].degrees
    reg_alt_m   = site.at(t_m).observe(ctx.regulus).apparent().altaz(temperature_C=10, pressure_mbar=1013)[0].degrees
    sun_alt_m   = site.at(t_m).observe(ctx.sun).apparent().altaz(temperature_C=10, pressure_mbar=1013)[0].degrees

    vis_m = (moon_alt_m > 0) & (reg_alt_m > 0) & (sun_alt_m < 0)
    sep_vis_m = np.where(vis_m, sep_m, np.inf)
    best_jd = jd_m[int(np.argmin(sep_vis_m))] if np.any(vis_m) else jd_hr[best_hr]

    t_best = ctx.ts.tt_jd(np.array([best_jd]))

    moon_app = site.at(t_best).observe(ctx.moon).apparent()
    moon_alt, moon_az, _ = moon_app.altaz(temperature_C=10, pressure_mbar=1013)
    moon_ecl_lat, moon_ecl_lon, _ = moon_app.frame_latlon(ecliptic_frame)

    jup_app  = site.at(t_best).observe(ctx.jup).apparent()
    jup_alt,  jup_az,  _ = jup_app.altaz(temperature_C=10, pressure_mbar=1013)
    jup_ecl_lat,  jup_ecl_lon,  _ = jup_app.frame_latlon(ecliptic_frame)

    reg_app  = site.at(t_best).observe(ctx.regulus).apparent()
    reg_alt,  reg_az,  _ = reg_app.altaz(temperature_C=10, pressure_mbar=1013)
    reg_ecl_lat,  reg_ecl_lon,  _ = reg_app.frame_latlon(ecliptic_frame)

    sun_app  = site.at(t_best).observe(ctx.sun).apparent()
    sun_alt2, _, _ = sun_app.altaz(temperature_C=10, pressure_mbar=1013)

    mr_sep = moon_app.separation_from(reg_app).degrees[0]
    jr_sep = jup_app.separation_from(reg_app).degrees[0]
    jm_sep = jup_app.separation_from(moon_app).degrees[0]

    earth_at   = ctx.earth.at(t_best)
    moon_elong = (earth_at.observe(ctx.moon).apparent()
                  .separation_from(earth_at.observe(ctx.sun).apparent()).degrees[0])
    illumination = (1 - np.cos(np.radians(moon_elong))) / 2 * 100

    t_scalar = ctx.ts.tt_jd(best_jd)
    print()
    print(f"  Time (TT)          : {fmt(t_scalar, hhmm=True)}  (local {_lst_str(t_scalar, _JER_LON)})")
    print()
    print(f"  Moon–Regulus sep   : {mr_sep * 60:.2f}′  ({mr_sep:.4f}°)")
    print(f"  Jup–Regulus sep    : {jr_sep * 60:.2f}′  ({jr_sep:.4f}°)")
    print(f"  Jup–Moon sep       : {jm_sep:.4f}°")
    print()
    print(f"  Moon    : alt {moon_alt.degrees[0]:+.2f}°  az {moon_az.degrees[0]:.2f}° ({_compass_pt(moon_az.degrees[0])})"
          f"  ecl lon {moon_ecl_lon.degrees[0]:.3f}°  lat {moon_ecl_lat.degrees[0]:+.3f}°"
          f"  phase {illumination:.1f}%  (elong {moon_elong:.1f}°)")
    print(f"  Jupiter : alt {jup_alt.degrees[0]:+.2f}°  az {jup_az.degrees[0]:.2f}° ({_compass_pt(jup_az.degrees[0])})"
          f"  ecl lon {jup_ecl_lon.degrees[0]:.3f}°  lat {jup_ecl_lat.degrees[0]:+.3f}°")
    print(f"  Regulus : alt {reg_alt.degrees[0]:+.2f}°  az {reg_az.degrees[0]:.2f}° ({_compass_pt(reg_az.degrees[0])})"
          f"  ecl lon {reg_ecl_lon.degrees[0]:.3f}°  lat {reg_ecl_lat.degrees[0]:+.3f}°")
    print(f"  Sun     : alt {sun_alt2.degrees[0]:+.2f}°")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Star of Bethlehem ephemeris")
    parser.add_argument(
        "--av", type=float, default=9.0, metavar="DEG",
        help="Arcus visionis for Jupiter heliacal rising (default: 9.0°)",
    )
    args = parser.parse_args()

    print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
    eph = load("de422.bsp")
    ts  = load.timescale()
    print("Loaded.\n")

    t_start = ts.tt(-2, 1, 1)
    t_end   = ts.tt(0, 1, 1)
    n_days  = int(round(t_end.tt - t_start.tt)) + 1
    jd_daily = np.linspace(t_start.tt, t_end.tt, n_days)
    times_d  = ts.tt_jd(jd_daily)

    print(f"Daily scan: {n_days} epochs  ({fmt(t_start)} – {fmt(t_end)})\n", flush=True)
    print("Computing ecliptic longitudes & separations (vectorised)…", flush=True)

    earth   = eph["earth"]
    sun     = eph["sun"]
    jup     = eph["jupiter barycenter"]
    ven     = eph["venus barycenter"]
    moon    = eph["moon"]
    regulus = Star(ra_hours=(10, 8, 22.311), dec_degrees=(11, 58, 1.95))
    babylon  = wgs84.latlon(32.5427 * N, 44.4215 * E)
    jerusalem = wgs84.latlon(31.7683 * N, 35.2137 * E)

    # Build a temporary partial ctx just for the precomputation helpers
    _ctx0 = Ctx(
        ts=ts, eph=eph, earth=earth, sun=sun, jup=jup, ven=ven,
        moon=moon, regulus=regulus, babylon=babylon, jerusalem=jerusalem,
        jd_daily=jd_daily, n_days=n_days, times_d=times_d,
        jup_lon_jer=np.empty(0), sun_lon_jer=np.empty(0),
        jup_elong_jer=np.empty(0), morning_jer=np.empty(0, dtype=bool),
    )
    jup_lon_jer  = ecl_lon_arr(_ctx0, jerusalem, jup, times_d)
    sun_lon_jer  = ecl_lon_arr(_ctx0, jerusalem, sun, times_d)
    jup_elong_jer = elong_arr(_ctx0, jerusalem, jup, times_d)
    morning_jer  = ((jup_lon_jer - sun_lon_jer) % 360) > 180

    print("Done.\n")

    ctx = Ctx(
        ts=ts, eph=eph, earth=earth, sun=sun, jup=jup, ven=ven,
        moon=moon, regulus=regulus, babylon=babylon, jerusalem=jerusalem,
        jd_daily=jd_daily, n_days=n_days, times_d=times_d,
        jup_lon_jer=jup_lon_jer, sun_lon_jer=sun_lon_jer,
        jup_elong_jer=jup_elong_jer, morning_jer=morning_jer,
    )

    event_1_jv_conjunctions(ctx)
    t_jr2_jd, t_jr3_jd = event_2_jr_conjunctions(ctx)
    event_3_moon_at_regulus(ctx, t_jr2_jd)
    event_4_stationary_points(ctx)
    event_regulus_heliacal_rising(ctx)
    event_5_heliacal_rising(ctx, args.av)
    event_6_weekly_altaz(ctx, args.av)
    event_7_lunar_eclipse(ctx)
    event_moon_regulus_occultations(ctx)
    event_8_moon_regulus_after_jr3(ctx, t_jr3_jd)

    print()
    print(SEP)
    print("NOTES")
    print(SEP)
    print(f"""
  All dates proleptic Gregorian.  Astronomical year numbering:
    3 BC = year -2,  2 BC = year -1,  1 BC = year 0.

  Positions are geocentric apparent (includes aberration, refraction ignored).
  Ecliptic frame: J2000 mean ecliptic and equinox.
  Jupiter arcus visionis threshold: {args.av}°.
  DE422 ephemeris, ΔT from Skyfield built-in tables.

  Observer coordinates:
    Babylon    32.54°N  44.42°E
    Jerusalem  31.77°N  35.21°E
""")


if __name__ == "__main__":
    main()
