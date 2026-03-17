#!/usr/bin/env python3
"""
gibeon_eclipse.py

Search for "horizontal eclipses" (selenelions) visible from Gibeon:
lunar eclipses where the eclipsed moon and the sun are both above the
apparent horizon at the same time, on opposite sides of the sky.

Biblical reference: Joshua 10:12-13 — "Sun, stand still over Gibeon,
and you, Moon, over the Valley of Aijalon."

Period: 1253–1175 BC (astronomical years -1252 to -1174)
Observer: Gibeon (el-Jib), 31.835°N 35.183°E, ~840 m elevation

Eclipse detection uses Skyfield's eclipselib (adapts ESAA 11.2.3).
Shadow geometry for contact-time bisection uses the same formulas
as eclipselib (Danjon enlargement, ERAD equatorial radius, Meeus).

ΔT UNNOMINALTY
--------------
Terrestrial Time (TT) and Universal Time (UT1) differ by ΔT ≈ 8 hours
for this period. Crucially, ΔT is uncertain: Earth's rotation rate has
irregular fluctuations that cannot be reconstructed perfectly from
historical records. Morrison & Stephenson (2004) estimate ±0.5 h for
1253–1175 BC.

Shifting ΔT by δ hours is geometrically identical to rotating Earth by
δ×15°, i.e. moving the observer δ×15° of longitude. So an eclipse
"visible within ±σ of ΔT" means visible from somewhere within ±σ×15°
of Gibeon's longitude — still broadly in the Levant / Mesopotamia.

Three tiers are reported:
  NOMINAL    : visible at nominal Skyfield ΔT               (δ = 0)
  POSSIBLE   : requires |δ| ≤ SIGMA_1 (±0.5 h, ±7.5° lon)
  SPECULATIVE: requires |δ| ≤ SIGMA_2 (±1 h, ±15° lon)
"""

import argparse
import numpy as np
from skyfield.api import load, wgs84, N, E
from skyfield import eclipselib

# ---------------------------------------------------------------------------
# Constants — match Skyfield / eclipselib exactly
# ---------------------------------------------------------------------------
ERAD_KM    = 6378.1366   # Earth equatorial radius (Skyfield's ERAD / 1e3)
MOON_R_KM  = 1737.1      # lunar radius used by eclipselib
SOLAR_R_KM = 696340.0    # solar radius used by eclipselib

GIBEON_LAT  = 31.835     # °N  (el-Jib, Israel)
GIBEON_LON  = 35.183     # °E
GIBEON_ELEV = 840        # m

ATMO_TEMP_C   = 15.0     # standard atmosphere for Skyfield refraction
ATMO_PRESS_MB = 1013.25

# Apparent altitude threshold: -0.25° lets the disc's upper limb just clear
# the visual horizon (sun/moon angular radius ≈ 0.25°).
MIN_ALT_DEG = -0.25

# Western horizon depression due to the Valley of Aijalon.
#
# From Gibeon (840 m, 31.835°N 35.183°E) the Aijalon valley floor lies
# ~22 km to the west at ~175 m elevation.  Correcting for Earth curvature:
#
#   dip = (840 − 175)/22000 − 22000/(2 × 6371000)  ≈  0.0285 rad  ≈ 1.6°
#
# We use 1.5° (conservative, allowing for possible intermediate ridges).
# This depresses the effective horizon for objects setting in the western
# sector, which is exactly where the biblical text places the Moon
# ("Moon over the Valley of Aijalon").
AIJALON_DEP_DEG = 1.5   # additional depression in the western sector
AIJALON_AZ_LO   = 210.0 # azimuth range of the "western" sector (°)
AIJALON_AZ_HI   = 330.0

# ΔT uncertainty tiers (Morrison & Stephenson 2004, rough estimates for period)
SIGMA_1H = 0.5   # ±1 tier: ~0.5 h
SIGMA_2H = 1.0   # ±2 tier: ~1 h

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def era(y: int) -> str:
    return f"{-y + 1} BC" if y <= 0 else f"{y} AD"


def fmt(t, hhmm: bool = False) -> str:
    y, mo, d, H, Mi, S = t.tt_calendar()
    s = f"{int(d):2d} {MONTHS[mo-1]} {era(y)}"
    if hhmm:
        fh = H + Mi / 60 + S / 3600
        s += f"  {fh:05.2f}h TT"
    return s


def shadow_params(t, earth, moon_body, sun_body):
    """
    Return (closest_approach_rad, umbra_r_rad, moon_r_rad) using the same
    formulas as Skyfield's eclipselib (Danjon 1% enlargement, ERAD).
    """
    geo_moon = earth.at(t).observe(moon_body)
    geo_sun  = earth.at(t).observe(sun_body)
    d_moon   = geo_moon.distance().km
    d_sun    = geo_sun.distance().km

    mv = geo_moon.position.km
    sv = geo_sun.position.km
    moon_hat = mv / np.linalg.norm(mv, axis=0)
    sun_hat  = sv / np.linalg.norm(sv,  axis=0)

    dot = np.clip(np.einsum('i...,i...->...', moon_hat, -sun_hat), -1.0, 1.0)
    ca  = np.arccos(dot)

    pi_m  = ERAD_KM    / d_moon
    pi_s  = ERAD_KM    / d_sun
    s_s   = SOLAR_R_KM / d_sun
    pi_1  = 1.01 * pi_m          # Danjon enlargement

    return ca, pi_1 + pi_s - s_s, MOON_R_KM / d_moon


def bisect_contact(ts, earth, moon_body, sun_body, t_lo, t_hi,
                   threshold_fn, rising=True, tol_min=0.5):
    """
    Find the zero-crossing of threshold_fn by bisection.
    Invariant: fn < 0 at lo side (rising=True) or > 0 at lo side (rising=False).
    """
    lo, hi = t_lo.tt, t_hi.tt
    for _ in range(50):
        mid_tt = (lo + hi) / 2
        t_mid  = ts.tt_jd(mid_tt)
        ca, ur, mr = shadow_params(t_mid, earth, moon_body, sun_body)
        val = threshold_fn(ca, ur, mr)
        if (val < 0) == rising:
            lo = mid_tt
        else:
            hi = mid_tt
        if (hi - lo) * 1440 < tol_min:
            break
    return ts.tt_jd((lo + hi) / 2)


def altaz_deg(observer, body, t):
    """Return (altitude_deg, azimuth_deg) with standard atmospheric refraction."""
    astr = observer.at(t).observe(body).apparent()
    alt, az, _ = astr.altaz(temperature_C=ATMO_TEMP_C, pressure_mbar=ATMO_PRESS_MB)
    return alt.degrees, az.degrees


def sep_deg(observer, body1, body2, t) -> float:
    """Angular separation in degrees between two bodies as seen from observer."""
    a1 = observer.at(t).observe(body1).apparent()
    a2 = observer.at(t).observe(body2).apparent()
    return a1.separation_from(a2).degrees


def moon_angular_diam_arcmin(earth, moon_body, t) -> float:
    """Angular diameter of the Moon in arcminutes."""
    dist_km = earth.at(t).observe(moon_body).distance().km
    return 2.0 * np.degrees(np.arcsin(MOON_R_KM / dist_km)) * 60.0


def moon_phase_info(earth, moon_body, sun_body, t):
    """Return (phase_angle_deg, illumination_fraction, approx_V_magnitude).

    Phase angle is the Sun-Moon-Earth angle.
    Magnitude formula: Allen's Astrophysical Quantities (approximate).
    """
    geo_moon = earth.at(t).observe(moon_body).position.km
    geo_sun  = earth.at(t).observe(sun_body).position.km
    moon_to_earth = -geo_moon
    moon_to_sun   = geo_sun - geo_moon
    cos_i = np.dot(moon_to_earth, moon_to_sun) / (
        np.linalg.norm(moon_to_earth) * np.linalg.norm(moon_to_sun))
    i_deg  = np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))
    illum  = (1.0 + np.cos(np.radians(i_deg))) / 2.0
    vmag   = -12.73 + 0.026 * i_deg + 4e-9 * i_deg ** 4
    return i_deg, illum, vmag


def horizon_min(az_deg: float) -> float:
    """Effective horizon altitude threshold for a given azimuth.

    In the western sector the Valley of Aijalon depresses the horizon by
    AIJALON_DEP_DEG below the standard flat-terrain value.
    """
    if AIJALON_AZ_LO <= az_deg <= AIJALON_AZ_HI:
        return MIN_ALT_DEG - AIJALON_DEP_DEG
    return MIN_ALT_DEG


def joint_fn(observer, sun_body, moon_body, t):
    """min margin above effective horizon; positive = both visible."""
    sa, s_az = altaz_deg(observer, sun_body,  t)
    ma, m_az = altaz_deg(observer, moon_body, t)
    return min(sa - horizon_min(s_az), ma - horizon_min(m_az))


def bisect_joint(observer, sun_body, moon_body, ts, lo_tt, hi_tt, rising):
    """Bisect the edge of the joint-visibility window to ~0.25 min precision."""
    for _ in range(30):
        mid = (lo_tt + hi_tt) / 2
        val = joint_fn(observer, sun_body, moon_body, ts.tt_jd(mid))
        if (val < 0) == rising:
            lo_tt = mid
        else:
            hi_tt = mid
        if (hi_tt - lo_tt) * 1440 < 0.25:
            break
    return (lo_tt + hi_tt) / 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Find selenelions (horizontal eclipses) visible from Gibeon.")
    parser.add_argument("--start", type=int, default=1253, metavar="BC_YEAR",
                        help="Start year in BC (default: 1253)")
    parser.add_argument("--end",   type=int, default=1175, metavar="BC_YEAR",
                        help="End year in BC, exclusive (default: 1175)")
    args = parser.parse_args()

    # Convert BC years to astronomical years (1 BC = 0, 2 BC = -1, …)
    ast_start = -(args.start - 1)
    ast_end   = -(args.end   - 1)
    period_label = f"{args.start}-{args.end} BC"

    print("Loading DE422 ephemeris (will download ~623 MB on first run)…", flush=True)
    eph = load("de422.bsp")
    ts  = load.timescale()
    print("Loaded.\n")

    earth     = eph["earth"]
    moon_body = eph["moon"]
    sun_body  = eph["sun"]

    gibeon   = wgs84.latlon(GIBEON_LAT * N, GIBEON_LON * E, elevation_m=GIBEON_ELEV)
    observer = earth + gibeon

    t_start = ts.tt(ast_start, 1, 1)
    t_end   = ts.tt(ast_end,   1, 1)

    # -----------------------------------------------------------------------
    # Step 1: eclipselib finds all validated lunar eclipses
    # -----------------------------------------------------------------------
    print(f"Finding all lunar eclipses {period_label} (eclipselib)…", flush=True)
    t_ecl, y_ecl, details = eclipselib.lunar_eclipses(t_start, t_end, eph)

    umbral_mask = y_ecl > 0
    t_ecl  = t_ecl[umbral_mask]
    y_ecl  = y_ecl[umbral_mask]
    u_mags = details["umbral_magnitude"][umbral_mask]
    n_ecl  = len(t_ecl)
    print(f"  Umbral eclipses: {n_ecl}  ({sum(y_ecl==2)} total, {sum(y_ecl==1)} partial)\n")

    # -----------------------------------------------------------------------
    # Step 2: process each eclipse
    # -----------------------------------------------------------------------
    def umbra_f(ca, ur, mr): return ur + mr - ca   # > 0 when overlapping umbra
    def total_f(ca, ur, mr): return ur - mr - ca   # > 0 when fully inside umbra

    results = []
    print(f"Checking horizon visibility (this may take several minutes)…", flush=True)

    for idx in range(n_ecl):
        t_fm  = t_ecl[idx]
        etype = eclipselib.LUNAR_ECLIPSES[y_ecl[idx]]
        mag   = u_mags[idx]

        # --- moon size & preceding-night brightness -------------------------
        moon_diam_fm = moon_angular_diam_arcmin(earth, moon_body, t_fm)
        t_prev       = ts.tt_jd(t_fm.tt - 1.0)   # ~1 day before eclipse max
        prev_i, prev_illum, prev_vmag = moon_phase_info(earth, moon_body, sun_body, t_prev)
        prev_diam = moon_angular_diam_arcmin(earth, moon_body, t_prev)

        # --- contact times --------------------------------------------------
        dt   = 6.0 / 24.0
        t_lo = ts.tt_jd(t_fm.tt - dt)
        t_hi = ts.tt_jd(t_fm.tt + dt)

        t_u1 = bisect_contact(ts, earth, moon_body, sun_body,
                               t_lo, t_fm, umbra_f, rising=True)
        t_u4 = bisect_contact(ts, earth, moon_body, sun_body,
                               t_fm, t_hi, umbra_f, rising=False)
        t_u2 = t_u3 = None
        if etype == "Total":
            t_u2 = bisect_contact(ts, earth, moon_body, sun_body,
                                   t_u1, t_fm, total_f, rising=True)
            t_u3 = bisect_contact(ts, earth, moon_body, sun_body,
                                   t_fm, t_u4, total_f, rising=False)
        eclip_dur = (t_u4.tt - t_u1.tt) * 1440

        # --- NOMINAL check: 1-min sampling of umbral window -----------------
        n_nom   = max(int(eclip_dur) + 2, 2)
        nom_tt  = np.linspace(t_u1.tt, t_u4.tt, n_nom)
        nom_t   = ts.tt_jd(nom_tt)
        s_altaz = [altaz_deg(observer, sun_body,  t) for t in nom_t]
        m_altaz = [altaz_deg(observer, moon_body, t) for t in nom_t]
        s_alts  = np.array([x[0] for x in s_altaz])
        m_alts  = np.array([x[0] for x in m_altaz])
        s_azs   = np.array([x[1] for x in s_altaz])
        m_azs   = np.array([x[1] for x in m_altaz])
        s_min   = np.array([horizon_min(az) for az in s_azs])
        m_min   = np.array([horizon_min(az) for az in m_azs])
        nom_jnt = (s_alts > s_min) & (m_alts > m_min)

        if np.any(nom_jnt):
            # Refine start/end of joint window with bisection
            i_s = int(np.argmax(nom_jnt))
            i_e = int(len(nom_jnt) - 1 - np.argmax(nom_jnt[::-1]))

            vs_tt = (bisect_joint(observer, sun_body, moon_body, ts,
                                   nom_tt[i_s-1], nom_tt[i_s], rising=True)
                     if i_s > 0 else nom_tt[i_s])
            ve_tt = (bisect_joint(observer, sun_body, moon_body, ts,
                                   nom_tt[i_e], nom_tt[i_e+1], rising=False)
                     if i_e < len(nom_tt)-1 else nom_tt[i_e])

            t_vs = ts.tt_jd(vs_tt)
            t_ve = ts.tt_jd(ve_tt)
            vis_dur = (ve_tt - vs_tt) * 1440

            ms_alt, ms_az = altaz_deg(observer, moon_body, t_vs)
            ss_alt, ss_az = altaz_deg(observer, sun_body,  t_vs)
            me_alt, me_az = altaz_deg(observer, moon_body, t_ve)
            se_alt, se_az = altaz_deg(observer, sun_body,  t_ve)

            if ss_az > 180:
                continue   # Sun must be in east half of sky for Joshua 10:13

            results.append(dict(
                tier="NOMINAL", delta_h=0.0,
                t_fm=t_fm, etype=etype, mag=mag,
                t_u1=t_u1, t_u2=t_u2, t_u3=t_u3, t_u4=t_u4,
                eclip_dur=eclip_dur,
                t_vs=t_vs, t_ve=t_ve, vis_dur=vis_dur,
                ms_alt=ms_alt, ms_az=ms_az, ss_alt=ss_alt, ss_az=ss_az,
                me_alt=me_alt, me_az=me_az, se_alt=se_alt, se_az=se_az,
                moon_diam_fm=moon_diam_fm,
                prev_i=prev_i, prev_illum=prev_illum,
                prev_vmag=prev_vmag, prev_diam=prev_diam,
            ))
            continue

        # --- POSSIBLE / SPECULATIVE: precompute extended altitude array -----
        # Shifting ΔT by δ hours ≡ sliding the eclipse window by δ hours in
        # the precomputed altitude array.  We need altitudes from
        # (t_u1 - SIGMA_2) to (t_u4 + SIGMA_2) to cover all offsets.
        ext_h   = SIGMA_2H + 0.5
        n_ext   = max(int((eclip_dur/60 + 2*ext_h) * 60 / 5) + 2, 10)
        ext_tt  = np.linspace(t_u1.tt - ext_h/24, t_u4.tt + ext_h/24, n_ext)
        ext_t   = ts.tt_jd(ext_tt)
        se_altaz = [altaz_deg(observer, sun_body,  t) for t in ext_t]
        me_altaz = [altaz_deg(observer, moon_body, t) for t in ext_t]
        se_arr  = np.array([x[0] for x in se_altaz])
        me_arr  = np.array([x[0] for x in me_altaz])
        se_azs  = np.array([x[1] for x in se_altaz])
        me_azs  = np.array([x[1] for x in me_altaz])
        se_min  = np.array([horizon_min(az) for az in se_azs])
        me_min  = np.array([horizon_min(az) for az in me_azs])
        jnt_ext = (se_arr > se_min) & (me_arr > me_min)

        if not np.any(jnt_ext):
            continue   # No visibility possible even with maximum ΔT shift

        # Scan offsets from smallest to largest |δ|, stop at first hit
        best_delta = None
        for mag_h in np.arange(0.25, SIGMA_2H + 0.125, 0.25):
            for sign in (+1, -1):
                d = sign * mag_h
                # With ΔT offset +d, the eclipse window in the ext array is
                # [t_u1 - d, t_u4 - d]:
                lo = t_u1.tt - d / 24
                hi = t_u4.tt - d / 24
                mask = (ext_tt >= lo - 1e-9) & (ext_tt <= hi + 1e-9)
                if np.any(jnt_ext[mask]):
                    best_delta = d
                    break
            if best_delta is not None:
                break

        if best_delta is None:
            continue

        tier = "POSSIBLE" if abs(best_delta) <= SIGMA_1H else "SPECULATIVE"

        # Find the visibility window at this offset
        d   = best_delta
        lo  = t_u1.tt - d / 24
        hi  = t_u4.tt - d / 24
        mask = (ext_tt >= lo - 1e-9) & (ext_tt <= hi + 1e-9)
        seg  = jnt_ext & mask
        i_s  = int(np.argmax(seg))
        i_e  = int(len(seg) - 1 - np.argmax(seg[::-1]))

        # ext_tt[i] is the shifted TT time (= actual TT - delta).
        # Actual TT of eclipse event = shifted TT + delta.
        vs_shifted = ext_tt[i_s]
        ve_shifted = ext_tt[i_e]
        vs_tt_ecl  = vs_shifted + d / 24   # actual eclipse TT
        ve_tt_ecl  = ve_shifted + d / 24
        vis_dur    = (ve_tt_ecl - vs_tt_ecl) * 1440

        # Azimuths are computed at the *shifted* TT (that's what the observer sees)
        t_vs_sh = ts.tt_jd(vs_shifted)
        t_ve_sh = ts.tt_jd(ve_shifted)
        ms_alt, ms_az = altaz_deg(observer, moon_body, t_vs_sh)
        ss_alt, ss_az = altaz_deg(observer, sun_body,  t_vs_sh)
        me_alt, me_az = altaz_deg(observer, moon_body, t_ve_sh)
        se_alt, se_az = altaz_deg(observer, sun_body,  t_ve_sh)

        if ss_az > 180:
            continue   # Sun must be in east half of sky for Joshua 10:13

        results.append(dict(
            tier=tier, delta_h=best_delta,
            t_fm=t_fm, etype=etype, mag=mag,
            t_u1=t_u1, t_u2=t_u2, t_u3=t_u3, t_u4=t_u4,
            eclip_dur=eclip_dur,
            t_vs=ts.tt_jd(vs_tt_ecl), t_ve=ts.tt_jd(ve_tt_ecl),
            vis_dur=vis_dur,
            ms_alt=ms_alt, ms_az=ms_az, ss_alt=ss_alt, ss_az=ss_az,
            me_alt=me_alt, me_az=me_az, se_alt=se_alt, se_az=se_az,
            moon_diam_fm=moon_diam_fm,
            prev_i=prev_i, prev_illum=prev_illum,
            prev_vmag=prev_vmag, prev_diam=prev_diam,
        ))

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    certain    = [r for r in results if r["tier"] == "NOMINAL"]
    possible   = [r for r in results if r["tier"] == "POSSIBLE"]
    speculative= [r for r in results if r["tier"] == "SPECULATIVE"]

    TIER_META = {
        "NOMINAL":     ("NOMINAL     (δ = 0 h,    nominal ΔT)",     certain),
        "POSSIBLE":    (f"POSSIBLE    (|δ| ≤ {SIGMA_1H:.1f} h,  ±{SIGMA_1H*15:.0f}° longitude)", possible),
        "SPECULATIVE": (f"SPECULATIVE (|δ| ≤ {SIGMA_2H:.1f} h,  ±{SIGMA_2H*15:.0f}° longitude)", speculative),
    }

    print()
    print("=" * 72)
    print(f"  HORIZONTAL ECLIPSES (SELENELIONS) — GIBEON  {period_label}")
    print(f"  Observer: {GIBEON_LAT}°N {GIBEON_LON}°E  elev {GIBEON_ELEV} m")
    print(f"  Terrain: western horizon (Aijalon valley) depressed {AIJALON_DEP_DEG}°  "
          f"(az {AIJALON_AZ_LO:.0f}°–{AIJALON_AZ_HI:.0f}°)")
    print(f"  ΔT uncertainty model: ±{SIGMA_1H:.0f} h (tier 1), ±{SIGMA_2H:.0f} h (tier 2)")
    print("=" * 72)
    print()
    print(f"  {'NOMINAL':15s} {len(certain):4d} events  (nominal ΔT)")
    print(f"  {'POSSIBLE':15s} {len(possible):4d} additional events  (|δ| ≤ {SIGMA_1H:.0f} h, ≡ ≤{SIGMA_1H*15:.0f}° lon)")
    print(f"  {'SPECULATIVE':15s} {len(speculative):4d} additional events  (|δ| ≤ {SIGMA_2H:.0f} h, ≡ ≤{SIGMA_2H*15:.0f}° lon)")
    print(f"  {'TOTAL':15s} {len(results):4d} events")

    event_n = 0
    for tier_key in ("NOMINAL", "POSSIBLE", "SPECULATIVE"):
        label, tier_list = TIER_META[tier_key]
        if not tier_list:
            continue
        print()
        print("─" * 72)
        print(f"  ── {label} ──")
        print("─" * 72)

        for r in tier_list:
            event_n += 1
            d  = r["delta_h"]
            print()
            print(f"  Event {event_n}: {r['etype'].upper()} LUNAR ECLIPSE  |  "
                  f"Max eclipse: {fmt(r['t_fm'], hhmm=True)}")
            print(f"  Umbral magnitude: {r['mag']:.3f}")

            if tier_key != "NOMINAL":
                sign_word = "larger" if d > 0 else "smaller"
                eff_lon   = GIBEON_LON - d * 15
                print(f"  ΔT offset required: {abs(d):.2f} h {sign_word} than nominal")
                print(f"  (≡ observer {abs(d*15):.0f}° {'west' if d>0 else 'east'} of Gibeon "
                      f"→ effective longitude {eff_lon:.1f}°E)")

            print()
            print(f"  Eclipse contacts (TT):")
            print(f"    Umbra entry   (U1): {fmt(r['t_u1'], hhmm=True)}")
            if r["etype"] == "Total":
                print(f"    Totality start (U2): {fmt(r['t_u2'], hhmm=True)}")
                print(f"    Totality end   (U3): {fmt(r['t_u3'], hhmm=True)}")
            print(f"    Umbra exit    (U4): {fmt(r['t_u4'], hhmm=True)}")
            print(f"    Umbral duration: {r['eclip_dur']:.0f} min")
            print()
            print(f"  Both sun & moon visible:")
            print(f"    Window start: {fmt(r['t_vs'], hhmm=True)}")
            print(f"    Window end:   {fmt(r['t_ve'], hhmm=True)}")
            print(f"    Duration:     {r['vis_dur']:.0f} min")
            print()
            az_diff  = abs(r["ms_az"] - r["ss_az"])
            opposite = az_diff > 90
            side     = "opposite sides of sky" if opposite else "same side of sky"
            print(f"  Azimuths at window start  ({side}):")
            print(f"    Moon: az {r['ms_az']:6.1f}°  alt {r['ms_alt']:+5.1f}°")
            print(f"    Sun:  az {r['ss_az']:6.1f}°  alt {r['ss_alt']:+5.1f}°")
            print(f"  Azimuths at window end:")
            print(f"    Moon: az {r['me_az']:6.1f}°  alt {r['me_alt']:+5.1f}°")
            print(f"    Sun:  az {r['se_az']:6.1f}°  alt {r['se_alt']:+5.1f}°")
            print()
            print(f"  Moon at eclipse maximum:")
            print(f"    Angular diameter: {r['moon_diam_fm']:.1f}′  "
                  f"({'large' if r['moon_diam_fm'] >= 32 else 'small'} disc)")
            print(f"  Moon on preceding evening (~24 h before max eclipse):")
            print(f"    Angular diameter: {r['prev_diam']:.1f}′")
            print(f"    Phase angle:      {r['prev_i']:.1f}°  "
                  f"(illumination {r['prev_illum']*100:.1f}%)")
            print(f"    Apparent magnitude ≈ {r['prev_vmag']:.1f}  "
                  f"({'nearly full' if r['prev_illum'] > 0.95 else 'gibbous'})")

    print()
    print("=" * 72)
    print(f"  Total events reported: {len(results)}")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
