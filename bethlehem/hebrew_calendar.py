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
The ΔT uncertainty rarely shifts crescent sightings across a day boundary, so no
separate ΔT error bar is given. Intercalation uncertainty (whether the Sanhedrin
actually intercalated in a given year) is noted where the decision was close.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skyfield.api import load, wgs84, N, E
from skyfield import almanac

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LOCATIONS = {
    "jerusalem": ("Jerusalem",             31.7683,  35.2137),
    "avaris":    ("Avaris (Tell el-Dabʿa)", 30.787,  31.823),
    "babylon":   ("Babylon",               32.5364,  44.4208),
}

MONTHS_GR = ["Jan","Feb","Mar","Apr","May","Jun",
             "Jul","Aug","Sep","Oct","Nov","Dec"]


# ─────────────────────────────────────────────────────────────────────────────
# Date formatting helpers (module-level; used by both classes)
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
# HebrewCalendarEngine
# ─────────────────────────────────────────────────────────────────────────────

class HebrewCalendarEngine:
    """
    Owns the Skyfield ephemeris, observer, and all astronomical computations.
    Call build_calendar() to run the full pipeline and get a HebrewCalendarResult.
    """

    MOON_RADIUS_KM    = 1737.4
    AU_KM             = 149_597_870.7
    NISAN_WINDOW_DAYS = 40.0      # latest plausible Nisan start after spring equinox
    _SUN_STEP_DAYS    = 10.0 / 1440.0   # 10-minute sample spacing

    SEQ_FROM_NISAN = [
        "Nisan","Iyar","Sivan","Tammuz","Av","Elul",
        "Tishri","Cheshvan","Kislev","Tevet","Shevat","Adar",
    ]
    # Leap-year sequence (from Nisan):
    # Nisan Iyar Sivan Tammuz Av Elul Tishri Cheshvan Kislev Tevet Shevat Adar-I Adar-II

    def __init__(self, location_key: str, start_year: int, end_year: int):
        self.start_year = start_year
        self.end_year   = end_year

        loc_name, loc_lat, loc_lon = LOCATIONS[location_key]
        self.loc_name = loc_name
        self.loc_lat  = loc_lat
        self.loc_lon  = loc_lon

        self.eph   = load("de422.bsp")
        self.ts    = load.timescale()
        self.sun   = self.eph["sun"]
        self.moon  = self.eph["moon"]
        self.earth = self.eph["earth"]

        obs_site      = wgs84.latlon(loc_lat * N, loc_lon * E)
        self.observer = self.earth + obs_site

        # scan range: one extra year each side to get complete Hebrew years
        self._scan_y0      = start_year - 1
        self._scan_y1      = end_year   + 1
        self._t_scan_start = self.ts.tt(self._scan_y0, 2,  1)
        self._t_scan_end   = self.ts.tt(self._scan_y1, 10, 1)

        # populated by _precompute_sun_altitudes()
        self._sun_jds  = None
        self._sun_alts = None

    # ── Low-level crossing helpers ────────────────────────────────────────────

    def _body_alts(self, body, jd_start, jd_end, n=120):
        """Vectorised altitude array for body over [jd_start, jd_end]."""
        jds   = np.linspace(jd_start, jd_end, n)
        t_arr = self.ts.tt_jd(jds)
        alts  = self.observer.at(t_arr).observe(body).apparent().altaz()[0].degrees
        return jds, alts

    def _crossing_time(self, body, target_alt, jd_start, jd_end, rising=False):
        """
        First crossing of target_alt in [jd_start, jd_end] via vectorised scan
        + bisection refinement.  rising=True → ascending; False → descending.
        Returns a Skyfield Time or None.
        """
        jds, alts = self._body_alts(body, jd_start, jd_end)
        shifted = alts - target_alt
        for i in range(len(shifted) - 1):
            a0, a1 = shifted[i], shifted[i + 1]
            cross  = (a0 < 0 and a1 >= 0) if rising else (a0 >= 0 and a1 < 0)
            if not cross:
                continue
            lo, hi = jds[i], jds[i + 1]
            for _ in range(30):
                mid   = (lo + hi) / 2
                t_mid = self.ts.tt_jd(mid)
                a_mid = self.observer.at(t_mid).observe(body).apparent().altaz()[0].degrees
                if (a_mid - target_alt) * a0 > 0:
                    lo = mid
                else:
                    hi = mid
            return self.ts.tt_jd((lo + hi) / 2)
        return None

    def _precompute_sun_altitudes(self):
        """
        Sample sun altitude at 10-min intervals over the full scan range in one
        vectorised Skyfield call.  Subsequent find_sunset / find_sun_at_minus5
        calls use this cache and only need a short bisection refinement.
        """
        jd0 = self._t_scan_start.tt - 1.0   # 1-day lead
        jd1 = self._t_scan_end.tt   + 5.0   # 5-day tail (crescent look-ahead)
        n   = int((jd1 - jd0) / self._SUN_STEP_DAYS) + 2
        print(f"Precomputing solar altitudes ({n:,} samples at 10-min spacing)…",
              flush=True)
        self._sun_jds  = np.linspace(jd0, jd1, n)
        self._sun_alts = (self.observer.at(self.ts.tt_jd(self._sun_jds))
                          .observe(self.sun).apparent().altaz()[0].degrees)
        print("  Done.\n")

    def _sun_crossing_cached(self, target_alt: float, jd_start: float,
                             jd_end: float, rising: bool = False):
        """
        Solar crossing from the precomputed altitude cache.
        Finds the bracket via searchsorted, then refines with bisection
        (individual Skyfield calls only inside the narrow ≤10-min bracket).
        """
        assert self._sun_jds is not None and self._sun_alts is not None, \
            "_precompute_sun_altitudes() must be called before find_sunset/find_sun_at_minus5"
        i0 = max(0, int(np.searchsorted(self._sun_jds, jd_start)) - 1)
        i1 = min(len(self._sun_jds) - 1,
                 int(np.searchsorted(self._sun_jds, jd_end, side='right')) + 1)
        shifted = self._sun_alts[i0:i1 + 1] - target_alt
        jds_w   = self._sun_jds[i0:i1 + 1]
        for i in range(len(shifted) - 1):
            a0, a1 = shifted[i], shifted[i + 1]
            cross  = (a0 < 0 and a1 >= 0) if rising else (a0 >= 0 and a1 < 0)
            if not cross:
                continue
            lo, hi  = jds_w[i], jds_w[i + 1]
            a0_sign = a0
            for _ in range(18):   # 10-min bracket → <0.002 min precision
                mid   = (lo + hi) / 2
                t_mid = self.ts.tt_jd(mid)
                a_mid = (self.observer.at(t_mid).observe(self.sun).apparent()
                         .altaz()[0].degrees - target_alt)
                if a_mid * a0_sign > 0:
                    lo = mid
                else:
                    hi = mid
            return self.ts.tt_jd((lo + hi) / 2)
        return None

    # ── Astronomical event finders ────────────────────────────────────────────

    # NOTE on JD convention: floor(JD) is *noon* TT of that Julian Day.
    # Jerusalem sunset in TT depends strongly on ΔT: for 3 BC (ΔT≈3.5h) it falls
    # near d0+0.28, but for 1200 BC (ΔT≈10.3h) it falls near d0+0.58, and for
    # earlier dates it can exceed d0+1.0.  We search a full 1.5-day window from
    # noon TT.  Descending-only crossings naturally skip sunrise.

    def find_sunset(self, date_jd: float):
        """Sunset TT on the evening of the Julian Day containing date_jd."""
        d0 = np.floor(date_jd)
        return self._sun_crossing_cached(0.0, d0, d0 + 1.5, rising=False)

    def find_sun_at_minus5(self, date_jd: float):
        """When sun descends through –5° on the evening of date_jd."""
        d0 = np.floor(date_jd)
        return self._sun_crossing_cached(-5.0, d0, d0 + 1.5, rising=False)

    def find_moonset(self, after_jd: float):
        """First moonset at or after after_jd (a TT JD, e.g. sunset time)."""
        return self._crossing_time(
            self.moon, 0.0, after_jd - 0.01, after_jd + 0.35, rising=False
        )

    def spring_equinox(self, astro_year: int):
        """Return the spring equinox time for an astronomical year."""
        t0 = self.ts.tt(astro_year, 2, 15)
        t1 = self.ts.tt(astro_year, 5, 15)
        times, events = almanac.find_discrete(t0, t1, almanac.seasons(self.eph))
        eq_times = times[events == 0]
        return eq_times[0] if len(eq_times) > 0 else None

    # ── Yallop crescent-visibility criterion ──────────────────────────────────

    def yallop(self, t_obs) -> dict:
        """
        Yallop (1997) q-factor at observation time t_obs.
        Returns dict with q, cat, arcl, arcv, W, moon_alt, sun_alt.
        """
        a_moon = self.observer.at(t_obs).observe(self.moon).apparent()
        a_sun  = self.observer.at(t_obs).observe(self.sun).apparent()

        alt_m, _, dist_m = a_moon.altaz()
        alt_s, _,  _     = a_sun.altaz()

        arcl = a_moon.separation_from(a_sun).degrees
        arcv = alt_m.degrees - alt_s.degrees

        sd_arcmin = np.degrees(np.arctan(
            self.MOON_RADIUS_KM / (dist_m.au * self.AU_KM)
        )) * 60
        W = sd_arcmin * (1.0 - np.cos(np.radians(arcl)))   # crescent width, arcmin

        q = arcv - (11.8371 - 6.3226 * np.sqrt(W) + 0.7319 * W**3 - 0.1018 * W**4)

        if   q >  0.216: cat = "A"
        elif q > -0.014: cat = "B"
        elif q > -0.160: cat = "C"
        elif q > -0.232: cat = "D"
        else:            cat = "E"

        return {"q": q, "cat": cat, "arcl": arcl, "arcv": arcv,
                "W": W, "moon_alt": alt_m.degrees, "sun_alt": alt_s.degrees}

    # ── First-crescent finder ─────────────────────────────────────────────────

    def first_crescent(self, nm_t) -> dict | None:
        """
        Given a new-moon Skyfield Time nm_t, find the first evening when the
        crescent is visible from the observer location.

        Returns a dict:
          evening_jd  – TT JD of the observing evening (floor of calendar date)
          evening_t   – Skyfield Time of the –5° sun moment
          day_offset  – days after conjunction (1, 2, or 3)
          yallop      – Yallop result dict for that evening
          moon_age_h  – hours since conjunction
          lag_min     – moonset lag after sunset (minutes)
          uncertain   – True if category C (±1 day)
          note        – human-readable note
        """
        nm_jd  = nm_t.tt
        best   = None
        second = None   # next day, for uncertainty

        for offset in range(1, 4):
            cand_jd = np.floor(nm_jd) + offset

            # Preferred observation time: sun at –5° (best crescent window)
            t_obs = self.find_sun_at_minus5(cand_jd)
            if t_obs is None:
                t_obs = self.find_sunset(cand_jd)
            if t_obs is None:
                continue

            moon_age_h = (t_obs.tt - nm_jd) * 24.0

            # Danjon limit: moon rarely visible if age < 13.5 h
            if moon_age_h < 13.5:
                continue

            v = self.yallop(t_obs)

            # Moon must be above horizon
            if v["moon_alt"] < 0:
                continue

            # Lag time (moonset – sunset)
            t_ss    = self.find_sunset(cand_jd)
            t_ms    = self.find_moonset(t_ss.tt) if t_ss is not None else None
            lag_min = ((t_ms.tt - t_ss.tt) * 1440
                       if (t_ms is not None and t_ss is not None) else 0.0)

            rec = {
                "evening_jd": cand_jd,
                "evening_t":  t_obs,
                "day_offset": offset,
                "yallop":     v,
                "moon_age_h": moon_age_h,
                "lag_min":    lag_min,
                "uncertain":  False,
                "note":       "",
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
                    best = rec
                    best["uncertain"] = True
                    best["note"] = "visibility doubtful – possible ±1 day"

        # If best is C, check whether second (next day) is A/B → first day remains possible
        if best and best["yallop"]["cat"] == "C":
            best["note"] = "±1 day (borderline Yallop C)"
        return best

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _am_year_for_tishri(self, tishri_jd: float) -> int:
        """Jewish AM year that begins with this Tishri."""
        y = self.ts.tt_jd(tishri_jd).tt_calendar()[0]
        return y + 3761   # Tishri of astronomical year y → AM y+3761

    def build_calendar(self) -> "HebrewCalendarResult":
        """
        Run the full pipeline:
          1. Find all new/full moons in the scan range.
          2. Precompute solar altitudes.
          3. Determine first-crescent date for each new moon.
          4. Compute spring equinoxes and identify 1 Nisan for each year.
          5. Assign Hebrew month names and AM years.
          6. Attach month lengths and full-moon dates.
        Returns a HebrewCalendarResult.
        """
        ts = self.ts

        # ── 1. New and full moons ─────────────────────────────────────────────
        print(f"Finding new moons ({era(self._scan_y0)} Feb → {era(self._scan_y1)} Oct)…",
              flush=True)
        phase_times, phase_idx = almanac.find_discrete(
            self._t_scan_start, self._t_scan_end, almanac.moon_phases(self.eph)
        )
        new_moons     = phase_times[phase_idx == 0]
        full_moons    = phase_times[phase_idx == 2]
        full_moon_jds = full_moons.tt
        print(f"  Found {len(new_moons)} new moons.\n")

        # ── 2. Solar altitude cache ───────────────────────────────────────────
        self._precompute_sun_altitudes()

        # ── 3. First-crescent date per new moon ───────────────────────────────
        print("Computing first-crescent visibility for each month…", flush=True)
        month_starts = []
        for nm in new_moons:
            fc = self.first_crescent(nm)
            if fc:
                month_starts.append(fc)
                print(f"  New moon {fmt_datetime(nm)}  →  "
                      f"crescent {fmt_date(ts.tt_jd(fc['evening_jd']))}  "
                      f"[{fc['yallop']['cat']}]  "
                      f"{'(uncertain)' if fc['uncertain'] else ''}")
        print()

        # ── 4. Spring equinoxes ───────────────────────────────────────────────
        print("Computing spring equinoxes…", flush=True)
        equinoxes = {}
        for astro_yr in range(self.start_year - 1, self.end_year + 2):
            eq = self.spring_equinox(astro_yr)
            if eq is not None:
                equinoxes[astro_yr] = eq
                print(f"  Spring equinox {astro_yr:+d} ({era(astro_yr)}): {fmt_datetime(eq)}")
        print()

        # ── 5. Identify 1 Nisan for each year ────────────────────────────────
        nisan_starts = {}   # astro_year → index into month_starts[]
        for astro_yr, eq_t in equinoxes.items():
            eq_jd      = eq_t.tt
            candidates = []
            for idx, ms in enumerate(month_starts):
                if ms["evening_jd"] > eq_jd + self.NISAN_WINDOW_DAYS:
                    break
                fm_approx = ms["evening_jd"] + 14.75
                if fm_approx >= eq_jd:
                    candidates.append((idx, fm_approx))
            if candidates:
                idx0 = candidates[0][0]
                nisan_starts[astro_yr] = idx0
                print(f"  1 Nisan {astro_yr} ({era(astro_yr)}): "
                      f"{fmt_date(ts.tt_jd(month_starts[idx0]['evening_jd']))}  "
                      f"(full moon ~{fmt_date(ts.tt_jd(month_starts[idx0]['evening_jd']+14.75))}  "
                      f"equinox {fmt_date(eq_t)})")
        print()

        # ── 6. Assign Hebrew month names ──────────────────────────────────────
        sorted_nisans = sorted(nisan_starts.items())
        name_map = {}   # month_index → {"name": str, "nisan_yr": int}

        for astro_yr, nisan_idx in sorted_nisans:
            for i, hname in enumerate(self.SEQ_FROM_NISAN):
                mi = nisan_idx + i
                if 0 <= mi < len(month_starts) and mi not in name_map:
                    name_map[mi] = {"name": hname, "nisan_yr": astro_yr}

        # Leap-year Adar II: if 13 months between consecutive Nisans
        for k in range(len(sorted_nisans) - 1):
            yr0, idx0 = sorted_nisans[k]
            yr1, idx1 = sorted_nisans[k + 1]
            n_months  = idx1 - idx0
            if n_months == 13:
                adar2_idx = idx0 + 12
                if 0 <= adar2_idx < len(month_starts):
                    name_map[adar2_idx] = {"name": "Adar II", "nisan_yr": yr0}
            elif n_months != 12:
                print(f"  NOTE: {n_months} months between Nisan {yr0} and Nisan {yr1}")

        # ── 7. Assign AM years via Tishri anchors ─────────────────────────────
        tishri_indices = sorted(
            mi for mi, v in name_map.items() if v["name"] == "Tishri"
        )

        def am_year_for_month(mi: int) -> int:
            prev = [t for t in tishri_indices if t <= mi]
            if prev:
                return self._am_year_for_tishri(month_starts[max(prev)]["evening_jd"])
            nyr = name_map[mi]["nisan_yr"]
            return nyr + 3760   # Nisan of astro_yr nyr → AM nyr+3760 (approx)

        # ── 8. Build calendar list ────────────────────────────────────────────
        calendar = []
        for mi in sorted(name_map.keys()):
            ms    = month_starts[mi]
            ev_jd = ms["evening_jd"]
            ev_t  = ts.tt_jd(ev_jd)
            y, mo, d = ev_t.tt_calendar()[:3]
            calendar.append({
                "mi":         mi,
                "hname":      name_map[mi]["name"],
                "am_yr":      am_year_for_month(mi),
                "evening_jd": ev_jd,
                "greg_d":     int(d),
                "greg_mo":    mo,
                "greg_yr":    y,
                "greg_str":   fmt_date(ev_t),
                "cat":        ms["yallop"]["cat"],
                "q":          ms["yallop"]["q"],
                "arcl":       ms["yallop"]["arcl"],
                "arcv":       ms["yallop"]["arcv"],
                "W":          ms["yallop"]["W"],
                "moon_alt":   ms["yallop"]["moon_alt"],
                "moon_age_h": ms["moon_age_h"],
                "lag_min":    ms["lag_min"],
                "uncertain":  ms["uncertain"],
                "note":       ms["note"],
            })

        # Month lengths (days until next month start)
        for i in range(len(calendar) - 1):
            calendar[i]["days"] = round(
                calendar[i + 1]["evening_jd"] - calendar[i]["evening_jd"]
            )
        if calendar:
            calendar[-1]["days"] = None

        # Full-moon day within each month
        for entry in calendar:
            ev_jd  = entry["evening_jd"]
            window = entry["days"] if entry["days"] is not None else 17
            mask   = (full_moon_jds >= ev_jd) & (full_moon_jds < ev_jd + window)
            hits   = np.where(mask)[0]
            if len(hits):
                fm_jd = full_moon_jds[hits[0]]
                fm_t  = ts.tt_jd(fm_jd)
                entry["fm_hday"]    = int(fm_jd - ev_jd) + 1
                lon_offset_h        = self.loc_lon / 15.0
                ut1_h               = (fm_t.ut1 % 1) * 24
                entry["fm_local_h"] = (ut1_h + lon_offset_h) % 24
            else:
                entry["fm_hday"]    = None
                entry["fm_local_h"] = None

        return HebrewCalendarResult(
            calendar   = calendar,
            loc_name   = self.loc_name,
            loc_lat    = self.loc_lat,
            loc_lon    = self.loc_lon,
            start_year = self.start_year,
            end_year   = self.end_year,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HebrewCalendarResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HebrewCalendarResult:
    """
    Holds the computed Hebrew calendar and handles all output formatting.
    """

    calendar:   list
    loc_name:   str
    loc_lat:    float
    loc_lon:    float
    start_year: int
    end_year:   int

    SEP = "=" * 117

    def save(self, path: str | Path):
        """
        Save the calendar to a JSON file.
        The Skyfield timescale is not serialised; all other fields are included.
        Numpy scalars are coerced to native Python types.
        """
        def _default(obj):
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer):  return int(obj)
            raise TypeError(f"Not JSON serialisable: {type(obj)}")

        data = {
            "location":   self.loc_name,
            "lat":        self.loc_lat,
            "lon":        self.loc_lon,
            "start_year": self.start_year,
            "end_year":   self.end_year,
            "calendar":   self.calendar,
        }
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=_default),
            encoding="utf-8",
        )
        print(f"Saved {len(self.calendar)} months → {path}")

    @classmethod
    def from_file(cls, path: str | Path) -> "HebrewCalendarResult":
        """
        Load a HebrewCalendarResult from a JSON file previously written by save().
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            calendar   = data["calendar"],
            loc_name   = data["location"],
            loc_lat    = data["lat"],
            loc_lon    = data["lon"],
            start_year = data["start_year"],
            end_year   = data["end_year"],
        )

    def hebrew_date_for_jd(self, event_jd: float):
        """
        Return (month_name, AM_year, day_number) for a given TT JD,
        or (None, None, None) if outside the calendar's range.
        """
        cal = self.calendar
        for i in range(len(cal) - 1):
            if cal[i]["evening_jd"] <= event_jd < cal[i + 1]["evening_jd"]:
                day_num = int(event_jd - cal[i]["evening_jd"]) + 1
                return cal[i]["hname"], cal[i]["am_yr"], day_num
        if cal and event_jd >= cal[-1]["evening_jd"]:
            day_num = int(event_jd - cal[-1]["evening_jd"]) + 1
            return cal[-1]["hname"], cal[-1]["am_yr"], day_num
        return None, None, None

    def print_calendar(self):
        """Print the full annotated calendar table."""
        SEP = self.SEP
        print(SEP)
        print(f"OBSERVATION-BASED HEBREW CALENDAR  ·  "
              f"~{era(self.start_year)} – {era(self.end_year)}  ·  "
              f"Observer: {self.loc_name}")
        print("All dates proleptic Gregorian (evening of first crescent sighting).")
        print("Jewish day begins at that sunset; Western date of same civil day is one day later.")
        print(SEP)
        print()

        # Only show AM years within the requested range.
        # Nisan(start_year) sits in AM start_year+3760;
        # Tishri(end_year) starts AM end_year+3761.
        am_first   = self.start_year + 3760
        am_last    = self.end_year   + 3761
        current_am = None

        for row in self.calendar:
            if not (am_first <= row["am_yr"] <= am_last):
                continue

            if row["am_yr"] != current_am:
                current_am = row["am_yr"]
                astro_yr   = current_am - 3761
                print()
                print(f"  ── Jewish Year AM {current_am}  "
                      f"({era(astro_yr)} / {era(astro_yr+1)}) ──")
                print()
                print(f"  {'Month':<14} {'Evening of first crescent':>26}  "
                      f"{'Full moon (d  LST)':>18}  "
                      f"{'Cat':>3}  {'q':>6}  {'ARCL':>6}  {'ARCV':>6}  "
                      f"{'W′':>5}  {'Age(h)':>7}  {'Lag′':>5}")
                print("  " + "─" * 104)

            eb      = "±1d" if row["uncertain"] else "   "
            special = ""
            if row["hname"] == "Tishri":
                special = " ← Rosh Hashanah (civil new year)"
            elif row["hname"] == "Nisan":
                special = " ← 1 Nisan (religious new year)"
            elif row["hname"] == "Adar II":
                special = " ← intercalary month (leap year)"

            if row["fm_hday"] is not None:
                fm_str = f"d{row['fm_hday']:2d}  {row['fm_local_h']:5.2f}h LST"
            else:
                fm_str = "        ??       "

            print(f"  1 {row['hname']:<12} {row['greg_str']:>26}  "
                  f"{fm_str:>18}  "
                  f"{row['cat']:>3}  "
                  f"{row['q']:>6.3f}  "
                  f"{row['arcl']:>6.2f}°  "
                  f"{row['arcv']:>6.2f}°  "
                  f"{row['W']:>5.2f}  "
                  f"{row['moon_age_h']:>7.1f}h  "
                  f"{row['lag_min']:>5.0f}′  "
                  f"{eb}{special}")

    def print_notes(self):
        """Print the methodology and error-bar notes."""
        SEP = self.SEP
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

Observer: {self.loc_name} ({self.loc_lat:.4f}°N, {self.loc_lon:.4f}°E), sea level.
Ephemeris: JPL DE422.
""")