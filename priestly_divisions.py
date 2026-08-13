#!/usr/bin/env python3
"""
priestly_divisions.py

Reconstructs the weekly schedule of the 24 priestly divisions (courses) of
1 Chronicles 24:7-18 for a given Jewish year, following Beckwith's model:

  - The cycle always restarts with Jehoiarib (the first-listed course) on
    the Sabbath on or immediately before 1 Tishri.
  - Each division serves one week (Sabbath to Sabbath).
  - The 24 divisions are served in the order listed in 1 Chronicles 24:7-18;
    after Maaziah (the 24th) the cycle repeats from Jehoiarib.
  - This continues until the next 1 Tishri, at which point the cycle again
    resets to Jehoiarib regardless of where a strict 24-week rotation would
    otherwise have landed.

Dates are computed from the observation-based Hebrew calendar reconstruction
in bethlehem/hebrew_calendar.py (Skyfield + DE422).

Date convention (deliberately DIFFERENT from hebrew_calendar.py's default):
hebrew_calendar.py labels every Hebrew day by the evening on which it
*begins* (so a day's main daytime activity actually falls on the next
civil date). For this script, the Gregorian date shown is instead the
civil *daytime* on which priestly service happened/changed hands — i.e.
for the Sabbath, that's the civil Saturday, not the Friday evening on
which the Hebrew calendar day technically starts. The Hebrew date shown
for that row is still the correct Hebrew calendar day for that Sabbath
(the one whose evening began the preceding Friday).

Usage:
  python priestly_divisions.py --year -2                # Jewish year beginning Tishri 3 BC
  python priestly_divisions.py --am-year 3758
  python priestly_divisions.py --year -2 --location babylon
"""

import argparse
import json
from pathlib import Path

from bethlehem import HebrewCalendarEngine, LOCATIONS, fmt_date
from bethlehem.hebrew_calendar import era

# ─────────────────────────────────────────────────────────────────────────────
# The 24 priestly divisions, in lot order (1 Chronicles 24:7-18)
# ─────────────────────────────────────────────────────────────────────────────

DIVISIONS = [
    "Jehoiarib", "Jedaiah", "Harim", "Seorim", "Malchijah", "Mijamin",
    "Hakkoz", "Abijah", "Jeshua", "Shecaniah", "Eliashib", "Jakim",
    "Huppah", "Jeshebeab", "Bilgah", "Immer", "Hezir", "Happizzez",
    "Pethahiah", "Jehezkel", "Jachin", "Gamul", "Delaiah", "Maaziah",
]

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct the weekly priestly-division schedule (Beckwith's model) for a Jewish year.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Years use astronomical year numbering unless --am-year is given:
  3 BC = -2,  2 BC = -1,  1 BC = 0,  1 AD = 1, ...

Examples:
  python priestly_divisions.py --year -2                # Jewish year beginning Tishri 3 BC
  python priestly_divisions.py --am-year 3758
  python priestly_divisions.py --year -2 --location babylon
""",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--year",
        type=int,
        metavar="ASTRO_YEAR",
        help="astronomical year in which 1 Tishri falls (0 = 1 BC, -2 = 3 BC, ...)",
    )
    g.add_argument(
        "--am-year",
        type=int,
        metavar="AM_YEAR",
        help="Jewish Anno Mundi year that begins with this Tishri (e.g. 3758)",
    )
    p.add_argument(
        "--location",
        choices=LOCATIONS.keys(),
        default="jerusalem",
        help="observation site for crescent visibility (default: jerusalem)",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="save the schedule to a JSON file",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Sabbath arithmetic
# ─────────────────────────────────────────────────────────────────────────────


def sabbath_evening_on_or_before(jd: float) -> float:
    """
    evening_jd of the Sabbath Hebrew-calendar-day on or before the given
    (integer-valued, evening_jd-style) JD.

    hebrew_calendar.py labels a Hebrew day by the evening_jd on which it
    *begins*; the Sabbath's evening_jd is therefore a Friday (its evening
    begins Sabbath, its daytime the following civil Saturday is Shabbat
    itself). JD 2451545 (1 Jan 2000, noon TT) was a Saturday and
    2451545 % 7 == 5, so JDN % 7 == 4 identifies Fridays under this
    module's JD convention (see hebrew_calendar.py's note on JD/noon-TT
    alignment). Feeding the returned value into
    HebrewCalendarResult.hebrew_date_for_jd() yields the correct Hebrew
    date for that Sabbath; add 1 to get the civil (Saturday) JDN.
    """
    wd = int(round(jd)) % 7
    return jd - ((wd - 4) % 7)


def find_tishri(result, am_year: int):
    for entry in result.calendar:
        if entry.hname == "Tishri" and entry.am_yr == am_year:
            return entry
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    astro_year = args.year if args.year is not None else args.am_year - 3761
    am_year = astro_year + 3761

    engine = HebrewCalendarEngine(args.location, astro_year, astro_year + 1)
    result = engine.build_calendar()

    tishri0 = find_tishri(result, am_year)
    tishri1 = find_tishri(result, am_year + 1)
    if tishri0 is None or tishri1 is None:
        raise SystemExit(
            f"Could not locate 1 Tishri for AM {am_year} and/or AM {am_year + 1} "
            f"in the computed calendar range."
        )

    # evening_jd of the Sabbath (a Friday); add 1 to get the civil Saturday.
    sab0_ev = sabbath_evening_on_or_before(tishri0.evening_jd)
    sab1_ev = sabbath_evening_on_or_before(tishri1.evening_jd)
    n_weeks = round((sab1_ev - sab0_ev) / 7)

    SEP = "=" * 90
    print()
    print(SEP)
    print(
        f"PRIESTLY DIVISION SCHEDULE  ·  Jewish Year AM {am_year}  "
        f"({era(astro_year)} / {era(astro_year + 1)})"
    )
    print("Beckwith's model: cycle restarts at Jehoiarib on the Sabbath on/before 1 Tishri.")
    print(
        "Gregorian dates below are the DAYTIME of service (the civil Saturday), not the"
    )
    print(
        "preceding evening on which the Hebrew calendar day technically begins."
    )
    print(f"Observer: {result.loc_name}")
    print(SEP)
    print()
    print(f"  1 Tishri AM {am_year}:              {tishri0.greg_str}")
    print(f"  Sabbath (daytime) on/before:    {fmt_date(engine.ts.tt_jd(sab0_ev + 1))}")
    print(f"  1 Tishri AM {am_year + 1}:              {tishri1.greg_str}")
    print(f"  Sabbath (daytime) on/before:    {fmt_date(engine.ts.tt_jd(sab1_ev + 1))}")
    print(f"  Weeks in cycle:                 {n_weeks}")
    print()
    print(
        f"  {'#':>3}  {'Division':<12} {'Gregorian date (daytime)':>25}  {'Hebrew date':>22}"
    )
    print("  " + "─" * 71)

    schedule = []
    for i in range(n_weeks):
        ev_jd = sab0_ev + 7 * i
        division = DIVISIONS[i % 24]
        t = engine.ts.tt_jd(ev_jd + 1)  # civil Saturday, not the Friday evening_jd
        greg_str = fmt_date(t)
        hmo, hyr, hday = result.hebrew_date_for_jd(ev_jd)
        hebrew_str = f"{hday} {hmo} AM {hyr}" if hmo else "?"
        print(f"  {i + 1:>3}  {division:<12} {greg_str:>25}  {hebrew_str:>22}")
        schedule.append(
            {
                "week": i + 1,
                "division": division,
                "jd": ev_jd + 1,
                "greg_date_daytime": greg_str,
                "hebrew_date": hebrew_str,
            }
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "am_year": am_year,
                    "astro_year": astro_year,
                    "location": result.loc_name,
                    "schedule": schedule,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nSaved {len(schedule)} weeks → {args.output}")


if __name__ == "__main__":
    main()
