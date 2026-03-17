import argparse

from bethlehem import (
    HebrewCalendarEngine, HebrewCalendarResult, LOCATIONS, fmt_date,
)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Reconstruct the observation-based Hebrew lunisolar calendar.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Years use astronomical year numbering:
  3 BC = -2,  2 BC = -1,  1 BC = 0,  1 AD = 1,  2 AD = 2, ...
  3 AD = 3,   etc.

Examples:
  python hebrew_calendar.py                           # default: -3 to 0 (4 BC – 1 BC)
  python hebrew_calendar.py --start -5 --end 0
  python hebrew_calendar.py --location babylon
  python hebrew_calendar.py --location avaris --start -5 --end 0
""",
    )
    p.add_argument(
        "--start", type=int, default=-3, metavar="ASTRO_YEAR",
        help="first equinox year to include (astronomical, default: -3 = 4 BC)",
    )
    p.add_argument(
        "--end", type=int, default=0, metavar="ASTRO_YEAR",
        help="last equinox year to include (astronomical, default: 0 = 1 BC)",
    )
    p.add_argument(
        "--location", choices=LOCATIONS.keys(), default="jerusalem",
        help="observation site (default: jerusalem)",
    )
    p.add_argument(
        "--output", metavar="FILE",
        help="save results to a JSON file (e.g. calendar.json)",
    )
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Key-events cross-reference (Star of Bethlehem)
# ─────────────────────────────────────────────────────────────────────────────

def print_key_events(result: "HebrewCalendarResult", ts) -> None:
    """
    Cross-reference key Star-of-Bethlehem candidate events against a computed
    HebrewCalendarResult.  Only printed when the result range covers 3–2 BC.
    ts is a Skyfield timescale (engine.ts).
    """
    if not (result.start_year <= -2 and result.end_year >= -1):
        return

    SEP = HebrewCalendarResult.SEP
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

    for label, t_event in events:
        hmo, am_yr, hday = result.hebrew_date_for_jd(t_event.tt)
        greg_str = fmt_date(t_event)
        if hmo:
            print(f"  {label:<40} {greg_str}  →  {hday} {hmo} AM {am_yr}")
        else:
            print(f"  {label:<40} {greg_str}  →  (outside computed range)")
    print()


def main():
    args   = parse_args()
    engine = HebrewCalendarEngine(args.location, args.start, args.end)
    result = engine.build_calendar()
    result.print_calendar()
    result.print_notes()
    print_key_events(result, engine.ts)
    if args.output:
        result.save(args.output)


if __name__ == "__main__":
    main()
