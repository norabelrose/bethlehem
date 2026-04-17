#!/usr/bin/env python3
"""
Export daily RA/Dec positions of Sun, Moon, Jupiter, Venus, Mercury, Mars
from Jerusalem, 3 BC – Jan 1 BC, as JSON for interactive.html.

Coordinate system: equatorial RA/Dec (J2000), apparent from Jerusalem.
The HTML canvas maps RA → x (right-to-left, since east is left on sky charts)
and Dec → y.
"""

import json
from skyfield.api import load, Star, wgs84, N, E


print("Loading DE422…", flush=True)
eph = load("de422.bsp")
ts = load.timescale()
print("Loaded.\n")

earth = eph["earth"]
sun   = eph["sun"]
moon  = eph["moon"]
jup   = eph["jupiter barycenter"]
ven   = eph["venus barycenter"]
mer   = eph["mercury barycenter"]
mar   = eph["mars barycenter"]

regulus = Star(ra_hours=(10, 8, 22.311), dec_degrees=(11, 58, 1.95))
spica   = Star(ra_hours=(13, 25, 11.579), dec_degrees=(-11, 9, 40.75))

jerusalem = wgs84.latlon(31.7683 * N, 35.2137 * E)

# ---------------------------------------------------------------------------
# Date range: 1 Jan 3 BC (astro year -2) to 31 Jan 1 BC (astro year 0)
# We export one entry per day at midnight TT.
# ---------------------------------------------------------------------------
START = ts.tt(-2, 1, 1)   # 1 Jan 3 BC
END   = ts.tt(0, 1, 31)   # 31 Jan 1 BC

# Build list of integer JD values
import math
start_jd = math.floor(START.tt + 0.5)
end_jd   = math.floor(END.tt + 0.5)

days = list(range(start_jd, end_jd + 1))
print(f"Exporting {len(days)} days…")

times = ts.tt_jd(days)
observer = earth + jerusalem

def apparent_radec(body):
    astrometric = observer.at(times).observe(body)
    apparent = astrometric.apparent()
    ra, dec, _ = apparent.radec()
    return ra.hours, dec.degrees

# Compute for each body
bodies = {
    "sun":     sun,
    "moon":    moon,
    "jupiter": jup,
    "venus":   ven,
    "mercury": mer,
    "mars":    mar,
    "regulus": regulus,
    "spica":   spica,
}

results = {}
for name, body in bodies.items():
    print(f"  {name}…", flush=True)
    ra_h, dec_d = apparent_radec(body)
    # Round to 4 decimal places to keep file size reasonable
    results[name] = {
        "ra":  [round(float(v), 4) for v in ra_h],
        "dec": [round(float(v), 4) for v in dec_d],
    }

# Also export the JD and calendar date for each day
def jd_to_calendar(jd):
    # Proleptic Gregorian via algorithm
    z = int(jd + 0.5)
    a = z + 32044
    b = (4*a + 3) // 146097
    c = a - (146097*b)//4
    d = (4*c + 3) // 1461
    e = c - (1461*d)//4
    m = (5*e + 2) // 153
    day   = e - (153*m + 2)//5 + 1
    month = m + 3 - 12*(m//10)
    year  = 100*b + d - 4800 + m//10
    return year, month, day

month_names = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

dates = []
for jd in days:
    y, m, d = jd_to_calendar(jd)
    # Convert astronomical year to BC label
    if y <= 0:
        label = f"{1 - y} B.C."
    else:
        label = f"A.D. {y}"
    dates.append(f"{d} {month_names[m-1]} {label}")

output = {
    "startJD": days[0],
    "dates": dates,
    "bodies": results,
}

out_path = "ephemeris_data.json"
with open(out_path, "w") as f:
    json.dump(output, f, separators=(",", ":"))

import os
size_mb = os.path.getsize(out_path) / 1e6
print(f"\nWrote {out_path} ({size_mb:.1f} MB, {len(days)} days)")
