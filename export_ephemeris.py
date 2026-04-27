#!/usr/bin/env python3
"""
Export daily RA/Dec positions of Sun, Moon, Jupiter, Venus, Mercury, Mars
from Jerusalem, 3 BC – Jan 1 BC, as JSON for interactive.html.

Coordinate system: equatorial RA/Dec (J2000), apparent from Jerusalem.
The HTML canvas maps RA → x (right-to-left, since east is left on sky charts)
and Dec → y.
"""

import json
import numpy as np
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

# Virgo constellation stars — J2000 catalog coordinates
# Heze and Zaniah have Dec between -1° and 0°, so decimal degrees avoid the -0 tuple problem
VIRGO_STARS = {
    "spica":        Star(ra_hours=(13, 25, 11.579), dec_degrees=(-11,  9, 40.75)),
    "zavijava":     Star(ra_hours=(11, 50, 41.718), dec_degrees=(  1, 45, 52.99)),
    "porrima":      Star(ra_hours=(12, 41, 39.644), dec_degrees=( -1, 26, 57.75)),
    "auva":         Star(ra_hours=(12, 55, 36.208), dec_degrees=(  3, 23, 50.89)),
    "vindemiatrix": Star(ra_hours=(13,  2, 10.598), dec_degrees=( 10, 57, 32.94)),
    "heze":         Star(ra_hours=(13, 34, 41.591), dec_degrees=-0.5958),
    "zaniah":       Star(ra_hours=(12, 19, 54.358), dec_degrees=-0.6668),
    "syrma":        Star(ra_hours=(14, 16,  0.874), dec_degrees=( -6,  0,  2.03)),
    "mu_vir":       Star(ra_hours=(14, 43,  3.522), dec_degrees=( -5, 39, 29.53)),
    "tau_vir":      Star(ra_hours=(14,  1, 38.780), dec_degrees=(  1, 32, 40.50)),
    "109_vir":      Star(ra_hours=(14, 46, 14.990), dec_degrees=(  1, 53, 34.60)),
}

# Hydra constellation stars — J2000 catalog coordinates (HIP catalog)
# ι Hya Dec is -1°08′33.6″: tuple (-1,8,33.6) works since degree component is non-zero negative
HYDRA_STARS = {
    "delta_hya":   Star(ra_hours=( 8, 37, 39.41), dec_degrees=( 5, 42, 13.7)),
    "sigma_hya":   Star(ra_hours=( 8, 38, 45.45), dec_degrees=( 3, 20, 29.3)),  # Minchir
    "eta_hya":     Star(ra_hours=( 8, 43, 13.49), dec_degrees=( 3, 23, 55.2)),
    "epsilon_hya": Star(ra_hours=( 8, 46, 46.65), dec_degrees=( 6, 25,  8.1)),  # Ashlesha
    "rho_hya":     Star(ra_hours=( 8, 48, 25.98), dec_degrees=( 5, 50, 16.4)),
    "zeta_hya":    Star(ra_hours=( 8, 55, 23.68), dec_degrees=( 5, 56, 43.9)),  # Hydrobius
    "theta_hya":   Star(ra_hours=( 9, 14, 21.79), dec_degrees=( 2, 18, 54.1)),
    "alphard":     Star(ra_hours=( 9, 27, 35.25), dec_degrees=(-8, 39, 31.3)),  # α Hya
    "iota_hya":    Star(ra_hours=( 9, 39, 51.33), dec_degrees=(-1,  8, 33.6)),
    "lambda_hya":  Star(ra_hours=(10, 10, 35.40), dec_degrees=(-12, 21, 13.8)),
    "mu_hya":      Star(ra_hours=(10, 26,  5.51), dec_degrees=(-16, 50,  9.9)),
    "nu_hya":      Star(ra_hours=(10, 49, 37.43), dec_degrees=(-16, 11, 38.9)),
    "xi_hya":      Star(ra_hours=(11, 33,  0.26), dec_degrees=(-31, 51, 27.1)),
    "beta_hya":    Star(ra_hours=(11, 52, 54.56), dec_degrees=(-33, 54, 29.3)),
    "gamma_hya":   Star(ra_hours=(13, 18, 55.25), dec_degrees=(-23, 10, 17.1)),  # Naga
    "pi_hya":      Star(ra_hours=(14,  6, 22.27), dec_degrees=(-26, 40, 55.3)),
}

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

# Fixed Virgo stars — one apparent position at midpoint of date range
t_mid = ts.tt(-1, 7, 1)  # ~Jul 2 BC
virgo_positions = {}
for name, star in VIRGO_STARS.items():
    print(f"  {name} (fixed)…", flush=True)
    app = observer.at(t_mid).observe(star).apparent()
    ra, dec, _ = app.radec()
    virgo_positions[name] = {
        "ra":  round(float(ra.hours),   4),
        "dec": round(float(dec.degrees), 4),
    }

hydra_positions = {}
for name, star in HYDRA_STARS.items():
    print(f"  {name} (fixed)…", flush=True)
    app = observer.at(t_mid).observe(star).apparent()
    ra, dec, _ = app.radec()
    hydra_positions[name] = {
        "ra":  round(float(ra.hours),   4),
        "dec": round(float(dec.degrees), 4),
    }

# Moon illumination fraction (geocentric elongation → (1 - cos θ) / 2)
print("  moon_phase…", flush=True)
_moon_app = earth.at(times).observe(moon).apparent()  # type: ignore[union-attr]
_sun_app  = earth.at(times).observe(sun).apparent()   # type: ignore[union-attr]
_elong    = _moon_app.separation_from(_sun_app).degrees
moon_phase = list(np.round((1 - np.cos(np.radians(_elong))) / 2, 3).tolist())

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
    "moon_phase": moon_phase,
    "virgo_stars": virgo_positions,
    "hydra_stars": hydra_positions,
}

out_path = "ephemeris_data.json"
with open(out_path, "w") as f:
    json.dump(output, f, separators=(",", ":"))

import os
size_mb = os.path.getsize(out_path) / 1e6
print(f"\nWrote {out_path} ({size_mb:.1f} MB, {len(days)} days)")

# Build standalone HTML with JSON embedded
import re
with open("interactive.html") as f:
    html = f.read()

json_str = json.dumps(output, separators=(",", ":"))
inline_script = f"<script>const EPH_INLINE={json_str};</script>"

# Replace the fetch(...) block with direct assignment from the inline data
html = re.sub(
    r"fetch\('ephemeris_data\.json'\)\s*\n"
    r"\s*\.then\(r=>r\.json\(\)\)\s*\n"
    r"\s*\.then\(data=>\{EPH=data;.*?\}\)\s*\n"
    r"\s*\.catch\(e=>console\.error\('Failed to load ephemeris:',e\)\);",
    "setTimeout(()=>{EPH=EPH_INLINE;if(EPH.virgo_stars)buildVirgo(EPH.virgo_stars);if(EPH.hydra_stars)buildHydra(EPH.hydra_stars);setT(209);},0);",
    html,
)

# Inject the data script just before the closing </head>
html = html.replace("</head>", f"{inline_script}\n</head>", 1)

standalone_path = "interactive_standalone.html"
with open(standalone_path, "w") as f:
    f.write(html)

standalone_mb = os.path.getsize(standalone_path) / 1e6
print(f"Wrote {standalone_path} ({standalone_mb:.1f} MB)")
