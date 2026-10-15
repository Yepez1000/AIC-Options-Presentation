#!/usr/bin/env python3
"""Build a dependency-free SVG implied-volatility surface from contract CSV data."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path


DEFAULT_DATE = "2026-10-14"
DEFAULT_TICKER = "NVDA"
DATA_ROOT = Path("/srv/data/options_model_features/Default_Strike/contracts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TICKER.upper(), help="Underlying symbol")
    parser.add_argument("--date", default=DEFAULT_DATE, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--input", type=Path, help="Contract CSV path; defaults to the standard data path")
    parser.add_argument("--output", type=Path, help="SVG path; defaults to <ticker>_options_surface_<date>.svg")
    parser.add_argument("--tikz-output", type=Path, help="Optional Beamer/TikZ fragment path")
    parser.add_argument("--min-moneyness", type=float, default=0.80, help="Minimum strike / spot to include")
    parser.add_argument("--max-moneyness", type=float, default=1.20, help="Maximum strike / spot to include")
    parser.add_argument("--max-iv", type=float, default=1.50, help="Maximum IV to include (1.50 = 150%%)")
    return parser.parse_args()


def load_surface_rows(path: Path, ticker: str, trade_date: str, min_m: float, max_m: float, max_iv: float):
    grouped = defaultdict(list)
    filtered = 0
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row.get("underlying", "").upper() != ticker.upper() or row.get("trade_date") != trade_date:
                continue
            try:
                strike = float(row["strike"])
                spot = float(row["underlying_close"])
                iv = float(row["implied_volatility"])
                expiry = row["expiration_date"]
                dte = int(row["dte_days"])
            except (KeyError, TypeError, ValueError):
                filtered += 1
                continue
            moneyness = strike / spot if spot > 0 else math.nan
            if (
                row.get("iv_failure_reason") != "ok"
                or not math.isfinite(iv)
                or iv <= 0
                or iv > max_iv
                or not min_m <= moneyness <= max_m
            ):
                filtered += 1
                continue
            grouped[(expiry, dte, moneyness)].append(iv)

    # Calls and puts at the same strike/expiry contribute one averaged surface point.
    by_expiry = defaultdict(list)
    for (expiry, dte, moneyness), ivs in grouped.items():
        by_expiry[(expiry, dte)].append((moneyness, sum(ivs) / len(ivs)))
    curves = []
    for (expiry, dte), points in sorted(by_expiry.items(), key=lambda item: item[0][1]):
        curves.append((expiry, dte, sorted(points)))
    return curves, filtered


def interpolate(points: list[tuple[float, float]], x: float) -> float:
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return (y0 + y1) / 2
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError(f"Cannot interpolate outside observed range at moneyness {x:.4f}")


def color_for(value: float, low: float, high: float) -> str:
    t = (value - low) / (high - low) if high > low else 0.5
    # Deep blue through teal to warm yellow.
    stops = ((0.0, (35, 75, 160)), (0.5, (34, 176, 170)), (1.0, (250, 205, 70)))
    for (ta, ca), (tb, cb) in zip(stops, stops[1:]):
        if t <= tb:
            f = max(0.0, min(1.0, (t - ta) / (tb - ta)))
            rgb = tuple(round(a + (b - a) * f) for a, b in zip(ca, cb))
            return "#%02x%02x%02x" % rgb
    return "#facd46"


def svg_surface(ticker: str, trade_date: str, curves, filtered: int, min_m: float, max_m: float) -> str:
    if len(curves) < 2:
        raise ValueError("A surface needs at least two distinct expirations with usable contracts")
    low_x = max(points[0][0] for _, _, points in curves)
    high_x = min(points[-1][0] for _, _, points in curves)
    if low_x >= high_x:
        raise ValueError("Expirations have no overlapping moneyness range after filtering")

    steps = 36
    xs = [low_x + (high_x - low_x) * i / (steps - 1) for i in range(steps)]
    grid = [[interpolate(points, x) for x in xs] for _, _, points in curves]
    all_iv = [iv for row in grid for iv in row]
    low_z, high_z = min(all_iv), max(all_iv)
    min_d, max_d = curves[0][1], curves[-1][1]

    def project(i: int, j: int, include_iv: bool = True) -> tuple[float, float]:
        u = i / (steps - 1)
        v = (curves[j][1] - min_d) / (max_d - min_d) if max_d > min_d else 0
        z = (grid[j][i] - low_z) / (high_z - low_z) if high_z > low_z else 0.5
        if not include_iv:
            z = 0
        return 190 + 650 * u + 145 * v, 700 - 40 * u - 125 * v - 300 * z

    faces = []
    for j in range(len(curves) - 1):
        for i in range(steps - 1):
            quad = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                points = [project(*node) for node in tri]
                avg = sum(grid[y][x] for x, y in tri) / 3
                faces.append((sum(y for _, y in points) / 3, points, color_for(avg, low_z, high_z)))
    faces.sort(key=lambda face: face[0])

    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1000" viewBox="0 0 1200 1000">',
        '<rect width="1200" height="1000" fill="#f5f7fb"/>',
        '<defs><linearGradient id="ivscale" x1="0" y1="1" x2="0" y2="0"><stop offset="0%" stop-color="#234ba0"/><stop offset="50%" stop-color="#22b0aa"/><stop offset="100%" stop-color="#facd46"/></linearGradient></defs>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172438}.muted{fill:#43536a}.axis{stroke:#26364d;stroke-width:2}.mesh{stroke:#ffffff;stroke-width:.55;stroke-opacity:.45}.grid{stroke:#9aa8b8;stroke-width:.8;stroke-opacity:.32}</style>',
        f'<text x="72" y="58" font-size="28" font-weight="700">{ticker.upper()} implied volatility surface</text>',
        f'<text x="72" y="88" class="muted" font-size="15">Trade date {trade_date}  |  {len(curves)} expirations  |  spot-normalized strikes</text>',
        '<line x1="72" y1="112" x2="1128" y2="112" stroke="#d8dee8"/>',
    ]

    # Draw a subtle floor grid before the colored surface patches.
    origin = project(0, 0, False)
    x_end = project(steps - 1, 0, False)
    y_end = project(0, len(curves) - 1, False)
    far_corner = project(steps - 1, len(curves) - 1, False)
    out.append(f'<polygon points="{origin[0]:.1f},{origin[1]:.1f} {x_end[0]:.1f},{x_end[1]:.1f} {far_corner[0]:.1f},{far_corner[1]:.1f} {y_end[0]:.1f},{y_end[1]:.1f}" fill="#e9eef5" stroke="#c8d1dd"/>')
    for fraction in (0.25, 0.5, 0.75):
        i = round(fraction * (steps - 1))
        a = project(i, 0, False)
        b = project(i, len(curves) - 1, False)
        out.append(f'<line class="grid" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
    for j in range(1, len(curves)):
        a = project(0, j, False)
        b = project(steps - 1, j, False)
        out.append(f'<line class="grid" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')

    # Draw colored triangular patches; restrained wire lines keep the mesh legible.
    for _, points, color in faces:
        out.append('<polygon class="mesh" fill="%s" points="%s"/>' % (color, " ".join(f"{x:.1f},{y:.1f}" for x, y in points)))
    for j in range(len(curves)):
        pts = [project(i, j) for i in range(steps)]
        out.append('<polyline fill="none" stroke="#17324d" stroke-opacity=".5" stroke-width="1" points="%s"/>' % " ".join(f"{x:.1f},{y:.1f}" for x, y in pts))
    for i in range(0, steps, 6):
        pts = [project(i, j) for j in range(len(curves))]
        out.append('<polyline fill="none" stroke="#17324d" stroke-opacity=".32" stroke-width=".8" points="%s"/>' % " ".join(f"{x:.1f},{y:.1f}" for x, y in pts))

    # Axes, readable ticks, and a separate color scale.
    z_end = (origin[0], origin[1] - 300)
    for a, b in ((origin, x_end), (origin, y_end), (origin, z_end)):
        out.append(f'<line class="axis" x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}"/>')
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        i = round(fraction * (steps - 1))
        px, py = project(i, 0, False)
        out.append(f'<line class="axis" x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{py + 7:.1f}"/>')
        out.append(f'<text x="{px:.1f}" y="{py + 27:.1f}" text-anchor="middle" fill="#26364d" font-size="15" font-weight="600">{xs[i]:.2f}</text>')
    for j, (expiry, dte, _) in enumerate(curves):
        px, py = project(0, j, False)
        out.append(f'<text x="{px - 20:.1f}" y="{py + 20:.1f}" text-anchor="end" fill="#26364d" font-size="13" font-weight="600">{expiry[5:]}  ({dte}d)</text>')
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        value = low_z + fraction * (high_z - low_z)
        py = origin[1] - 300 * fraction
        out.append(f'<line class="axis" x1="{origin[0] - 5:.1f}" y1="{py:.1f}" x2="{origin[0] + 5:.1f}" y2="{py:.1f}"/>')
        out.append(f'<text x="{origin[0] - 15:.1f}" y="{py + 5:.1f}" text-anchor="end" fill="#26364d" font-size="14" font-weight="600">{value * 100:.0f}%</text>')

    out.extend([
        '<text x="540" y="835" text-anchor="middle" fill="#172438" font-size="18" font-weight="700">Moneyness (strike / spot)</text>',
        '<text x="815" y="775" text-anchor="middle" fill="#172438" font-size="18" font-weight="700">Days to expiration</text>',
        '<text x="92" y="540" transform="rotate(-90 92 540)" text-anchor="middle" fill="#172438" font-size="18" font-weight="700">Implied volatility</text>',
        '<text x="1000" y="345" font-size="13" font-weight="600">IV scale</text>',
        '<rect x="1000" y="360" width="22" height="260" rx="3" fill="url(#ivscale)"/>',
        f'<text x="1035" y="374" class="muted" font-size="12">{high_z * 100:.0f}%</text>',
        f'<text x="1035" y="625" class="muted" font-size="12">{low_z * 100:.0f}%</text>',
        f'<text x="72" y="910" class="muted" font-size="13">{sum(len(points) for _, _, points in curves)} observed strike/expiry points shown; calls and puts averaged at matching strikes.</text>',
        f'<text x="72" y="935" class="muted" font-size="12">Filtered {filtered} source rows outside quality, IV, or moneyness limits (0.80–1.20 moneyness; IV at most 150%).</text>',
        '<text x="72" y="960" class="muted" font-size="12">Interpolated for display across sparse expirations; this is not an arbitrage-free fitted model.</text>',
        '</svg>',
    ])
    return "\n".join(out)


def tikz_surface(curves) -> str:
    """Emit a compact native-TikZ surface so the chart can be used in Beamer."""
    steps = 36
    low_x = max(points[0][0] for _, _, points in curves)
    high_x = min(points[-1][0] for _, _, points in curves)
    xs = [low_x + (high_x - low_x) * i / (steps - 1) for i in range(steps)]
    grid = [[interpolate(points, x) for x in xs] for _, _, points in curves]
    all_iv = [iv for row in grid for iv in row]
    low_z, high_z = min(all_iv), max(all_iv)
    min_d, max_d = curves[0][1], curves[-1][1]
    scale = 0.0085

    def project(i: int, j: int, include_iv: bool = True) -> tuple[float, float]:
        u = i / (steps - 1)
        v = (curves[j][1] - min_d) / (max_d - min_d) if max_d > min_d else 0
        z = (grid[j][i] - low_z) / (high_z - low_z) if high_z > low_z else 0.5
        if not include_iv:
            z = 0
        return scale * (650 * u + 145 * v), scale * (40 * u + 125 * v + 300 * z)

    def rgb(hex_color: str) -> tuple[int, int, int]:
        return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))

    lines = [r"\begin{tikzpicture}[line join=round,line cap=round]"]
    floor = [project(0, 0, False), project(steps - 1, 0, False), project(steps - 1, len(curves) - 1, False), project(0, len(curves) - 1, False)]
    lines.append(r"\fill[gray!8] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in floor) + r" -- cycle;")
    for j in range(len(curves) - 1):
        for i in range(steps - 1):
            quad = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
            for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
                points = [project(*node) for node in tri]
                avg = sum(grid[y][x] for x, y in tri) / 3
                r, g, b = rgb(color_for(avg, low_z, high_z))
                path = " -- ".join(f"({x:.3f},{y:.3f})" for x, y in points)
                lines.append(rf"\path[fill={{rgb,255:red,{r};green,{g};blue,{b}}},draw=white,line width=.18pt] {path} -- cycle;")
    for j in range(len(curves)):
        points = [project(i, j) for i in range(steps)]
        lines.append(r"\draw[black!45,line width=.28pt] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in points) + ";")
    for i in range(0, steps, 6):
        points = [project(i, j) for j in range(len(curves))]
        lines.append(r"\draw[black!30,line width=.22pt] " + " -- ".join(f"({x:.3f},{y:.3f})" for x, y in points) + ";")

    origin = project(0, 0, False)
    x_end = project(steps - 1, 0, False)
    y_end = project(0, len(curves) - 1, False)
    z_end = (origin[0], origin[1] + scale * 300)
    for end in (x_end, y_end, z_end):
        lines.append(rf"\draw[->,>=stealth,thick] ({origin[0]:.3f},{origin[1]:.3f}) -- ({end[0]:.3f},{end[1]:.3f});")
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        i = round(fraction * (steps - 1))
        x, y = project(i, 0, False)
        lines.append(rf"\node[below,font=\scriptsize,text=black] at ({x:.3f},{y - .12:.3f}) {{{xs[i]:.2f}}};")
    for j, (expiry, dte, _) in enumerate(curves):
        x, y = project(0, j, False)
        lines.append(rf"\node[left,font=\tiny,text=black] at ({x - .04:.3f},{y - .11:.3f}) {{{expiry[5:]} ({dte}d)}};")
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        value = low_z + fraction * (high_z - low_z)
        y = origin[1] + scale * 300 * fraction
        lines.append(rf"\node[left,font=\scriptsize,text=black] at ({origin[0] - .06:.3f},{y:.3f}) {{{value * 100:.0f}\%}};")
    lines.extend([
        r"\node[below,font=\small\bfseries] at (2.76,-.47) {Moneyness (strike / spot)};",
        r"\node[rotate=90,font=\small\bfseries] at (-.63,1.35) {Implied volatility};",
        r"\end{tikzpicture}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    trade_day = date.fromisoformat(args.date)
    source = args.input or DATA_ROOT / trade_day.strftime("%Y") / trade_day.strftime("%m") / f"{args.date}.csv"
    output = args.output or Path(f"{args.ticker.lower()}_options_surface_{args.date}.svg")
    curves, filtered = load_surface_rows(source, args.ticker, args.date, args.min_moneyness, args.max_moneyness, args.max_iv)
    svg = svg_surface(args.ticker, args.date, curves, filtered, args.min_moneyness, args.max_moneyness)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    tikz_output = args.tikz_output or output.with_name(output.stem + "_tikz.tex")
    tikz_output.write_text(tikz_surface(curves), encoding="utf-8")
    print(f"Wrote {output}: {len(curves)} expirations; {sum(len(p) for _, _, p in curves)} strike/expiry points; {filtered} rows excluded")
    print(f"Wrote {tikz_output}")


if __name__ == "__main__":
    main()
