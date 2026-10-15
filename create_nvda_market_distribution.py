#!/usr/bin/env python3
"""Create an option-implied risk-neutral terminal-price distribution for NVDA.

The script smooths the implied-volatility smile for one expiration, converts the
smile to Black--Scholes call prices, and applies the Breeden--Litzenberger
relationship q(K) = exp(rT) * d2C/dK2. It uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from datetime import date
from html import escape
from pathlib import Path


DEFAULT_TICKER = "NVDA"
DEFAULT_TRADE_DATE = "2026-10-14"
DEFAULT_EXPIRATION = "2026-12-18"
DATA_ROOT = Path("/srv/data/options_model_features/Default_Strike/contracts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--date", default=DEFAULT_TRADE_DATE, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--expiration", default=DEFAULT_EXPIRATION, help="Expiration date (YYYY-MM-DD)")
    parser.add_argument("--input", type=Path, help="Contract CSV; defaults to the standard data path")
    parser.add_argument("--output", type=Path, help="SVG output path")
    parser.add_argument("--csv-output", type=Path, help="Optional density CSV output path")
    parser.add_argument("--min-moneyness", type=float, default=0.60)
    parser.add_argument("--max-moneyness", type=float, default=1.50)
    parser.add_argument("--max-iv", type=float, default=1.50, help="Maximum accepted IV; 1.50 = 150%%")
    parser.add_argument("--bandwidth", type=float, default=0.055, help="Smile smoothing bandwidth in log-moneyness")
    parser.add_argument("--grid-points", type=int, default=481)
    return parser.parse_args()


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, years: float, rate: float, dividend: float, iv: float) -> float:
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * iv * iv) * years) / (iv * root_t)
    d2 = d1 - iv * root_t
    return spot * math.exp(-dividend * years) * normal_cdf(d1) - strike * math.exp(-rate * years) * normal_cdf(d2)


def load_smile(
    source: Path,
    ticker: str,
    trade_date: str,
    expiration: str,
    min_moneyness: float,
    max_moneyness: float,
    max_iv: float,
):
    by_strike: dict[float, list[float]] = defaultdict(list)
    spots: list[float] = []
    rates: list[float] = []
    dividends: list[float] = []
    matched_rows = 0

    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("underlying", "").upper() != ticker.upper()
                or row.get("trade_date") != trade_date
                or row.get("expiration_date") != expiration
                or row.get("iv_failure_reason") != "ok"
            ):
                continue
            try:
                strike = float(row["strike"])
                spot = float(row["underlying_close"])
                iv = float(row["implied_volatility"])
                rate = float(row["risk_free_rate"])
                dividend = float(row["dividend_yield"])
            except (KeyError, TypeError, ValueError):
                continue
            moneyness = strike / spot if spot > 0 else math.nan
            if not (
                math.isfinite(iv)
                and 0 < iv <= max_iv
                and min_moneyness <= moneyness <= max_moneyness
            ):
                continue
            by_strike[strike].append(iv)
            spots.append(spot)
            rates.append(rate)
            dividends.append(dividend)
            matched_rows += 1

    if len(by_strike) < 12:
        raise ValueError(f"Only {len(by_strike)} usable strikes found for {ticker} {expiration}")

    spot = statistics.median(spots)
    rate = statistics.median(rates)
    dividend = statistics.median(dividends)
    points = sorted((math.log(strike / spot), statistics.median(ivs)) for strike, ivs in by_strike.items())
    return points, spot, rate, dividend, matched_rows


def smooth_iv(points: list[tuple[float, float]], target_x: float, bandwidth: float) -> float:
    """Gaussian local-linear smoothing, including boundary-bias correction."""
    weighted = []
    for x_value, iv in points:
        distance = (x_value - target_x) / bandwidth
        if abs(distance) <= 4.5:
            weighted.append((math.exp(-0.5 * distance * distance), x_value - target_x, iv))
    if len(weighted) < 3:
        nearest = sorted(points, key=lambda point: abs(point[0] - target_x))[:5]
        weighted = [(1.0, x_value - target_x, iv) for x_value, iv in nearest]

    sw = sum(weight for weight, _, _ in weighted)
    sx = sum(weight * dx for weight, dx, _ in weighted)
    sy = sum(weight * iv for weight, _, iv in weighted)
    sxx = sum(weight * dx * dx for weight, dx, _ in weighted)
    sxy = sum(weight * dx * iv for weight, dx, iv in weighted)
    denominator = sw * sxx - sx * sx
    estimate = (sy * sxx - sx * sxy) / denominator if abs(denominator) > 1e-14 else sy / sw
    return min(1.50, max(0.03, estimate))


def gaussian_smooth(values: list[float], sigma: float = 2.2) -> list[float]:
    radius = max(1, round(4 * sigma))
    kernel = [math.exp(-0.5 * (offset / sigma) ** 2) for offset in range(-radius, radius + 1)]
    result = []
    for index in range(len(values)):
        numerator = 0.0
        denominator = 0.0
        for offset, weight in zip(range(-radius, radius + 1), kernel):
            source_index = min(len(values) - 1, max(0, index + offset))
            numerator += weight * values[source_index]
            denominator += weight
        result.append(numerator / denominator)
    return result


def trapezoid_area(xs: list[float], ys: list[float]) -> float:
    return sum((ys[i - 1] + ys[i]) * 0.5 * (xs[i] - xs[i - 1]) for i in range(1, len(xs)))


def build_distribution(
    points: list[tuple[float, float]],
    spot: float,
    years: float,
    rate: float,
    dividend: float,
    min_moneyness: float,
    max_moneyness: float,
    bandwidth: float,
    grid_points: int,
):
    if grid_points < 101:
        raise ValueError("--grid-points must be at least 101")
    observed_low = spot * math.exp(points[0][0])
    observed_high = spot * math.exp(points[-1][0])
    low_strike = max(spot * min_moneyness, observed_low)
    high_strike = min(spot * max_moneyness, observed_high)
    step = (high_strike - low_strike) / (grid_points - 1)
    strikes = [low_strike + step * index for index in range(grid_points)]
    ivs = [smooth_iv(points, math.log(strike / spot), bandwidth) for strike in strikes]
    calls = [black_scholes_call(spot, strike, years, rate, dividend, iv) for strike, iv in zip(strikes, ivs)]

    scale = math.exp(rate * years) / (step * step)
    raw_density = [0.0]
    for index in range(1, grid_points - 1):
        curvature = calls[index + 1] - 2.0 * calls[index] + calls[index - 1]
        raw_density.append(max(0.0, scale * curvature))
    raw_density.append(0.0)
    density = gaussian_smooth(raw_density)
    area = trapezoid_area(strikes, density)
    if area <= 0:
        raise ValueError("Could not recover a positive density from the fitted option curve")
    density = [value / area for value in density]

    cdf = [0.0]
    for index in range(1, len(strikes)):
        increment = 0.5 * (density[index - 1] + density[index]) * (strikes[index] - strikes[index - 1])
        cdf.append(min(1.0, cdf[-1] + increment))
    cdf[-1] = 1.0
    return strikes, ivs, calls, density, cdf


def interpolate_cdf(strikes: list[float], cdf: list[float], target: float) -> float:
    if target <= strikes[0]:
        return 0.0
    if target >= strikes[-1]:
        return 1.0
    for index in range(1, len(strikes)):
        if strikes[index] >= target:
            fraction = (target - strikes[index - 1]) / (strikes[index] - strikes[index - 1])
            return cdf[index - 1] + fraction * (cdf[index] - cdf[index - 1])
    return 1.0


def quantile(strikes: list[float], cdf: list[float], probability: float) -> float:
    for index in range(1, len(cdf)):
        if cdf[index] >= probability:
            width = cdf[index] - cdf[index - 1]
            fraction = (probability - cdf[index - 1]) / width if width > 0 else 0.0
            return strikes[index - 1] + fraction * (strikes[index] - strikes[index - 1])
    return strikes[-1]


def nice_ceiling(value: float) -> float:
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    choice = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return choice * magnitude


def nice_tick_step(span: float, target_ticks: int = 10) -> float:
    raw_step = span / target_ticks
    magnitude = 10 ** math.floor(math.log10(raw_step))
    scaled = raw_step / magnitude
    for choice in (1.0, 2.0, 2.5, 5.0, 10.0):
        if scaled <= choice:
            return choice * magnitude
    return 10.0 * magnitude


def make_svg(
    ticker: str,
    trade_date: str,
    expiration: str,
    spot: float,
    forward: float,
    years: float,
    strikes: list[float],
    density: list[float],
    cdf: list[float],
    matched_rows: int,
) -> str:
    width, height = 1400, 900
    left, right, top, bottom = 110, 1325, 165, 735
    plot_width, plot_height = right - left, bottom - top
    density_pct = [value * 100.0 for value in density]
    y_max = nice_ceiling(max(density_pct) * 1.08)
    q10 = quantile(strikes, cdf, 0.10)
    q90 = quantile(strikes, cdf, 0.90)

    def px(strike: float) -> float:
        return left + (strike - strikes[0]) / (strikes[-1] - strikes[0]) * plot_width

    def py(value: float) -> float:
        return bottom - value / y_max * plot_height

    curve = " ".join(f"{px(strike):.1f},{py(value):.1f}" for strike, value in zip(strikes, density_pct))
    area = f"{left},{bottom} {curve} {right},{bottom}"
    interval_points = [(strike, value) for strike, value in zip(strikes, density_pct) if q10 <= strike <= q90]
    interval = " ".join(f"{px(strike):.1f},{py(value):.1f}" for strike, value in interval_points)
    interval_area = f"{px(q10):.1f},{bottom} {interval} {px(q90):.1f},{bottom}"

    x_step = nice_tick_step(strikes[-1] - strikes[0])
    first_tick = math.ceil(strikes[0] / x_step) * x_step
    x_ticks = []
    tick = first_tick
    while tick <= strikes[-1]:
        x_ticks.append(tick)
        tick += x_step

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1400" height="900" fill="#f7f8fb"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#172438}.muted{fill:#556277}.grid{stroke:#cfd6df;stroke-width:1}.axis{stroke:#26364d;stroke-width:2}</style>',
        f'<text x="70" y="60" font-size="30" font-weight="700">{escape(ticker.upper())} option-implied terminal-price distribution</text>',
        f'<text x="70" y="96" class="muted" font-size="17">Trade date {escape(trade_date)}  |  Expiration {escape(expiration)} ({round(years * 365)} days)  |  Risk-neutral pricing probabilities</text>',
        '<line x1="70" y1="120" x2="1330" y2="120" stroke="#d6dce5"/>',
    ]

    for index in range(6):
        value = y_max * index / 5
        y_value = py(value)
        lines.append(f'<line class="grid" x1="{left}" y1="{y_value:.1f}" x2="{right}" y2="{y_value:.1f}"/>')
        lines.append(f'<text x="{left - 14}" y="{y_value + 5:.1f}" text-anchor="end" font-size="14">{value:.1f}%</text>')
    for value in x_ticks:
        x_value = px(value)
        lines.append(f'<line class="grid" x1="{x_value:.1f}" y1="{top}" x2="{x_value:.1f}" y2="{bottom}"/>')
        lines.append(f'<text x="{x_value:.1f}" y="{bottom + 30}" text-anchor="middle" font-size="14">${value:.0f}</text>')

    lines.extend([
        f'<polygon points="{area}" fill="#dbe6f2" opacity="0.82"/>',
        f'<polygon points="{interval_area}" fill="#8ab6d6" opacity="0.68"/>',
        f'<polyline points="{curve}" fill="none" stroke="#500000" stroke-width="4"/>',
        f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>',
        f'<text x="{(left + right) / 2:.1f}" y="{bottom + 68}" text-anchor="middle" font-size="18" font-weight="700">{escape(ticker.upper())} price at expiration</text>',
        f'<text x="32" y="{(top + bottom) / 2:.1f}" transform="rotate(-90 32 {(top + bottom) / 2:.1f})" text-anchor="middle" font-size="18" font-weight="700">Probability density (% per $1)</text>',
    ])

    for value, label, color, dash, label_x, label_y, anchor in (
        (spot, f"Spot ${spot:.2f}", "#500000", "8 6", -10, top + 24, "end"),
        (forward, f"Forward ${forward:.2f}", "#2374a6", "3 5", 10, top + 50, "start"),
    ):
        x_value = px(value)
        lines.append(f'<line x1="{x_value:.1f}" y1="{top}" x2="{x_value:.1f}" y2="{bottom}" stroke="{color}" stroke-width="2.5" stroke-dasharray="{dash}"/>')
        lines.append(f'<text x="{x_value + label_x:.1f}" y="{label_y}" text-anchor="{anchor}" fill="{color}" font-size="15" font-weight="700">{escape(label)}</text>')

    lines.extend([
        f'<rect x="980" y="185" width="315" height="82" rx="10" fill="#ffffff" stroke="#c9d1dc"/>',
        '<text x="1005" y="218" font-size="16" font-weight="700">Market-implied summary</text>',
        '<rect x="1005" y="237" width="24" height="12" fill="#8ab6d6" opacity="0.8"/>',
        '<text x="1039" y="249" class="muted" font-size="13">Central 80% interval</text>',
        f'<text x="70" y="835" class="muted" font-size="14">Derived from {matched_rows} valid call/put IV observations using a smoothed smile and Breeden-Litzenberger density.</text>',
        '<text x="70" y="862" class="muted" font-size="13">Risk-neutral probabilities reflect option prices and risk premia; they are not a forecast of real-world outcome frequencies. Density is normalized over the displayed range.</text>',
        '</svg>',
    ])
    return "\n".join(lines)


def write_density_csv(path: Path, strikes, ivs, calls, density, cdf) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["terminal_price", "smoothed_iv", "model_call_price", "risk_neutral_density", "cdf"])
        writer.writerows(zip(strikes, ivs, calls, density, cdf))


def main() -> None:
    args = parse_args()
    ticker = args.ticker.upper()
    trade_day = date.fromisoformat(args.date)
    expiration_day = date.fromisoformat(args.expiration)
    if expiration_day <= trade_day:
        raise ValueError("Expiration must be after the trade date")
    source = args.input or DATA_ROOT / trade_day.strftime("%Y") / trade_day.strftime("%m") / f"{args.date}.csv"
    output = args.output or Path(f"{ticker.lower()}_market_distribution_{args.expiration}.svg")
    csv_output = args.csv_output or output.with_suffix(".csv")
    years = (expiration_day - trade_day).days / 365.0

    points, spot, rate, dividend, matched_rows = load_smile(
        source,
        ticker,
        args.date,
        args.expiration,
        args.min_moneyness,
        args.max_moneyness,
        args.max_iv,
    )
    strikes, ivs, calls, density, cdf = build_distribution(
        points,
        spot,
        years,
        rate,
        dividend,
        args.min_moneyness,
        args.max_moneyness,
        args.bandwidth,
        args.grid_points,
    )
    forward = spot * math.exp((rate - dividend) * years)
    svg = make_svg(ticker, args.date, args.expiration, spot, forward, years, strikes, density, cdf, matched_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    write_density_csv(csv_output, strikes, ivs, calls, density, cdf)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")
    print(f"Expiration {args.expiration}: {matched_rows} valid option rows, spot ${spot:.2f}, forward ${forward:.2f}")
    print(
        "Risk-neutral 10th/median/90th percentiles: "
        f"${quantile(strikes, cdf, 0.10):.2f} / ${quantile(strikes, cdf, 0.50):.2f} / ${quantile(strikes, cdf, 0.90):.2f}"
    )


if __name__ == "__main__":
    main()
