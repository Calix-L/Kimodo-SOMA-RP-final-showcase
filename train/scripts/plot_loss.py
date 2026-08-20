#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_loss_rows(csv_path: Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((int(row["step"]), float(row["loss"])))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def save_matplotlib(rows: list[tuple[int, float]], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = zip(*rows)
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, losses, linewidth=1.8)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title("Kimodo human7951 overfit loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    return out


def save_svg(rows: list[tuple[int, float]], out: Path) -> Path:
    width, height = 960, 540
    left, right, top, bottom = 76, 28, 34, 64
    plot_w = width - left - right
    plot_h = height - top - bottom

    min_step, max_step = rows[0][0], rows[-1][0]
    min_loss = min(loss for _, loss in rows)
    max_loss = max(loss for _, loss in rows)
    if max_loss == min_loss:
        max_loss += 1.0

    def x_pos(step: int) -> float:
        return left + (step - min_step) / (max_step - min_step) * plot_w

    def y_pos(loss: float) -> float:
        return top + (max_loss - loss) / (max_loss - min_loss) * plot_h

    points = " ".join(f"{x_pos(step):.2f},{y_pos(loss):.2f}" for step, loss in rows)
    y_ticks = [min_loss + (max_loss - min_loss) * i / 4 for i in range(5)]
    x_ticks = [min_step + round((max_step - min_step) * i / 5) for i in range(6)]

    grid = []
    for loss in y_ticks:
        y = y_pos(loss)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        grid.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12" fill="#374151">{loss:.3f}</text>'
        )
    for step in x_ticks:
        x = x_pos(step)
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="#f3f4f6"/>'
        )
        grid.append(
            f'<text x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" fill="#374151">{step}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{width/2:.0f}" y="24" text-anchor="middle" font-size="18" font-family="Arial, sans-serif" fill="#111827">Kimodo human7951 overfit loss</text>
  <g font-family="Arial, sans-serif">
    {''.join(grid)}
    <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>
    <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>
    <polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>
    <text x="{width/2:.0f}" y="{height-18}" text-anchor="middle" font-size="14" fill="#111827">step</text>
    <text x="20" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 20 {height/2:.0f})" font-size="14" fill="#111827">train loss</text>
  </g>
</svg>
'''
    out.write_text(svg, encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    rows = read_loss_rows(args.run_dir / "loss_history.csv")
    try:
        out = save_matplotlib(rows, args.run_dir / "loss_curve.png")
    except ModuleNotFoundError:
        out = save_svg(rows, args.run_dir / "loss_curve.svg")
    print(out)


if __name__ == "__main__":
    main()
