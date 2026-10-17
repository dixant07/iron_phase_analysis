#!/usr/bin/env python3
"""Count Fe atoms by oxygen coordination number in exported OVITO dumps.

main.py exports type 1 for Fe and stores the number of oxygen neighbors
selected by CrystalNN in coordination_number (see count.py). Reuse that
property rather than recomputing neighbors or filtering by geometry: distorted
CN=4 and CN=6 Fe atoms are included in their respective coordination counts.

Run with the project's Python environment:
    .venv/bin/python fe_coordination.py
    .venv/bin/python fe_coordination.py --from-frame 0 --to-frame 1001

Outputs: count and CN4/CN6 percentage CSVs, PNG/PDF plots, and a missing-frame list. Missing
frames have blank counts in the CSV and gaps in the plots. Only matplotlib
is required beyond the Python standard library.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def read_coordination_counts(path: Path, fe_type: int = 1) -> tuple[int, Counter]:
    """Read one complete LAMMPS snapshot, finding columns by header name.

    Return the exported timestep and a histogram of Fe oxygen coordination.
    Fail on malformed/truncated files instead of writing partial counts.
    """
    with path.open() as stream:
        def expect(prefix: str) -> str:
            line = stream.readline().strip()
            if line != prefix:
                raise ValueError(f"Expected {prefix!r}, got {line!r}")
            return line

        expect("ITEM: TIMESTEP")
        timestep = int(stream.readline())
        expect("ITEM: NUMBER OF ATOMS")
        n_atoms = int(stream.readline())
        if n_atoms < 0:
            raise ValueError("Negative number of atoms")
        if not stream.readline().startswith("ITEM: BOX BOUNDS"):
            raise ValueError("Missing BOX BOUNDS header")
        for _ in range(3):
            bounds = stream.readline().split()
            if len(bounds) < 2:
                raise ValueError("Incomplete box bounds")
            for value in bounds:
                float(value)
        header = stream.readline().split()
        if header[:2] != ["ITEM:", "ATOMS"]:
            raise ValueError("Missing ATOMS header")
        columns = header[2:]
        if "type" not in columns or "coordination_number" not in columns:
            raise ValueError("ATOMS must contain type and coordination_number")
        type_index = columns.index("type")
        cn_index = columns.index("coordination_number")
        counts = Counter()
        for row in range(n_atoms):
            fields = stream.readline().split()
            if len(fields) != len(columns):
                raise ValueError(f"Incomplete atom row {row + 1} of {n_atoms}")
            if int(fields[type_index]) == fe_type:
                cn = int(fields[cn_index])
                if cn < 0:
                    raise ValueError(f"Negative Fe coordination at atom row {row + 1}")
                counts[cn] += 1
        if any(line.strip() for line in stream):
            raise ValueError("Expected exactly one snapshot per dump file")
    return timestep, counts


def save_outputs(results: dict, frames: range, output_dir: Path) -> list[Path]:
    """Save all observed CN series, retaining missing-frame gaps."""
    import matplotlib
    matplotlib.use("Agg")  # Works on machines without a graphical display.
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    coordination_numbers = sorted({cn for _, counts in results.values() for cn in counts})
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fe_coordination_counts.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "timestep", "status", "total_fe"] +
                        [f"cn_{cn}" for cn in coordination_numbers])
        for frame in frames:
            if frame not in results:
                writer.writerow([frame, "", "missing", ""] + [""] * len(coordination_numbers))
            else:
                timestep, counts = results[frame]
                writer.writerow([frame, timestep, "ok", sum(counts.values())] +
                                [counts[cn] for cn in coordination_numbers])

    missing_path = output_dir / "missing_frames.txt"
    missing_path.write_text("".join(f"{frame}\n" for frame in frames if frame not in results))

    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    for cn in coordination_numbers:
        values = [results[frame][1][cn] if frame in results else float("nan")
                  for frame in frames]
        ax.plot(list(frames), values, linewidth=1.2, label=f"CN = {cn}")
    # A lone available frame would otherwise be invisible in a line plot.
    if len(results) == 1:
        frame = next(iter(results))
        for line, cn in zip(ax.lines, coordination_numbers):
            ax.scatter([frame], [results[frame][1][cn]], color=line.get_color(), s=18)
    ax.set(title="Fe atoms by oxygen coordination number", xlabel="Frame number",
           ylabel="Number of Fe atoms")
    ax.set_xlim(frames.start - 0.5, frames.stop - 0.5)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)
    ax.legend(title="Oxygen neighbors (CrystalNN)", ncol=min(4, len(coordination_numbers)))
    fig.suptitle(f"Frames {frames.start}–{frames.stop - 1} | "
                 f"{len(results)} analyzed, {len(frames) - len(results)} missing", fontsize=10)
    fig.text(0.5, 0.012, "Source: reduction.<frame>.dump; exported coordination_number; "
             "missing frames are gaps", ha="center", fontsize=8)
    plots = [output_dir / f"fe_coordination_vs_frame.{suffix}" for suffix in ("png", "pdf")]
    for path in plots:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    percentage_outputs = save_cn4_cn6_percentages(results, frames, output_dir)
    return [csv_path, *plots, missing_path, *percentage_outputs]


def save_cn4_cn6_percentages(results: dict, frames: range, output_dir: Path) -> list[Path]:
    """Normalize only CN4 and CN6 counts; all other CNs are excluded.

    Percentages are 100 * count / (CN4 + CN6). Missing frames and zero
    denominators have undefined percentages, stored blank and plotted as gaps.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    csv_path = output_dir / "fe_cn4_cn6_percentages.csv"
    series = {4: [], 6: []}
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "timestep", "status", "cn_4", "cn_6",
                         "cn_4_plus_cn_6", "pct_cn4", "pct_cn6"])
        for frame in frames:
            percentages = [float("nan"), float("nan")]
            if frame not in results:
                writer.writerow([frame, "", "missing", "", "", "", "", ""])
            else:
                timestep, counts = results[frame]
                total = counts[4] + counts[6]
                if total:
                    percentages = [100.0 * counts[cn] / total for cn in (4, 6)]
                writer.writerow([frame, timestep, "ok" if total else "no_cn4_or_cn6",
                                 counts[4], counts[6], total,
                                 *percentages] if total else
                                [frame, timestep, "no_cn4_or_cn6", counts[4], counts[6],
                                 total, "", ""])
            for cn, percentage in zip((4, 6), percentages):
                series[cn].append(percentage)

    fig, ax = plt.subplots(figsize=(13, 7), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94))
    for cn, values in series.items():
        ax.plot(list(frames), values, linewidth=1.4, label=f"CN = {cn}",
                marker="o" if len(results) == 1 else None)
    ax.set(title="Fe coordination: CN4 and CN6 percentages", xlabel="Frame number",
           ylabel="Share of Fe atoms with CN4 or CN6 (%)", ylim=(0, 100),
           xlim=(frames.start - 0.5, frames.stop - 0.5))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.suptitle(f"Frames {frames.start}–{frames.stop - 1} | "
                 f"{len(results)} analyzed, {len(frames) - len(results)} missing", fontsize=10)
    fig.text(0.5, 0.015,
             "Percentage = 100 × count(CN) / [count(CN4) + count(CN6)]. Other CNs excluded.\n"
             "Source: exported CrystalNN oxygen coordination; missing/undefined values are gaps.",
             ha="center", fontsize=8)
    plots = [output_dir / f"fe_cn4_cn6_percentages_vs_frame.{suffix}"
             for suffix in ("png", "pdf")]
    for path in plots:
        fig.savefig(path, dpi=200)
    plt.close(fig)
    return [csv_path, *plots]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "analysis_results/output_ovito")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "analysis_results/fe_coordination")
    parser.add_argument("--from-frame", type=int, default=0, help="First frame, inclusive (default: 0)")
    parser.add_argument("--to-frame", type=int, default=1001, help="Last frame, inclusive (default: 1001)")
    parser.add_argument("--fe-type", type=int, default=1, help="Fe atom type in dump (default: 1, as in main.py)")
    parser.add_argument("--strict", action="store_true", help="Fail if any requested dump is missing")
    args = parser.parse_args()
    if args.from_frame < 0 or args.to_frame < args.from_frame:
        parser.error("Require 0 <= --from-frame <= --to-frame")
    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")
    frames = range(args.from_frame, args.to_frame + 1)
    missing = [f for f in frames if not (args.input_dir / f"reduction.{f}.dump").is_file()]
    if args.strict and missing:
        parser.error(f"{len(missing)} requested dumps missing; first missing frame: {missing[0]}")
    if missing:
        print(f"Warning: {len(missing)} requested dumps are missing; counts will be blank, "
              "with gaps in the plots.", file=sys.stderr)
    results = {}
    for frame in frames:
        path = args.input_dir / f"reduction.{frame}.dump"
        if frame in missing:
            continue
        try:
            timestep, counts = read_coordination_counts(path, args.fe_type)
            if not counts:
                raise ValueError(f"No Fe atoms found with type {args.fe_type}")
        except (OSError, ValueError) as exc:
            parser.error(f"{path}: {exc}")
        results[frame] = (timestep, counts)
        if len(results) % 100 == 0:
            print(f"Read {len(results)} frames (latest: {frame})", flush=True)
    if not results:
        parser.error("No frames were available for analysis")
    for path in save_outputs(results, frames, args.output_dir):
        print(f"Saved: {path}")
    totals = [sum(counts.values()) for _, counts in results.values()]
    print(f"Analyzed {len(results)}/{len(frames)} frames; Fe atoms per frame: "
          f"{min(totals)}–{max(totals)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
