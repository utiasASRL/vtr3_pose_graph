import re
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

ROBOTS = {0: ("mr_green", "#2E8B3A"),
          1: ("prof_plum", "#8E4585"),
          2: ("col_mustard", "#C9971A"),
          3: ("mrs_peacock", "#0082A8")}

len_re = re.compile(r"Path (0x[0-9a-f]+) was ([\d.]+)m long with (\d+) vertices and is a (teach|repeat)")
time_re = re.compile(r"Path (0x[0-9a-f]+) started at (\S+ \S+) took ([\d.]+)s")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Plot run gantt',
                        description = 'Gantt chart of teach and repeat runs per robot, built from the output of print_pose_graph_info.py')
    parser.add_argument('info', nargs="+", help="One or more text files containing the output of print_pose_graph_info.py")
    parser.add_argument('-o', '--output', default=None, help="Save the figure to this file instead of showing it.")
    args = parser.parse_args()

    runs = {}
    for fname in args.info:
        with open(fname) as f:
            for line in f:
                if m := len_re.match(line):
                    r = runs.setdefault(m[1], {"id": m[1]})
                    r["length"] = max(r.get("length", 0), float(m[2]))
                    r["kind"] = m[4]
                elif m := time_re.match(line):
                    r = runs.setdefault(m[1], {"id": m[1]})
                    r["start"] = datetime.fromisoformat(m[2])
                    # Logs from different graphs can disagree slightly; keep the longest
                    r["duration"] = max(r.get("duration", 0), float(m[3]))

    # Run ids are 48 bits; the top hex digit is the robot id
    for r in runs.values():
        r["robot"] = (int(r["id"], 16) >> 44) & 0xF

    ordered = sorted(runs.values(), key=lambda r: r["start"])
    print(f"{'start':<20} {'end':<9} {'robot':<12} {'kind':<7} {'length':>9} {'dur':>8}  id")
    for r in ordered:
        end = r["start"] + timedelta(seconds=r["duration"])
        print(f"{r['start']:%Y-%m-%d %H:%M:%S}  {end:%H:%M:%S} {ROBOTS[r['robot']][0]:<12} "
              f"{r['kind']:<7} {r['length']:8.1f}m {r['duration']:7.1f}s  {r['id']}")

    plt.rcParams["hatch.linewidth"] = 1.5
    fig, ax = plt.subplots(figsize=(14, 4.5))
    min_width = 10  # seconds, so zero-length runs stay visible
    for r in ordered:
        name, colour = ROBOTS[r["robot"]]
        y = r["robot"]
        start = mdates.date2num(r["start"])
        width = max(r["duration"], min_width) / 86400
        if r["kind"] == "teach":
            ax.barh(y, width, left=start, height=0.6, color=colour, edgecolor="white", linewidth=0.5)
        else:
            ax.barh(y, width, left=start, height=0.6, facecolor="white", edgecolor=colour,
                    hatch="////", linewidth=1.0)

    ax.set_yticks(list(ROBOTS), [n for n, _ in ROBOTS.values()])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel(f"Time on {ordered[0]['start']:%Y-%m-%d}")
    ax.set_title("Teach and repeat runs by robot")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(handles=[Patch(facecolor="#555555", label="teach"),
                       Patch(facecolor="white", edgecolor="#555555", hatch="////", label="repeat")],
              loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1, 1.12))
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
    else:
        plt.show()
