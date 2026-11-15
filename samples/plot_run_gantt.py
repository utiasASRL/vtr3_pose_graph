import re
import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch

ROBOTS = {0: ("Robot 0", "#2E8B3A"),
          1: ("Robot 1", "#8E4585"),
          2: ("Robot 2", "#C9971A"),
          3: ("Robot 3", "#0082A8")}

len_re = re.compile(r"Path (0x[0-9a-f]+) was ([\d.]+)m long with (\d+) vertices and is a (teach|repeat)")
teacher_re = re.compile(r"Path (0x[0-9a-f]+) was taught by path (0x[0-9a-f]+)")
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
        try:
            with open(fname) as f:
                lines = f.readlines()
        except (UnicodeDecodeError, OSError) as err:
            print(f"Skipping {fname}: {err}")
            continue
        if True:
            for line in lines:
                if m := len_re.match(line):
                    r = runs.setdefault(m[1], {"id": m[1]})
                    r["length"] = max(r.get("length", 0), float(m[2]))
                    r["kind"] = m[4]
                elif m := teacher_re.match(line):
                    r = runs.setdefault(m[1], {"id": m[1]})
                    r["teacher"] = (int(m[2], 16) >> 44) & 0xF
                    r["teacher_id"] = m[2]
                elif m := time_re.match(line):
                    r = runs.setdefault(m[1], {"id": m[1]})
                    r["start"] = datetime.fromisoformat(m[2])
                    # Logs from different graphs can disagree slightly; keep the longest
                    r["duration"] = max(r.get("duration", 0), float(m[3]))

    # Run ids are 48 bits; the top hex digit is the robot id
    for r in runs.values():
        r["robot"] = (int(r["id"], 16) >> 44) & 0xF

    # Runs skipped by print_pose_graph_info.py have no start time; drop them
    ordered = sorted((r for r in runs.values() if "start" in r), key=lambda r: r["start"])
    print(f"{'start':<20} {'end':<9} {'robot':<12} {'kind':<7} {'length':>9} {'dur':>8} {'taught by':<12} id")
    for r in ordered:
        end = r["start"] + timedelta(seconds=r["duration"])
        print(f"{r['start']:%Y-%m-%d %H:%M:%S}  {end:%H:%M:%S} {ROBOTS[r['robot']][0]:<12} "
              f"{r['kind']:<7} {r['length']:8.1f}m {r['duration']:7.1f}s "
              f"{ROBOTS[r['teacher']][0] if 'teacher' in r else '-':<12} {r['id']}")

 
    plt.rcParams["hatch.linewidth"] = 1.5
    fig, ax = plt.subplots(figsize=(14, 3.2))
    t0 = ordered[0]["start"]
    min_width = 10  # seconds, so zero-length runs stay visible
    bar_height = 0.85
    for r in ordered:
        name, colour = ROBOTS[r["robot"]]
        y = r["robot"]
        start = (r["start"] - t0).total_seconds() / 60  # minutes elapsed
        width = max(r["duration"], min_width) / 60
        if r["kind"] == "teach":
            ax.barh(y, width, left=start, height=bar_height, color=colour, edgecolor="white", linewidth=0.5)
        else:
            # Hatch colour = robot that taught the route (falls back to the repeating robot)
            teach_colour = ROBOTS[r["teacher"]][1] if "teacher" in r else colour
            ax.barh(y, width, left=start, height=bar_height, facecolor="white", edgecolor=teach_colour,
                    hatch="////", linewidth=1.0)
 
    ax.set_yticks(list(ROBOTS), [n for n, _ in ROBOTS.values()])
    ax.set_ylim(len(ROBOTS) - 0.5 - (1 - bar_height) / 2, -0.5 + (1 - bar_height) / 2)  # tight, inverted
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, steps=[1, 2, 5, 10]))
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Elapsed time (min)")
    ax.grid(axis="x", color="#e0e0e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Legend lives in the figure margin on the left; tight_layout's rect reserves that space
    fig.legend(handles=[Patch(facecolor="#555555", label="teach (solid)"),
                        Patch(facecolor="white", edgecolor="#555555", hatch="////", label="repeat (hatched)")]
                       + [Patch(facecolor=c, label=f"taught by {n}") for n, c in ROBOTS.values()],
               loc="center right", frameon=False, ncol=1, bbox_to_anchor=(0.99, 0.5), fontsize=8)
    fig.tight_layout(rect=(0, 0, 0.89, 1))
 
    if args.output:
        fig.savefig(args.output, dpi=150)
    else:
        plt.show()
