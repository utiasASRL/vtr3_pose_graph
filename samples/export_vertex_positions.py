#!/usr/bin/env python3
"""
export_vertex_positions.py -g <graph dir> [-o positions.csv]

Write every vertex of a VTR posegraph with its position in the root vertex's frame:
    vid,x,y,z,stamp
Uses the same VTR tools and world frame as plot_teach_path.py, so the result lines up
with that plot. Run it once per teach graph on a machine with the VTR Python packages;
animate_robots.py then needs only this CSV.
"""
import argparse
import csv
import os

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export vertex positions of a VTR posegraph")
    ap.add_argument("-g", "--graph", default=os.getenv("VTRDATA"), help="graph directory (as for plot_teach_path.py)")
    ap.add_argument("-o", "--out", default="vertex_positions.csv")
    args = ap.parse_args()

    graph = Rosbag2GraphFactory(args.graph).buildGraph()
    print(f"{graph.number_of_vertices} vertices, {graph.number_of_edges} edges")
    g_utils.set_world_frame(graph, graph.root)

    n = 0
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vid", "x", "y", "z", "stamp"])
        for v, _ in PriviledgedIterator(graph.root):
            r = v.T_v_w.r_ba_ina()
            w.writerow([int(v.id), float(r[0][0]), float(r[1][0]), float(r[2][0]), int(v.stamp)])
            n += 1
    print(f"wrote {n} vertices to {args.out}")