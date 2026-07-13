import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np
import pickle as pkl
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator, TemporalIterator, BreadthFirstSearchIterator, DepthFirstSearchIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

import pdb

custom_colors = [
  "#3cb44b", #  0 - mr_green
  "#911eb4", #  1 - prof_plum
  "#ffe119", #  2 - col_mustard
  "#4363d8", #  3 - mrs_peacock
  "#e6194b", #  4 - red
  "#f58231", #  5 - orange
  "#bfef45", #  6 - lime
  "#42d4f4", #  7 - cyan
  "#f032e6", #  8 - magenta
  "#fabebe", #  9 - pink
  "#ffd8b1", # 10 - apricot
  "#fffac8", # 11 - cream
  "#aaffc3", # 12 - mint
  "#a9a9a9", # 13 - grey
  "#ffffff", # 14 - white
  "#000000", # 15 - black
]

# Create the custom colormap
custom_cmap = mcolors.ListedColormap(custom_colors)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"))      # option that takes a value
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")
    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    x = []
    y = []
    t = []
    c = []

    path_len=0
    teach_count = 0
    count = 0

    for v, e in BreadthFirstSearchIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)
        if e.mode == 1 and (e.from_id < e.to_id):
            teach_count += 1
        count += 1
        # c.append(e.mode)

        c.append((v.id >> 60) & 0xF) # color based on rid
            
        path_len += np.linalg.norm(e.T.r_ba_ina())


    fig, ax = plt.subplots()

    c_map = mcolors.ListedColormap(custom_colors)
    print(f"path length : {path_len}")
    print(f"num taught vertices : {teach_count}, num total vertices : {count}")    
    sc = ax.scatter(x, y, label="Teach", c=c, cmap=c_map, vmin=-0.5, vmax=15.5)
    # ax.set_title()
    id = args.graph[-8]
    ax.set_title(
        f"vtr_pose_graph", 
        fontsize=14, 
        fontweight='bold', 
        pad=15,
        color=c_map(int(id))
    )
    cbar = plt.colorbar(sc, ticks=range(16))
    # plt.colorbar(label='Autonomous (0), Manual (1), Unknown (2)', ax=None)
    cbar.set_label('Run ID (rid)')

    plt.axis('equal')
    plt.show()
