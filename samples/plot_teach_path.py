import os
import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator, TemporalIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"))
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root
    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))
    print(path_matrix.shape)

    x = []
    y = []
    t = []
    p = []         # List to hold cumulative path lengths
    path_len = 0.0 # Initialize path length for PrivilegedIterator

    # Loop over vertices and edges using PrivilegedIterator,
    # accumulate positions and path length iteratively.
    for v, e in PriviledgedIterator(v_start):
        pos = v.T_v_w.r_ba_ina()
        x.append(pos[0])
        y.append(pos[1])
        t.append(v.stamp / 1e9)
        # Only update path length if an edge is valid
        if e is not None:
            path_len += np.linalg.norm(e.T.r_ba_ina())
        p.append(path_len)

    # Print the cumulative path length computed iteratively
    print(f"Iterative total path length using PrivilegedIterator: {path_len:.3f}")

    plt.figure(0)
    plt.plot(x, y, label="Teach", linewidth=5)
    plt.axis('equal')
    plt.show()
