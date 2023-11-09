import os
import matplotlib.pyplot as plt
import numpy as np

from src.vtr_pose_graph.graph_factory import Rosbag2GraphFactory
from src.vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator
import src.vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default="graph")      # option that takes a value
    args = parser.parse_args()

    offline_graph_dir = os.path.join(os.getenv("VTRROOT"), "vtr_testing_lidar", "tmp", args.graph)
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

    for v, e in PriviledgedIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        t.append(v.stamp / 1e9)

    plt.figure(0)
    plt.plot(x, y, label="Teach", linewidth=5)
    plt.axis('equal')

    plt.show()

