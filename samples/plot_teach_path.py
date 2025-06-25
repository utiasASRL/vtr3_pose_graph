import os
import matplotlib.pyplot as plt
import numpy as np

from pylgmath import Transformation
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse
import pylgmath.so3.operations as so3op

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('graph')      # option that takes a value
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)

    v_start = test_graph.root

    path_matrix = vtr_path.path_to_matrix(test_graph, PriviledgedIterator(v_start))

    x = []
    y = []
    xg = []
    yg = []
    r = []
    p = []
    h = []
    t = []

    for v, e in PriviledgedIterator(v_start):
        x.append(v.T_v_w.r_ba_ina()[0])
        y.append(v.T_v_w.r_ba_ina()[1])
        orientation = so3op.rot2vec(v.T_w_v.C_ba()) * 180/np.pi
        r.append(orientation[0])
        p.append(orientation[1])
        h.append(orientation[2])
        t.append(v.stamp / 1e9)

    plt.figure(0)
    plt.plot(x, y, label="Teach", linewidth=5)
    plt.axis('equal')
    plt.legend()

    plt.figure(1)
    plt.plot(t, r, label="Roll")
    plt.plot(t, p, label="Pitch")
    plt.plot(t, h, label="Yaw")
    plt.legend()

    plt.show()

