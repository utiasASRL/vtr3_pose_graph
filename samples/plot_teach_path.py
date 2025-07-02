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

    pose_vec = np.zeros((6, 0))
    t = []

    for v, e in PriviledgedIterator(v_start):
        pose_vec = np.hstack((pose_vec, np.vstack((v.T_v_w.r_ba_ina(), so3op.rot2vec(v.T_w_v.C_ba()) * 180/np.pi))))
        t.append(v.stamp / 1e9)
    t = np.array(t)
    t_sort = np.argsort(t)
    t = t[t_sort]
    pose_vec = pose_vec[:, t_sort]

    plt.figure(0)
    plt.plot(pose_vec[0], pose_vec[1], label="Teach", linewidth=5)
    plt.axis('equal')
    plt.legend()

    plt.figure(1)
    plt.plot(t, pose_vec[3], label="Roll")
    plt.plot(t, pose_vec[4], label="Pitch")
    plt.plot(t, pose_vec[5], label="Yaw")
    plt.title("Orientation")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle ($^\circ$)")
    plt.legend()

    plt.figure(2)
    plt.plot(t, pose_vec[0], label="X")
    plt.plot(t, pose_vec[1], label="Y")
    plt.plot(t, pose_vec[2], label="Z")
    plt.title("Position")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()

    plt.show()

