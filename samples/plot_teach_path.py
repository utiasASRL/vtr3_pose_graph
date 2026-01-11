import matplotlib.pyplot as plt
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_pose_graph.graph_utils as g_utils
import vtr_regression_testing.path_comparison as vtr_path
import argparse
import pylgmath.so3.operations as so3op

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'plot_teach_path.py',
                        description = 'Plots the route of the teach path for a VR pose graph')
    parser.add_argument('graph', help="File path to VR pose graph folder")
    args = parser.parse_args()

    offline_graph_dir = args.graph
    factory = Rosbag2GraphFactory(offline_graph_dir)

    full_graph = factory.buildGraph()
    teach_graph = full_graph.get_privileged_subgraph()
    print(f"Full graph {full_graph} has {full_graph.number_of_vertices} vertices and {full_graph.number_of_edges} edges")
    print(f"Teach subgraph {teach_graph} has {teach_graph.number_of_vertices} vertices and {teach_graph.number_of_edges} edges")

    g_utils.set_world_frame(teach_graph, teach_graph.root)

    v_start = teach_graph.root

    path_matrix = vtr_path.path_to_matrix(teach_graph, PriviledgedIterator(v_start))

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
    plt.scatter(pose_vec[0], pose_vec[1], label="Teach", linewidth=5)
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

