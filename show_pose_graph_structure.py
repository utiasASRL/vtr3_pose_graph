import os
import matplotlib.pyplot as plt
import numpy as np

from src.vtr_pose_graph.graph_factory import Rosbag2GraphFactory
from src.vtr_pose_graph.graph_iterators import TemporalIterator
from src.vtr_utils.plot_utils import plot_graph
import src.vtr_pose_graph.graph_utils as g_utils
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Verify Point Cloud',
                        description = 'Plots point cloud to verify alignment')
    parser.add_argument('-g', '--graph', default="graph")      # option that takes a value
    args = parser.parse_args()

    #offline_graph_dir = os.path.join(os.getenv("VTRTEMP"), "lidar/2023-02-13/2023-02-13/long_repeat")
    offline_graph_dir = '/home/alec/ASRL/vtr3/temp/lidar/2023-02-20/2023-02-20/graph_test2'
    offline_graph_dir = os.path.join(os.getenv("VTRROOT"), "vtr_testing_lidar", "tmp", args.graph)
    factory = Rosbag2GraphFactory(offline_graph_dir)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)
    plot_graph(test_graph)

    
    plt.show()