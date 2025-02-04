import os
import time

import numpy as np
import matplotlib.pyplot as plt

from sensor_msgs_py.point_cloud2 import read_points
import open3d as o3d
from pylgmath import Transformation
import sys
sys.path.append('/home/desiree/ASRL/vtr3/vtr3_posegraph_tools/vtr3_pose_graph/src')

from vtr_utils.plot_utils import extract_map_from_vertex
import argparse


from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator, SpatialIterator
import vtr_pose_graph.graph_utils as g_utils



if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'Plot Point Clouds Path',
                            description = 'Plots point clouds')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")      # option that takes a value
    parser.add_argument('-f', '--filter', type=int, nargs="*", help="Select only some of the repeat runs. Default plots all runs.")
    args = parser.parse_args()

    factory = Rosbag2GraphFactory(args.graph)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    g_utils.set_world_frame(test_graph, test_graph.root)
    v_start = test_graph.root

    x = []
    y = []
    live_2_map = []
    map_2_live = []


    first = True
    paused = False
    def toggle(vis):
        global paused
        paused = not paused
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_key_callback(ord(' '), toggle)
    vis.create_window()

    pcd = o3d.geometry.PointCloud()
    vis.poll_events()
    vis.update_renderer()

    radius_of_interest = 100

    for i in range(test_graph.major_id + 1):
        v_start = test_graph.get_vertex((i, 0))
        paused = True
        for vertex, e in TemporalIterator(v_start):

            new_points, map_ptr = extract_map_from_vertex(test_graph, vertex)

            print(new_points.shape)

            robot_position = vertex.T_v_w.r_ba_ina().reshape((3,) )

            x.append(vertex.T_v_w.r_ba_ina()[0]) 
            y.append(vertex.T_v_w.r_ba_ina()[1])

            pcd.points = o3d.utility.Vector3dVector(new_points.T)
            if np.allclose(map_ptr.matrix(), np.eye(4)):
                pcd.paint_uniform_color((1.0, 0.0, 0.0))  # Red color for identity matrix
            else:
                pcd.paint_uniform_color((0.1*vertex.run, 0.25*vertex.run, 0.45))

            colors = np.asarray(pcd.colors)

            if first:
                first = False
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            t = time.time()
            while time.time() - t < 0.1 or paused:
                vis.poll_events()
                vis.update_renderer()
