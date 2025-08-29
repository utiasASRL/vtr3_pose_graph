import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs_py.point_cloud2 import read_points
import open3d as o3d
import sys
from vtr_utils.plot_utils import extract_map_from_vertex
import argparse
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator, SpatialIterator
import vtr_pose_graph.graph_utils as g_utils

sys.path.append('/home/desiree/ASRL/vtr3/vtr3_posegraph_tools/vtr3_pose_graph/src')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'Plot Point Clouds Path',
                            description = 'Plots point clouds')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('--save_pc', type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to save the accumulated point cloud (True/False).")
    parser.add_argument('--save_path', type=str, default="accumulated_cloud.ply", help="Path to save the accumulated point cloud (PLY format).")
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

    # For saving point clouds and poses
    all_points = []  # List of np.arrays (N,3)
    all_poses = []   # List of pose matrices (4x4)

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

    #cloud_count = 0

    for i in range(test_graph.major_id + 1):
        v_start = test_graph.get_vertex((i, 0))
        paused = True
        vertices = list(TemporalIterator(v_start))
        vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices

        for idx, (vertex, e) in enumerate(vertices_to_plot):

            new_points, map_ptr = extract_map_from_vertex(test_graph, vertex)

            num_points = new_points.shape[1]
            print(f'Number of points: {num_points}')

            robot_position = vertex.T_v_w.r_ba_ina().reshape((3,) )
            print('robot position = ', robot_position)
            robot_pose = vertex.T_v_w.matrix()
            print('robot_pose = ', vertex.T_v_w.matrix())

            x.append(vertex.T_v_w.r_ba_ina()[0]) 
            y.append(vertex.T_v_w.r_ba_ina()[1])

            all_points.append(new_points.T)
            all_poses.append(robot_pose)
            print('Shape is', new_points.T.shape)

            pcd.points = o3d.utility.Vector3dVector(new_points.T)
            if np.allclose(map_ptr.matrix(), np.eye(4)):
                pcd.paint_uniform_color((1.0, 0.0, 0.0))  # Red color for identity matrix
            else:
                pcd.paint_uniform_color((0.1*vertex.run, 0.25*vertex.run, 0.45))
            # Create coordinate frame for the vertex
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=robot_position)

            if first:
                first = False
                vis.add_geometry(pcd)
                vis.add_geometry(frame)
            else:
                vis.update_geometry(pcd)
                vis.remove_geometry(frame, reset_bounding_box=False)
                vis.add_geometry(frame)
            t = time.time()
            while time.time() - t < 0.1 or paused:
                vis.poll_events()
                vis.update_renderer()

    print("Finished processing all point clouds.")

    vis.run()
    vis.destroy_window()

    # # Save all accumulated points as a single point cloud in PLY format if requested
    # if args.save_pc and len(all_points) > 0:
    #     accumulated_points = np.vstack(all_points)
    #     acc_pcd = o3d.geometry.PointCloud()
    #     acc_pcd.points = o3d.utility.Vector3dVector(accumulated_points)
    #     o3d.io.write_point_cloud(args.save_path, acc_pcd)
    #     print(f"Saved accumulated point cloud to {args.save_path}")

    #     # Also save all point clouds into a single .bin file and poses into a CSV file
    #     import struct
    #     import csv
    #     bin_filename = "all_pointclouds.bin"
    #     csv_filename = "all_poses.csv"
    #     with open(bin_filename, "wb") as fbin, open(csv_filename, "w", newline="") as fcsv:
    #         csv_writer = csv.writer(fcsv)
    #         csv_writer.writerow(["cloud_index", "pose_00", "pose_01", "pose_02", "pose_03",
    #                              "pose_10", "pose_11", "pose_12", "pose_13",
    #                              "pose_20", "pose_21", "pose_22", "pose_23",
    #                              "pose_30", "pose_31", "pose_32", "pose_33",
    #                              "num_points", "offset_in_bin"])
    #         offset = 0
    #         for idx, (pts, pose) in enumerate(zip(all_points, all_poses)):
    #             num_points = pts.shape[0]
    #             # Write number of points (int)
    #             fbin.write(struct.pack('I', num_points))
    #             # Write all points (float32)
    #             fbin.write(pts.astype(np.float32).tobytes())
    #             # Write pose and metadata to CSV
    #             csv_writer.writerow([idx] + pose.flatten().tolist() + [num_points, offset])
    #             # Each entry: 4 bytes for num_points + 12*num_points*4 bytes for points
    #             offset += 4 + num_points * 3 * 4
    #     print(f"Saved {len(all_points)} point clouds to {bin_filename} and poses to {csv_filename}")

