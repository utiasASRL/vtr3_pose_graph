import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs_py.point_cloud2 import read_points
import open3d as o3d
import sys
from vtr_utils.plot_utils import extract_map_from_vertex, extract_points_from_vertex
import argparse
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import TemporalIterator, PriviledgedIterator, SpatialIterator
import vtr_pose_graph.graph_utils as g_utils
from pylgmath.se3.transformation import Transformation

sys.path.append('/home/desiree/ASRL/vtr3/vtr3_posegraph_tools/vtr3_pose_graph/src')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='Plot Repeat Path Submaps',
                        description='Plots submaps from repeat path.')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="Path to the pose graph folder")
    parser.add_argument('-r', '--run', type=int, help="Select a repeat run.")
    args = parser.parse_args()

    factory = Rosbag2GraphFactory(args.graph)

    test_graph = factory.buildGraph()
    print(f"Graph {test_graph} has {test_graph.number_of_vertices} vertices and {test_graph.number_of_edges} edges")

    T0_dome_pix4d = np.array([[0.7794990609, 0.6263708317, -0.006387118736, 13.45056318],
                              [-0.6263390774, 0.7792323893, -0.0222765252, -7.393389117],
                              [-0.008976315826, 0.02136503253, 0.9997314445, -1.692823882],
                              [0, 0, 0, 1]])
    T0_dome_nerf = np.array([[0.6744665106, 0.7376960755, 0.02999043816, 9.175750405],
                              [-0.7348657571, 0.6746866472, -0.06906697525, -9.727562991],
                              [-0.07118458476, 0.02454441575, 0.9971611337, -2.716972725],
                              [0, 0, 0, 1]])
    T0_bigpath_pix4d = np.array([[0.8569029348, 0.5153315648, 0.01227756775, -133.7701596],
                                 [-0.515373839, 0.8569654008, 0.0003285861324, 74.15833147],
                                 [-0.01035211997, -0.006609103649, 0.9999245738, -2.773400032],
                                 [0, 0, 0, 1]])
    T0_parking_pix4d = np.array([[-0.9678134329, 0.2515043753, 0.009094405128, -7.582900147],
                                 [-0.2516049208, -0.9677498363, -0.01245866088, 1.0105025],
                                 [0.005667701353, -0.01434585643, 0.9998810297, -2.032632195],
                                 [0, 0, 0, 1]])
    T0_parking_nerf = np.array([[-0.9986654977, -0.05010478211, -0.01251936303, -1.894547981],
                                 [0.05026439505, -0.998654208, -0.01277745633, 2.403520724],
                                 [-0.01186230291, -0.01338968299, 0.9998399883, -2.453529562],
                                 [0, 0, 0, 1]])
    T0_grassy_pix4d = np.array([[-0.2705899184, -0.9626250275, -0.01158241845, -7.611904567],
                                [0.9301976365, -0.2645364746, 0.2544657357, 29.50752703],
                                [-0.248019058, 0.05808192438, 0.9670124285, -2.830707544],
                                [0, 0, 0, 1]])
    T0_transform = Transformation(T_ba=T0_grassy_pix4d)

    g_utils.set_world_frame(test_graph, test_graph.root, T0_transform)
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

    v_start_repeat = test_graph.get_vertex((args.run, 0))
    paused = True
    vertices = list(TemporalIterator(v_start_repeat))
    vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices

    for vertex, e in vertices_to_plot:

        new_points, map_ptr = extract_map_from_vertex(test_graph, vertex)
        # new_points = extract_points_from_vertex(vertex, msg="filtered_point_cloud")

        robot_position = vertex.T_v_w.r_ba_ina().reshape((3,) )
        print('robot position = ', robot_position)
        robot_pose = vertex.T_v_w.matrix()
        print('robot_pose = ', vertex.T_v_w.matrix())

        x.append(vertex.T_v_w.r_ba_ina()[0]) 
        y.append(vertex.T_v_w.r_ba_ina()[1])

        all_points.append(new_points.T)
        robot_pose_inv = np.linalg.inv(robot_pose)
        all_poses.append(robot_pose_inv)
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
        while time.time() - t < 0.01 or paused:
            vis.poll_events()
            vis.update_renderer()

    print("Finished processing all point clouds.")

    vis.run()
    vis.destroy_window()

    # Save all point clouds into a single .bin file and poses into a CSV file
    import struct
    import csv

    bin_filename = "point_clouds/grassy_pix4d_lidar.bin"
    csv_filename = "point_clouds/grassy_pix4d_poses_lidar.csv"

    # Save point clouds: each cloud is saved as [num_points][points...], all concatenated
    with open(bin_filename, "wb") as fbin, open(csv_filename, "w", newline="") as fcsv:
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow(["cloud_index", "pose_00", "pose_01", "pose_02", "pose_03",
                             "pose_10", "pose_11", "pose_12", "pose_13",
                             "pose_20", "pose_21", "pose_22", "pose_23",
                             "pose_30", "pose_31", "pose_32", "pose_33",
                             "num_points", "offset_in_bin"])
        offset = 0
        for idx, (pts, pose) in enumerate(zip(all_points, all_poses)):
            num_points = pts.shape[0]
            # Write number of points (int)
            fbin.write(struct.pack('I', num_points))
            # Write all points (float32)
            fbin.write(pts.astype(np.float32).tobytes())
            # Write pose and metadata to CSV
            csv_writer.writerow([idx] + pose.flatten().tolist() + [num_points, offset])
            # Each entry: 4 bytes for num_points + 12*num_points*4 bytes for points
            offset += 4 + num_points * 3 * 4

    print(f"Saved {len(all_points)} point clouds to {bin_filename} and poses to {csv_filename}")
