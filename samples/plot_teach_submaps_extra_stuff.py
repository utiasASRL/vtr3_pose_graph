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
### THIS VERSION OF PLOT_TEACH_SUBMAPS INCLUDES CREATING AN AGGRAGATED POINT CLOUD AT THE END WITH THE PATH OVERLAYED ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog = 'Plot Point Clouds Path',
                            description = 'Plots point clouds')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")      # option that takes a value
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

    for i in range(test_graph.major_id + 1):
        v_start = test_graph.get_vertex((i, 0))
        paused = True
        vertices = list(TemporalIterator(v_start))
        vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices

        for vertex, e in vertices_to_plot:

            new_points, map_ptr = extract_map_from_vertex(test_graph, vertex)

            robot_position = vertex.T_v_w.r_ba_ina().reshape((3,) )
            print('robot position = ', robot_position)
            robot_pose = vertex.T_v_w.matrix()
            print('robot_pose = ', vertex.T_v_w.matrix())

            x.append(vertex.T_v_w.r_ba_ina()[0]) 
            y.append(vertex.T_v_w.r_ba_ina()[1])

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

    # New visualization for triangle mesh path geometry and submaps
    vis.clear_geometries()
    path_points = []
    submap_pcds = []

    for i in range(test_graph.major_id + 1):
        v_start = test_graph.get_vertex((i, 0))
        vertices = list(TemporalIterator(v_start))
        vertices_to_plot = vertices[:-10] if len(vertices) > 10 else vertices
        vertex_count = 0
        for vertex, e in vertices_to_plot:
            robot_position = vertex.T_v_w.r_ba_ina().reshape((3,))
            path_points.append(robot_position)

            if vertex_count % 40 == 0:  # Plot every 40 vertices
                new_points, map_ptr = extract_map_from_vertex(test_graph, vertex)
                if not np.allclose(map_ptr.matrix(), np.eye(4)):
                    submap_pcd = o3d.geometry.PointCloud()
                    submap_pcd.points = o3d.utility.Vector3dVector(new_points.T)
                    submap_pcd.paint_uniform_color((0.0, 0.0, 1.0))  # Blue color for non-identity matrix
                    submap_pcds.append(submap_pcd)
            vertex_count += 1

    # Create line set for path
    path_lines = [[i, i + 1] for i in range(len(path_points) - 1)]
    path_colors = [[1, 0, 0] for _ in range(len(path_lines))]  # Green color for path
    path_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(path_points),
        lines=o3d.utility.Vector2iVector(path_lines),
    )
    path_line_set.colors = o3d.utility.Vector3dVector(path_colors)

    # Create a cylinder mesh for each line segment to make the path thicker
    cylinder_radius = 0.1  # Adjust the radius to make the path thicker
    for start, end in path_lines:
        start_point = path_points[start]
        end_point = path_points[end]
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=np.linalg.norm(np.array(end_point) - np.array(start_point)))
        cylinder.paint_uniform_color([0, 0, 0])  # Green color for path
        transformation = np.eye(4)
        direction = np.array(end_point) - np.array(start_point)
        direction /= np.linalg.norm(direction)
        transformation[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0, np.arctan2(direction[1], direction[0]), np.arccos(direction[2])])
        transformation[:3, 3] = (np.array(start_point) + np.array(end_point)) / 2
        cylinder.transform(transformation)
        vis.add_geometry(cylinder)

    for submap_pcd in submap_pcds:
        vis.add_geometry(submap_pcd)

    vis.run()
    vis.destroy_window()

    # Save the visualized blue point cloud and path to the specified directory
    #output_dir = '/home/desiree/ASRL/vtr3/finalUsedPoseGraphs/dome/pointcloudvideo'
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)

    # Combine path line set and submap point clouds into one point cloud
    combined_pcd = o3d.geometry.PointCloud()
    combined_points = []
    combined_colors = []

    # Add path points to the combined point cloud
    for start, end in path_lines:
        start_point = path_points[start]
        end_point = path_points[end]
        combined_points.append(start_point)
        combined_points.append(end_point)
        combined_colors.append([0, 0, 0])  # Black color for path
        combined_colors.append([0, 0, 0])  # Black color for path

    # Add submap point clouds to the combined point cloud
    for submap_pcd in submap_pcds:
        combined_points.append(np.asarray(submap_pcd.points))
        combined_colors.append(np.full((len(submap_pcd.points), 3), [0, 0, 1]))  # Blue color for submaps

    combined_pcd.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
    combined_pcd.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))

    # Save the combined point cloud
    #combined_pcd_path = os.path.join(output_dir, 'combined_point_cloud.ply')
    #o3d.io.write_point_cloud(combined_pcd_path, combined_pcd)

    #print(f"Saved combined point cloud to {combined_pcd_path}")
