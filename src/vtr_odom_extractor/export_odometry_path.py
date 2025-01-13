import os
import numpy as np
import argparse
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_regression_testing.path_comparison as vtr_path
from scipy.spatial.transform import Rotation as R

def export_odometry_path(graph_path, output_path):
    # Check if output_path is a directory and append default filename
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "odometry_path.csv")

    # Load the pose graph
    print(f"Loading graph from {graph_path}...")
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()

    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    # Set world frame
    g_utils.set_world_frame(graph, graph.root)

    # Initialize iterator
    v_start = graph.root
    path_matrix = vtr_path.path_to_matrix(graph, PriviledgedIterator(v_start))
    print(f"Path matrix shape: {path_matrix.shape}")
    
    # Extract 6dof relative transforms between consecutive vertices
    odometry_data = []
    prev_T = None
    for v, e in PriviledgedIterator(v_start):
            current_T = v.T_v_w #T_v_w is the transformation from vertex to world frame
            if prev_T is not None:
                # Compute relative transform: T_curr_prev = T_curr_w * T_w_prev = T_curr_w * inv(T_prev_w)
                T_rel = current_T * prev_T.inverse()
                
                # Extract relative position and orientation
                rel_position = T_rel.r_ba_ina()
                rel_rotation = R.from_matrix(T_rel.C_ba()).as_euler('xyz', degrees=False)
                timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds
                
                # Combine timestamp, relative position (x,y,z) and orientation (roll,pitch,yaw)
                odometry_data.append([
                        float(timestamp),
                        float(rel_position[0]), float(rel_position[1]), float(rel_position[2]),
                        float(rel_rotation[0]), float(rel_rotation[1]), float(rel_rotation[2])
                    ])
            
            prev_T = current_T

    # Convert to numpy array
    odometry_data = np.array(odometry_data, dtype=float)

    # Convert relative poses to absolute poses FOR PLOTTING
    absolute_poses = []
    current_pose = np.eye(4)  # Start at origin
    absolute_poses.append(current_pose[:3, 3])  # Add initial position
    
    for rel_pose in odometry_data:
        # Create transformation matrix from relative pose
        rot = R.from_euler('xyz', rel_pose[4:7])  # roll, pitch, yaw
        trans = np.eye(4)
        trans[:3, :3] = rot.as_matrix()
        trans[:3, 3] = rel_pose[1:4]  # x, y, z
        
        # Update current pose
        current_pose = current_pose @ trans
        absolute_poses.append(current_pose[:3, 3])
    
    absolute_poses = np.array(absolute_poses)
    
    # Plot the accumulated path
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.plot(absolute_poses[:, 0], absolute_poses[:, 1], 'b-', label='Odometry Path')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Accumulated Odometry Path')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save to CSV
    header = "timestamp,dx,dy,dz,droll,dpitch,dyaw"
    np.savetxt(output_path, odometry_data, delimiter=",", header=header, comments="")
    print(f"Relative odometry path saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export odometry path from pose graph.")
    parser.add_argument("-g", "--graph", required=True, help="Path to the pose graph directory.")
    parser.add_argument("-o", "--output", required=True, help="File path or directory to save the exported odometry file.")
    args = parser.parse_args()

    # Ensure the output directory exists if a directory is provided
    if os.path.isdir(args.output):
        print(f"Output directory exists: {args.output}")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    export_odometry_path(args.graph, args.output)


'''

import os
import numpy as np
import argparse
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
import vtr_regression_testing.path_comparison as vtr_path


def export_odometry_path(graph_path, output_path):
    # Check if output_path is a directory and append default filename
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, "odometry_path.csv")

    # Load the pose graph
    print(f"Loading graph from {graph_path}...")
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()

    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    # Set world frame
    g_utils.set_world_frame(graph, graph.root)

    # Initialize iterator
    v_start = graph.root
    path_matrix = vtr_path.path_to_matrix(graph, PriviledgedIterator(v_start))
    print(f"Path matrix shape: {path_matrix.shape}")

    # Extract relative transformations between vertices
    odometry_data = []
    prev_vertex = None
    for v, e in PriviledgedIterator(v_start):
        if prev_vertex is not None:
            # Get relative transformation between current and previous vertex
            T_curr_prev = prev_vertex.T_v_w.inverse() @ v.T_v_w
            position = T_curr_prev.r_ba_ina()  # Get relative position
            timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds
        else:
            # For the first vertex, use zero relative transformation
            position = np.zeros(3)
            timestamp = v.stamp / 1e9
        prev_vertex = v

        odometry_data.append([timestamp, position[0], position[1], position[2]])

    # Ensure all rows in odometry_data have the same length
    odometry_data = np.array(odometry_data, dtype=float)
    # Plot the odometry path
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.plot(odometry_data[:, 1], odometry_data[:, 2], 'b-', label='Odometry Path')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Odometry Path')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    # Save to CSV
    header = "timestamp,x,y,z"
    np.savetxt(output_path, odometry_data, delimiter=",", header=header, comments="")
    print(f"Odometry path saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export odometry path from pose graph.")
    parser.add_argument("-g", "--graph", required=True, help="Path to the pose graph directory.")
    parser.add_argument("-o", "--output", required=True, help="File path or directory to save the exported odometry file.")
    args = parser.parse_args()

    # Ensure the output directory exists if a directory is provided
    if os.path.isdir(args.output):
        print(f"Output directory exists: {args.output}")
    else:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

    export_odometry_path(args.graph, args.output)
'''