import os
import csv
import numpy as np
from pylgmath import se3op
import argparse
import vtr_pose_graph.graph_utils as g_utils
from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.graph_iterators import PriviledgedIterator
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import open3d as o3d  # Added for mesh creation

def create_path_tube_mesh(path_points, radius=0.1, resolution=20):
    """Creates a 3D mesh (tube) from a series of path points and colors it red."""
    path_mesh = o3d.geometry.TriangleMesh()
    if len(path_points) < 2:
        return path_mesh

    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i+1]
        
        segment_vector = p2 - p1
        segment_length = np.linalg.norm(segment_vector)
        if np.isclose(segment_length, 0):
            continue

        # Create a default cylinder aligned with Z-axis
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=radius, height=segment_length, resolution=resolution
        )
        # Set the cylinder's color to red.
        cylinder.paint_uniform_color([1.0, 0.0, 0.0])
        
        # Compute rotation to align the cylinder with the segment vector
        segment_norm = segment_vector / segment_length
        z_axis = np.array([0, 0, 1])
        
        if np.allclose(segment_norm, z_axis):
            R_mat = np.identity(3)
        elif np.allclose(segment_norm, -z_axis):
            # 180-degree rotation around X-axis
            R_mat = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
        else:
            axis = np.cross(z_axis, segment_norm)
            angle = np.arccos(np.dot(z_axis, segment_norm))
            rotation_vector = axis / np.linalg.norm(axis) * angle
            R_mat = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)
        
        # Apply transformation: rotate then translate cylinder to the segment midpoint.
        cylinder.rotate(R_mat, center=(0, 0, 0))
        cylinder.translate((p1 + p2) / 2)
        
        path_mesh += cylinder

    return path_mesh

def export_relative_transforms(graph_path, output_path):
    if os.path.isdir(output_path):
        output_path_mat = os.path.join(output_path, "relative_transforms_mat_test.csv")
        output_path_abs = os.path.join(output_path, "relative_transforms_test.csv")
    else:
        output_path_mat = output_path.replace(".csv", "_mat_test.csv")
        output_path_abs = output_path.replace(".csv", "_test.csv")

    # Load the pose graph
    print(f"Loading graph from {graph_path}...")
    factory = Rosbag2GraphFactory(graph_path)
    graph = factory.buildGraph()

    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    # Set world frame
    g_utils.set_world_frame(graph, graph.root)

    # Initialize iterator
    v_start = graph.root
    relative_transforms = []
    relative_transforms_mat = []

    for v, e in PriviledgedIterator(v_start):
        if e is not None:
            # Extract relative transform
            T_rel = e.T
            print(T_rel)
            timestamp = v.stamp / 1e9  # Convert nanoseconds to seconds

            T_rel_matrix = T_rel.matrix()
            relative_transforms_mat.append([float(timestamp)] + T_rel_matrix.flatten().tolist())

            # Extract relative position and orientation
            rel_position = T_rel.r_ba_ina()
            rel_rotation = R.from_matrix(T_rel.C_ba()).as_euler('xyz', degrees=False)
        
            # Combine timestamp, relative position (x,y,z) and orientation (roll,pitch,yaw)
            relative_transforms.append([
                float(timestamp),
                float(rel_position[0]), float(rel_position[1]), float(rel_position[2]),
                float(rel_rotation[0]), float(rel_rotation[1]), float(rel_rotation[2])
            ])
    
    # Ensure the output file has a .csv extension.
    if not output_path_mat.endswith(".csv"):
        output_path_mat += ".csv"
    
    # Save relative transforms matrix to CSV file
    with open(output_path_mat, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
        header = ['timestamp'] + [f'm{i}{j}' for i in range(4) for j in range(4)]
        csvwriter.writerow(header)
        # Write data
        csvwriter.writerows(relative_transforms_mat)
        print(f"Saved relative transforms matrix to {output_path_mat}")

    # Compute the absolute path by compounding the relative transforms.
    transforms_with_timestamps = [
        (np.array(row[1:]).reshape(4, 4), row[0])
        for row in relative_transforms_mat
    ]
    if not transforms_with_timestamps:
        raise RuntimeError("No transformation data available.")

    # Compute the actual path from relative transformations
    actual_positions = []
    current_position = np.array([0.0, 0.0, 0.0])
    current_orientation = R.from_euler('xyz', [0.0, 0.0, 0.0])

    for transform in relative_transforms:
        timestamp, x, y, z, roll, pitch, yaw = transform
        rel_position = np.array([x, y, z])
        rel_orientation = R.from_euler('xyz', [roll, pitch, yaw])

        # Update current position and orientation
        current_position += current_orientation.apply(rel_position)
        current_orientation *= rel_orientation
        actual_positions.append(current_position.copy())

    actual_positions = np.array(actual_positions)

    # Plot the odometry path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], label='Odometry Path')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    max_range = np.array([actual_positions[:,0].max()-actual_positions[:,0].min(), 
                          actual_positions[:,1].max()-actual_positions[:,1].min(), 
                          actual_positions[:,2].max()-actual_positions[:,2].min()]).max()
    mid_x = (actual_positions[:,0].max()+actual_positions[:,0].min()) * 0.5
    mid_y = (actual_positions[:,1].max()+actual_positions[:,1].min()) * 0.5
    mid_z = (actual_positions[:,2].max()+actual_positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)

    ax.legend()
    plt.title('Odometry Path')
    plt.show()

    # Create tube mesh of the path and save as OBJ file.
    tube_mesh = create_path_tube_mesh(actual_positions, radius=0.1, resolution=20)
    if os.path.isdir(output_path):
        output_obj = os.path.join(output_path, "odometry_path_tube.obj")
    else:
        output_obj = output_path.replace(".csv", "_tube.obj")
    if not output_obj.lower().endswith(".obj"):
        output_obj += ".obj"
    o3d.io.write_triangle_mesh(output_obj, tube_mesh)
    print(f"Saved tube mesh to {output_obj}")

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

    export_relative_transforms(args.graph, args.output)
