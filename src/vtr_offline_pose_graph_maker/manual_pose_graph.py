from vtr_pose_graph.graph import Graph
from vtr_pose_graph.vertex import Vertex
from vtr_pose_graph.edge import Edge
from pylgmath.se3.transformation import Transformation
import numpy as np
import sqlite3
import yaml
import os
from datetime import datetime

# Input data: LiDAR scans and their positions
lidar_scans = [
    "/path/to/scan1.pcd",
    "/path/to/scan2.pcd",
    "/path/to/scan3.pcd",
]
positions = [
    np.eye(4),  # Identity matrix for the first position
    np.eye(4) @ Transformation(T_xyz=[1, 0, 0]).matrix(),  # Move 1m along X
    np.eye(4) @ Transformation(T_xyz=[2, 0, 0]).matrix(),  # Move 2m along X
]

# Output directory
output_dir = "./pose_graph_output"
vertices_dir = f"{output_dir}/vertices"
edges_dir = f"{output_dir}/edges"
index_dir = f"{output_dir}/index"

# Compute metadata values
starting_time = int(datetime.now().timestamp() * 1e9)  # Current time in nanoseconds
duration = len(positions) * int(1e9)  # Assume 1 second between scans
message_count = len(positions)


def create_vertices_db(vertices_dir, graph):
    """Create the vertices folder with vertices.db3 and metadata.yaml."""
    os.makedirs(vertices_dir, exist_ok=True)
    conn = sqlite3.connect(f"{vertices_dir}/vertices_0.db3")
    cursor = conn.cursor()

    # Create vertices table
    cursor.execute("""
    CREATE TABLE vertices (
        id INTEGER PRIMARY KEY,
        pose TEXT,
        lidar_scan TEXT
    )
    """)

    # Insert data
    for vertex in graph.vertices:
        cursor.execute(
            "INSERT INTO vertices (id, pose, lidar_scan) VALUES (?, ?, ?)",
            (vertex.id, str(vertex.T_w_v.matrix.tolist()), vertex.data["lidar_scan"]),
        )

    conn.commit()
    conn.close()

    # Create metadata.yaml
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 4,
            "storage_identifier": "sqlite3",
            "relative_file_paths": ["vertices_0.db3"],
            "duration": {"nanoseconds": duration},
            "starting_time": {"nanoseconds_since_epoch": starting_time},
            "message_count": message_count,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "vertices",
                        "type": "vtr_pose_graph_msgs/msg/Vertex",
                        "serialization_format": "cdr",
                        "offered_qos_profiles": "",
                    },
                    "message_count": message_count,
                }
            ],
            "compression_format": "",
            "compression_mode": "",
        }
    }

    with open(f"{vertices_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


def create_edges_db(edges_dir, graph):
    """Create the edges folder with edges.db3 and metadata.yaml."""
    os.makedirs(edges_dir, exist_ok=True)
    conn = sqlite3.connect(f"{edges_dir}/edges_0.db3")
    cursor = conn.cursor()

    # Create edges table
    cursor.execute("""
    CREATE TABLE edges (
        id INTEGER PRIMARY KEY,
        from_id INTEGER,
        to_id INTEGER,
        transform TEXT
    )
    """)

    # Insert edges
    for edge in graph.edges:
        cursor.execute(
            "INSERT INTO edges (id, from_id, to_id, transform) VALUES (?, ?, ?, ?)",
            (edge.id, edge.from_id, edge.to_id, str(edge.T.matrix.tolist())),
        )

    conn.commit()
    conn.close()

    # Create metadata.yaml
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 4,
            "storage_identifier": "sqlite3",
            "relative_file_paths": ["edges_0.db3"],
            "duration": {"nanoseconds": duration},
            "starting_time": {"nanoseconds_since_epoch": starting_time},
            "message_count": message_count - 1,  # One less edge than vertices
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "edges",
                        "type": "vtr_pose_graph_msgs/msg/Edge",
                        "serialization_format": "cdr",
                        "offered_qos_profiles": "",
                    },
                    "message_count": message_count - 1,
                }
            ],
            "compression_format": "",
            "compression_mode": "",
        }
    }

    with open(f"{edges_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


def create_index_db(index_dir):
    """Create the index folder with index.db3 and metadata.yaml."""
    os.makedirs(index_dir, exist_ok=True)
    conn = sqlite3.connect(f"{index_dir}/index_0.db3")
    cursor = conn.cursor()

    # Create a dummy index table
    cursor.execute("""
    CREATE TABLE index_data (
        id INTEGER PRIMARY KEY,
        description TEXT
    )
    """)

    cursor.execute(
        "INSERT INTO index_data (id, description) VALUES (?, ?)",
        (0, "Index metadata for the pose graph."),
    )

    conn.commit()
    conn.close()

    # Create metadata.yaml
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 4,
            "storage_identifier": "sqlite3",
            "relative_file_paths": ["index_0.db3"],
            "duration": {"nanoseconds": 1},
            "starting_time": {"nanoseconds_since_epoch": starting_time},
            "message_count": 1,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "index",
                        "type": "vtr_pose_graph_msgs/msg/Graph",
                        "serialization_format": "cdr",
                        "offered_qos_profiles": "",
                    },
                    "message_count": 1,
                }
            ],
            "compression_format": "",
            "compression_mode": "",
        }
    }

    with open(f"{index_dir}/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


# Main function to create the pose graph
if __name__ == "__main__":
    # Initialize pose graph
    graph = Graph(name="ManualPoseGraph")

    # Add vertices
    for i, (pose, scan) in enumerate(zip(positions, lidar_scans)):
        vertex = Vertex(id=i, T_w_v=Transformation(pose))
        vertex.data["lidar_scan"] = scan
        graph.add_vertex(vertex)

    # Add edges
    for i in range(len(positions) - 1):
        T_relative = np.linalg.inv(positions[i]) @ positions[i + 1]
        edge = Edge(from_id=i, to_id=i + 1, T=Transformation(T_relative))
        graph.add_edge(edge)

    # Create vertices, edges, and index databases
    create_vertices_db(vertices_dir, graph)
    create_edges_db(edges_dir, graph)
    create_index_db(index_dir)

    print("Pose graph created successfully!")
