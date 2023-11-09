from matplotlib import pyplot as plt
import numpy as np
from vtr_pose_graph.graph import Graph
from vtr_pose_graph.graph_iterators import DepthFirstSearchIterator
from vtr_pose_graph.vertex import Vertex

import vtr_pose_graph.graph_utils as g_utils
from sensor_msgs_py.point_cloud2 import read_points
from pylgmath import Transformation


def plot_graph(g: Graph):
    f = plt.figure("Plotting Pose Graph")
    ax = f.add_axes([0, 0, 0.5, 1])
    ax.set_title(f"{g}")
    for v, e in DepthFirstSearchIterator(g.root):
        ax.scatter([v.minor_id], [v.run])

        try:
            from_v = g.get_vertex(e.from_id)
            to_v = g.get_vertex(e.to_id)
            colour = "red" if e.is_temporal() else "blue";
            ax.plot([from_v.minor_id, to_v.minor_id], [from_v.run, to_v.run], c=colour)
        except:
            continue


def extract_points_from_vertex(v: Vertex, msg="raw_point_cloud", labels=False, return_tf=False):
    raw_pc_msg = v.get_data(msg)
    new_pc = read_points(raw_pc_msg.point_cloud)
    T_v_m = Transformation(xi_ab=np.array(raw_pc_msg.t_vertex_this.xi).reshape(6, 1))


    if return_tf and labels:
        return convert_points_to_frame(np.vstack((new_pc['x'], new_pc['y'], new_pc['z'])), T_v_m).astype(np.float32), T_v_m, new_pc['flex14'] > 5000
    elif return_tf:
        return convert_points_to_frame(np.vstack((new_pc['x'], new_pc['y'], new_pc['z'])), T_v_m).astype(np.float32), T_v_m
    elif labels:
        return convert_points_to_frame(np.vstack((new_pc['x'], new_pc['y'], new_pc['z'])), T_v_m).astype(np.float32), new_pc['flex14'] > 5000
    else:
        return convert_points_to_frame(np.vstack((new_pc['x'], new_pc['y'], new_pc['z'])), T_v_m).astype(np.float32)


def convert_points_to_frame(pts: np.ndarray, frame: Transformation):
    if pts.shape[0] != 3:
        raise RuntimeError(f"Expecting 3D points shape was {pts.shape}")
    new_points = np.vstack((pts, np.ones(pts[0].shape, dtype=np.float32)))
    new_points = (frame.matrix() @ new_points)
    return new_points[:3, :]


def extract_points_and_map(graph: Graph, v: Vertex, labels=False, world_frame=True):
    if labels:
        curr_pts, T_v_s, curr_labels = extract_points_from_vertex(v, msg='nn_point_cloud', labels=labels, return_tf=True)
    else:
        curr_pts, T_v_s = extract_points_from_vertex(v, return_tf=True)
    
    map_ptr = v.get_data("pointmap_ptr")
    map_v = g_utils.get_closest_teach_vertex(graph.get_vertex(map_ptr.map_vid))
    map_ptr = map_v.get_data("pointmap_ptr")
    teach_v = graph.get_vertex(map_ptr.map_vid)

    map_pts = extract_points_from_vertex(teach_v, msg="pointmap")
    
    if world_frame:
        curr_pts = convert_points_to_frame(curr_pts, v.T_w_v)
        map_pts = convert_points_to_frame(map_pts, teach_v.T_w_v)
    else:
        #curr_pts = convert_points_to_frame(curr_pts, T_v_s.inverse())
        #T_v_s.inverse() *
        map_pts = convert_points_to_frame(map_pts,  v.T_w_v.inverse() * teach_v.T_w_v)

    if labels:
        return curr_pts.T, map_pts.T, curr_labels
    else:
        return curr_pts.T, map_pts.T

def range_crop(pts, center, radius, labels=None):
    points_filter = np.linalg.norm((pts - center), axis=1) < radius
    if labels is None:
        return pts[points_filter, :]
    else:
    	return pts[points_filter, :], labels[points_filter]

def downsample(pc: np.ndarray, grid_size=0.05):
    import open3d as o3d

    pts = o3d.utility.Vector3dVector(pc)
    pcd = o3d.geometry.PointCloud(pts)
    downpcd = pcd.voxel_down_sample(voxel_size=grid_size)
    return np.asarray(downpcd.points)
