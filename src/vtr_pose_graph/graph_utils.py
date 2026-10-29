import numpy as np
from pylgmath.se3.transformation import Transformation
from pylgmath.se3.transformation_with_covariance import TransformationWithCovariance


from vtr_pose_graph.graph import Graph, GraphError
from vtr_pose_graph.graph_iterators import DepthFirstSearchIterator, SpatialIterator, TemporalIterator, BreadthFirstSearchIterator, PriviledgedIterator
from vtr_pose_graph.vertex import Vertex
from vtr_pose_graph.edge import Edge


def get_closest_teach_vertex(v: Vertex) -> Vertex:
    if v.taught:
        return v
    for r_v, _ in BreadthFirstSearchIterator(v):
        if r_v.taught:
            return r_v
    raise GraphError("Graph is malformed, repeat pass does not connect to teach vertex.")

def collect_path(v_start: Vertex, v_goal: Vertex):

    def _collect_path(it, v_goal: Vertex):
        p = []
        for v, e, c in it:
            if e.from_id == e.to_id:
                continue
            p.append((e, c))
            if v == v_goal:
                current_vid = v_goal.id
                current_cost = c + 1
                final_branch = []
                for e, e_c in reversed(p):
                    if e.from_id == current_vid and e_c < current_cost:
                        current_vid = e.to_id
                        final_branch.insert(0, e)
                        current_cost = e_c
                    elif e.to_id == current_vid and e_c < current_cost:
                        current_vid = e.from_id
                        final_branch.insert(0, e)
                        current_cost = e_c
                    else:
                        continue
                    if current_vid == v_start.id:
                        return final_branch
                raise GraphError(f"{v_start} was not connected to the iterator")

        raise GraphError(f"{v_goal} was not connected to the iterator")

    if v_start == v_goal:
        return []
    path = []
    if v_start.taught and v_goal.taught:
        path += _collect_path(PriviledgedIterator(v_start, return_cost=True), v_goal)
    else:
        if not v_start.taught:
            for start_teach_v, e in BreadthFirstSearchIterator(v_start):
                if start_teach_v.taught:
                    path.append(e)
                    break
            v_start = start_teach_v
        if not v_goal.taught:
            for goal_teach_v, e in BreadthFirstSearchIterator(v_goal):
                if goal_teach_v.taught:
                    break
            path += collect_path(v_start, goal_teach_v)
            path.append(e)
        else:
            path += collect_path(v_start, v_goal)

    
    return path
   
def get_path_transform(v_start: Vertex, v_goal: Vertex, path=[]):
    if len(path) == 0:
        path = collect_path(v_start, v_goal)
    
    T_g_s = TransformationWithCovariance(init_cov_to_zero=True)
    current_vid = v_goal.id
    for e in reversed(path):
        if e.from_id == current_vid:
            T_g_s *= e.T
            current_vid = e.to_id
        elif e.to_id == current_vid:
            T_g_s *= e.T.inverse()
            current_vid = e.from_id
        else:
            continue
    if (current_vid != v_start.id):
        raise GraphError("Path didn't end at the goal!")
    return T_g_s.inverse()
        


def count_runs(graph: Graph) -> int :
    max_run = 0
    for r_v, _ in BreadthFirstSearchIterator(graph.root):
        max_run = max(r_v.run, max_run)
    return max_run + 1

def set_world_frame(graph: Graph, vertex: Vertex, T_w_v0=Transformation()):
    vertex.T_w_v = T_w_v0

    for v, e in DepthFirstSearchIterator(vertex):
        if v == vertex:
            continue
        
        v_from = graph.get_vertex(e.from_id)
        v_to = graph.get_vertex(e.to_id)

        v.T_w_v = v_to.T_w_v @ e.T if v == v_from else v_from.T_w_v @ e.T.inverse()
        #print(v.T_w_v)

def mask_points_near_vertex(vertex: Vertex, points:np.ndarray, distance: float):
    return np.linalg.norm(points[:, :2] - vertex.T_v_w.r_ba_ina()[:2].reshape(1, 2), axis=1) < distance

def filter_points_along_path(root_v: Vertex, points: np.ndarray, distance: float):
    mask = np.zeros((points.shape[0],), dtype=bool)
    for v, _ in TemporalIterator(root_v):
        mask |= mask_points_near_vertex(v, points, distance)
    return mask

def collapse_pose_graph(original_graph: Graph) -> Graph:
    """
    Collapses all vertices and edges in a pose graph with exactly two neighbours
    into single long edges containing the accumulated transform.
    """
    collapsed_graph = Graph()
    collapsed_graph.major_id = original_graph.major_id
    collapsed_graph.minor_id = original_graph.minor_id
    
    # Step 1: Identify "junction" vertices (degree != 2) and add them to the new graph
    junction_vertices = set()
    for v in original_graph._vertices.values():
        if len(v.neighbours) != 2:
            junction_vertices.add(v.id)
            
            # Create a copy of the vertex for the new graph
            v_copy = Vertex(bagfile_cache=v.cache)
            v_copy.stamp = v.stamp
            v_copy.id = v.id
            collapsed_graph.add_vertex(v_copy)
            
    # Handle the edge case where the entire graph is a single closed loop (all degrees == 2)
    if not junction_vertices and original_graph.number_of_vertices > 0:
        root_v = original_graph.root
        junction_vertices.add(root_v.id)
        collapsed_graph.add_vertex(root_v)

    visited_edges = set()

    # Step 2: Traverse from each junction to find and collapse chains
    for start_id in junction_vertices:
        start_vertex = original_graph.get_vertex(start_id)
        
        for next_v, first_edge in start_vertex.get_neighbours():
            if first_edge.id in visited_edges:
                continue
                
            current_vertex = next_v
            current_edge = first_edge
            
            # Start accumulating the transform
            # Assuming T represents t_to_from; invert if traversing from 'to' to 'from'
            if current_edge.from_id == start_vertex.id:
                accumulated_T = current_edge.T
            else:
                accumulated_T = current_edge.T.inverse()
                
            visited_edges.add(current_edge.id)
            
            # Walk along the chain of degree-2 vertices
            while current_vertex.id not in junction_vertices:
                # Find the next edge that we haven't visited yet
                for neighbor_v, neighbor_e in current_vertex.get_neighbours():
                    if neighbor_e.id not in visited_edges:
                        next_edge = neighbor_e
                        next_vertex = neighbor_v
                        break
                        
                if next_edge.from_id == current_vertex.id:
                    # Traversed in forward direction
                    accumulated_T = next_edge.T * accumulated_T 
                else:
                    # Traversed in reverse direction
                    accumulated_T = next_edge.T.inverse() * accumulated_T
                
                visited_edges.add(next_edge.id)
                current_vertex = next_vertex

            # Step 3: Create the new collapsed edge connecting the two junctions
            new_edge = Edge()
            new_edge.from_id = start_vertex.id
            new_edge.to_id = current_vertex.id
            new_edge.T = accumulated_T
            
            # Inherit type and mode from the starting edge of the chain
            new_edge.type = first_edge.type
            new_edge.mode = first_edge.mode
            
            collapsed_graph.add_edge(new_edge)

    return collapsed_graph
