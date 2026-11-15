import datetime
import numpy as np

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory

from vtr_pose_graph.graph_iterators import TemporalIterator
from collections import Counter
import argparse

def teacher_run(graph, run_id):
    """Run that a repeat was localized against, from its spatial edges.
 
    A spatial edge joins a vertex of this run to a vertex of another run; vertex ids are
    (run_id << 16) | index. Returns the most common other run, or None.
    """
    counts = Counter()
    edges = graph._edges  # NOTE: assumed container of Edge objects; adjust if the Graph API differs
    for e in (edges.values() if isinstance(edges, dict) else edges):
        if not e.is_spatial():
            continue

        a, b = (int(x) for x in (e.from_id, e.to_id))  # NOTE: attribute names assumed
        ra, rb = a >> 16, b >> 16
        if ra == run_id and rb != run_id:
            counts[rb] += 1
        elif rb == run_id and ra != run_id:
            counts[ra] += 1
    return counts.most_common(1)[0][0] if counts else None
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Plot Repeat Path',
                        description = 'Plots scatter of points to show path. Also calculates RMS error')
    parser.add_argument('graph', help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    args = parser.parse_args()

    factory = Rosbag2GraphFactory(args.graph)

    graph = factory.buildGraph()
    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")


    for i in graph.runs:
        try:
            v_start = graph.get_vertex((i,0))
        except KeyError as e:
            print(f"Vertex ({hex(i)},0) not found in graph. Skipping run.")
            continue

        x = []
        y = []
        t = []
        dist = []
        path_len = 0

        for v, e in TemporalIterator(v_start):
            x.append(v.T_v_w.r_ba_ina()[0])
            y.append(v.T_v_w.r_ba_ina()[1])
            t.append(v.stamp / 1e9)
            path_len += np.linalg.norm(e.T.r_ba_ina())
                
        print(f"Path {hex(i)} was {path_len:.3f}m long with {len(x)} vertices and is a {'teach' if v.taught else 'repeat'}.")
        if not v.taught:
            teacher = teacher_run(graph, i)
            if teacher is not None:
                print(f"Path {hex(i)} was taught by path {hex(teacher)}")
        print(f"Path {hex(i)} started at {datetime.datetime.fromtimestamp(t[0])} took {t[-1] - t[0]:.1f}s to complete")
 


