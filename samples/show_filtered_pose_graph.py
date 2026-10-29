import os
import datetime
import argparse

import plotly.graph_objects as go
from pylgmath import Transformation

from vtr_utils.bag_file_parsing import Rosbag2GraphFactory
from vtr_pose_graph.edge import Edge
import vtr_pose_graph.graph_utils as g_utils


def robot_of(vid):
    return vid >> 60


def run_of(vid):
    return vid >> 16


def minor_of(vid):
    return vid & 0xFFFF


def edge_category(e: Edge, graph):
    if not (graph.contains_vertex(e.from_id) and graph.contains_vertex(e.to_id)):
        return "Dangling (missing vertex)"
    if e.is_spatial():
        return "Spatial"
    return "Taught temporal" if e.is_teach() else "Repeat temporal"


def temporal_poses(graph, vids):
    """Accumulate T_w_v along temporal edges only, so each chain keeps its local shape.
    Each temporal chain is then placed rigidly using a single spatial edge to an already placed chain,
    which avoids the distortion from over-constrained (possibly wrong) loop closures."""
    T_w_v = {}
    chain_of = {}
    chains = []
    for vid in sorted(v for v in vids if graph.contains_vertex(v)):
        if vid in chain_of:
            continue
        start = graph.get_vertex(vid)
        T_w_v[vid] = Transformation()
        chain_of[vid] = len(chains)
        chain, frontier = [vid], [start]
        while frontier:
            v = frontier.pop()
            for n, e in v.neighbours:
                if not e.is_temporal() or n.id in chain_of:
                    continue
                T_w_v[n.id] = T_w_v[v.id] @ e.T if n.id == e.from_id else T_w_v[v.id] @ e.T.inverse()
                chain_of[n.id] = len(chains)
                chain.append(n.id)
                frontier.append(n)
        chains.append(chain)

    # Grow from the chain holding the earliest vertex, anchoring each new chain with one spatial edge
    order = sorted(range(len(chains)), key=lambda c: min(graph.get_vertex(v).stamp for v in chains[c]))
    placed = {order[0]}
    changed = True
    while changed:
        changed = False
        for c in order:
            if c in placed:
                continue
            for vid in chains[c]:
                anchor = next(((n, e) for n, e in graph.get_vertex(vid).neighbours
                               if e.is_spatial() and chain_of.get(n.id) in placed), None)
                if anchor is None:
                    continue
                n, e = anchor
                T_w_b = T_w_v[n.id] @ e.T if vid == e.from_id else T_w_v[n.id] @ e.T.inverse()
                T_offset = T_w_b @ T_w_v[vid].inverse()
                for v in chains[c]:
                    T_w_v[v] = T_offset @ T_w_v[v]
                print(f"  anchored chain of run {hex(run_of(vid))} ({len(chains[c])} vertices) with {e}")
                placed.add(c)
                changed = True
                break
    for c in order:
        if c not in placed:
            print(f"  chain of run {hex(run_of(chains[c][0]))} ({len(chains[c])} vertices) has no spatial link, left at the origin")
    return T_w_v


EDGE_STYLES = {
    "Taught temporal": dict(color="#d62728", width=2),
    "Repeat temporal": dict(color="#ff9896", width=1.5),
    "Spatial": dict(color="#1f77b4", width=1),
    "Dangling (missing vertex)": dict(color="#000000", width=2.5, dash="dot"),
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog = 'Show filtered pose graph',
                        description = 'Interactive plot of the pose graph edges touching a robot and/or run. '
                                      'Includes edges that reference vertices missing from the graph.')
    parser.add_argument('-g', '--graph', default=os.getenv("VTRDATA"), help="The filepath to the pose graph folder. (Usually /a/path/graph)")
    parser.add_argument('-r', '--robot', type=int, default=None, help="Only keep edges with an endpoint on this robot (e.g. 3).")
    parser.add_argument('--run', type=lambda s: int(s, 0), nargs="*", default=None,
                        help="Only keep edges with an endpoint in these runs (e.g. 0x3503a049463d).")
    parser.add_argument('-l', '--layout', choices=["index", "world", "temporal"], default="index",
                        help="index: x=vertex minor id, y=run. world: x/y from the vertex poses chained through the whole graph. "
                             "temporal: poses chained along temporal edges only, each chain placed by a single spatial edge.")
    parser.add_argument('-o', '--output', default=None, help="Write an html file instead of opening a browser.")
    args = parser.parse_args()

    factory = Rosbag2GraphFactory(args.graph)
    graph = factory.buildGraph()
    print(f"Graph {graph} has {graph.number_of_vertices} vertices and {graph.number_of_edges} edges")

    if args.layout == "world":
        g_utils.set_world_frame(graph, graph.root)

    def keep(vid):
        return (args.robot is None or robot_of(vid) == args.robot) and \
               (args.run is None or run_of(vid) in args.run)

    # Read the raw edges so ones dropped by buildGraph for missing vertices are still shown
    edges = [Edge(msg) for _, msg in factory.cache[factory.edge_path].get_bag_msgs_iter("edges")]
    edges = [e for e in edges if keep(e.from_id) or keep(e.to_id)]

    vids = {vid for e in edges for vid in (e.from_id, e.to_id)}
    missing = sorted(vid for vid in vids if not graph.contains_vertex(vid))
    print(f"Kept {len(edges)} edges touching {len(vids)} vertices, {len(missing)} of which are missing from the graph")
    for vid in missing:
        print(f"  missing vertex {hex(vid)} (run {hex(run_of(vid))}, minor {minor_of(vid)})")

    if args.layout == "temporal":
        T_w_v = temporal_poses(graph, vids)

    # Order runs by their earliest vertex so the index layout reads top to bottom in time
    def run_start(run):
        stamps = [graph.get_vertex(v).stamp for v in vids if run_of(v) == run and graph.contains_vertex(v)]
        return min(stamps) if stamps else float("inf")
    runs = sorted({run_of(v) for v in vids}, key=run_start)
    run_row = {run: i for i, run in enumerate(runs)}

    def position(vid):
        if args.layout == "index":
            return minor_of(vid), run_row[run_of(vid)]
        if not graph.contains_vertex(vid):
            return None
        T = T_w_v[vid].inverse() if args.layout == "temporal" else graph.get_vertex(vid).T_v_w
        r = T.r_ba_ina()
        return float(r[0, 0]), float(r[1, 0])

    fig = go.Figure()

    by_category = {name: ([], []) for name in EDGE_STYLES}
    dangling_ends = []
    for e in edges:
        cat = edge_category(e, graph)
        p_from, p_to = position(e.from_id), position(e.to_id)
        if p_from is None or p_to is None:
            # Pose layouts cannot place a missing vertex, so mark the end that exists
            dangling_ends += [p for p in (p_from, p_to) if p is not None]
            continue
        xs, ys = by_category[cat]
        xs += [p_from[0], p_to[0], None]
        ys += [p_from[1], p_to[1], None]

    for name, (xs, ys) in by_category.items():
        if xs:
            fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", name=name, line=EDGE_STYLES[name], hoverinfo="skip",
                                       opacity=0.35 if name == "Spatial" else 1.0))

    for run in runs:
        run_vids = sorted(v for v in vids if run_of(v) == run)
        pts, text, symbols = [], [], []
        for vid in run_vids:
            p = position(vid)
            if p is None:
                continue
            pts.append(p)
            present = graph.contains_vertex(vid)
            stamp = datetime.datetime.fromtimestamp(graph.get_vertex(vid).stamp / 1e9) if present else None
            text.append(f"{hex(vid)}<br>run {hex(run)} minor {minor_of(vid)}<br>"
                        f"{stamp:%H:%M:%S.%f}" if present else f"{hex(vid)}<br><b>MISSING from graph</b>")
            symbols.append("circle" if present else "x")
        if not pts:
            continue
        fig.add_trace(go.Scattergl(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers",
                                   name=f"run {hex(run)}", legendgroup="runs",
                                   marker=dict(size=6, symbol=symbols, line=dict(width=0)),
                                   text=text, hovertemplate="%{text}<extra></extra>"))

    if dangling_ends:
        fig.add_trace(go.Scattergl(x=[p[0] for p in dangling_ends], y=[p[1] for p in dangling_ends], mode="markers",
                                   name="Dangling edge endpoint", marker=dict(size=14, symbol="x-open", color="black")))

    filters = []
    if args.robot is not None:
        filters.append(f"robot {args.robot}")
    if args.run:
        filters.append("runs " + ", ".join(hex(r) for r in args.run))
    fig.update_layout(title=f"{graph}: {' & '.join(filters) or 'all edges'} "
                            f"({len(edges)} edges, {len(missing)} missing vertices)",
                      template="plotly_white", hovermode="closest")
    if args.layout == "index":
        fig.update_xaxes(title="vertex minor id")
        fig.update_yaxes(title="run", tickvals=list(run_row.values()), ticktext=[hex(r) for r in runs], autorange="reversed")
    else:
        fig.update_xaxes(title="x (m)")
        fig.update_yaxes(title="y (m)", scaleanchor="x", scaleratio=1)

    if args.output:
        fig.write_html(args.output)
        print(f"Wrote {args.output}")
    else:
        fig.show()
