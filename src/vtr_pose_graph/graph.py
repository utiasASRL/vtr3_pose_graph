from vtr_pose_graph import INVALID_ID
from vtr_pose_graph.edge import Edge
from vtr_pose_graph.vertex import Vertex
from typing import Union
from typing_extensions import Self


class Graph:
    def __init__(self, graph_msg=None):
        self._edges = {}
        self._vertices = {}
        if graph_msg:
            self.major_id = graph_msg.curr_major_id
            self.minor_id = graph_msg.curr_minor_id
        else:
            self.major_id = INVALID_ID
            self.minor_id = INVALID_ID

    def add_edge(self, edge: Edge):
        if not (self.contains_vertex(edge.from_id) and self.contains_vertex(edge.to_id)):
            raise RuntimeError(f"Edge {edge} references vertices that are not present in the graph")
        v_from = self.get_vertex(edge.from_id)
        v_to = self.get_vertex(edge.to_id)
        v_from.link(v_to, edge)
        v_to.link(v_from, edge)

        self._edges[edge.id] = edge

    def add_vertex(self, v: Vertex):
        self._vertices[v.id] = v

    def contains_vertex(self, v_id):
        return self._handle_vid(v_id) in self._vertices.keys()

    def contains_edge(self, e_id):
        return e_id in self._edges.keys()

    def get_vertex(self, vid: Union[int, tuple]) -> Vertex:
        return self._vertices[self._handle_vid(vid)]
    
    def get_privileged_subgraph(self) -> Self:
        teach_graph = Graph()
        teach_graph.major_id = self.major_id
        teach_graph.minor_id = self.minor_id

        for v in self._vertices.values():
            if v.taught:
                v_copy = Vertex(bagfile_cache=v.cache)
                v_copy.stamp = v.stamp
                v_copy.id = v.id
                teach_graph.add_vertex(v_copy)

        for e in self._edges.values():
            try:
                teach_graph.add_edge(e)
            except RuntimeError:
                continue
            
        return teach_graph


    def _handle_vid(self, vid: Union[int, tuple]) -> int:
        if isinstance(vid, int):
            return vid
        elif isinstance(vid, tuple) and len(vid) == 2 and isinstance(vid[0], int):
            return Vertex.compose_id(vid[0], vid[1])
        raise IndexError("Vertices must be a single unique integer or a tuple of two integers")


    @property
    def root(self) -> Vertex:
        return self._vertices.get(0)


    @property
    def number_of_edges(self):
        return len(self._edges)

    @property
    def number_of_vertices(self):
        return len(self._vertices)

    def __repr__(self):
        return f"Graph V{self.major_id}.{self.minor_id}"

class GraphError (Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)