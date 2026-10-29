from queue import Queue, LifoQueue

from vtr_pose_graph.edge import Edge
from vtr_pose_graph.vertex import Vertex

class GraphIterator:
    def __init__(self, q, vertex_start: Vertex, t_filter_fcn: callable, d_filter_fcn: callable, e_cost_fcn: callable, return_cost=False):
        self.queue = q
        self.type_filter = t_filter_fcn
        self.dir_filter = d_filter_fcn
        self.edge_cost = e_cost_fcn
        self.include_cost = return_cost
        self.visited_vertices = set()

        self._put(vertex_start, Edge(), 0)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Vertex, Edge]:
        if self.queue.empty():
            raise StopIteration
        to_return = self.queue.get()
        for v, e in to_return[0].get_neighbours():
            if self.type_filter(v, e) and self.dir_filter(v, e):
                self._put(v, e, to_return[2] + self.edge_cost(e))

        if self.include_cost:
            return to_return
        else:
            return to_return[:2]

    def _put(self, v: Vertex, e: Edge, cost_to_come: float):
        if not v.id in self.visited_vertices:
            self.visited_vertices.add(v.id)
            self.queue.put((v, e, cost_to_come))



class BreadthFirstSearchIterator(GraphIterator):

    def __init__(self, vertex_start: Vertex, t_filter_fcn: callable = lambda v,e: True,
                  d_filter_fcn: callable = lambda v,e: True, e_cost_fcn: callable = lambda e: 1, include_cost=False):
        GraphIterator.__init__(self, Queue(), vertex_start, t_filter_fcn, d_filter_fcn, e_cost_fcn, include_cost)


class DepthFirstSearchIterator(GraphIterator):

    def __init__(self, vertex_start: Vertex, t_filter_fcn: callable = lambda v,e: True,
                  d_filter_fcn: callable = lambda v,e: True, e_cost_fcn: callable = lambda e: 1, include_cost=False):
        GraphIterator.__init__(self, LifoQueue(), vertex_start, t_filter_fcn, d_filter_fcn, e_cost_fcn, include_cost)


def forward_filter(vertex: Vertex, edge: Edge):
    return vertex.id == edge.to_id

def reverse_filter(vertex: Vertex, edge: Edge):
    return vertex.id == edge.from_id

def spatial_filter(vertex: Vertex, edge: Edge):
    return edge.is_spatial()


class SpatialIterator(DepthFirstSearchIterator):

    def __init__(self, vertex_start: Vertex, to_teach=True):
        DepthFirstSearchIterator.__init__(self, vertex_start, spatial_filter, forward_filter if to_teach else reverse_filter)


def temporal_filter(vertex: Vertex, edge: Edge):
    return edge.is_temporal()


class TemporalIterator(DepthFirstSearchIterator):

    def __init__(self, vertex_start: Vertex, to_goal=True):
        DepthFirstSearchIterator.__init__(self, vertex_start, temporal_filter, forward_filter if to_goal else reverse_filter)

def teach_filter(vertex: Vertex, edge: Edge):
    return vertex.taught

class PriviledgedIterator(BreadthFirstSearchIterator):

    def __init__(self, vertex_start: Vertex, return_cost=False):
        super().__init__(vertex_start, t_filter_fcn=teach_filter, include_cost=return_cost)



class SlidingWindowIterator(BreadthFirstSearchIterator):

    def __init__(self, vertex_start: Vertex, max_nodes: int):
        BreadthFirstSearchIterator.__init__(self, vertex_start, self.temporal_nodal_filter)
        self.max_nodes = max_nodes

    def temporal_nodal_filter(self, vertex: Vertex, edge: Edge):
        if temporal_filter(vertex, edge) and (len(self.visited_vertices) <= self.max_nodes):
            return True
        return False    

