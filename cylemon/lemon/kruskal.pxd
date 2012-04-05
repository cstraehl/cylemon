cimport list_graph
from list_graph cimport Node, NodeIt, Arc, ArcIt, NodeMap, ArcMap, ListDigraph


cdef extern from "<lemon/kruskal.h>" namespace "lemon":
  # there seems to be no  to define a templated function in cython
  # therefore all required combinations have to defined explicitly:
  cdef double kruskal(ListDigraph&,ArcMap[float]&,ArcMap[bint]&)

