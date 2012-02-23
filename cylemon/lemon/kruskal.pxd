cimport smart_graph
from smart_graph cimport Node, NodeIt, Arc, ArcIt, NodeMap, ArcMap, SmartDigraph


cdef extern from "<lemon/kruskal.h>" namespace "lemon":
  # there seems to be no  to define a templated function in cython
  # therefore all required combinations have to defined explicitly:
  cdef double kruskal(SmartDigraph&,ArcMap[float]&,ArcMap[bint]&)

