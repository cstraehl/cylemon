
import cython
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import cython
import numpy as np
cimport numpy as np

from cylemon.lemon cimport smart_graph
from cylemon.lemon cimport preflow
from cylemon.lemon.smart_graph cimport Arc,ArcIt,Node,NodeIt
from cylemon.lemon.smart_graph cimport SmartDigraph as Graph

ctypedef smart_graph.ArcMap[float] ArcMap
ctypedef preflow.Preflow[Graph,ArcMap] Maxflow

cdef extern from "stdlib.h":
  long libc_random "random"()


cdef Graph g 
cdef smart_graph.Arc a
cdef smart_graph.Node n1
cdef smart_graph.Node n2
cdef ArcMap *am = new ArcMap(g)


cdef numArcs  = 100
cdef numNodes = 10


print "Constructing graph..."

g.reserveArc(numArcs)
g.reserveNode(numNodes)

cdef int i
for i in range(numNodes):
  g.addNode()

n1 = g.nodeFromId(0)
n2 = g.nodeFromId(numNodes-1)

cdef int na
cdef int nb

for i in range(numArcs):
  na = i % numNodes
  nb = libc_random() % numNodes
  g.addArc(g.nodeFromId(na), g.nodeFromId(nb))

cdef ArcIt arcit
arcit = ArcIt(g)

while g.id(arcit) > 0 :
  arcit = inc(arcit)
  i = g.id(arcit) 
  am[0][arcit]=np.random.rand()

cdef Maxflow *flow

print "Computing Max-Flow..."

flow = new Maxflow(g,deref(am),n1,n2)
flow.init()
flow.runMinCut()

cdef NodeIt nodeit = NodeIt(g)

while g.id(nodeit) > 0:
  inc(nodeit)
  print "Cut of Node %r is %r" % (g.id(nodeit),flow.minCut(nodeit))


cdef float f
arcit = ArcIt(g)
while g.id(arcit) > 0:
  inc(arcit)
  f = flow.flow(arcit)

  print "Flow on Arc %r = %f" % (g.id(arcit), f)



"""
build watershed adjacency graph
"""
import cython
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import cython
import numpy as np
cimport numpy as np

from cylemon.lemon cimport smart_graph
from cylemon.lemon cimport preflow
from cylemon.lemon.smart_graph cimport Arc,ArcIt,Node,NodeIt
from cylemon.lemon.smart_graph cimport SmartDigraph as Graph

import vigra
import numpy

cdef ArcMap[float] *arcm = new ArcMap[float](g)

dv = numpy.ndarray((10,20,30),numpy.float32)
dl = numpy.random.randint(0,40,(10,20,30)).astype(numpy.int32)
smart_graph.arcMapByLabels(&g,am,dl, dv)

#g = adjacencyGraph.CooGraph()
#g.fromLabelVolume(labelVolume, volumeFeatures)

