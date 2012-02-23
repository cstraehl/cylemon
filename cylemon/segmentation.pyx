import pyximport; pyximport.install(pyimport=False)
import h5py
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import cython

import numpy as np
cimport numpy as np

from libcpp.queue cimport queue, priority_queue
from libcpp.deque cimport deque
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref, preincrement as inc


from cylemon.lemon cimport smart_graph
from cylemon.lemon cimport preflow
from cylemon.lemon cimport kruskal
from cylemon.lemon.smart_graph cimport Arc,ArcIt,Node,NodeIt,Snapshot,OutArcIt,InArcIt, ArcMap, NodeMap, INVALID
from cylemon.lemon.smart_graph cimport SmartDigraph as Graph


ctypedef smart_graph.ArcMap[float] ArcMapF
ctypedef smart_graph.ArcMap[bint] ArcMapBool
ctypedef ArcMap[int] ArcMapI
ctypedef preflow.Preflow[Graph,ArcMapF] Maxflow
ctypedef NodeMap[int] NodeMapI


cdef fused value_t:
  char
  int
  long
  float
  double

cdef struct neighborhood_t_t:
  int a
  int b
  float val


cdef inline value_t maximum(value_t a, value_t b): return a if a>= b else b

cdef ArcMap[float]* arcMapByLabels(Graph *digraph,
                      ArcMapF *am,
                      np.ndarray[np.int32_t, ndim=3, mode="strided"] labelMap,
                      np.ndarray[np.float32_t, ndim=3, mode="strided"] nodeMapIn,
                      float (*edgeValueCallback)(float[:]),
                      ):
  """
  builds the adjacency graph structure for an 3D image.

  Copyright 2011 Christoph Straehle cstraehl@iwr.uni-heidelberg.de

  Arguments:
  labelMap -- a 3D np.int32 label image
  edgeMap  -- a 3D np.float32 edge indicator image

  Returns:
  a coo matrix as a tuple (coo_ind, coo_data) (see scipy.sparse) 
  """

  print "Constructing RAG..."
  cdef np.ndarray[np.float32_t, ndim=3] edgeMap =  nodeMapIn


  cdef int sizeX = labelMap.shape[0]
  cdef int sizeY = labelMap.shape[1]
  cdef int sizeZ = labelMap.shape[2]

  cdef int x,y,z,a,b

  # determine maximum label
  cdef int maxLabel = 0
  for x in range(0,sizeX):
    for y in range(0,sizeY):
      for z in range(0,sizeZ):
        maxLabel = maximum(maxLabel,<int>labelMap[x,y,z])  

  digraph.reserveNode(maxLabel)

  cdef np.ndarray[dtype=np.int32_t,ndim=1] neighborCount = np.zeros((maxLabel+1,), dtype=np.int32)

  cdef int totalNeighborhoods = 0

  print "   counting neighborhood sizes"
  # count the number of labels
  for x in range(sizeX):
    for y in range(sizeY):
      for z in range(sizeZ):
        a = labelMap[x,y,z]
        if x < sizeX-1:
          b = labelMap[x+1,y,z]
          if a != b:
            neighborCount[a] += 1
            neighborCount[b] += 1
            totalNeighborhoods += 2
        if y < sizeY-1:
          b = labelMap[x,y+1,z]
          if a != b:
            neighborCount[a] += 1
            neighborCount[b] += 1
            totalNeighborhoods += 2
        if z < sizeZ-1:
          b = labelMap[x,y,z+1]
          if a != b:
            neighborCount[a] += 1
            neighborCount[b] += 1
            totalNeighborhoods += 2

  cdef np.ndarray[dtype=np.int32_t,ndim=1] neighborOffset, offsetBackup
  neighborOffset = np.cumsum(neighborCount).astype(np.int32)
  assert neighborOffset[-1] == totalNeighborhoods
  offsetBackup = neighborOffset.copy()
  offsetBackup[1:] = neighborOffset[:-1]
  offsetBackup[0] = 0
  neighborOffset[:] = offsetBackup[:]

  neighborhood_t = np.dtype([('a', np.int32), ('b', np.int32), ('val', np.float32)], align = True)

  bbb = np.ndarray((totalNeighborhoods,),dtype=neighborhood_t)
  cdef np.ndarray[dtype=neighborhood_t_t, ndim=1]  neighbors = bbb

  print "   adding values to neighborhoods (count=%r) ..." % (totalNeighborhoods,)
  cdef float av,bv
  # add everything to the neighborhood array
  for x in range(0,sizeX):
    for y in range(0,sizeY):
      for z in range(0,sizeZ):
        a = labelMap[x,y,z]
        av = edgeMap[x,y,z]
        if x < sizeX-1:
          b = labelMap[x+1,y,z]
          bv = edgeMap[x+1,y,z]
          if a != b:
            neighbors[neighborOffset[a]].val = (av+bv)/2.0
            neighbors[neighborOffset[a]].a = a
            neighbors[neighborOffset[a]].b = b
            neighborOffset[a]+=1
            neighbors[neighborOffset[b]].val = (av+bv)/2.0
            neighbors[neighborOffset[b]].a = b
            neighbors[neighborOffset[b]].b = a
            neighborOffset[b]+=1
        if y < sizeY-1:
          b = labelMap[x,y+1,z]
          bv = edgeMap[x,y+1,z]
          if a != b:
            neighbors[neighborOffset[a]].val = (av+bv)/2.0
            neighbors[neighborOffset[a]].a = a
            neighbors[neighborOffset[a]].b = b
            neighborOffset[a]+=1
            neighbors[neighborOffset[b]].val = (av+bv)/2.0
            neighbors[neighborOffset[b]].a = b
            neighbors[neighborOffset[b]].b = a
            neighborOffset[b]+=1
        if z < sizeZ-1:
          b = labelMap[x,y,z+1]
          bv = edgeMap[x,y,z+1]
          if a != b:
            neighbors[neighborOffset[a]].val = (av+bv)/2.0
            neighbors[neighborOffset[a]].a = a
            neighbors[neighborOffset[a]].b = b
            neighborOffset[a]+=1
            neighbors[neighborOffset[b]].val = (av+bv)/2.0
            neighbors[neighborOffset[b]].a = b
            neighbors[neighborOffset[b]].b = a
            neighborOffset[b]+=1


  cdef int nsize
  cdef int lastA = -1
  cdef int lastB = -1
  cdef int i
  cdef int l

  print "   sorting neighborhoods..."
  neighborOffset[:] = offsetBackup[:]

  # sort teh neighborhood information
  for i in range(1, neighborOffset.shape[0]-1):
    neighbors[neighborOffset[i]:neighborOffset[i+1]].sort(order=('b')) 

  neighbors[neighborOffset[-1]:].sort(order=('b')) 
  print "a"
  nsize = 0
  lastA = -1
  lastB = -1
  # determine size of coo matrix
  for i in range(neighbors.shape[0]):
    if neighbors[i].a != lastA or neighbors[i].b != lastB:
      lastA = neighbors[i].a
      lastB = neighbors[i].b
      nsize += 1

  print "b"
  bbb = np.ndarray((nsize,2),dtype=np.int32)
  cdef np.ndarray[dtype=np.int32_t, ndim=2]  coo_ind = bbb
  bbb = np.ndarray((nsize,),dtype=np.float32)
  cdef np.ndarray[dtype=np.float32_t, ndim=1]  coo_data = bbb

  cdef int j
  j = 0
  print "c0", neighbors.shape[0]
  lastA = neighbors[0].a
  print "c1"
  lastB = neighbors[0].b
  cdef int lastPos = 0
  cdef np.ndarray[dtype=np.float32_t,ndim=1] bordervalues
  print "   constructing coo graph..."
  # FINALLY, construct the true graph
  for i in range(neighbors.shape[0]):
    if neighbors[i].a != lastA or neighbors[i].b != lastB:
      coo_ind[j,0] = lastA
      coo_ind[j,1] = lastB
      bordervalues = neighbors['val'][lastPos:i]
      assert (neighbors['a'][lastPos:i] == lastA).all()
      assert (neighbors['b'][lastPos:i] == lastB).all()
      coo_data[j] = edgeValueCallback(bordervalues)
      lastA = neighbors[i].a
      lastB = neighbors[i].b
      lastPos = i
      j += 1

  coo_ind[j,0] = lastA
  coo_ind[j,1] = lastB
  bordervalues = neighbors['val'][lastPos:]
  coo_data[j] = edgeValueCallback(bordervalues)
 

  print "   constructing lemon graph..."
  digraph.clear()
  digraph.reserveNode(maxLabel+1)
  digraph.reserveArc(coo_ind.shape[0])

  for i in range(maxLabel+1):
    digraph.addNode()

  cdef Arc aaa
  for i in range(coo_ind.shape[0]):
    aaa = digraph.addArc(digraph.nodeFromId(coo_ind[i,0]),digraph.nodeFromId(coo_ind[i,1]))
    am.set(aaa,coo_data[i])

  print "   constructed graph with"
  print "      num Nodes", maxLabel+1
  print "      num Arcs ", coo_ind.shape[0]






"""
      
      Neighborhood -> edge weight callbacks

"""

cdef float callbackMinimum(float[:] values) nogil:
  cdef float value = 10e10
  cdef int i
  for i in range(values.shape[0]):
    if values[i] < value:
      value = values[i]
  return value

cdef float callbackMaximum(float[:] values) nogil:
  cdef float value = -10e10
  cdef int i
  for i in range(values.shape[0]):
    if values[i] > value:
      value = values[i]
  return value

@cython.cdivision(True)
cdef float callbackAverage(float[:] values) nogil:
  cdef double value = 0
  cdef int i
  for i in range(values.shape[0]):
    value += values[i]
  value = value / values.shape[0]
  return value

cdef float callbackSum(float[:] values) nogil:
  cdef double value = 0
  cdef int i
  for i in range(values.shape[0]):
    value += values[i]
  return value



cdef class IndexAccessor(object):
  cdef object _indexVol
  cdef object _lut

  property lut:
    def __get__(self):
      return self._lut

  def __init__(self,indexVol, lut):
    self._indexVol = indexVol
    self._lut = lut

  def __getitem__(self,key):
    indices = self._indexVol[key]
    result = self._lut[indices]
    return result

  def __setitem__(self,key,value):
    """
    values of zero mean ignore
    value of -1 mean set to zero
    other values mean set to this value
    """
    indices = self._indexVol[key].ravel()
    values = value.ravel()
    
    indicesindices = np.where(values != 0)[0]
    subindices = indices[indicesindices]
    self._lut[subindices] = values[indicesindices]

# cdef fusion int32_2and3dimension:
#   np.ndarray[dtype=np.int32_t,ndim=3]
#   np.ndarray[dtype=np.int32_t,ndim=2]

cdef class Segmentor(object):
  # data
  cdef Graph *graph
  cdef ArcMapF *arcMap

  cdef object _labelVol
  cdef object edgeVol
  cdef object _rawData
  cdef object _seeds
  cdef int    _numNodes
  cdef object _segmentation
  cdef object _edgeWeightFunctor

  property numNodes:
    def __get__(self):
      return self._numNodes

  property segmentation:
    def __get__(self):
      return IndexAccessor(self._labelVol,self._segmentation)
  
  property seeds:
    def __get__(self):
      if self._seeds is None:
        self._seeds = np.ndarray((self._numNodes,),np.uint8)
      return IndexAccessor(self._labelVol,self._seeds)

  property labelVol:
    def __get__(self):
      return self._labelVol

  property raw:
    def __get__(self):
      return self._rawData
    def __set__(self,raw):
      self._rawData = raw

  def __init__(self, labels,
                     edgePMap = None,
                     edgeWeightFunctor = "average",
                     rawData = None):

    """
    build lemon adjacency graph
      labels   :  a two or three dimensional label ndarray
      edgePMap :  a two or three dimensional ndarray that
                  represents the cost of a node being separated from its neighbors
    """
    self._seeds = None
    self._rawData = None
    if edgePMap is None:
      return
    assert labels.dtype == np.int32
    assert edgePMap.dtype == np.float32
    assert labels.ndim <= 3 and labels.ndim >=2
    assert edgePMap.ndim <= 3 and edgePMap.ndim >=2

    self._labelVol = labels
    self.edgeVol = edgePMap
    
    cdef Graph *g = new Graph()
    cdef ArcMapF *am = new ArcMapF(deref(g))

    cdef float (*mycallback)(float[:])
    
    self._edgeWeightFunctor = edgeWeightFunctor


    if edgeWeightFunctor == "average":
      mycallback = callbackAverage
    elif edgeWeightFunctor == "minimum":
      mycallback = callbackMinimum
    elif edgeWeightFunctor == "maximum":
      mycallback = callbackMaximum
    elif edgeWeightFunctor == "sum":
      mycallback = callbackSum

    if edgePMap.ndim == 2:
      edgePMap.shape += (1,)
    if labels.ndim == 2:
      labels.shape += (1,)

    arcMapByLabels(g,am,self._labelVol, self.edgeVol, mycallback)

    cdef NodeIt node
    cdef OutArcIt arcit
    cdef int a,b,i
    cdef float value = 10e10
    cdef float value2

    self.graph = g
    self.arcMap = am
    self._numNodes = g.maxNodeId()+1
    self._seeds = np.zeros((self._numNodes,),np.uint8)
    self._segmentation = np.zeros((self._numNodes,),np.uint8)

    self.printMinimum()

  def printMinimum(self):
    cdef Graph *g = self.graph
    cdef ArcMapF *am = self.arcMap

    cdef NodeIt node
    cdef OutArcIt arcit
    cdef int a,b,i
    cdef float value
    cdef float value2 = 10e10
    cdef float value3 = -10e10

    node = NodeIt(deref(g))
    i = 0
    while node != INVALID:
      arcit = OutArcIt(deref(g),node)
      while arcit != INVALID:
        value = deref(am)[arcit]
        if value < value2:
          value2 = value
        if value > value3:
          value3 = value
        inc(arcit)
      inc(node)
    print "MINIMUM EDGE WEIGHT: ", value2
    print "MAXIMUM EDGE WEIGHT: ", value3



  def saveH5(self, filename, group):
    print "saving segmentor to %r[%r] ..." % (filename, group)
    f = h5py.File(filename,"w")
    try:
      f.create_group("graph")
    except:
      pass

    
    g = f["graph"]
    g.attrs["numNodes"] = self._numNodes
    g.attrs["edgeWeightFunctor"] = self._edgeWeightFunctor
    d_labels = g.create_dataset("labels", data = self._labelVol)

    cdef np.ndarray[dtype=np.int32_t, ndim = 2] indices = np.ndarray((self.graph.maxArcId()+1,2),dtype=np.int32)
    cdef np.ndarray[dtype=np.float32_t, ndim = 1] data = np.ndarray((self.graph.maxArcId()+1,),dtype=np.float32)
    cdef NodeIt node
    cdef OutArcIt arcit
    cdef int a,b,i

    node = NodeIt(deref(self.graph))
    i = 0
    while node != INVALID:
      arcit = OutArcIt(deref(self.graph),Node(node))
      while arcit != INVALID:
        a = self.graph.id(self.graph.source(arcit))
        b = self.graph.id(self.graph.target(arcit))
        indices[i,0] = a
        indices[i,1] = b
        data[i] = deref(self.arcMap)[arcit]
        i += 1
        inc(arcit)
      inc(node)
    
    g.create_dataset("coo_indices",data=indices)
    f.flush()
    g.create_dataset("coo_data",data=data)
    f.flush()
    g.create_dataset("regions", data=self._labelVol)
    g.create_dataset("seeds",data = self._seeds)
    if self._rawData is not None:
      g.create_dataset("raw",data=self._rawData)
    f.close()
    print "   done"

  @classmethod
  def loadH5(cls,filename,groupname):
    print "loading segmentor from %r[%r] ..." % (filename, groupname)
    f = h5py.File(filename,"r")
    gr = f[groupname]
    numNodes = gr.attrs["numNodes"]
    edgeWeightFunctor = gr.attrs["edgeWeightFunctor"]

    labels = gr["regions"][:]
    cdef np.ndarray[dtype=np.int32_t, ndim = 2] indices = gr["coo_indices"][:]
    cdef np.ndarray[dtype=np.float32_t, ndim = 1] data = gr["coo_data"][:]
    cdef np.ndarray[dtype=np.uint8_t,ndim=1] seeds = gr["seeds"][:]

    cdef Segmentor instance = cls(labels=labels)

    cdef Graph *g = new Graph()

    g.reserveNode(numNodes)
    cdef int i
    for i in range(numNodes):
      g.addNode()

    cdef ArcMapF *am = new ArcMapF(deref(g))
    cdef Arc a
    g.reserveArc(indices.shape[0])
    for i in range(indices.shape[0]):
      a = g.addArc(g.nodeFromId(indices[i,0]),g.nodeFromId(indices[i,1]))
      am.set(a,data[i])


    instance.graph = g
    instance.arcMap = am
    instance._edgeWeightFunctor = edgeWeightFunctor
    instance._numNodes = numNodes
    instance._labelVol = labels
    instance._seeds = seeds
    instance._rawData = gr["raw"][:]
    instance.printMinimum()
    print "   done"

    f.close()
    return instance


cdef class GCSegmentor(Segmentor):
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries):
    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color
    """
    print "segmenting..."
    self._segmentation = np.ndarray((self._numNodes,),dtype=np.int32)
    # take snapshot of the current graph
    print "   snapshotting graph..."
    cdef Snapshot sns = Snapshot(deref(self.graph))
    print "   adding unary potentials..."


    #shorthand
    cdef Graph *g = self.graph

    # add source and sink
    cdef Node source = g.addNode()
    cdef int sourceId = g.id(source)
    cdef Node sink = g.addNode()
    cdef int sinkId = g.id(sink)

    cdef int i,j
    cdef Arc a

    for i in range(unaries.shape[0]):
      if unaries[i,0] > 0:
        a = g.addArc(g.nodeFromId(i),sink)
        deref(self.arcMap)[a] = unaries[i,0]
      if unaries[i,1] > 0:
        a = g.addArc(source,g.nodeFromId(i))
        deref(self.arcMap)[a] = unaries[i,1]

      
    print "   running graph-cut..."
    cdef Maxflow *flow = new Maxflow(deref(g),deref(self.arcMap),source,sink)
    flow.init()
    flow.runMinCut()   

    cdef int count0 = 0
    cdef int count1 = 0
    cdef np.ndarray[dtype=np.int32_t,ndim=1] _segmentation = self._segmentation
    _segmentation[:] = -1

    for i in range(self._numNodes):
      if not flow.minCut(g.nodeFromId(i)):
        count0 += 1
        _segmentation[i] = 0
      else:
        count1 += 1
        _segmentation[i] = 1

    print "     color %r: count = %r" % (0,count0)
    print "     color %r: count = %r" % (1,count1)

    print "   restoring graph..."
    sns.restore()




cdef class MSTSegmentor(Segmentor):
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries):
    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color

    """
    print "segmenting..."
    self._segmentation = np.ndarray((self._numNodes,),dtype=np.int32)
    # take snapshot of the current graph
    print "   snapshotting graph..."
    cdef Snapshot sns = Snapshot(deref(self.graph))
    print "   adding unary potentials graph..."


    #shorthand
    cdef Graph *g = self.graph
    cdef ArcMapF *am = self.arcMap

    # add source and sink
    cdef int i,j
    cdef Node nodep
    cdef Arc a
    cdef vector[Node] sources

    # add meta root node 
    cdef Node root = g.addNode()
    cdef int rootId = g.id(root)

    # add color meta nodes
    for i in range(unaries.shape[1]):
      nodep = g.addNode()
      j = g.id(nodep)
      sources.push_back(nodep)
      a = g.addArc(root,nodep)

      # connect color node and root node
      # with large negative weight
      # to enforce inclusion in MST
      deref(am)[a] = -10e10
    
    cdef np.ndarray[dtype=np.uint8_t,ndim=1] seeds = self._seeds
    for i in range(seeds.shape[0]):
      if seeds[i] != 0:
        j = seeds[i]
        a = g.addArc(sources[j],g.nodeFromId(i))
        deref(am)[a] = -10e10 



    # # add unary potentials
    # for i in range(unaries.shape[0]):
    #   for j in range(unaries.shape[1]):
    #     if unaries[i,j] > 0:
    #       a = g.addArc(sources[j],g.nodeFromId(i))
    #       deref(am)[a] = unaries[i,j]

      
    print "   running kruskal..."
    cdef ArcMapBool *result = new ArcMapBool(deref(g))
    kruskal.kruskal(deref(g),deref(am),deref(result))

    print "   determing segmentation..."


    cdef OutArcIt oit
    cdef InArcIt iit
    cdef queue[Node] q
    cdef Node tnode 
    cdef int tid
    cdef np.ndarray[dtype=np.int32_t,ndim=1] seg = self._segmentation
    seg[:] = -1
    

    # retrieve the "segmentation color":
    for i in range(unaries.shape[1]):
      nodep = sources[i]
      q.push(nodep)
      while not q.empty():
        nodep = q.front()
        q.pop()
        oit = OutArcIt(deref(g),nodep)
        while oit != INVALID:
          tnode = g.target(oit)
          tid = g.id(tnode)
          if deref(result)[oit] == 1 and tid < self._numNodes and seg[tid] == -1 and tnode != root:
            q.push(tnode)
            seg[g.id(tnode)] = i
          inc(oit)
        iit = InArcIt(deref(g),nodep)
        while iit != INVALID:
          tnode = g.source(iit)
          tid = g.id(tnode)
          if deref(result)[iit] == 1 and tid < self._numNodes and seg[tid] == -1 and tnode != root:
            q.push(tnode)
            seg[g.id(tnode)] = i
          inc(iit)

    cdef int count
    cdef int unknown = 0
    for j in range(unaries.shape[1]):
      count = 0
      for i in range(unaries.shape[0]):
        if seg[i] == j:
          count += 1
        elif seg[i] == -1:
          unknown += 1
      print "     color %r: count = %r" % (j,count)
    print "     unknown color: count = %r" % (unknown,)

    print "   restoring graph..."
    sns.restore()


cdef extern from "segmentation.hxx":
  cdef void prioMST(Graph &, NodeMap[int] &segmentation, ArcMap[float] &weights, vector[float] &prio)



cdef class MSTSegmentor2(Segmentor):
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries, prios = None):
    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color

    """
    print "segmenting..."
    self._segmentation = np.ndarray((self._numNodes,),dtype=np.int32)
    # take snapshot of the current graph

    print "   snapshotting graph..."
    cdef Snapshot sns = Snapshot(deref(self.graph))
    print "   adding seeds to graph..."


    #shorthand
    cdef Graph *g = self.graph
    cdef ArcMapF *am = self.arcMap

    # add source and sink
    cdef int i,j
    cdef Node nodep
    cdef Arc a
    cdef vector[Node] sources

    cdef NodeMapI *segmentation = new NodeMapI(deref(g))
    cdef np.ndarray[dtype=np.uint8_t,ndim=1] seeds = self._seeds
    for i in range(seeds.shape[0]):
      segmentation.set(g.nodeFromId(i),seeds[i])
    

    cdef vector[float] prio
    prio.resize(unaries.shape[1], 1.0)
    if prios is not None:
      for i in range(prio.size()):
        prio[i] = prios[i]


    print "   running prioMST..."
    prioMST(deref(g), deref(segmentation), deref(am), prio)


    cdef np.ndarray[ndim=1,dtype=np.int32_t] outseg = self._segmentation
    for i in range(seeds.shape[0]):
      outseg[i] = deref(segmentation)[g.nodeFromId(i)]
    
    print "   restoring graph..."
    sns.restore()
