#cython: boundscheck=False
#import pyximport; pyximport.install(pyimport=False)
import h5py
cimport cython
from cython.operator cimport dereference as deref, preincrement as inc
import cython
import sys
import time
import math

import numpy as np
cimport numpy as np

from libcpp.queue cimport queue, priority_queue
from libcpp.deque cimport deque
from libcpp.vector cimport vector

from cython.operator cimport dereference as deref, preincrement as inc

from cylemon.lemon cimport preflow
from cylemon.lemon cimport kruskal
from cylemon.lemon.list_graph cimport Arc,ArcIt,Node,NodeIt,OutArcIt,InArcIt, ArcMap, NodeMap, INVALID
from cylemon.lemon.list_graph cimport ListDigraph as Graph


ctypedef ArcMap[float] ArcMapF
ctypedef ArcMap[bint] ArcMapBool
ctypedef ArcMap[int] ArcMapI
ctypedef preflow.Preflow[Graph,ArcMapF] Maxflow
ctypedef NodeMap[int] NodeMapI
ctypedef NodeMap[long] NodeMapL

cdef extern from "math.h":
    double sqrt(double x)
    double exp(double x)


cdef extern from "segmentation.hxx":
  cdef void prioMST(Graph &, NodeMap[int] &seeds, NodeMap[int] &segmentation, ArcMap[float] &weights,
                    ArcMap[bint] &intree, vector[float] &prio, int noBiasBelow)
  cdef void prioMSTmargin(Graph &, NodeMap[int] &, ArcMap[float] &, NodeMap[float] &, vector[float] &)
  cdef void prioMSTmav(Graph &, NodeMap[int] &seeds, NodeMap[int] &segmentation, ArcMap[float] &weights, ArcMap[bint] &intree, vector[float] &prio)
  cdef void edgeExchangeCount(Graph &, NodeMap[int] &seeds, NodeMap[int] &segmentation, ArcMap[float] &weights, ArcMap[bint] &intree, ArcMap[int] &exchangeCount, vector[float] &prio)
  cdef void prioMSTperturb(Graph &, NodeMap[int] &seeds, NodeMap[int] &segmentation, ArcMap[float] &weights, NodeMap[long] &cumSubtreeSize, ArcMap[int] &edgeExchangeCount, int trials, float perturbation, vector[float] & prio, bint moving_average) 
  cdef void maxDistanceNode(Graph &, ArcMap[float] &, NodeMap[int] &, Node &, float &)

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
                      object precomputedEdgeWeights,
                      object progressCallback
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

  ragTimeStart = time.time()
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
  print "   maximum label = %d" % maxLabel

  digraph.reserveNode(maxLabel)

  cdef np.ndarray[dtype=np.int32_t,ndim=1] neighborCount = np.zeros((maxLabel+1,), dtype=np.int32)
  print "   neighborCount: %f MB" % (neighborCount.nbytes / float(1024**2),) 

  cdef int totalNeighborhoods = 0

  timeStart = time.time()
  # count the number of labels
  for x in range(sizeX):
    if x % int(math.ceil(1.0 * sizeX / 1000.0)) == 0:
      progressCallback(10.0*x/float(sizeX)) # show progress 0..10
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
  
  print "\n   counting nhood sizes: %f sec.          " % (time.time()-timeStart)
  cdef np.ndarray[dtype=np.int32_t,ndim=1] neighborOffset, offsetBackup
  neighborOffset = np.cumsum(neighborCount).astype(np.int32)
  print "   neighborOffset: %f MB" % (neighborOffset.nbytes / float(1024**2),) 
  assert neighborOffset[-1] == totalNeighborhoods
  offsetBackup = neighborOffset.copy()
  print "   offsetBackup: %f MB" % (offsetBackup.nbytes / float(1024**2),) 
  offsetBackup[1:] = neighborOffset[:-1]
  offsetBackup[0] = 0
  neighborOffset[:] = offsetBackup[:]

  neighborhood_t = np.dtype([('a', np.int32), ('b', np.int32), ('val', np.float32)], align = True)

  bbb = np.ndarray((totalNeighborhoods,),dtype=neighborhood_t)
  print "   bbb: %f MB" % (bbb.nbytes / float(1024**2),) 
  cdef np.ndarray[dtype=neighborhood_t_t, ndim=1]  neighbors = bbb

  timeStart = time.time()
  print "   adding values to neighborhoods (count=%r) ..." % (totalNeighborhoods,)
  cdef float av,bv
  # add everything to the neighborhood array
  
  for x in range(0,sizeX):
    if x % int(math.ceil(1.0 * sizeX / 1000.0)) == 0:
      progressCallback(10+10.0*x/float(sizeX)) # show progress 10..20
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
  
  print "\n    add to nhood array: %f sec." % (time.time()-timeStart)
  
  cdef int nsize
  cdef int lastA = -1
  cdef int lastB = -1
  cdef int i
  cdef int l

  print "   sorting neighborhoods..."
  neighborOffset[:] = offsetBackup[:]

  # sort the neighborhood information
  timeStart = time.time()
  for i in range(1, neighborOffset.shape[0]-1):
    if i % int(math.ceil(1.0 * neighborOffset.shape[0] / 1000.0)) == 0:
        progressCallback(20+40.0*i/float(neighborOffset.shape[0])) # show progress 20..60
    neighbors[neighborOffset[i]:neighborOffset[i+1]].sort(order=('b'))
  
  print "\n   ... took %f sec." % (time.time()-timeStart)

  neighbors[neighborOffset[-1]:].sort(order=('b')) 
  nsize = 0
  lastA = -1
  lastB = -1
  # determine size of coo matrix
  for i in range(neighbors.shape[0]):
    if neighbors[i].a != lastA or neighbors[i].b != lastB:
      lastA = neighbors[i].a
      lastB = neighbors[i].b
      nsize += 1

  bbb = np.ndarray((nsize,2),dtype=np.int32)
  cdef np.ndarray[dtype=np.int32_t, ndim=2]  coo_ind = bbb
  bbb = np.ndarray((nsize,),dtype=np.float32)
  cdef np.ndarray[dtype=np.float32_t, ndim=1]  coo_data = bbb

  cdef int j
  j = 0
  lastA = neighbors[0].a
  lastB = neighbors[0].b
  cdef int lastPos = 0
  cdef np.ndarray[dtype=np.float32_t,ndim=1] bordervalues
  print "   constructing coo graph..."
  # FINALLY, construct the true graph
  cdef int ll
  ll = neighbors.shape[0]
  
  for i in range(neighbors.shape[0]):
    if i % int(math.ceil(1.0 * neighbors.shape[0] / 1000.0)) == 0:
        progressCallback(60+40.0*i/float(ll)) # show progress 60..100
    if neighbors[i].a != lastA or neighbors[i].b != lastB:
      coo_ind[j,0] = lastA
      coo_ind[j,1] = lastB

      assert (neighbors['a'][lastPos:i] == lastA).all()
      assert (neighbors['b'][lastPos:i] == lastB).all()
     
      if precomputedEdgeWeights is None:
        bordervalues = neighbors['val'][lastPos:i]
        coo_data[j] = edgeValueCallback(bordervalues)
      else:
        if lastA > lastB:
          t = (lastB, lastA)
        else:
          t = (lastA, lastB)

        if t not in precomputedEdgeWeights:
          raise RuntimeError("%s not in precomputedEdgeWeights" % (t,))
        coo_data[j] = 255*precomputedEdgeWeights[t]

      lastA = neighbors[i].a
      lastB = neighbors[i].b
      lastPos = i
      j += 1
  coo_ind[j,0] = lastA
  coo_ind[j,1] = lastB

  if precomputedEdgeWeights is None:
    bordervalues = neighbors['val'][lastPos:]
    coo_data[j] = edgeValueCallback(bordervalues)
  else:
    if lastA > lastB:
      t = (lastB, lastA)
    else:
      t = (lastA, lastB)
    if t not in precomputedEdgeWeights:
      raise RuntimeError("%s not in precomputedEdgeWeights" % (t,))
    coo_data[j] = 255*precomputedEdgeWeights[t]

  print "\n   constructing lemon graph..."
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
  print "      took %f sec." % (time.time() - ragTimeStart)
  progressCallback(100)





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

cdef np.ndarray[ndim=2, dtype=np.int32_t] calcRegionCenters( np.ndarray[np.int32_t, ndim=3, mode="strided"] labelMap, int labelCount ):
  """
  Calculate the center of mass for all connected regions in a labelMap
  """
  cdef np.ndarray[ndim=2,dtype=np.int32_t] centers = np.zeros((labelCount,3), np.int32)
  cdef np.ndarray[ndim=2,dtype=np.int64_t] axisSum = np.zeros((labelCount,3), np.int64)
  cdef np.ndarray[ndim=1,dtype=np.int64_t] count = np.zeros((labelCount,), np.int64)

  cdef int sizeX = labelMap.shape[0]
  cdef int sizeY = labelMap.shape[1]
  cdef int sizeZ = labelMap.shape[2]

  cdef int i,x,y,z,label

  for x in range(0,sizeX):
    for y in range(0,sizeY):
      for z in range(0,sizeZ):
        label = labelMap[x,y,z]
        axisSum[label,0] += x
        axisSum[label,1] += y
        axisSum[label,2] += z
        count[label] += 1
  
  for i in range(labelCount):
    if count[i] > 0:
      centers[i,0] = <int> (axisSum[i,0] / count[i])
      centers[i,1] = <int> (axisSum[i,1] / count[i])
      centers[i,2] = <int> (axisSum[i,2] / count[i])

  return centers

cdef np.ndarray[ndim=1, dtype=np.int32_t] calcRegionSizes( np.ndarray[np.int32_t, ndim=3, mode="strided"] labelMap, int labelCount ):

  cdef int sizeX = labelMap.shape[0]
  cdef int sizeY = labelMap.shape[1]
  cdef int sizeZ = labelMap.shape[2]
  cdef np.ndarray[ndim=1, dtype=np.int32_t] sizes = np.zeros((labelCount,), np.int32)
  cdef int i,x,y,z,label

  for x in range(0,sizeX):
    for y in range(0,sizeY):
      for z in range(0,sizeZ):
        label = labelMap[x,y,z]
        sizes[label] += 1
  
  return sizes

cdef class IndexAccessor(object):
  cdef object _indexVol
  cdef object _lut

  property lut:
    def __get__(self):
      return self._lut

    def __set__(self,value):
      self._lut[:] = value

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
    if not isinstance(value, np.ndarray):
      a = np.ndarray(indices.shape, self._lut.dtype)
      a[:] = value
      value = a
    values = value.ravel()
    
    indicesindices = np.where((values != 0) * (values != 255))[0]
    deleteindices = np.where(values == 255)[0]
    subindices = indices[indicesindices]
    self._lut[subindices] = values[indicesindices]

    deletesubindices = indices[deleteindices]
    self._lut[deletesubindices] = 0

# cdef fusion int32_2and3dimension:
#   np.ndarray[dtype=np.int32_t,ndim=3]
#   np.ndarray[dtype=np.int32_t,ndim=2]

cdef class Segmentor(object):
  # data
  cdef Graph *graph
  cdef ArcMapF *arcMap

  cdef object _regionVol
  cdef object edgeVol
  cdef object _rawData
  cdef object _seeds
  cdef object _objects
  cdef object _object_names
  cdef object _object_lut
  cdef object _object_seeds_fg
  cdef object _object_seeds_bg
  cdef object _object_seeds_fg_voxels
  cdef object _object_seeds_bg_voxels
  cdef object _no_bias_below
  cdef object _bg_priority
  cdef int    _numNodes
  cdef object _segmentation
  cdef object _uncertainty
  cdef object _edgeWeightFunctor
  cdef object _regionCenter
  cdef object _regionSize

  property object_names:
    def __get__(self):
      return self._object_names
    def __set__(self, value):
      self._object_names = value

  property no_bias_below:
    def __get__(self):
      return self._no_bias_below
    def __set__(self, value):
      self._no_bias_below = value

  property bg_priority:
    def __get__(self):
      return self._bg_priority
    def __set__(self, value):
      self._bg_priority = value

  property object_seeds_fg_voxels:
    def __get__(self):
      return self._object_seeds_fg_voxels
    def __set__(self, value):
      self._object_seeds_fg_voxels = value

  property object_seeds_bg_voxels:
    def __get__(self):
      return self._object_seeds_bg_voxels
    def __set__(self, value):
      self._object_seeds_bg_voxels = value

  property object_lut:
    def __get__(self):
      return self._object_lut
    def __set__(self, value):
      self._object_lut = value

  property object_seeds_fg:
    def __get__(self):
      return self._object_seeds_fg
    def __set__(self, value):
      self._object_seeds_fg = value

  property object_seeds_bg:
    def __get__(self):
      return self._object_seeds_bg
    def __set__(self, value):
      self._object_seeds_bg = value

  property regionSize:
    def __get__(self):
      return self._regionSize

  property regionCenter:
    def __get__(self):
      return self._regionCenter

  property numNodes:
    def __get__(self):
      return self._numNodes

  property segmentation:
    def __get__(self):
      return IndexAccessor(self._regionVol,self._segmentation)
  
  property uncertainty:
    def __get__(self):
      return IndexAccessor(self._regionVol, self._uncertainty)

  property seeds:
    def __get__(self):
      if self._seeds is None:
        self._seeds = np.ndarray((self._numNodes,),np.uint8)
      return IndexAccessor(self._regionVol,self._seeds)
  
  property objects:
    def __get__(self):
      if self._objects is None:
        self._objects = np.ndarray((self._numNodes,),np.uint32)
      return IndexAccessor(self._regionVol,self._objects)

  property regionVol:
    def __get__(self):
      return self._regionVol

  property raw:
    def __get__(self):
      return self._rawData
    def __set__(self,raw):
      self._rawData = raw

  def __init__(self, labels,
                     edgePMap = None,
                     edgeWeightFunctor = "average",
                     rawData = None,
                     precomputedEdgeWeights=None,
                     progressCallback = lambda x:None):

    """
    build lemon adjacency graph
      labels   :  a two or three dimensional label ndarray
      edgePMap :  a two or three dimensional ndarray that
                  represents the cost of a node being separated from its neighbors
    """
    self._seeds = None
    self._rawData = None  
    self.object_names = dict()
    self.object_seeds_fg = dict()
    self.object_seeds_bg = dict()      
    self._object_seeds_fg_voxels = dict()
    self._object_seeds_bg_voxels = dict()
    self._no_bias_below = dict()
    self._bg_priority = dict()
    self.object_lut = dict()
    if edgePMap is None:
      return
    assert labels.dtype == np.int32
    assert edgePMap.dtype == np.float32
    assert labels.ndim <= 3 and labels.ndim >=2
    assert edgePMap.ndim <= 3 and edgePMap.ndim >=2

    self._regionVol = labels
    minv = np.min(edgePMap)
    maxv = np.max(edgePMap)

    self.edgeVol = ((edgePMap - minv) / ( maxv - minv) * 255).astype(np.float32) #normalize edge probability map to 0-255
    
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

    arcMapByLabels(g,am,self._regionVol, self.edgeVol, mycallback, precomputedEdgeWeights,progressCallback)

    cdef NodeIt node
    cdef OutArcIt arcit
    cdef int a,b,i
    cdef float value = 10e10
    cdef float value2

    self.graph = g
    self.arcMap = am
    self._numNodes = g.maxNodeId()+1
    self._seeds = np.zeros((self._numNodes,),np.uint8)

    self._objects = np.zeros((self._numNodes,),np.uint32)


    self._segmentation = np.zeros((self._numNodes,),np.int32)
    self._uncertainty = np.zeros((self._numNodes,),np.uint8)
    self._regionCenter = calcRegionCenters(self._regionVol, self._numNodes)
    self._regionSize = calcRegionSizes(self._regionVol, self._numNodes)
  
  def saveH5(self, filename, groupname, mode="w"):
    print "saving segmentor to %r[%r] ..." % (filename, groupname)
    f = h5py.File(filename, mode)
    try:
      f.create_group(groupname)
    except:
      pass
    h5g = f[groupname]
    self.saveH5G(h5g)

  def saveH5G(self, h5g):
    g = h5g
    g.attrs["numNodes"] = self._numNodes
    g.attrs["edgeWeightFunctor"] = self._edgeWeightFunctor
    g.create_dataset("labels", data = self._regionVol)

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
    g.create_dataset("coo_data",data=data)
    g.create_dataset("regions", data=self._regionVol)
    g.create_dataset("regionCenter", data=self._regionCenter)
    g.create_dataset("regionSize", data=self._regionSize)
    g.create_dataset("seeds",data = self._seeds)
    g.create_dataset("objects",data = self._objects)

    sg = g.create_group("objects_seeds")
    
    g.file.flush()
    
    # delete old attributes
    for k in sg.attrs.keys():
      del sg.attrs[k]
      

    # insert new attributes
    for k,v in self.object_names.items():
      print "   -> saving object %r with Nr=%r" % (k,v)
      sg.attrs[k] = v
      og = sg.create_group(k)
      og.create_dataset("foreground", data = self.object_seeds_fg[k])
      og.create_dataset("background", data = self.object_seeds_bg[k])

    if self._rawData is not None:
      g.create_dataset("raw",data=self._rawData)

    g.file.flush()
    print "   done"
  
  @classmethod
  def loadH5(cls,file_name, group_name):
    print "loading segmentor from %r[%r] ..." % (file_name, group_name)
    h5f = h5py.File(file_name, "r")
    h5g = h5f[group_name]
    return cls.loadH5G(h5g)

  @classmethod
  def loadH5G(cls,h5g):
    gr = h5g
    numNodes = gr.attrs["numNodes"]
    edgeWeightFunctor = gr.attrs["edgeWeightFunctor"]

    labels = gr["regions"][:]
    cdef np.ndarray[dtype=np.int32_t, ndim = 2] indices = gr["coo_indices"][:]
    cdef np.ndarray[dtype=np.float32_t, ndim = 1] data = gr["coo_data"][:]
    cdef np.ndarray[dtype=np.uint8_t,ndim=1] seeds = gr["seeds"][:]
    cdef np.ndarray[dtype=np.uint32_t,ndim=1] objects 
    if "objects" in gr.keys():
      objects = gr["objects"][:]
    else:
      objects = np.zeros((numNodes,), np.uint32)

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
    instance._regionVol = labels
    instance._seeds = seeds
    instance._objects = objects
    if "raw" in gr.keys():
        instance._rawData = gr["raw"][:]
    else:
        instance._rawData = None
    instance._regionCenter = gr["regionCenter"][:]
    instance._regionSize = gr["regionSize"][:]
    instance._segmentation = np.zeros((numNodes,),np.int32)
    instance._uncertainty = np.zeros((numNodes,),np.uint8)

    if "objects" in gr.keys():
      for k,v in gr["objects_seeds"].attrs.items():
        print "   -> loading object %r with Nr=%r" % (k,v)
        instance.object_names[k] = v
        instance.object_seeds_fg[k] = gr["objects_seeds"][k]["foreground"][:]
        instance.object_seeds_bg[k] = gr["objects_seeds"][k]["background"][:]
    
    print "   done"

    return instance


  @classmethod
  def fromOtherSegmentor(cls, Segmentor seg):
    cdef Segmentor instance = cls(labels=seg._regionVol)
    instance.graph = seg.graph
    instance.arcMap = seg.arcMap
    instance._edgeWeightFunctor = seg._edgeWeightFunctor
    instance._numNodes = seg._numNodes
    instance._regionVol = seg._regionVol
    instance._seeds = seg._seeds
    instance._objects = seg._objects
    instance.object_names = seg.object_names
    instance.object_seeds_fg = seg.object_seeds_fg
    instance.object_seeds_bg = seg.object_seeds_bg
    instance._rawData = seg._rawData
    instance._regionCenter = seg._regionCenter
    instance._regionSize = seg._regionSize
    instance._segmentation = seg._segmentation
    instance._uncertainty = seg._uncertainty
    return instance



  def getCenterOfRegion(self, np.ndarray[ndim=1, dtype=np.int32_t] regions):
    """
    determine the center node from a bunch of labeled nodes (given by regions!=0)
    the number of the center node, i.e. the index into the region array, is returned
    """
    assert regions.shape[0] == self.numNodes
    cdef Graph* g = self.graph
    cdef ArcMapF *dm = new ArcMapF(deref(g))
    cdef NodeMapI *nm = new NodeMapI(deref(g))
    cdef np.ndarray[ndim=2, dtype=np.int32_t] centers = self._regionCenter

    cdef int i, counter
    counter = 0
    for i in range(self.numNodes):
      if regions[i] == 0:
        deref(nm)[g.nodeFromId(i)] = 1
        counter += 1
      else:
        deref(nm)[g.nodeFromId(i)] = 0
    
    if counter == self.numNodes:
      del dm
      del nm
      return -1, 0
    
    cdef int a,b
    cdef ArcIt ait = ArcIt(deref(g)) 
    cdef double sqrt_arg
    while ait != INVALID:
      a = g.id(g.source(ait))
      b = g.id(g.target(ait))
      # calculate euclidean distance
      sqrt_arg = (centers[a,0] - centers[b,0])**2 + (centers[a,1] - centers[b,1])**2 +(centers[a,2] - centers[b,2])**2
      deref(dm)[ait] = sqrt(sqrt_arg)
      ##print "Distance between node %d and %d : %f" % (a,b, deref(dm)[ait])
      inc(ait)
    
    cdef Node n
    cdef float distance = 0
    maxDistanceNode(deref(g), deref(dm), deref(nm), n, distance)

    del dm
    del nm

    return g.id(n), distance

  def volumeLabelsToRegionLabels(self, np.ndarray[ndim=3,dtype=np.int32_t] labelMap):
    """
    Map a labelVolume to the regionVolume such that
    each region is assigned the label that appears most often in it
    """
    cdef int sizeX = labelMap.shape[0]
    cdef int sizeY = labelMap.shape[1]
    cdef int sizeZ = labelMap.shape[2]
    cdef np.ndarray[ndim=3, dtype=np.int32_t] regionVol = self._regionVol
    cdef int x,y,z,region, label, i, j, oldLabel, bestLabel, bestLabelSize, counter
    cdef np.ndarray[ndim=1, dtype=np.int32_t] result = np.ndarray((self._numNodes,),np.int32)
    cdef np.ndarray[ndim=1, dtype=np.int32_t] offsets = np.cumsum(self._regionSize).astype(np.int32)
    cdef int totalSize = offsets[-1]
    offsets[1:] = offsets[0:-1]
    offsets[0] = 0

    cdef np.ndarray[ndim=1, dtype=np.int32_t] regionLabels = np.ndarray((totalSize,), np.int32)

    # determine the labels that appear in one region
    for x in range(0,sizeX):
      for y in range(0,sizeY):
        for z in range(0,sizeZ):
          region = regionVol[x,y,z]
          label = labelMap[x,y,z]
          regionLabels[offsets[region]] = label
          offsets[region] += 1
    
    cdef np.ndarray[ndim=1, dtype=np.int32_t] regionStop = offsets
    cdef np.ndarray[ndim=1, dtype=np.int32_t] regionStart = np.cumsum(self._regionSize).astype(np.int32)
    regionStart[1:] = regionStop[0:-1]
    regionStart[0] = 0
    
    # sort the labels that appear in one region
    for i in range(self._numNodes):
      regionLabels[regionStart[i]:regionStop[i]].sort()

    # determine the maximum label of each region
    for i in range(self._numNodes):
      oldLabel = -1
      bestLabel = -1
      bestLabelSize = 0
      counter = 0
      for j in range(regionStart[i], regionStop[i]):
        label = regionLabels[j]
        if label != oldLabel:
          if counter > bestLabelSize:
            bestLabel = oldLabel
            bestLabelSize = counter
          counter = 1
          oldLabel = label
        else:
          counter += 1
      if counter > bestLabelSize:
        bestLabel = oldLabel
      result[i] = bestLabel
    
    return result


cdef class GCSegmentor(Segmentor):
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries):
    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color
    """
    print "segmenting..."
    # take snapshot of the current graph
    print "   snapshotting graph..."
    #cdef Snapshot sns = Snapshot(deref(self.graph))

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
    #sns.restore()

    del flow




cdef class MSTSegmentorKruskal(Segmentor):
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries):
    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color

    """
    print "segmenting..."
    # take snapshot of the current graph
    print "   snapshotting graph..."
    #cdef Snapshot sns = Snapshot(deref(self.graph))
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

    del result

    #sns.restore()


cdef class MSTSegmentor(Segmentor):    
  
  cdef object _exchangeCount

  property exchangeCount:
    def __get__(self):
      if self._exchangeCount is None:
        self._exchangeCount = np.ndarray((self._numNodes,),np.int32)
      return IndexAccessor(self._regionVol,self._exchangeCount)
 
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries, prios = None, uncertainty="exchangeCount",
          moving_average = False, noBiasBelow = 0, **kwargs):
    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color

    """
    print "segmenting..."
    # take snapshot of the current graph

    print "   snapshotting graph..."
    #cdef Snapshot sns = Snapshot(deref(self.graph))

    #shorthand
    cdef Graph *g = self.graph
    cdef ArcMapF *am_backup = self.arcMap

    cdef ArcMapF *am = new ArcMapF(deref(g))
    cdef ArcIt cait  = ArcIt(deref(g))
    while cait != INVALID:
      deref(am)[cait] = deref(am_backup)[cait]
      inc(cait)

    # add source and sink
    cdef int i,j
    cdef Arc a

    cdef NodeMapI *segmentation = new NodeMapI(deref(g))
    cdef NodeMapI *origSeeds = new NodeMapI(deref(g))
    cdef np.ndarray[dtype=np.uint8_t,ndim=1] seeds = self._seeds
    for i in range(seeds.shape[0]):
      segmentation.set(g.nodeFromId(i),seeds[i])
      origSeeds.set(g.nodeFromId(i),seeds[i])
    

    cdef vector[float] prio
    prio.resize(unaries.shape[1], 1.0)
    if prios is not None:
      for i in range(prio.size()):
        prio[i] = prios[i]



    print "   running prioMST..."
    cdef ArcMapBool *intree = new ArcMapBool(deref(g))
    if moving_average:
      prioMSTmav(deref(g), deref(origSeeds), deref(segmentation), deref(am), deref(intree),prio)

    else:
      prioMST(deref(g), deref(origSeeds), deref(segmentation), deref(am), deref(intree),prio, noBiasBelow)


    cdef ArcMapI *exchangeCount = new ArcMapI(deref(g))
    cdef NodeMap[float] *localMargin = new NodeMap[float](deref(g))

    # store the segmentatino in the result ndarray
    cdef np.ndarray[ndim=1,dtype=np.int32_t] outseg = self._segmentation


    for i in range(seeds.shape[0]):
      outseg[i] = deref(segmentation)[g.nodeFromId(i)]

    cdef np.ndarray[ndim=1, dtype=np.int32_t] outExchangeCount 
    cdef np.ndarray[ndim=1, dtype=np.float32_t] outMargin = np.ndarray((self.numNodes,), np.float32)
    cdef InArcIt ait
    if uncertainty == "exchangeCount":
      print "   calculating edge exchange count..."
      edgeExchangeCount(deref(g), deref(origSeeds), deref(segmentation), deref(am), deref(intree), deref(exchangeCount), prio)
      temp = self.exchangeCount # just ensure the array self._exchangeCount exists
      outExchangeCount = self._exchangeCount
      outExchangeCount[:] = 0
      for i in range(outseg.shape[0]):
        ait = InArcIt(deref(g), g.nodeFromId(i))
        while ait != INVALID:
          outExchangeCount[i] += deref(exchangeCount)[ait]
          inc(ait)
      ecMin = outExchangeCount.min()
      ecMax = outExchangeCount.max()
      print "   ecMin %d, exMax %d" % (ecMin, ecMax)
      if ecMin == ecMax:
        ecMax += 1
      outExchangeCount[:] = (outExchangeCount - ecMin) * 255 / (ecMax - ecMin) 
    
      print "   using uncertainty: %s" % (uncertainty,)
      self._uncertainty[:] = outExchangeCount[:]
    elif uncertainty == "localMargin":
      print "   calculating local margin..."
      prioMSTmargin(deref(g), deref(origSeeds), deref(am), deref(localMargin), prio)
      for i in range(seeds.shape[0]):
        outMargin[i] = deref(localMargin)[g.nodeFromId(i)]
      ecMin = outMargin.min()
      ecMax = outMargin.max()
      self._uncertainty[:] = (ecMax - ecMin - outMargin) * 255 / (ecMax - ecMin) 
    elif uncertainty == "none":
      pass
    else:
      print "ERROR: U N K N O W N  U N C E R T A I N T Y ! %s", uncertainty
      assert 1 == 2

    print "   restoring original graph..."
    #sns.restore()

    del segmentation
    del origSeeds
    del intree
    del exchangeCount
    del am
    del localMargin


cdef class PerturbMSTSegmentor(Segmentor):    
  
  cdef object _cumSubtreeSize
  cdef object _cumExchangeCount

  property cumExchangeCount:
    def __get__(self):
      if self._cumExchangeCount is None:
        self._cumExchangeCount = np.ndarray((self._numNodes,),np.int32)
      return IndexAccessor(self._regionVol,self._cumExchangeCount)
  
  property cumSubtreeSize:
    def __get__(self):
      if self._cumSubtreeSize is None:
        self._cumSubtreeSize = np.ndarray((self._numNodes,),np.int64)
      return IndexAccessor(self._regionVol,self._cumSubtreeSize)
 
  def run(self, np.ndarray[dtype=np.float32_t, ndim=2] unaries, prios = None, trials = 5, perturbation = 0.1, uncertainty="cumSubtreeSize", moving_average = False, **kwargs):


    """
    Run Graph Cut algorithm with the parameters
      unaries    : a 2D float array, column = region, row = color

    """
    print "segmenting..."
    # take snapshot of the current graph

    print "   snapshotting graph..."
    #cdef Snapshot sns = Snapshot(deref(self.graph))

    #shorthand
    cdef Graph *g = self.graph
    cdef ArcMapF *am = self.arcMap

    # add source and sink
    cdef int i,j
    cdef Arc a

    cdef NodeMapI *segmentation = new NodeMapI(deref(g))
    cdef NodeMapI *origSeeds = new NodeMapI(deref(g))
    cdef np.ndarray[dtype=np.uint8_t,ndim=1] seeds = self._seeds
    for i in range(seeds.shape[0]):
      segmentation.set(g.nodeFromId(i),seeds[i])
      origSeeds.set(g.nodeFromId(i),seeds[i])



    cdef vector[float] prio
    prio.resize(unaries.shape[1], 1.0)
    if prios is not None:
      assert len(prios) == unaries.shape[1]
      for i in range(prio.size()):
        prio[i] = prios[i]


    print "   running prioMST..."
    cdef ArcMapI *exchangeCount = new ArcMapI(deref(g))
    cdef NodeMapL *cumSubtreeSize = new NodeMapL(deref(g))


    prioMSTperturb(deref(g), deref(origSeeds), deref(segmentation), deref(am), deref(cumSubtreeSize), deref(exchangeCount), trials, perturbation, prio, moving_average)
    print "   finished..."

    # store the segmentatino in the result ndarray
    cdef np.ndarray[ndim=1,dtype=np.int32_t] outseg = self._segmentation
    for i in range(seeds.shape[0]):
      outseg[i] = deref(segmentation)[g.nodeFromId(i)]

    # calculate the exchange count uncertainty
    temp = self.cumExchangeCount # just ensure the array self._exchangeCount exists
    cdef np.ndarray[ndim=1, dtype=np.int32_t] outExchangeCount = self._cumExchangeCount
    outExchangeCount[:] = 0
    cdef InArcIt ait

    for i in range(outseg.shape[0]):
      ait = InArcIt(deref(g), g.nodeFromId(i))
      while ait != INVALID:
        outExchangeCount[i] += deref(exchangeCount)[ait]
        inc(ait)
    outExchangeCount[:] = (outExchangeCount - outExchangeCount.min()) * 255 / (outExchangeCount.max() - outExchangeCount.min()) 

    # calculate the subtree size uncertainty
    temp = self.cumSubtreeSize # just ensure the array self._exchangeCount exists
    cdef np.ndarray[ndim=1, dtype=np.int64_t] outSubtreeSize = self._cumSubtreeSize
    cdef NodeIt nit
    for i in range(outSubtreeSize.shape[0]):
      outSubtreeSize[i] = deref(cumSubtreeSize)[g.nodeFromId(i)]
      inc(nit)

    cdef long tmin = outSubtreeSize.min()
    cdef long tmax = outSubtreeSize.max()
    outSubtreeSize[:] = (outSubtreeSize - tmin) * 255 / (tmax-tmin) 

    if uncertainty == "cumSubtreeSize":
      print "   using uncertainty: %s" % (uncertainty,)
      self._uncertainty[:] = outSubtreeSize.astype(np.uint8)
    elif uncertainty == "cumExchangeCount":
      print "   using uncertainty: %s" % (uncertainty,)
      self._uncertainty[:] = outExchangeCount[:].astype(np.uint8)
    else:
      print "ERROR: U N K N O W N  U N C E R T A I N T Y ! %s", uncertainty
      assert 1 == 2


    print "   restoring original graph..."
    #sns.restore()

    del exchangeCount
    del cumSubtreeSize
    del segmentation
    del origSeeds

# vim: set expandtab tw=120 shiftwidth=2 softtabstop=2 tabstop=2:
