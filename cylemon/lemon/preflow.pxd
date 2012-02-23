cimport smart_graph
from smart_graph cimport Node, NodeIt, Arc, ArcIt, NodeMap, ArcMap, SmartDigraph

cdef extern from "<lemon/preflow.h>" namespace "lemon":
  cdef cppclass Preflow[GT,CM]:
    Preflow(GT,CM,Node,Node)
    Preflow(GT,CM,NodeIt,NodeIt)

    Preflow capacityMap(CM)
    Preflow flowMap(CM)
    CM flowMap()
    Preflow source(Node)
    Preflow source(NodeIt)
    Preflow target(Node)
    Preflow target(NodeIt)

    void init()
    bint flowInit(CM)
    void startFirstPhase()
    void startSecondPhase()
    void run()
    void runMinCut()

    inline bint minCut(Node)
    inline bint minCut(NodeIt)

    # due to cythons inability to access template paramer
    # members, we have to define the method with a fixed
    # typet we use double and hope the compiler optimizes
    # any type casts away
    inline double flow(Arc)
    inline double flow(ArcIt)

