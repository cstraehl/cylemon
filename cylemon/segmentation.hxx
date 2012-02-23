#include <lemon/bucket_heap.h>
#include <lemon/concepts/digraph.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace lemon;
using namespace std;
                                                          
template <typename GT, typename NMS, typename AMF>
void prioMST(const GT &graph, NMS &segmentation, const AMF & weights, const vector<float> &prio) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template ArcMap<int> ArcMapI;


  NodeMapB visited(graph,false);
  ArcMapI heap_state(graph);
  BucketHeap< ArcMapI, true > heap(heap_state);

  //add seeded nodes to heap
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(segmentation[nit] != 0) {
      visited[nit] = true;
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        heap.push(ait,(int) (weights[ait]*prio[segmentation[nit]])); // add arc to heap
      }
    }
  }

  int curPrio = 0;

  while(!heap.empty()) {
    Arc a = heap.top();
    curPrio = heap.prio();
    heap.pop();

    Node source = graph.source(a);
    Node target = graph.target(a);

    if(!visited[target]) {
      visited[target] = true;
      segmentation[target] = segmentation[source];
      for(OutArcIt ait(graph,target); ait != INVALID; ++ait) {
        int tprio = (int) (weights[ait]*prio[segmentation[target]]); 
        heap.push(ait,tprio); // add arc to heap
      }
    }
  }
}


template <typename GT, typename NMS, typename AMF, typename NMF>
void prioMSTperturb(const GT &graph, NMS &segmentation, AMF & weights, NMF &certainty, int trials, float perturbation, const vector<float> &prio) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template ArcMap<int> ArcMapI;
  typedef AMF ArcMap;

  for(int i=0;i<trials;++i) {
    ArcMap am(weights);
    // perturb the edge weights
    for(ArcIt ait(graph); ait != INVALID; ++ait) {
      am[ait] *= (1.0 + perturbation*rand() / RAND_MAX);
    }
    // run normal prioMST
    prioMST(graph, segmentation, am, prio);
  }
}


