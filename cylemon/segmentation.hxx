#include <lemon/bucket_heap.h>
#include <lemon/concepts/digraph.h>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace lemon;
using namespace std;
                                                          
template <typename GT, typename NMS, typename AMF, typename AMB>
void prioMST(const GT &graph, NMS &segmentation, const AMF & weights, AMB & intree, const vector<float> &prio) {
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

  // mark seeded nodes
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(segmentation[nit] != 0) {
      visited[nit] = true;
    }
  }
  //add seeded outgoing arcs of nodes to heap
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(segmentation[nit] != 0) {
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        Node target = graph.target(ait);
        if(!visited[target]) {
          heap.push(ait,(int) (weights[ait]*prio[segmentation[nit]])); // add arc to heap
        }
      }
    }
  }

  // calculate the MST and the corresponding segmentation
  int curPrio = 0;
  while(!heap.empty()) {
    Arc a = heap.top();
    curPrio = heap.prio();
    heap.pop();

    Node source = graph.source(a);
    Node target = graph.target(a);

    if(!visited[target]) {
      visited[target] = true;
      intree[a] = true;
      segmentation[target] = segmentation[source];
      for(OutArcIt ait(graph,target); ait != INVALID; ++ait) {
        int tprio = (int) (weights[ait]*prio[segmentation[target]]); 
        heap.push(ait,tprio);
      }
    }
  }
}


template <typename GT, typename NMS, typename NMS2, typename AMF, typename AMB, typename AMI>
void edgeExchangeCount(const GT &graph, const NMS &seeds,const NMS2 &segmentation, const AMF & weights, const AMB & intree, AMI &exchangeCount, const vector<float> &prio) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template ArcMap<int> ArcMapI;

  typename GT::template ArcMap<Arc> arc_strongestArc(graph, INVALID);
  typename GT::template NodeMap<Arc> node_strongestArc(graph, INVALID);
  deque<Arc> bfs;

  // reset edge excahnge count to 0
  for(ArcIt ait(graph); ait != INVALID; ++ait) {
    exchangeCount[ait] = 0;
  }
  
  // fill a ArcMap with the with the largest weight arc
  // encountered on the path from the seeded nodes
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        if(intree[ait]) {
          // the root arcs have no parent are
          // thus the strongest encountered so far
          arc_strongestArc[ait] = ait;
          bfs.push_back(ait);
        }
      }
    }
  }

  while(!bfs.empty()) {
    Arc a = bfs.front();
    bfs.pop_front();
    Node n = graph.target(a);
    node_strongestArc[n] = arc_strongestArc[a];
    for(OutArcIt ait(graph,n);ait != INVALID; ++ait) {
      // is it a spanning tree arc ?
      if(intree[ait]) {
        if(weights[ait] > weights[arc_strongestArc[ait]]) {
          arc_strongestArc[ait] = ait; // we have a new strongest arc
        }
        else {
          arc_strongestArc[ait] = arc_strongestArc[a]; // predecessors strongest arc is also for this arc
        }
        bfs.push_back(ait); // add arc to queue
      }
      // not a spanning tree arc
      else {
        Node source = graph.source(ait);
        Node target = graph.target(ait);
        // is the edge in the cut set ?
        if(segmentation[source] != segmentation[target]) {
          // increase the exchange count of the stronger of the two edges
          if( weights[node_strongestArc[source]] * prio[segmentation[source]] > weights[node_strongestArc[target]] * prio[segmentation[target]]) {
            exchangeCount[node_strongestArc[source]] += 1;
          }
          else {
            exchangeCount[node_strongestArc[target]] += 1;
          }
        }
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


