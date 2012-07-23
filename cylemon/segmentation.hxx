#include <lemon/bucket_heap.h>
#include <lemon/bin_heap.h>
#include <vector>
#include <deque>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include <omp.h>

using namespace lemon;
using namespace std;
                                                          
template <typename GT, typename NMS, typename AMF, typename NMF>
void prioMSTmargin(const GT &graph, const NMS & seeds, const AMF & weights, NMF & margin, const vector<float> &prio) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::InArcIt InArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template NodeMap<int> NodeMapI;
  typedef typename GT::template NodeMap<float> NodeMapF;
  typedef typename GT::template ArcMap<int> ArcMapI;
  typedef typename GT::template ArcMap<float> ArcMapF;

  std::vector< NodeMapF *> height;
  height.resize(prio.size());
  for(int round = 0; round < prio.size(); ++round) {
    height[round] = new NodeMapF(graph,0);
  }

  
  for(int round = 1; round < prio.size(); ++round) {
    NodeMapB visited(graph,false);
    ArcMapI heap_state(graph);
    BucketHeap< ArcMapI, true > heap(heap_state);


    // mark seeded nodes
    for(NodeIt nit(graph); nit != INVALID; ++nit) {
      visited[nit] = false;
      if(seeds[nit] != 0) {
        visited[nit] = true;
      }
    }
    //add seeded outgoing arcs of nodes to heap
    for(NodeIt nit(graph); nit != INVALID; ++nit) {
      if(seeds[nit] == round) {
        for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
          Node target = graph.runningNode(ait);
          if(!visited[target]) {
            heap.push(ait,(int) (weights[ait]*prio[seeds[nit]])); // add arc to heap
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
        (*height[round])[target] = curPrio;
        visited[target] = true;
        int label =  round;
        for(OutArcIt ait(graph,target); ait != INVALID; ++ait) {
          if(!visited[graph.runningNode(ait)]) {
            int tprio;
            tprio = (int) (weights[ait]*prio[label]);
            tprio = std::max(tprio,curPrio);
            heap.push(ait,tprio);
          }
        }
      }
    }
  }
  
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    (*height[0])[nit] = 9e9;
  }
  //determine lowest height and second lowest height
  // lowest height is stored in height
  // second lowest one in margin
  for(int round = 1; round < prio.size(); ++round) {
    for(NodeIt nit(graph); nit != INVALID; ++nit) {
      if((*height[0])[nit] >= (*height[round])[nit]) {
        margin[nit] = (*height[0])[nit];
        (*height[0])[nit] = (*height[round])[nit];
      }
      else if(margin[nit] > (*height[round])[nit]) {
        margin[nit] = (*height[round])[nit];
      }
    }
  }
  float maxMargin = 0;
  // calculate second lowest height minus lowest height, i.e. margin
  // and store in margin
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    margin[nit] -= (*height[0])[nit];
    if(margin[nit] > maxMargin) {
      maxMargin = margin[nit];
    }
  }
  //set margin of seeded nodes to maximum
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      margin[nit] = maxMargin;
    }
  }
  
  for(int round = 0; round < prio.size(); ++round) {
    delete height[round];
  }
}


template <typename GT, typename NMS, typename NMS2, typename AMF, typename AMB>
void prioMST(const GT &graph, const NMS & seeds, NMS2 & segmentation, const AMF & weights, AMB & intree, const vector<float> &prio) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::InArcIt InArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template NodeMap<int> NodeMapI;
  typedef typename GT::template ArcMap<int> ArcMapI;
  typedef typename GT::template ArcMap<float> ArcMapF;
  
  NodeMapB visited(graph,false);
  ArcMapI heap_state(graph);
  BucketHeap< ArcMapI, true > heap(heap_state);

  for(ArcIt ait(graph); ait != INVALID; ++ait) {
    intree[ait] = false;
  }

  // mark seeded nodes
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    visited[nit] = false;
    if(seeds[nit] != 0) {
      segmentation[nit] = seeds[nit];
      visited[nit] = true;
    }
  }
  //add seeded outgoing arcs of nodes to heap
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        Node target = graph.runningNode(ait);
        if(!visited[target]) {
          heap.push(ait,(int) (weights[ait]*prio[seeds[nit]])); // add arc to heap
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
      int label =  segmentation[source]; 
      segmentation[target] = label;
      for(OutArcIt ait(graph,target); ait != INVALID; ++ait) {
        if(!visited[graph.runningNode(ait)]) {
          int tprio;
          tprio = (int) (weights[ait]*prio[label]);
          heap.push(ait,tprio);
        }
      }
    }
  }
}


template <typename GT, typename NMS, typename NMS2, typename AMF, typename AMB>
void prioMSTmav(const GT &graph, const NMS & seeds, NMS2 & segmentation, AMF & weights, AMB & intree,  const vector<float> &prio) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::InArcIt InArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template ArcMap<int> ArcMapI;
  typedef typename GT::template ArcMap<float> ArcMapF;
  
  ArcMapF ew_average(graph);
  float factor = 0.7;

  NodeMapB visited(graph,false);
  ArcMapI heap_state(graph);
  BucketHeap< ArcMapI, true > heap(heap_state);

  for(ArcIt ait(graph); ait != INVALID; ++ait) {
    ew_average[ait] = weights[ait];
  }

  // mark seeded nodes
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      segmentation[nit] = seeds[nit];
      visited[nit] = true;
    }
  }
  //add seeded outgoing arcs of nodes to heap
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        Node target = graph.runningNode(ait);
        if(!visited[target]) {
          heap.push(ait,(int) (weights[ait]*prio[seeds[nit]])); // add arc to heap
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

    float moving_average = ew_average[a];
    weights[a] = (weights[a]*2 - moving_average );

    if(!visited[target]) {
      visited[target] = true;
      intree[a] = true;
      int label =  segmentation[source]; 
      segmentation[target] = label;
      for(OutArcIt ait(graph,target); ait != INVALID; ++ait) {
        if(!visited[graph.target(ait)]) {
          ew_average[ait] = moving_average*(1-factor) + weights[ait]*factor;
          int tprio;
          if(prio[label] == 1) { // for foreground nodes
            tprio = (int) (weights[ait]*2 - moving_average );
          }
          else {
            tprio = (int) (weights[ait]*prio[label]);
          }
          tprio = std::max(tprio,0);
          heap.push(ait,tprio);
        }   
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
    OutArcIt a(graph,bfs.front());
    bfs.pop_front();
    Node n = graph.runningNode(a);
    node_strongestArc[n] = arc_strongestArc[a];
    for(OutArcIt ait(graph,n);ait != INVALID; ++ait) {
      // is it a spanning tree arc ?
      if(intree[ait]) {
        if(weights[ait] > weights[arc_strongestArc[a]]) {
          arc_strongestArc[ait] = ait; // we have a new strongest arc
        }
        else {
          arc_strongestArc[ait] = arc_strongestArc[a]; // predecessors strongest arc is also for this arc
        }
        bfs.push_back(ait); // add arc to queue
      }
      // not a spanning tree arc
      else {
        Node source = graph.baseNode(ait);
        Node target = graph.runningNode(ait);
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

template <typename GT, typename NMS, typename NMS2, typename AMB, typename NML>
void subtreeSize(const GT &graph, const NMS &seeds, const NMS2 &segmentation, const AMB & intree, NML &subtreeSize) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template ArcMap<int> ArcMapI;

  deque<Node> bfs;
  deque<Node> bfsn;

  // initialize to 0
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    subtreeSize[nit] = 1;
  }
  
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        if(intree[ait]) {
          bfs.push_back(graph.target(ait));
        }
      }
    }
  }

  //construct breadth first search queue
  while(!bfs.empty()) {
    Node n = bfs.front();
    bfs.pop_front();
    bfsn.push_back(n);
    for(OutArcIt oit(graph,n); oit != INVALID; ++oit) {
      if(intree[oit]) {
        bfs.push_back(graph.target(oit));
      }
    }
  }
  int counter = 0;
  // traverse in reverse order and add up subtree sizes
  while(!bfsn.empty()) {
    counter += 1;
    Node n = bfsn.back();
    bfsn.pop_back();
    for(OutArcIt oit(graph,n); oit != INVALID; ++oit) {
      if(intree[oit]) {
        Node target = graph.target(oit);
        subtreeSize[n] += subtreeSize[target];
      }
    }
  }
}



template <typename GT, typename NMS, typename NMS2, typename AMF, typename NML, typename AMI>
void prioMSTperturb(const GT &graph, const NMS & seeds, NMS2 &segmentation, const AMF & weights, NML &uncertainty, AMI &exchangeCount, int trials, float perturbation, const vector<float> &prios, bool moving_average) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template NodeMap<char> NodeMapC;
  typedef typename GT::template NodeMap<long> NodeMapL;
  typedef typename GT::template ArcMap<int> ArcMapI;
  typedef typename GT::template ArcMap<long> ArcMapL;
  typedef typename GT::template ArcMap<bool> ArcMapB;
  typedef typename GT::template ArcMap<float> ArcMapF;
  typedef typename GT::template NodeMap<int> NodeMapI;
  typedef typename GT::template NodeMap<Node> NodeMapN;
  typedef typename GT::template NodeMap< vector< char > > NodeMapS;
  typedef typename GT::template NodeMap< vector< int > > NodeMapVI;

  vector< typename GT::template NodeMap< int >* > segmentations;
  vector< typename GT::template NodeMap< long >* > subtreeSizes;

  // resize the segmentations
  segmentations.resize(prios.size());
  subtreeSizes.resize(prios.size());
  for(int i = 0; i < prios.size(); ++i) {
    segmentations[i] = new NodeMapI(graph,0);
    subtreeSizes[i] = new NodeMapL(graph,0);
  }

  for(ArcIt ait(graph); ait != INVALID; ++ait) {
    exchangeCount[ait] = 0;
  }
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    segmentation[nit] = 0;
  }
 
  bool haveSegmentation = false;
  bool haveUncertainty = false;
  int maxTries = 4;
  int numTries = 0;
  float perturbationScalingFactor = 1.6;

  while((!haveUncertainty)&&(numTries < maxTries)) {
    numTries += 1;
      #pragma omp parallel for
      for(int i=0;i<trials;++i) {
        ArcMapF tweights(graph);
        mapCopy(graph, weights, tweights);
        NodeMapI tseg(graph, 0);
        ArcMapB intree(graph, false);
        ArcMapI tedgeExchangeCount(graph, 0);

        // perturb the edge weights
        for(ArcIt ait(graph); ait != INVALID; ++ait) {
          int range = 1 + weights[ait] * perturbation;
          //printf("perturb range: %d\n", range);
          tweights[ait] = weights[ait] + (rand() % range);
          //printf("tweight %d: %f\n", graph.id(ait), tweights[ait]);
        }

        if(moving_average) {
        // run normal prioMST
          prioMSTmav(graph, seeds, tseg, tweights, intree, prios);
        }
        else {
          prioMST(graph, seeds, tseg, tweights, intree, prios);
        }
        
        // calculate subtree size
        NodeMapL tsubtreeSize(graph);
        subtreeSize(graph, seeds, tseg, intree, tsubtreeSize);

        // calculate edge exchange count
        edgeExchangeCount(graph, seeds, tseg, tweights, intree, tedgeExchangeCount, prios);
        
        #pragma omp critical
        {
          // add the edge exchange count to the global result
          for(ArcIt ait(graph); ait != INVALID; ++ait) {
            exchangeCount[ait] += tedgeExchangeCount[ait];
          }

          
          // add the segmentation of the trial to the global segresult    
          for(NodeIt nit(graph); nit != INVALID; ++nit) {
            //printf("tseg %d: %d\n", graph.id(nit), tsubtreeSize[nit]);
            (*segmentations[tseg[nit]])[nit] += 1;
            (*subtreeSizes[tseg[nit]])[nit] += tsubtreeSize[nit];
          }
        }
      }
    haveSegmentation = true;
    // determine final segmentation
    for(int i = 1; i < prios.size(); ++i) {
      for(NodeIt nit(graph); nit != INVALID; ++nit) {
        int cur = segmentation[nit];
        if((*segmentations[i])[nit] >= (*segmentations[cur])[nit]) {
          segmentation[nit] = i;
        }
      }
    }
    
    long maxUncertainty = 0;

    // calculate the sum of the subtree sizes
    // in the runs that differed from the final segmentation
    for(int i = 1; i < prios.size(); ++i) {
      for(NodeIt nit(graph); nit != INVALID; ++nit) {
        if(segmentation[nit] != i) {
          uncertainty[nit] += (*subtreeSizes[i])[nit];
          if(uncertainty[nit] != 0) {
            haveUncertainty = true;
            maxUncertainty = std::max(maxUncertainty, uncertainty[nit]);
          }
          //printf("subtreeSize %d: %d\n", graph.id(nit), uncertainty[nit]);
        }
      }
    }
    perturbation = perturbation * perturbationScalingFactor;
    if(!haveUncertainty) {
      printf("SCALING UNCERTAINTY TO %f (ROUND %d)\n",perturbation,numTries);
    }
    else if(numTries > 1) {
      printf("FOUND UNCERTAINTY OF %d (ROUND %d)\n",maxUncertainty, numTries);
    }
  }

  for(int i = 0; i < prios.size(); ++i) {
    delete segmentations[i];
    delete subtreeSizes[i];
  }

}



template <typename GT, typename AMF, typename NMS, typename NT>
void maxDistanceNode(const GT &graph, const AMF & distances, const NMS & seeds, NT & maxDistanceNode, float & distance) {
  typedef typename GT::Node Node;
  typedef typename GT::Arc Arc;
  typedef typename GT::NodeIt NodeIt;
  typedef typename GT::ArcIt ArcIt;
  typedef typename GT::OutArcIt OutArcIt;
  typedef typename GT::template NodeMap<bool> NodeMapB;
  typedef typename GT::template ArcMap<int> ArcMapI;


  NodeMapB visited(graph,false);
  ArcMapI heap_state(graph);
  BinHeap< typename AMF::Value, ArcMapI > heap(heap_state);

  // mark seeded nodes
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      visited[nit] = true;
    }
  }

  //add seeded outgoing arcs of nodes to heap
  for(NodeIt nit(graph); nit != INVALID; ++nit) {
    if(seeds[nit] != 0) {
      for(OutArcIt ait(graph,nit); ait != INVALID; ++ait) {
        Node target = graph.target(ait);
        if(!visited[target]) {
          heap.push(ait,(distances[ait])); // add arc to heap
        }
      }
    }
  }

  // calculate the MST and the corresponding segmentation
  float curPrio = 0;
  while(!heap.empty()) {
    Arc a = heap.top();
    curPrio = heap.prio();
    heap.pop();

    Node source = graph.source(a);
    Node target = graph.target(a);

    if(!visited[target]) {
      maxDistanceNode = graph.target(a);
      distance = curPrio;
      visited[target] = true;
      for(OutArcIt ait(graph,target); ait != INVALID; ++ait) {
        if(!visited[graph.target(ait)]) {
          float tprio = (distances[ait]+curPrio); 
          heap.push(ait,tprio);
        }
      }
    }
  }
}
