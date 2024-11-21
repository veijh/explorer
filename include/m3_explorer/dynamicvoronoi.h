#ifndef _DYNAMICVORONOI_H_
#define _DYNAMICVORONOI_H_

#include <Eigen/Dense>
#include <limits.h>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

#include "bucketedqueue.h"

struct IntPointHash {
  size_t operator()(const IntPoint &point) const {
    // x max_range: [0, 1023], 10 bit
    // y max_range: [0, 1023], 10 bit
    const int x0 = point.x << 16;
    const int y0 = point.y << 0;
    const int hash_value = x0 | y0;
    return std::hash<int>()(hash_value);
  }
};

class VGraphNode {
public:
  IntPoint point_;
  std::unordered_map<int, float> edges_;
  VGraphNode() = default;
  VGraphNode(const IntPoint &point);
  void RemoveEdge(const int dest_id);
};

class VGraph {
public:
  std::vector<VGraphNode> nodes_;
  std::unordered_map<IntPoint, int, IntPointHash> node_id_;
  void AddOneWayEdge(const IntPoint &src, const IntPoint &dest,
                     const float weight);
  void AddTwoWayEdge(const IntPoint &src, const IntPoint &dest,
                     const float weight);
  bool isNodeExist(const IntPoint &point);
  VGraph() = default;
};

class NodeProperty {
public:
  enum class AstarState { kNull = 0, kOpen, kClose };
  AstarState state_ = AstarState::kNull;
  float g_score_;
  float h_score_;
  int father_id_ = -1;
  NodeProperty() = default;
  NodeProperty(const AstarState state, const float g_score, const float h_score,
               const int father_id);
};

class QueueNode {
public:
  IntPoint point_;
  float f_score_;
  QueueNode() = default;
  QueueNode(const IntPoint &point, const float f_score);
};

struct QueueNodeCmp {
  bool operator()(const QueueNode &lhs, const QueueNode &rhs) const {
    return lhs.f_score_ > rhs.f_score_;
  }
};

//! A DynamicVoronoi object computes and updates a distance map and Voronoi
//! diagram.
class DynamicVoronoi {

public:
  DynamicVoronoi();
  ~DynamicVoronoi();

  //! Initialization with an empty map
  void initializeEmpty(int _sizeX, int _sizeY, bool initGridMap = true);
  //! Initialization with a given binary map (false==free, true==occupied)
  void initializeMap(int _sizeX, int _sizeY, bool **_gridMap);

  //! add an obstacle at the specified cell coordinate
  void occupyCell(int x, int y);
  //! remove an obstacle at the specified cell coordinate
  void clearCell(int x, int y);
  //! remove old dynamic obstacles and add the new ones
  void exchangeObstacles(std::vector<INTPOINT> &newObstacles);

  //! update distance map and Voronoi diagram to reflect the changes
  void update(bool updateRealDist = true);
  //! prune the Voronoi diagram
  void prune();
  //! prune the Voronoi diagram by globally revisiting all Voronoi nodes. Takes
  //! more time but gives a more sparsely pruned Voronoi graph. You need to call
  //! this after every call to update()
  void updateAlternativePrunedDiagram();
  //! retrieve the alternatively pruned diagram. see
  //! updateAlternativePrunedDiagram()
  int **alternativePrunedDiagram() { return alternativeDiagram; };
  //! retrieve the number of neighbors that are Voronoi nodes (4-connected)
  int getNumVoronoiNeighborsAlternative(int x, int y) const;
  //! returns whether the specified cell is part of the alternatively pruned
  //! diagram. See updateAlternativePrunedDiagram.
  bool isVoronoiAlternative(int x, int y);

  //! returns the obstacle distance at the specified location
  float getDistance(int x, int y) const;
  //! returns the obstacle distance at the specified location
  int getSquaredDistance(int x, int y) const;
  //! returns whether the specified cell is part of the (pruned) Voronoi graph
  bool isVoronoi(int x, int y) const;
  //! checks whether the specficied location is occupied
  bool isOccupied(int x, int y) const;
  //! write the current distance map and voronoi diagram as ppm file
  void visualize(const char *filename = "result.ppm");

  //! returns the horizontal size of the workspace/map
  unsigned int getSizeX() { return sizeX; }
  //! returns the vertical size of the workspace/map
  unsigned int getSizeY() { return sizeY; }

  // Get Voronoi neighbors of a cell.
  std::vector<IntPoint> GetVoronoiNeighbors(const int x, const int y);
  // Get the number of Voronoi neighbors of a cell.
  int GetNumVoronoiNeighbors(const int x, const int y) const;
  //
  float GetUnionVolume(const IntPoint &p1, const IntPoint &p2) const;
  // Generate sparse graph.
  void ConstructSparseGraph();
  void ConstructSparseGraphBK();
  //
  std::vector<IntPoint> GetAstarPath(const IntPoint &start,
                                     const IntPoint &goal);
  // Get sparse graph.
  const VGraph &GetSparseGraph() const;
  bool isInSparseGraph(const IntPoint &point) const;
  // Get the distance between two cells.
  float GetDistanceBetween(const IntPoint &p1, const IntPoint &p2) const;
  // Get the squared distance between two cells.
  int GetSquaredDistanceBetween(const IntPoint &p1, const IntPoint &p2) const;
  // Calculate the heuristic value for A* search.
  float GetHeuristic(const IntPoint &start, const IntPoint &goal) const;

  // iLQR related methods.
  std::vector<IntPoint> GetiLQRPath(const std::vector<IntPoint> &path);
  std::pair<Eigen::Matrix4f, Eigen::Vector4f>
  GetCost(const Eigen::Vector4f &xu, const IntPoint &bubble_1,
          const float radius_1, const IntPoint &bubble_2, const float radius_2);
  float GetRealCost(const Eigen::Vector4f &xu, const IntPoint &bubble_1,
                    const float radius_1, const IntPoint &bubble_2,
                    const float radius_2);
  std::pair<Eigen::Matrix4f, Eigen::Vector4f>
  GetTermCost(const Eigen::Vector4f &xu, const IntPoint &goal);
  float GetRealTermCost(const Eigen::Vector4f &xu, const IntPoint &goal);

private:
  struct dataCell {
    float dist;
    char voronoi;
    char queueing;
    int obstX;
    int obstY;
    bool needsRaise;
    int sqdist;
  };

  typedef enum {
    voronoiKeep = -4,
    freeQueued = -3,
    voronoiRetry = -2,
    voronoiPrune = -1,
    free = 0,
    occupied = 1
  } State;
  typedef enum {
    fwNotQueued = 1,
    fwQueued = 2,
    fwProcessed = 3,
    bwQueued = 4,
    bwProcessed = 1
  } QueueingState;
  typedef enum { invalidObstData = SHRT_MAX / 2 } ObstDataState;
  typedef enum { pruned, keep, retry } markerMatchResult;

  // methods
  void setObstacle(int x, int y);
  void removeObstacle(int x, int y);
  inline void checkVoro(int x, int y, int nx, int ny, dataCell &c,
                        dataCell &nc);
  void recheckVoro();
  void commitAndColorize(bool updateRealDist = true);
  inline void reviveVoroNeighbors(int &x, int &y);

  inline bool isOccupied(int &x, int &y, dataCell &c);
  inline markerMatchResult markerMatch(int x, int y);
  inline bool markerMatchAlternative(int x, int y);
  inline int getVoronoiPruneValence(int x, int y);

  // queues

  BucketPrioQueue<INTPOINT> open;
  std::queue<INTPOINT> pruneQueue;
  BucketPrioQueue<INTPOINT> sortedPruneQueue;

  std::vector<INTPOINT> removeList;
  std::vector<INTPOINT> addList;
  std::vector<INTPOINT> lastObstacles;

  // maps
  int sizeY;
  int sizeX;
  dataCell **data;
  bool **gridMap;
  bool allocatedGridMap;

  // parameters
  int padding;
  double doubleThreshold;

  double sqrt2;

  //  dataCell** getData(){ return data; }
  int **alternativeDiagram;

  // Sparse graph.
  VGraph graph_;
};

#endif
