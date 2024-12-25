#ifndef _DYNAMICVORONOI3D_H_
#define _DYNAMICVORONOI3D_H_

#include <Eigen/Dense>
#include <limits.h>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>

#include "bucketedqueue.h"

#define STATE_DIM 9
#define CONTROL_DIM 10
#define ALL_DIM (STATE_DIM + CONTROL_DIM)

struct IntPoint3DHash {
  size_t operator()(const IntPoint3D &point) const {
    // x max_range: [0, 4095], 12 bit
    // y max_range: [0, 4095], 12 bit
    // z max_range: [0, 255], 8 bit
    const int x0 = point.x << 20;
    const int y0 = point.y << 8;
    const int z0 = point.z;
    const int hash_value = x0 | y0 | z0;
    return std::hash<int>()(hash_value);
  }
};

class VGraphNode3D {
public:
  IntPoint3D point_;
  std::unordered_map<int, float> edges_;
  VGraphNode3D() = default;
  VGraphNode3D(const IntPoint3D &point);
  void RemoveEdge(const int dest_id);
};

class VGraph3D {
public:
  std::vector<VGraphNode3D> nodes_;
  std::unordered_map<IntPoint3D, int, IntPoint3DHash> node_id_;
  void AddOneWayEdge(const IntPoint3D &src, const IntPoint3D &dest,
                     const float weight);
  void AddTwoWayEdge(const IntPoint3D &src, const IntPoint3D &dest,
                     const float weight);
  bool isNodeExist(const IntPoint3D &point);
  VGraph3D() = default;
};

class NodeProperty3D {
public:
  enum class AstarState { kNull = 0, kOpen, kClose };
  AstarState state_ = AstarState::kNull;
  float g_score_;
  float h_score_;
  int father_id_ = -1;
  NodeProperty3D() = default;
  NodeProperty3D(const AstarState state, const float g_score,
                 const float h_score, const int father_id);
};

class QueueNode3D {
public:
  IntPoint3D point_;
  float f_score_;
  QueueNode3D() = default;
  QueueNode3D(const IntPoint3D &point, const float f_score);
};

struct QueueNodeCmp3D {
  bool operator()(const QueueNode3D &lhs, const QueueNode3D &rhs) const {
    return lhs.f_score_ > rhs.f_score_;
  }
};

struct AstarOutput {
  std::vector<IntPoint3D> path;
  int num_expansions;
  float path_length;
  bool success;
};

struct iLQROutput {
  std::vector<IntPoint3D> path;
  int num_iter;
  float path_length;
};

struct iLQRTrajectory {
  std::vector<Eigen::Matrix<float, ALL_DIM, 1>> traj;
  int num_iter;
  float traj_length;
  float total_time;
};

struct RealCostOutput {
  float total_cost;
  float state_cost;
  float time_cost;
  float smooth_cost;
};

//! A DynamicVoronoi3D object computes and updates a distance map and Voronoi
//! diagram.
class DynamicVoronoi3D {
private:
  enum QueueState {
    kCellQueue = 0,
    kBfsQueue,
    kCandidate,
    kCellProcessed,
    kProcessed
  };

public:
  DynamicVoronoi3D();
  ~DynamicVoronoi3D();

  //! Initialization with an empty map
  void initializeEmpty(int _sizeX, int _sizeY, int _sizeZ,
                       bool initGridMap = true);
  //! Initialization with a given binary map (false==free, true==occupied)
  void initializeMap(int _sizeX, int _sizeY, int _sizeZ, bool ***_gridMap);

  //! add an obstacle at the specified cell coordinate
  void occupyCell(int x, int y, int z);
  //! remove an obstacle at the specified cell coordinate
  void clearCell(int x, int y, int z);
  //! remove old dynamic obstacles and add the new ones
  void exchangeObstacles(std::vector<IntPoint3D> &newObstacles);

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
  int ***alternativePrunedDiagram() { return alternativeDiagram; };
  //! retrieve the number of neighbors that are Voronoi nodes (4-connected)
  int getNumVoronoiNeighborsAlternative(int x, int y, int z) const;
  //! returns whether the specified cell is part of the alternatively pruned
  //! diagram. See updateAlternativePrunedDiagram.
  bool isVoronoiAlternative(int x, int y, int z);

  //! returns the obstacle distance at the specified location
  float getDistance(int x, int y, int z) const;
  //! returns the obstacle distance at the specified location
  int getSquaredDistance(int x, int y, int z) const;
  //! returns whether the specified cell is part of the (pruned) Voronoi graph
  bool isVoronoi(int x, int y, int z) const;
  //! checks whether the specficied location is occupied
  bool isOccupied(const int x, const int y, const int z) const;
  //! write the current distance map and voronoi diagram as ppm file
  void visualize(const char *filename = "result.ppm");

  //! returns the horizontal size of the workspace/map
  unsigned int getSizeX() { return sizeX; }
  //! returns the vertical size of the workspace/map
  unsigned int getSizeY() { return sizeY; }
  //! returns the depth size of the workspace/map
  unsigned int getSizeZ() { return sizeZ; }

  // Get Voronoi neighbors of a cell.
  std::vector<IntPoint3D> GetVoronoiNeighbors(const IntPoint3D &point) const;
  // Get the number of Voronoi neighbors of a cell.
  int GetNumVoronoiNeighbors(const IntPoint3D &point) const;
  //
  float GetUnionVolume(const IntPoint3D &p1, const IntPoint3D &p2) const;
  // Generate sparse graph.
  void ConstructSparseGraph();
  void ConstructSparseGraphBK();
  // Get A* path from start to goal.
  AstarOutput GetAstarPath(const IntPoint3D &start, const IntPoint3D &goal);
  // Get sparse graph.
  const VGraph3D &GetSparseGraph() const;
  bool isInSparseGraph(const IntPoint3D &point) const;
  // Get the distance between two cells.
  float GetDistanceBetween(const IntPoint3D &p1, const IntPoint3D &p2) const;
  // Get the squared distance between two cells.
  int GetSquaredDistanceBetween(const IntPoint3D &p1,
                                const IntPoint3D &p2) const;
  // Calculate the heuristic value for A* search.
  float GetHeuristic(const IntPoint3D &start, const IntPoint3D &goal) const;

  // iLQR Path related methods.
  iLQROutput GetiLQRPath(const std::vector<IntPoint3D> &path);
  std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>
  GetCost(const Eigen::Matrix<float, 6, 1> &xu, const IntPoint3D &bubble_1,
          const float radius_1, const IntPoint3D &bubble_2,
          const float radius_2, const float coeff);
  float GetRealCost(const Eigen::Matrix<float, 6, 1> &xu,
                    const IntPoint3D &bubble_1, const float radius_1,
                    const IntPoint3D &bubble_2, const float radius_2,
                    const float coeff);
  std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>
  GetTermCost(const Eigen::Matrix<float, 6, 1> &xu, const IntPoint3D &goal);
  float GetRealTermCost(const Eigen::Matrix<float, 6, 1> &xu,
                        const IntPoint3D &goal);

  // iLQR Trajectory related methods.
  iLQRTrajectory GetiLQRTrajectory(const std::vector<IntPoint3D> &path,
                                   const std::vector<IntPoint3D> &ilqr_path);
  Eigen::Matrix<float, 9, ALL_DIM>
  GetTransition(const Eigen::Matrix<float, ALL_DIM, 1> &xu);
  Eigen::Matrix<float, 9, 1>
  GetRealTransition(const Eigen::Matrix<float, ALL_DIM, 1> &xu);
  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
  GetTrajCost(const Eigen::Matrix<float, ALL_DIM, 1> &xu,
              const IntPoint3D &bubble_1, const float radius_1,
              const IntPoint3D &bubble_2, const float radius_2,
              const float max_vel, const float max_acc, const float coeff);
  RealCostOutput GetTrajRealCost(const Eigen::Matrix<float, ALL_DIM, 1> &xu,
                                 const IntPoint3D &bubble_1,
                                 const float radius_1,
                                 const IntPoint3D &bubble_2,
                                 const float radius_2, const float max_vel,
                                 const float max_acc, const float coeff);
  float GetSmoothCost(const Eigen::Vector3f &coeff, const float dt);
  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
  GetTrajTermCost(const Eigen::Matrix<float, ALL_DIM, 1> &xu,
                  const IntPoint3D &goal);
  float GetTrajRealTermCost(const Eigen::Matrix<float, ALL_DIM, 1> &xu,
                            const IntPoint3D &goal);

  // Graph related methods.
  void SparseAddTwoWayEdge(const IntPoint3D &core, const IntPoint3D &add,
                           const float weight);

private:
  struct dataCell {
    float dist;
    char voronoi;
    char queueing;
    int obstX;
    int obstY;
    int obstZ;
    bool needsRaise;
    int sqdist;
  };

  typedef enum {
    voronoiKeep = -4,
    freeQueued = -3,
    voronoiRetry = -2,
    voronoiPrune = -1,
    // On the edge of the GVD.
    free = 0,
    // Not on the edge of the GVD.
    occupied = 1
  } State;
  typedef enum {
    // Initial State.
    fwNotQueued = 1,
    // State after adding to queue.
    fwQueued = 2,
    // State after the lower cell poping from queue.
    fwProcessed = 3,
    // State after adding to queue from remove list.
    bwQueued = 4,
    // State after the raise cell poping from queue.
    bwProcessed = 1
  } QueueingState;
  typedef enum { invalidObstData = SHRT_MAX / 2 } ObstDataState;
  typedef enum { pruned, keep, retry } markerMatchResult;

  // methods
  void setObstacle(int x, int y, int z);
  void removeObstacle(int x, int y, int z);
  inline void checkVoro(int x, int y, int z, int nx, int ny, int nz,
                        dataCell &c, dataCell &nc);
  void commitAndColorize(bool updateRealDist = true);
  inline void reviveVoroNeighbors(int &x, int &y, int &z);

  inline bool isOccupied(const int &x, const int &y, const int &z,
                         const dataCell &c);
  inline markerMatchResult markerMatch(int x, int y, int z);
  inline bool markerMatchAlternative(int x, int y, int z);
  inline int getVoronoiPruneValence(int x, int y, int z);

  // queues

  BucketPrioQueue<IntPoint3D> open;
  std::queue<IntPoint3D> pruneQueue;
  BucketPrioQueue<IntPoint3D> sortedPruneQueue;

  std::vector<IntPoint3D> removeList;
  std::vector<IntPoint3D> addList;
  std::vector<IntPoint3D> lastObstacles;

  // maps
  int sizeY;
  int sizeX;
  int sizeZ;
  dataCell ***data;
  bool ***gridMap;
  bool allocatedGridMap;

  // parameters
  int padding;
  double doubleThreshold;

  double sqrt2;

  //  dataCell** getData(){ return data; }
  int ***alternativeDiagram;

  // Sparse graph.
  VGraph3D graph_;
};

#endif
