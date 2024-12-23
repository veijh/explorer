#include "explorer/dynamicvoronoi3D.h"
#include "explorer/time_track.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <queue>
#include <unordered_map>

namespace {
// Voronoi graph parameters.
constexpr int kDeadEndThreshold = 10;
constexpr int kObstacleWaveInibitDistance = 5;
// iLQR parameters.
constexpr int kMaxIteration = 100;
constexpr int kMaxLineSearchIter = 10;
// iLQR Path parameters.
constexpr float kConvergenceThreshold = 0.05f;
constexpr float kTermWeight = 1000.0f;
constexpr float kBndWeight = 10000.0f;
constexpr float kWeight = 100.0f;
// iLQR Trajectory parameters.
constexpr float kMaxVelocity = 15.0f;
constexpr float kMaxAcceleration = 10.0f;
constexpr float kMinTimeStep = 0.1f;
constexpr float kInitialTimeScale = 5.0f;
constexpr float kTrajConvergenceThreshold = 1.0f;
constexpr float kTrajTermWeight = 10.0f;
constexpr float kTrajBndWeight = 1.0f;
constexpr float kTimeBndWeight = 1.0f;
constexpr float kSmoothWeight = 0.1f;
constexpr float kTimeWeight = 1.0f;
// Rgularization parameters.
constexpr float kRegularization = 0.1f;
constexpr float kMaxRegularizationIter = 35;
constexpr float kRegularizationScale = 1.6f;
} // namespace

const std::vector<IntPoint3D> nbr_offsets = {
    // clang-format off
    // Bottom layer.
    {-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
    {-1,  0, -1}, {0,  0, -1}, {1,  0, -1},
    {-1,  1, -1}, {0,  1, -1}, {1,  1, -1},
    // Middle layer.
    {-1, -1, 0}, {0, -1, 0}, {1, -1, 0},
    {-1,  0, 0},             {1,  0, 0},
    {-1,  1, 0}, {0,  1, 0}, {1,  1, 0},
    // Top layer.
    {-1, -1, 1}, {0, -1, 1}, {1, -1, 1},
    {-1,  0, 1}, {0,  0, 1}, {1,  0, 1},
    {-1,  1, 1}, {0,  1, 1}, {1,  1, 1},
    // clang-format on
};

const std::vector<IntPoint3D> voronoi_nbr_offsets = {
    // clang-format off
    // Bottom layer.
    {-1, -1, -1}, {0, -1, -1}, {1, -1, -1},
    {-1,  0, -1}, {0,  0, -1}, {1,  0, -1},
    {-1,  1, -1}, {0,  1, -1}, {1,  1, -1},
    // Middle layer.
    {-1, -1, 0}, {0, -1, 0}, {1, -1, 0},
    {-1,  0, 0},             {1,  0, 0},
    {-1,  1, 0}, {0,  1, 0}, {1,  1, 0},
    // Top layer.
    {-1, -1, 1}, {0, -1, 1}, {1, -1, 1},
    {-1,  0, 1}, {0,  0, 1}, {1,  0, 1},
    {-1,  1, 1}, {0,  1, 1}, {1,  1, 1},
    // clang-format on
};

VGraphNode3D::VGraphNode3D(const IntPoint3D &point) : point_(point) {}

void VGraphNode3D::RemoveEdge(const int dest_id) {
  edges_.erase(dest_id);
  return;
}

void VGraph3D::AddOneWayEdge(const IntPoint3D &src, const IntPoint3D &dest,
                             const float weight) {
  // Skip self-loops.
  if (src == dest) {
    return;
  }
  // Check whether the nodes has been already in the graph.
  if (node_id_.find(src) == node_id_.end()) {
    nodes_.emplace_back(src);
    node_id_[src] = nodes_.size() - 1;
  }
  if (node_id_.find(dest) == node_id_.end()) {
    nodes_.emplace_back(dest);
    node_id_[dest] = nodes_.size() - 1;
  }
  // Add the edge to the graph.
  const int src_id = node_id_[src];
  const int dest_id = node_id_[dest];
  if (nodes_[src_id].edges_.find(dest_id) == nodes_[src_id].edges_.end()) {
    nodes_[node_id_[src]].edges_[node_id_[dest]] = weight;
  }
}

void VGraph3D::AddTwoWayEdge(const IntPoint3D &src, const IntPoint3D &dest,
                             const float weight) {
  // Skip self-loops.
  if (src == dest) {
    return;
  }
  // Check whether the nodes has been already in the graph.
  if (node_id_.find(src) == node_id_.end()) {
    nodes_.emplace_back(src);
    node_id_[src] = nodes_.size() - 1;
  }
  if (node_id_.find(dest) == node_id_.end()) {
    nodes_.emplace_back(dest);
    node_id_[dest] = nodes_.size() - 1;
  }
  // Add the edge to the graph.
  const int src_id = node_id_[src];
  const int dest_id = node_id_[dest];
  if (nodes_[src_id].edges_.find(dest_id) == nodes_[src_id].edges_.end()) {
    nodes_[node_id_[src]].edges_[node_id_[dest]] = weight;
  }
  if (nodes_[dest_id].edges_.find(src_id) == nodes_[dest_id].edges_.end()) {
    nodes_[node_id_[dest]].edges_[node_id_[src]] = weight;
  }
}

bool VGraph3D::isNodeExist(const IntPoint3D &point) {
  return node_id_.find(point) != node_id_.end();
}

NodeProperty3D::NodeProperty3D(const AstarState state, const float g_score,
                               const float h_score, const int father_id)
    : state_(state), g_score_(g_score), h_score_(h_score),
      father_id_(father_id) {}

QueueNode3D::QueueNode3D(const IntPoint3D &point, const float f_score)
    : point_(point), f_score_(f_score) {}

DynamicVoronoi3D::DynamicVoronoi3D() {
  sqrt2 = sqrt(2.0);
  data = nullptr;
  gridMap = nullptr;
  alternativeDiagram = nullptr;
  allocatedGridMap = false;
}

DynamicVoronoi3D::~DynamicVoronoi3D() {
  if (data) {
    for (int x = 0; x < sizeX; ++x) {
      for (int y = 0; y < sizeY; ++y) {
        delete[] data[x][y];
      }
      delete[] data[x];
    }
    delete[] data;
  }
  if (allocatedGridMap && gridMap) {
    for (int x = 0; x < sizeX; ++x) {
      for (int y = 0; y < sizeY; ++y) {
        delete[] gridMap[x][y];
      }
      delete[] gridMap[x];
    }
    delete[] gridMap;
  }
}

void DynamicVoronoi3D::initializeEmpty(int _sizeX, int _sizeY, int _sizeZ,
                                       bool initGridMap) {
  if (data) {
    for (int x = 0; x < sizeX; ++x) {
      for (int y = 0; y < sizeY; ++y) {
        delete[] data[x][y];
      }
      delete[] data[x];
    }
    delete[] data;
    data = nullptr;
  }
  if (alternativeDiagram) {
    for (int x = 0; x < sizeX; ++x) {
      for (int y = 0; y < sizeY; ++y) {
        delete[] alternativeDiagram[x][y];
      }
      delete[] alternativeDiagram[x];
    }
    delete[] alternativeDiagram;
    alternativeDiagram = nullptr;
  }
  if (initGridMap) {
    if (allocatedGridMap && gridMap) {
      for (int x = 0; x < sizeX; x++) {
        for (int y = 0; y < sizeY; y++) {
          delete[] gridMap[x][y];
        }
        delete[] gridMap[x];
      }
      delete[] gridMap;
      gridMap = nullptr;
      allocatedGridMap = false;
    }
  }

  sizeX = _sizeX;
  sizeY = _sizeY;
  sizeZ = _sizeZ;
  data = new dataCell **[sizeX];
  for (int x = 0; x < sizeX; ++x) {
    data[x] = new dataCell *[sizeY];
    for (int y = 0; y < sizeY; ++y) {
      data[x][y] = new dataCell[sizeZ];
    }
  }

  if (initGridMap) {
    gridMap = new bool **[sizeX];
    for (int x = 0; x < sizeX; ++x) {
      gridMap[x] = new bool *[sizeY];
      for (int y = 0; y < sizeY; ++y) {
        gridMap[x][y] = new bool[sizeZ];
      }
    }
    allocatedGridMap = true;
  }

  dataCell c;
  c.dist = INFINITY;
  c.sqdist = INT_MAX;
  c.obstX = invalidObstData;
  c.obstY = invalidObstData;
  c.obstZ = invalidObstData;
  c.voronoi = free;
  c.queueing = fwNotQueued;
  c.needsRaise = false;

  for (int x = 0; x < sizeX; ++x) {
    for (int y = 0; y < sizeY; ++y) {
      for (int z = 0; z < sizeZ; ++z) {
        data[x][y][z] = c;
      }
    }
  }

  if (initGridMap) {
    for (int x = 0; x < sizeX; ++x) {
      for (int y = 0; y < sizeY; ++y) {
        for (int z = 0; z < sizeZ; ++z) {
          gridMap[x][y][z] = false;
        }
      }
    }
  }
}

void DynamicVoronoi3D::initializeMap(int _sizeX, int _sizeY, int _sizeZ,
                                     bool ***_gridMap) {
  gridMap = _gridMap;
  initializeEmpty(_sizeX, _sizeY, _sizeZ, false);
  const int num_nbr = nbr_offsets.size();

  for (int x = 0; x < sizeX; ++x) {
    for (int y = 0; y < sizeY; ++y) {
      for (int z = 0; z < sizeZ; ++z) {
        if (gridMap[x][y][z]) {
          dataCell c = data[x][y][z];
          // In fact, isOccupied(x, y, z, c) is always true, when initializeMap
          // is called for the first time.
          if (!isOccupied(x, y, z, c)) {
            bool isSurrounded = true;
            // Check if the cell is surrounded by occupied cells.
            for (int nbr_iter = 0; nbr_iter < num_nbr; ++nbr_iter) {
              const int nx = x + nbr_offsets[nbr_iter].x;
              const int ny = y + nbr_offsets[nbr_iter].y;
              const int nz = z + nbr_offsets[nbr_iter].z;
              // One of the neighbors is not occupied.
              if ((nx >= 0) && (nx < sizeX) && (ny >= 0) && (ny < sizeY) &&
                  (nz >= 0) && (nz < sizeZ) && gridMap[nx][ny][nz] == false) {
                isSurrounded = false;
                break;
              }
            }
            if (isSurrounded) {
              c.obstX = x;
              c.obstY = y;
              c.obstZ = z;
              c.sqdist = 0;
              c.dist = 0.0f;
              c.voronoi = occupied;
              c.queueing = fwProcessed;
              data[x][y][z] = c;
            } else
              setObstacle(x, y, z);
          }
        }
      }
    }
  }
}

void DynamicVoronoi3D::occupyCell(int x, int y, int z) {
  gridMap[x][y][z] = true;
  setObstacle(x, y, z);
}

void DynamicVoronoi3D::clearCell(int x, int y, int z) {
  gridMap[x][y][z] = false;
  removeObstacle(x, y, z);
}

void DynamicVoronoi3D::setObstacle(int x, int y, int z) {
  dataCell c = data[x][y][z];
  if (isOccupied(x, y, z, c))
    return;

  addList.emplace_back(x, y, z);
  // Update the parent of the cell.
  data[x][y][z].obstX = x;
  data[x][y][z].obstY = y;
  data[x][y][z].obstZ = z;
}

void DynamicVoronoi3D::removeObstacle(int x, int y, int z) {
  dataCell c = data[x][y][z];
  if (isOccupied(x, y, z, c) == false)
    return;

  removeList.emplace_back(x, y, z);
  // Reset the parent of the cell.
  data[x][y][z].obstX = invalidObstData;
  data[x][y][z].obstY = invalidObstData;
  data[x][y][z].obstZ = invalidObstData;
  data[x][y][z].queueing = bwQueued;
}

void DynamicVoronoi3D::exchangeObstacles(std::vector<IntPoint3D> &points) {
  // Remove old dynamic obstacles.
  for (unsigned int i = 0; i < lastObstacles.size(); ++i) {
    const int x = lastObstacles[i].x;
    const int y = lastObstacles[i].y;
    const int z = lastObstacles[i].z;

    if (gridMap[x][y][z] == false) {
      removeObstacle(x, y, z);
    }
  }
  lastObstacles.clear();

  const int num_points = points.size();
  for (unsigned int i = 0; i < num_points; ++i) {
    const int x = points[i].x;
    const int y = points[i].y;
    const int z = points[i].z;
    if (gridMap[x][y][z] == false) {
      setObstacle(x, y, z);
      // Update the list of dynamic obstacles.
      lastObstacles.emplace_back(x, y, z);
    }
  }
}

void DynamicVoronoi3D::update(bool updateRealDist) {
  // Register the obstacles from the addList and removeList.
  commitAndColorize(updateRealDist);
  const int num_nbrs = nbr_offsets.size();

  while (!open.empty()) {
    const IntPoint3D p = open.pop();
    const int x = p.x;
    const int y = p.y;
    const int z = p.z;
    dataCell c = data[x][y][z];

    if (c.queueing == fwProcessed) {
      continue;
    }

    if (c.needsRaise) {
      // Raise.
      c.queueing = bwProcessed;
      c.needsRaise = false;
      for (int i = 0; i < num_nbrs; ++i) {
        const int nx = x + nbr_offsets[i].x;
        const int ny = y + nbr_offsets[i].y;
        const int nz = z + nbr_offsets[i].z;
        if ((nx >= 0) && (nx < sizeX) && (ny >= 0) && (ny < sizeY) &&
            (nz >= 0) && (nz < sizeZ)) {
          dataCell nc = data[nx][ny][nz];
          // If nearest obstacle of the neighbor is valid and the neighbor cell
          // is not raised.
          if (nc.obstX != invalidObstData && !nc.needsRaise) {
            if (!isOccupied(nc.obstX, nc.obstY, nc.obstZ,
                            data[nc.obstX][nc.obstY][nc.obstZ])) {
              open.push(nc.sqdist, IntPoint3D(nx, ny, nz));
              nc.queueing = fwQueued;
              nc.needsRaise = true;
              nc.obstX = invalidObstData;
              nc.obstY = invalidObstData;
              nc.obstZ = invalidObstData;
              if (updateRealDist)
                nc.dist = INFINITY;
              nc.sqdist = INT_MAX;
              data[nx][ny][nz] = nc;
            } else {
              if (nc.queueing != fwQueued) {
                open.push(nc.sqdist, IntPoint3D(nx, ny, nz));
                nc.queueing = fwQueued;
                data[nx][ny][nz] = nc;
              }
            }
          }
        }
      }
    } else if (c.obstX != invalidObstData &&
               isOccupied(c.obstX, c.obstY, c.obstZ,
                          data[c.obstX][c.obstY][c.obstZ])) {
      // Lower.
      c.queueing = fwProcessed;
      c.voronoi = occupied;
      for (int i = 0; i < num_nbrs; ++i) {
        const int nx = x + nbr_offsets[i].x;
        const int ny = y + nbr_offsets[i].y;
        const int nz = z + nbr_offsets[i].z;
        if ((nx >= 0) && (nx < sizeX) && (ny >= 0) && (ny < sizeY) &&
            (nz >= 0) && (nz < sizeZ)) {
          dataCell nc = data[nx][ny][nz];
          if (!nc.needsRaise) {
            const int distx = nx - c.obstX;
            const int disty = ny - c.obstY;
            const int distz = nz - c.obstZ;
            const int newSqDistance =
                distx * distx + disty * disty + distz * distz;
            bool overwrite = (newSqDistance < nc.sqdist);
            if (!overwrite && newSqDistance == nc.sqdist) {
              if (nc.obstX == invalidObstData ||
                  isOccupied(nc.obstX, nc.obstY, nc.obstZ,
                             data[nc.obstX][nc.obstY][nc.obstZ]) == false)
                overwrite = true;
            }
            if (overwrite) {
              open.push(newSqDistance, IntPoint3D(nx, ny, nz));
              nc.queueing = fwQueued;
              if (updateRealDist) {
                nc.dist = std::sqrt((float)newSqDistance);
              }
              nc.sqdist = newSqDistance;
              nc.obstX = c.obstX;
              nc.obstY = c.obstY;
              nc.obstZ = c.obstZ;
            } else {
              checkVoro(x, y, z, nx, ny, nz, c, nc);
            }
            data[nx][ny][nz] = nc;
          }
        }
      }
    }
    // Update the cell.
    data[x][y][z] = c;
  }
}

float DynamicVoronoi3D::getDistance(int x, int y, int z) const {
  if ((x > 0) && (x < sizeX) && (y > 0) && (y < sizeY) && (z > 0) &&
      (z < sizeZ))
    return data[x][y][z].dist;
  else
    return INFINITY;
}

int DynamicVoronoi3D::getSquaredDistance(int x, int y, int z) const {
  if ((x > 0) && (x < sizeX) && (y > 0) && (y < sizeY) && (z > 0) &&
      (z < sizeZ))
    return data[x][y][z].sqdist;
  else
    return INT_MAX;
}

bool DynamicVoronoi3D::isVoronoi(int x, int y, int z) const {
  const dataCell c = data[x][y][z];
  return (c.voronoi == free || c.voronoi == voronoiKeep);
}

bool DynamicVoronoi3D::isVoronoiAlternative(int x, int y, int z) {
  const int v = alternativeDiagram[x][y][z];
  return (v == free || v == voronoiKeep);
}

void DynamicVoronoi3D::commitAndColorize(bool updateRealDist) {
  // Add new obstacles.
  const int num_add = addList.size();
  for (unsigned int i = 0; i < num_add; ++i) {
    const IntPoint3D p = addList[i];
    const int x = p.x;
    const int y = p.y;
    const int z = p.z;
    dataCell &c = data[x][y][z];
    if (c.queueing != fwQueued) {
      if (updateRealDist) {
        c.dist = 0.0f;
      }
      c.sqdist = 0;
      c.obstX = x;
      c.obstY = y;
      c.obstZ = z;
      c.queueing = fwQueued;
      c.voronoi = occupied;
      // Insert obstacles into the open list.
      open.push(0, IntPoint3D(x, y, z));
    }
  }

  // Remove old obstacles.
  const int num_remove = removeList.size();
  for (unsigned int i = 0; i < num_remove; ++i) {
    const IntPoint3D p = removeList[i];
    const int x = p.x;
    const int y = p.y;
    const int z = p.z;
    dataCell &c = data[x][y][z];
    if (isOccupied(x, y, z, c) == false) {
      // obstacle was removed and reinserted
      open.push(0, IntPoint3D(x, y, z));
      if (updateRealDist)
        c.dist = INFINITY;
      c.sqdist = INT_MAX;
      c.needsRaise = true;
    }
  }
  removeList.clear();
  addList.clear();
}

void DynamicVoronoi3D::checkVoro(int x, int y, int z, int nx, int ny, int nz,
                                 dataCell &c, dataCell &nc) {

  if ((c.sqdist > 1 || nc.sqdist > 1) && nc.obstX != invalidObstData) {
    if (abs(c.obstX - nc.obstX) > kObstacleWaveInibitDistance ||
        abs(c.obstY - nc.obstY) > kObstacleWaveInibitDistance ||
        abs(c.obstZ - nc.obstZ) > kObstacleWaveInibitDistance) {
      // compute dist from x, y, z to obstacle of nx, ny, nz
      int ds_nox = x - nc.obstX;
      int ds_noy = y - nc.obstY;
      int ds_noz = z - nc.obstZ;
      int sqds_no = ds_nox * ds_nox + ds_noy * ds_noy + ds_noz * ds_noz;
      int stability_xyz = sqds_no - c.sqdist;
      if (sqds_no - c.sqdist < 0)
        return;

      // compute dist from nx,ny to obstacle of x,y
      int dn_sox = nx - c.obstX;
      int dn_soy = ny - c.obstY;
      int dn_soz = nz - c.obstZ;
      int sqdn_so = dn_sox * dn_sox + dn_soy * dn_soy + dn_soz * dn_soz;
      int stability_nxyz = sqdn_so - nc.sqdist;
      if (sqdn_so - nc.sqdist < 0)
        return;

      // which cell is added to the Voronoi diagram?
      if (stability_xyz <= stability_nxyz && c.sqdist > 2) {
        if (c.voronoi != free) {
          c.voronoi = free;
          reviveVoroNeighbors(x, y, z);
          pruneQueue.push(IntPoint3D(x, y, z));
        }
      }
      if (stability_nxyz <= stability_xyz && nc.sqdist > 2) {
        if (nc.voronoi != free) {
          nc.voronoi = free;
          reviveVoroNeighbors(nx, ny, nz);
          pruneQueue.push(IntPoint3D(nx, ny, nz));
        }
      }
    }
  }
}

void DynamicVoronoi3D::reviveVoroNeighbors(int &x, int &y, int &z) {
  const int num_nbrs = nbr_offsets.size();
  for (int i = 0; i < num_nbrs; ++i) {
    const int nx = x + nbr_offsets[i].x;
    const int ny = y + nbr_offsets[i].y;
    const int nz = z + nbr_offsets[i].z;
    dataCell nc = data[nx][ny][nz];
    if (nc.sqdist != INT_MAX && !nc.needsRaise &&
        (nc.voronoi == voronoiKeep || nc.voronoi == voronoiPrune)) {
      nc.voronoi = free;
      data[nx][ny][nz] = nc;
      pruneQueue.push(IntPoint3D(nx, ny, nz));
    }
  }
}

bool DynamicVoronoi3D::isOccupied(const int x, const int y, const int z) const {
  dataCell c = data[x][y][z];
  return (c.obstX == x && c.obstY == y && c.obstZ == z);
}

bool DynamicVoronoi3D::isOccupied(const int &x, const int &y, const int &z,
                                  const dataCell &c) {
  return (c.obstX == x && c.obstY == y && c.obstZ == z);
}

/*
void DynamicVoronoi3D::prune() {
  // filler
  while (!pruneQueue.empty()) {
    IntPoint3D p = pruneQueue.front();
    pruneQueue.pop();
    const int x = p.x;
    const int y = p.y;
    const int z = p.z;

    if (data[x][y][z].voronoi == occupied)
      continue;
    if (data[x][y][z].voronoi == freeQueued)
      continue;

    data[x][y][z].voronoi = freeQueued;
    sortedPruneQueue.push(data[x][y][z].sqdist, p);

    // l: left, r: right, u: up, d: down, t: top, m: middle, b: bottom
    // lub ub rub
    // lb   b  rb
    // ldb db rdb

    // lum um rum
    // lm   c  rm
    // ldm dm rdm

    // lut ut rut
    // lt   t  rt
    // ldt dt rdt

    dataCell tr, tl, br, bl;
    tr = data[x + 1][y + 1];
    tl = data[x - 1][y + 1];
    br = data[x + 1][y - 1];
    bl = data[x - 1][y - 1];

    dataCell r, b, t, l;
    r = data[x + 1][y];
    l = data[x - 1][y];
    t = data[x][y + 1];
    b = data[x][y - 1];

    if (x + 2 < sizeX && r.voronoi == occupied) {
      // fill to the right
      if (tr.voronoi != occupied && br.voronoi != occupied &&
          data[x + 2][y].voronoi != occupied) {
        r.voronoi = freeQueued;
        sortedPruneQueue.push(r.sqdist, IntPoint3D(x + 1, y));
        data[x + 1][y] = r;
      }
    }
    if (x - 2 >= 0 && l.voronoi == occupied) {
      // fill to the left
      if (tl.voronoi != occupied && bl.voronoi != occupied &&
          data[x - 2][y].voronoi != occupied) {
        l.voronoi = freeQueued;
        sortedPruneQueue.push(l.sqdist, IntPoint3D(x - 1, y));
        data[x - 1][y] = l;
      }
    }
    if (y + 2 < sizeY && t.voronoi == occupied) {
      // fill to the top
      if (tr.voronoi != occupied && tl.voronoi != occupied &&
          data[x][y + 2].voronoi != occupied) {
        t.voronoi = freeQueued;
        sortedPruneQueue.push(t.sqdist, IntPoint3D(x, y + 1));
        data[x][y + 1] = t;
      }
    }
    if (y - 2 >= 0 && b.voronoi == occupied) {
      // fill to the bottom
      if (br.voronoi != occupied && bl.voronoi != occupied &&
          data[x][y - 2].voronoi != occupied) {
        b.voronoi = freeQueued;
        sortedPruneQueue.push(b.sqdist, IntPoint3D(x, y - 1));
        data[x][y - 1] = b;
      }
    }
  }

  while (!sortedPruneQueue.empty()) {
    IntPoint3D p = sortedPruneQueue.pop();
    dataCell c = data[p.x][p.y];
    int v = c.voronoi;
    if (v != freeQueued && v != voronoiRetry) { // || v>free || v==voronoiPrune
                                                // || v==voronoiKeep) {
      //      assert(v!=retry);
      continue;
    }

    markerMatchResult r = markerMatch(p.x, p.y);
    if (r == pruned)
      c.voronoi = voronoiPrune;
    else if (r == keep)
      c.voronoi = voronoiKeep;
    else { // r==retry
      c.voronoi = voronoiRetry;
      //      printf("RETRY %d %d\n", x, sizeY-1-y);
      pruneQueue.push(p);
    }
    data[p.x][p.y] = c;

    if (sortedPruneQueue.empty()) {
      while (!pruneQueue.empty()) {
        IntPoint3D p = pruneQueue.front();
        pruneQueue.pop();
        sortedPruneQueue.push(data[p.x][p.y][p.z].sqdist, p);
      }
    }
  }
  //  printf("match: %d\nnomat: %d\n", matchCount, noMatchCount);
}

DynamicVoronoi3D::markerMatchResult DynamicVoronoi3D::markerMatch(int x, int y,
                                                                  int z) {
  // implementation of connectivity patterns
  bool f[8];

  int nx, ny;
  int dx, dy;

  int i = 0;
  int count = 0;
  //  int obstacleCount=0;
  int voroCount = 0;
  int voroCountFour = 0;

  for (dy = 1; dy >= -1; dy--) {
    ny = y + dy;
    for (dx = -1; dx <= 1; dx++) {
      if (dx || dy) {
        nx = x + dx;
        dataCell nc = data[nx][ny];
        int v = nc.voronoi;
        bool b = (v <= free && v != voronoiPrune);
        //	if (v==occupied) obstacleCount++;
        f[i] = b;
        if (b) {
          voroCount++;
          if (!(dx && dy))
            voroCountFour++;
        }
        if (b && !(dx && dy))
          count++;
        //	if (v<=free && !(dx && dy)) voroCount++;
        i++;
      }
    }
  }
  if (voroCount < 3 && voroCountFour == 1 && (f[1] || f[3] || f[4] || f[6])) {
    //    assert(voroCount<2);
    //    if (voroCount>=2) printf("voro>2 %d %d\n", x, y);
    return keep;
  }

  // 4-connected
  if ((!f[0] && f[1] && f[3]) || (!f[2] && f[1] && f[4]) ||
      (!f[5] && f[3] && f[6]) || (!f[7] && f[6] && f[4]))
    return keep;
  if ((f[3] && f[4] && !f[1] && !f[6]) || (f[1] && f[6] && !f[3] && !f[4]))
    return keep;

  // keep voro cells inside of blocks and retry later
  if (voroCount >= 5 && voroCountFour >= 3 &&
      data[x][y].voronoi != voronoiRetry) {
    return retry;
  }

  return pruned;
}
*/

std::vector<IntPoint3D>
DynamicVoronoi3D::GetVoronoiNeighbors(const IntPoint3D &point) const {
  std::vector<IntPoint3D> neighbors;
  const int num_neighbors = voronoi_nbr_offsets.size();
  for (int i = 0; i < num_neighbors; ++i) {
    const IntPoint3D &offset = voronoi_nbr_offsets[i];
    const int neighbor_x = point.x + offset.x;
    const int neighbor_y = point.y + offset.y;
    const int neighbor_z = point.z + offset.z;
    if (neighbor_x >= 0 && neighbor_x < sizeX && neighbor_y >= 0 &&
        neighbor_y < sizeY && neighbor_z >= 0 && neighbor_z < sizeZ) {
      if (isVoronoi(neighbor_x, neighbor_y, neighbor_z)) {
        neighbors.emplace_back(neighbor_x, neighbor_y, neighbor_z);
      }
    }
  }
  return neighbors;
}

int DynamicVoronoi3D::GetNumVoronoiNeighbors(const IntPoint3D &point) const {
  int num_voronoi_neighbors = 0;
  const int num_neighbors = voronoi_nbr_offsets.size();
  for (int i = 0; i < num_neighbors; ++i) {
    const IntPoint3D &offset = voronoi_nbr_offsets[i];
    const int neighbor_x = point.x + offset.x;
    const int neighbor_y = point.y + offset.y;
    const int neighbor_z = point.z + offset.z;
    if (neighbor_x >= 0 && neighbor_x < sizeX && neighbor_y >= 0 &&
        neighbor_y < sizeY && neighbor_z >= 0 && neighbor_z < sizeZ) {
      if (isVoronoi(neighbor_x, neighbor_y, neighbor_z)) {
        ++num_voronoi_neighbors;
      }
    }
  }
  return num_voronoi_neighbors;
}

float DynamicVoronoi3D::GetUnionVolume(const IntPoint3D &p1,
                                       const IntPoint3D &p2) const {
  return 0.0;
}

void DynamicVoronoi3D::ConstructSparseGraphBK() {
  std::unordered_map<IntPoint3D, QueueState, IntPoint3DHash> is_visited;
  std::queue<IntPoint3D> cell_queue;
  // Traverse all cells and add unvisited voronoi cells to the queue.
  for (int x = 0; x < sizeX; ++x) {
    for (int y = 0; y < sizeY; ++y) {
      for (int z = 0; z < sizeZ; ++z) {
        if (isVoronoi(x, y, z) &&
            is_visited.find(IntPoint3D(x, y, z)) == is_visited.end()) {
          cell_queue.emplace(x, y, z);
          while (!cell_queue.empty()) {
            const IntPoint3D core = cell_queue.front();
            cell_queue.pop();
            is_visited[core] = kCellProcessed;
            const float obstacle_dist = getDistance(core.x, core.y, core.z);
            // Determine the new vertex candidates to add to the graph.
            std::queue<IntPoint3D> bfs_queue;
            std::vector<std::pair<IntPoint3D, float>> candidates;
            candidates.reserve(64);
            bfs_queue.emplace(core);
            while (!bfs_queue.empty()) {
              const IntPoint3D point = bfs_queue.front();
              bfs_queue.pop();
              if (is_visited[point] != kCellProcessed) {
                is_visited[point] = kProcessed;
              }
              const int p_to_core = GetDistanceBetween(core, point);
              if (p_to_core >= obstacle_dist) {
                // Add the edge.
                candidates.emplace_back(point,
                                        getDistance(point.x, point.y, point.z));
                is_visited[point] = kCandidate;
              } else {
                // Expansion.
                const std::vector<IntPoint3D> nbrs = GetVoronoiNeighbors(point);
                for (const IntPoint3D &nbr : nbrs) {
                  if (is_visited.find(nbr) == is_visited.end()) {
                    bfs_queue.emplace(nbr);
                    is_visited[nbr] = kBfsQueue;
                  } else if (is_visited[nbr] == kCellQueue ||
                             is_visited[nbr] == kCellProcessed) {
                    const float nbr_to_core = GetDistanceBetween(core, nbr);
                    if (getDistance(nbr.x, nbr.y, nbr.z) >= kDeadEndThreshold) {
                      graph_.AddTwoWayEdge(core, nbr, nbr_to_core);
                    }
                  }
                }
              }
            }
            // std::cout << "Adding " << candidates.size() << " candidates.\n";
            const int num_candidates = candidates.size();
            std::vector<float> candidate_dists(num_candidates);
            // Sort the candidates by distance.
            std::sort(candidates.begin(), candidates.end(),
                      [](const std::pair<IntPoint3D, float> &lhs,
                         const std::pair<IntPoint3D, float> &rhs) {
                        return lhs.second > rhs.second;
                      });
            std::vector<int> is_selected(num_candidates, 0);
            for (int i = 0; i < num_candidates; ++i) {
              if (!is_selected[i]) {
                is_selected[i] = 1;
                const IntPoint3D candidate = candidates[i].first;
                const float candidate_dist = candidates[i].second;
                const float candidate_to_core =
                    GetDistanceBetween(core, candidate);
                if (candidate_dist >= kDeadEndThreshold) {
                  graph_.AddTwoWayEdge(core, candidate, candidate_to_core);
                  cell_queue.emplace(candidate);
                  is_visited[candidate] = kCellQueue;
                  for (int j = 0; j < num_candidates; ++j) {
                    if (!is_selected[j]) {
                      const IntPoint3D other_candidate = candidates[j].first;
                      const float c_to_c =
                          GetDistanceBetween(candidate, other_candidate);
                      if (c_to_c < candidate_dist) {
                        is_selected[j] = 1;
                        is_visited[other_candidate] = kProcessed;
                      }
                    }
                  }
                } else {
                  is_visited[candidate] = kProcessed;
                }
              }
            }
          }
        }
      }
    }
  }
  std::cout << "Number of nodes in the graph: " << graph_.nodes_.size()
            << std::endl;
}

void DynamicVoronoi3D::SparseAddTwoWayEdge(const IntPoint3D &core,
                                           const IntPoint3D &add,
                                           const float weight) {
  // Edges cannot get close to each other.
  bool can_add_edge = true;
  if (graph_.node_id_.find(core) == graph_.node_id_.end()) {
    graph_.AddTwoWayEdge(core, add, weight);
  } else {
    const int core_id = graph_.node_id_[core];
    for (const auto &edge : graph_.nodes_[core_id].edges_) {
      const IntPoint3D dst_point = graph_.nodes_[edge.first].point_;
      const float dst_to_point = GetDistanceBetween(add, dst_point);
      const float dst_obstacle_dist =
          getDistance(dst_point.x, dst_point.y, dst_point.z);
      if (dst_to_point < dst_obstacle_dist) {
        can_add_edge = false;
        break;
      }
    }
    if (can_add_edge) {
      graph_.AddTwoWayEdge(core, add, weight);
    }
  }
}

AstarOutput DynamicVoronoi3D::GetAstarPath(const IntPoint3D &start,
                                           const IntPoint3D &goal) {
  AstarOutput output;
  TimeTrack track;
  // Add the start and goal nodes to the graph.
  const int num_nodes = graph_.nodes_.size();
  for (int i = 0; i < num_nodes; ++i) {
    const IntPoint3D point = graph_.nodes_[i].point_;
    const int obstacle_distance = getSquaredDistance(point.x, point.y, point.z);
    const int start_distance = GetSquaredDistanceBetween(point, start);
    const int goal_distance = GetSquaredDistanceBetween(point, goal);
    if (obstacle_distance >= start_distance) {
      graph_.AddOneWayEdge(start, point, start_distance);
    }
    if (obstacle_distance >= goal_distance) {
      graph_.AddTwoWayEdge(point, goal, goal_distance);
    }
  }
  track.OutputPassingTime("Add start and goal nodes to the graph");

  track.SetStartTime();
  // Run A* to find the path.
  std::priority_queue<QueueNode3D, std::vector<QueueNode3D>, QueueNodeCmp3D>
      astar_q;
  std::unordered_map<IntPoint3D, NodeProperty3D, IntPoint3DHash>
      node_properties;
  if (graph_.node_id_.find(start) == graph_.node_id_.end()) {
    std::cout << "Start node not found in graph !" << std::endl;
  } else {
    node_properties[start] = NodeProperty3D(NodeProperty3D::AstarState::kOpen,
                                            0.0, GetHeuristic(start, goal), -1);
    astar_q.push(QueueNode3D(start, node_properties[start].g_score_ +
                                        node_properties[start].h_score_));
  }
  bool is_path_found = false;
  int count = 0;
  while (!astar_q.empty()) {
    ++count;
    // Selection.
    const QueueNode3D current_node = astar_q.top();
    astar_q.pop();
    // Check if the current node is the goal.
    if (current_node.point_ == goal) {
      is_path_found = true;
      break;
    }
    // Skip visited nodes due to the same
    if (node_properties[current_node.point_].state_ ==
        NodeProperty3D::AstarState::kClose) {
      continue;
    }
    node_properties[current_node.point_].state_ =
        NodeProperty3D::AstarState::kClose;
    // Expansion.
    const int current_node_id = graph_.node_id_[current_node.point_];
    const auto &edges = graph_.nodes_[current_node_id].edges_;
    for (const auto &edge : edges) {
      const IntPoint3D neighbor = graph_.nodes_[edge.first].point_;
      const float edge_weight = edge.second;
      const float g_score =
          node_properties[current_node.point_].g_score_ + edge_weight;
      if (node_properties.find(neighbor) == node_properties.end()) {
        const float h_score = GetHeuristic(neighbor, goal);
        node_properties[neighbor] =
            NodeProperty3D(NodeProperty3D::AstarState::kOpen, g_score, h_score,
                           current_node_id);
        astar_q.push(QueueNode3D(neighbor, g_score + h_score));
      } else if (node_properties[neighbor].state_ ==
                 NodeProperty3D::AstarState::kOpen) {
        if (g_score < node_properties[neighbor].g_score_) {
          node_properties[neighbor].g_score_ = g_score;
          node_properties[neighbor].father_id_ = current_node_id;
          astar_q.push(QueueNode3D(
              neighbor, g_score + node_properties[neighbor].h_score_));
        }
      }
    }
  }
  track.OutputPassingTime("Run A* to find the path");
  output.num_expansions = count;
  std::cout << "A* count: " << count << std::endl;

  track.SetStartTime();
  std::vector<IntPoint3D> path;
  if (is_path_found) {
    output.success = true;
    std::cout << "Path found !" << std::endl;
    IntPoint3D waypoint = goal;
    int waypoint_id = graph_.node_id_[waypoint];
    std::unordered_map<int, bool> visited;
    while (waypoint_id != -1) {
      if (visited.find(waypoint_id) == visited.end()) {
        visited[waypoint_id] = true;
        waypoint = graph_.nodes_[waypoint_id].point_;
        path.push_back(waypoint);
        waypoint_id = node_properties[waypoint].father_id_;
      } else {
        std::cout << waypoint_id << " already visited !" << std::endl;
        std::cout << "Loop detected !" << std::endl;
        break;
      }
    }
    std::reverse(path.begin(), path.end());
  } else {
    output.success = false;
    std::cout << "No path found from " << start.x << "," << start.y << " to "
              << goal.x << "," << goal.y << std::endl;
  }
  track.OutputPassingTime("Output the path");
  // Remove the start and goal nodes from the graph.
  // Remove last two nodes in the graph.
  for (int i = 0; i < 2; ++i) {
    VGraphNode3D node = graph_.nodes_.back();
    const int node_id = graph_.node_id_[node.point_];
    graph_.nodes_.pop_back();
    for (const auto &edge : node.edges_) {
      const int dest_id = edge.first;
      graph_.nodes_[dest_id].edges_.erase(node_id);
    }
  }
  graph_.node_id_.erase(start);
  graph_.node_id_.erase(goal);
  // Output the length of the path.
  if (is_path_found) {
    float len = 0.0f;
    const int num_path_points = path.size();
    for (int i = 0; i < num_path_points - 1; ++i) {
      const float dist = GetDistanceBetween(path[i], path[i + 1]);
      len += dist;
    }
    output.path_length = len;
    output.path = std::move(path);
    std::cout << "Path length: " << len << std::endl;
  }
  return output;
}

iLQROutput DynamicVoronoi3D::GetiLQRPath(const std::vector<IntPoint3D> &path) {
  iLQROutput output;
  TimeTrack track;
  std::vector<IntPoint3D> ilqr_path;
  if (path.empty() || path.size() < 2) {
    return output;
  }
  // Construct the constraints.
  const int num_bubbles = path.size() - 2;
  std::vector<IntPoint3D> bubbles(path.begin() + 1, path.end() - 1);
  std::vector<float> radius;
  radius.reserve(num_bubbles);
  for (int i = 0; i < num_bubbles; ++i) {
    radius.emplace_back(getDistance(bubbles[i].x, bubbles[i].y, bubbles[i].z));
  }
  // iLQR Path Optimization.
  const int num_steps = path.size() - 1;
  Eigen::Matrix<float, 3, 6> F;
  // clang-format off
  F <<
  1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f;
  // clang-format on
  std::vector<Eigen::Matrix3f> K_mats(num_steps, Eigen::Matrix3f::Zero());
  std::vector<Eigen::Vector3f> k_vecs(num_steps, Eigen::Vector3f::Zero());
  std::vector<Eigen::Matrix<float, 6, 1>> xu_vecs(
      num_steps, Eigen::Matrix<float, 6, 1>::Zero());
  std::vector<Eigen::Vector3f> x_hat_vecs(num_steps, Eigen::Vector3f::Zero());
  std::vector<Eigen::Vector3f> ov_center(num_bubbles - 1,
                                         Eigen::Vector3f::Zero());
  std::vector<float> coeff(num_steps, 0.0f);
  track.OutputPassingTime("Initialize");
  // Construct the initial guess.
  std::cout << "Construct the initial guess..." << std::endl;
  const IntPoint3D start = path.front();
  const IntPoint3D goal = path.back();
  xu_vecs[0] << start.x, start.y, start.z, 0.0f, 0.0f, 0.0f;
  x_hat_vecs[0] << start.x, start.y, start.z;
  for (int i = 1; i < num_steps - 1; ++i) {
    const float delta_c = GetDistanceBetween(bubbles[i - 1], bubbles[i]);
    const float dist =
        0.5f * (delta_c + (radius[i - 1] + radius[i]) *
                              (radius[i - 1] - radius[i]) / delta_c);
    const float x_initial =
        bubbles[i - 1].x + dist / delta_c * (bubbles[i].x - bubbles[i - 1].x);
    const float y_initial =
        bubbles[i - 1].y + dist / delta_c * (bubbles[i].y - bubbles[i - 1].y);
    const float z_initial =
        bubbles[i - 1].z + dist / delta_c * (bubbles[i].z - bubbles[i - 1].z);
    xu_vecs[i].block<3, 1>(0, 0) << x_initial, y_initial, z_initial;
    xu_vecs[i - 1].block<3, 1>(3, 0) =
        xu_vecs[i].block<3, 1>(0, 0) - xu_vecs[i - 1].block<3, 1>(0, 0);
    ov_center[i - 1] << x_initial, y_initial, z_initial;
  }
  xu_vecs[num_steps - 1] << goal.x, goal.y, goal.z, 0.0f, 0.0f, 0.0f;
  xu_vecs[num_steps - 2].block<3, 1>(3, 0) =
      xu_vecs[num_steps - 1].block<3, 1>(0, 0) -
      xu_vecs[num_steps - 2].block<3, 1>(0, 0);

  // Calculate the coefficents.
  coeff[0] = std::max(
      1.0f, (ov_center[0] - Eigen::Vector3f(start.x, start.y, start.z)).norm());
  coeff[num_bubbles - 1] =
      std::max(1.0f, (ov_center[num_bubbles - 2] -
                      Eigen::Vector3f(goal.x, goal.y, goal.z))
                         .norm());
  for (int i = 0; i < num_bubbles - 2; ++i) {
    coeff[i + 1] = std::max(1.0f, (ov_center[i + 1] - ov_center[i]).norm());
  }

  std::cout << "start ilqr optimization..." << std::endl;
  bool is_ilqr_success = false;
  float cost_sum = 0.0f;
  float last_cost_sum = cost_sum;
  float path_length = 0.0f;
  float last_path_length = path_length;
  // Iteration Loop.
  for (int iter = 0; iter < kMaxIteration; ++iter) {
    Eigen::Matrix3f V;
    Eigen::Vector3f v;
    cost_sum = 0.0f;
    std::pair<float, float> delta_V(0.0f, 0.0f);
    // Backward Pass.
    // track.SetStartTime();
    std::vector<
        std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>>
        costs(num_steps, std::make_pair(Eigen::Matrix<float, 6, 6>::Zero(),
                                        Eigen::Matrix<float, 6, 1>::Zero()));
    // Cost can be calculated in parallel.
    // Update: Actually, in this case, it is not worthwhile to parallelize the
    // cost. Time is mainly spent in the calculation of V and v.
    costs[0] = GetCost(xu_vecs[0], bubbles[0], radius[0], bubbles[0], radius[0],
                       coeff[0]);
    cost_sum += GetRealCost(xu_vecs[0], bubbles[0], radius[0], bubbles[0],
                            radius[0], coeff[0]);
    costs[num_steps - 1] = GetTermCost(xu_vecs[num_steps - 1], goal);
    cost_sum += GetRealTermCost(xu_vecs[num_steps - 1], goal);
    for (int k = 1; k < num_steps - 1; ++k) {
      costs[k] = GetCost(xu_vecs[k], bubbles[k - 1], radius[k - 1], bubbles[k],
                         radius[k], coeff[k]);
      cost_sum += GetRealCost(xu_vecs[k], bubbles[k - 1], radius[k - 1],
                              bubbles[k], radius[k], coeff[k]);
    }

    for (int k = num_steps - 1; k >= 0; --k) {
      Eigen::Matrix<float, 6, 6> Q;
      Eigen::Matrix<float, 6, 1> q;
      if (k == num_steps - 1) {
        // Terminal cost.
        const std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>
            cost = costs[k];
        Q = cost.first;
        q = cost.second;
      } else {
        const std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>
            cost = costs[k];
        Q = cost.first + F.transpose() * V * F;
        q = cost.second + F.transpose() * v;
      }
      const Eigen::Matrix3f Qxx = Q.block<3, 3>(0, 0);
      const Eigen::Matrix3f Qxu = Q.block<3, 3>(0, 3);
      const Eigen::Matrix3f Qux = Q.block<3, 3>(3, 0);
      const Eigen::Matrix3f Quu = Q.block<3, 3>(3, 3);
      const Eigen::Vector3f qx = q.block<3, 1>(0, 0);
      const Eigen::Vector3f qu = q.block<3, 1>(3, 0);
      K_mats[k] = -Quu.inverse() * Qux;
      k_vecs[k] = -Quu.inverse() * qu;
      const Eigen::Matrix3f K_mat = K_mats[k];
      const Eigen::Vector3f k_vec = k_vecs[k];
      V = Qxx + Qxu * K_mat + K_mat.transpose() * Qux +
          K_mat.transpose() * Quu * K_mat;
      v = qx + Qxu * k_vec + K_mat.transpose() * qu +
          K_mat.transpose() * Quu * k_vec;
      delta_V.first += k_vec.transpose() * qu;
      delta_V.second += 0.5f * k_vec.transpose() * Quu * k_vec;
    }
    // track.OutputPassingTime("Backward Pass");

    float alpha = 1.0f;
    bool is_line_search_done = false;
    int line_search_iter = 0;
    // TODO: Parellel line search.
    // Update: It is a pity that it is not worthwhile to parallelize the line
    // search, too. The overhead of creating threads is too high.
    // track.SetStartTime();
    const std::vector<Eigen::Matrix<float, 6, 1>> cur_xu_vecs(xu_vecs);
    while (!is_line_search_done && line_search_iter < kMaxLineSearchIter) {
      ++line_search_iter;
      // Forward Pass.
      float next_cost_sum = 0.0f;
      for (int k = 0; k < num_steps - 1; ++k) {
        const Eigen::Vector3f x = cur_xu_vecs[k].block<3, 1>(0, 0);
        const Eigen::Vector3f u = cur_xu_vecs[k].block<3, 1>(3, 0);
        xu_vecs[k].block<3, 1>(3, 0) =
            K_mats[k] * (x_hat_vecs[k] - x) + alpha * k_vecs[k] + u;
        xu_vecs[k].block<3, 1>(0, 0) = x_hat_vecs[k];
        x_hat_vecs[k + 1] = F * xu_vecs[k];
        // Calculate the new cost.
        if (k == 0) {
          next_cost_sum += GetRealCost(xu_vecs[k], bubbles[k], radius[k],
                                       bubbles[k], radius[k], coeff[k]);
        } else {
          next_cost_sum +=
              GetRealCost(xu_vecs[k], bubbles[k - 1], radius[k - 1], bubbles[k],
                          radius[k], coeff[k]);
        }
      }
      xu_vecs[num_steps - 1].block<3, 1>(0, 0) = x_hat_vecs[num_steps - 1];
      next_cost_sum += GetRealTermCost(xu_vecs[num_steps - 1], goal);
      // Check if J satisfy line search condition.
      const float ratio_decrease =
          (next_cost_sum - cost_sum) /
          (alpha * (delta_V.first + alpha * delta_V.second));
      if (line_search_iter < kMaxLineSearchIter &&
          (ratio_decrease <= 1e-4 || ratio_decrease >= 10.0f)) {
        alpha *= 0.5f;
      } else {
        is_line_search_done = true;
      }
    }
    // track.OutputPassingTime("Forward Pass");

    // Calculate the path length.
    path_length = 0.0f;
    for (int k = 0; k < num_steps - 1; ++k) {
      path_length += xu_vecs[k].block<3, 1>(3, 0).norm();
    }
    // Terminate condition.
    if (iter > 0 &&
        std::fabs(path_length - last_path_length) < kConvergenceThreshold) {
      std::cout << "Convergence reached ! iter: " << iter
                << " cost: " << cost_sum << " path_length: " << path_length
                << std::endl;
      output.num_iter = iter;
      output.path_length = path_length;
      break;
    } else {
      last_path_length = path_length;
      last_cost_sum = cost_sum;
      // std::cout << "iter: " << iter << " cost: " << cost_sum
      //           << " path_length: " << path_length << std::endl;
    }
  }
  // Output the path.
  ilqr_path.reserve(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    const Eigen::Vector3f x = xu_vecs[i].block<3, 1>(0, 0);
    ilqr_path.emplace_back(x(0), x(1), x(2));
  }
  output.path = std::move(ilqr_path);
  return output;
}

const VGraph3D &DynamicVoronoi3D::GetSparseGraph() const { return graph_; }

bool DynamicVoronoi3D::isInSparseGraph(const IntPoint3D &point) const {
  return graph_.node_id_.find(point) != graph_.node_id_.end();
}

float DynamicVoronoi3D::GetDistanceBetween(const IntPoint3D &p1,
                                           const IntPoint3D &p2) const {
  return std::hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

int DynamicVoronoi3D::GetSquaredDistanceBetween(const IntPoint3D &p1,
                                                const IntPoint3D &p2) const {
  const int dx = p1.x - p2.x;
  const int dy = p1.y - p2.y;
  const int dz = p1.z - p2.z;
  return dx * dx + dy * dy + dz * dz;
}

float DynamicVoronoi3D::GetHeuristic(const IntPoint3D &start,
                                     const IntPoint3D &goal) const {
  return GetDistanceBetween(start, goal);
}

std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>
DynamicVoronoi3D::GetCost(const Eigen::Matrix<float, 6, 1> &xu,
                          const IntPoint3D &bubble_1, const float radius_1,
                          const IntPoint3D &bubble_2, const float radius_2,
                          const float coeff) {
  float dcx = 0.0;
  float ddcx = 0.0;
  float dcy = 0.0;
  float ddcy = 0.0;
  float dcz = 0.0;
  float ddcz = 0.0;
  // Constraints on bubble 1.
  const int dx_1 = xu(0) - bubble_1.x;
  const int dy_1 = xu(1) - bubble_1.y;
  const int dz_1 = xu(2) - bubble_1.z;
  if (dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1 > radius_1 * radius_1) {
    dcx += kBndWeight * dx_1;
    dcy += kBndWeight * dy_1;
    dcz += kBndWeight * dz_1;
    ddcx += kBndWeight;
    ddcy += kBndWeight;
    ddcz += kBndWeight;
  }
  // Constraints on bubble 2.
  const int dx_2 = xu(0) - bubble_2.x;
  const int dy_2 = xu(1) - bubble_2.y;
  const int dz_2 = xu(2) - bubble_2.z;
  if (dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2 > radius_2 * radius_2) {
    dcx += kBndWeight * dx_2;
    dcy += kBndWeight * dy_2;
    dcz += kBndWeight * dz_2;
    ddcx += kBndWeight;
    ddcy += kBndWeight;
    ddcz += kBndWeight;
  }
  std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> cost;
  // clang-format off
  cost.first <<
  ddcx, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, ddcy, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, ddcz, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, kWeight / coeff, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, kWeight / coeff, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, kWeight / coeff;
  cost.second <<
  dcx,
  dcy,
  dcz,
  kWeight / coeff * xu(3),
  kWeight / coeff * xu(4),
  kWeight / coeff * xu(5);
  // clang-format on
  return cost;
}

float DynamicVoronoi3D::GetRealCost(const Eigen::Matrix<float, 6, 1> &xu,
                                    const IntPoint3D &bubble_1,
                                    const float radius_1,
                                    const IntPoint3D &bubble_2,
                                    const float radius_2, const float coeff) {
  float real_cost = 0.0;
  real_cost += kWeight * 0.5 / coeff * xu(3) * xu(3) +
               kWeight * 0.5 / coeff * xu(4) * xu(4) +
               kWeight * 0.5 / coeff * xu(5) * xu(5);
  // Constraints on bubble 1.
  const int dx_1 = xu(0) - bubble_1.x;
  const int dy_1 = xu(1) - bubble_1.y;
  const int dz_1 = xu(2) - bubble_1.z;
  if (dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1 > radius_1 * radius_1) {
    real_cost +=
        kBndWeight * 0.5 * dx_1 * dx_1 + kBndWeight * 0.5 * dy_1 * dy_1 +
        kBndWeight * 0.5 * dz_1 * dz_1 - kBndWeight * 0.5 * radius_1 * radius_1;
  }
  // Constraints on bubble 2.
  const int dx_2 = xu(0) - bubble_2.x;
  const int dy_2 = xu(1) - bubble_2.y;
  const int dz_2 = xu(2) - bubble_2.z;
  if (dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2 > radius_2 * radius_2) {
    real_cost +=
        kBndWeight * 0.5 * dx_2 * dx_2 + kBndWeight * 0.5 * dy_2 * dy_2 +
        kBndWeight * 0.5 * dz_2 * dz_2 - kBndWeight * 0.5 * radius_2 * radius_2;
  }
  return real_cost;
}

std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>>
DynamicVoronoi3D::GetTermCost(const Eigen::Matrix<float, 6, 1> &xu,
                              const IntPoint3D &goal) {
  std::pair<Eigen::Matrix<float, 6, 6>, Eigen::Matrix<float, 6, 1>> cost;
  // clang-format off
  cost.first <<
  kTermWeight, 0.0, 0.0, 0.0, 0.0, 0.0,
  0.0, kTermWeight, 0.0, 0.0, 0.0, 0.0,
  0.0, 0.0, kTermWeight, 0.0, 0.0, 0.0,
  0.0, 0.0, 0.0, kWeight, 0.0, 0.0,
  0.0, 0.0, 0.0, 0.0, kWeight, 0.0,
  0.0, 0.0, 0.0, 0.0, 0.0, kWeight;
  cost.second <<
  kTermWeight * (xu(0) - goal.x),
  kTermWeight * (xu(1) - goal.y),
  kTermWeight * (xu(2) - goal.z),
  kWeight * xu(3),
  kWeight * xu(4),
  kWeight * xu(5);
  // clang-format on
  return cost;
}

float DynamicVoronoi3D::GetRealTermCost(const Eigen::Matrix<float, 6, 1> &xu,
                                        const IntPoint3D &goal) {
  float real_cost = 0.0;
  const float dx = xu(0) - goal.x;
  const float dy = xu(1) - goal.y;
  const float dz = xu(2) - goal.z;
  real_cost += kTermWeight * 0.5 * dx * dx + kTermWeight * 0.5 * dy * dy +
               kTermWeight * 0.5 * dz * dz;
  real_cost += kWeight * 0.5 * xu(3) * xu(3) + kWeight * 0.5 * xu(4) * xu(4) +
               kWeight * 0.5 * xu(5) * xu(5);
  return real_cost;
}

iLQRTrajectory
DynamicVoronoi3D::GetiLQRTrajectory(const std::vector<IntPoint3D> &path,
                                    const std::vector<IntPoint3D> &ilqr_path) {

  iLQRTrajectory ilqr_traj;
  std::vector<Eigen::Matrix<float, ALL_DIM, 1>> traj;
  TimeTrack track;
  if (path.empty() || path.size() < 2) {
    return ilqr_traj;
  }
  // Construct the constraints.
  const int num_bubbles = path.size() - 2;
  std::vector<IntPoint3D> bubbles(path.begin() + 1, path.end() - 1);
  std::vector<float> radius;
  radius.reserve(num_bubbles);
  for (int i = 0; i < num_bubbles; ++i) {
    radius.emplace_back(getDistance(bubbles[i].x, bubbles[i].y, bubbles[i].z));
  }
  // iLQR Path Optimization.
  const int num_steps = path.size() - 1;
  std::vector<Eigen::Matrix<float, STATE_DIM, ALL_DIM>> F_mats(
      num_steps, Eigen::Matrix<float, STATE_DIM, ALL_DIM>::Zero());
  std::vector<Eigen::Matrix<float, CONTROL_DIM, STATE_DIM>> K_mats(
      num_steps, Eigen::Matrix<float, CONTROL_DIM, STATE_DIM>::Zero());
  std::vector<Eigen::Matrix<float, CONTROL_DIM, 1>> k_vecs(
      num_steps, Eigen::Matrix<float, CONTROL_DIM, 1>::Zero());
  std::vector<Eigen::Matrix<float, ALL_DIM, 1>> xu_vecs(
      num_steps, Eigen::Matrix<float, ALL_DIM, 1>::Zero());
  std::vector<Eigen::Matrix<float, STATE_DIM, 1>> x_hat_vecs(
      num_steps, Eigen::Matrix<float, STATE_DIM, 1>::Zero());
  std::vector<float> coeff(num_steps, 0.0f);
  track.OutputPassingTime("Initialize");
  // Construct the initial guess.
  std::cout << "Construct the initial guess..." << std::endl;
  const IntPoint3D start = path.front();
  const IntPoint3D goal = path.back();

  // Initial Guess Solution 1.
  // clang-format off
  x_hat_vecs[0] <<
  start.x, 0.0f, 0.0f,
  start.y, 0.0f, 0.0f,
  start.z, 0.0f, 0.0f;
  // clang-format on
  xu_vecs[0].block<STATE_DIM, 1>(0, 0) = x_hat_vecs[0];
  for (int i = 1; i < num_steps - 1; ++i) {
    const float delta_c = GetDistanceBetween(bubbles[i - 1], bubbles[i]);
    const float dist =
        0.5f * (delta_c + (radius[i - 1] + radius[i]) *
                              (radius[i - 1] - radius[i]) / delta_c);
    const float x_initial =
        bubbles[i - 1].x + dist / delta_c * (bubbles[i].x - bubbles[i - 1].x);
    const float y_initial =
        bubbles[i - 1].y + dist / delta_c * (bubbles[i].y - bubbles[i - 1].y);
    const float z_initial =
        bubbles[i - 1].z + dist / delta_c * (bubbles[i].z - bubbles[i - 1].z);
    const float dx_initial = x_initial - xu_vecs[i - 1](0);
    const float dy_initial = y_initial - xu_vecs[i - 1](3);
    const float dz_initial = z_initial - xu_vecs[i - 1](6);
    const float dt_initial = kInitialTimeScale *
                             std::hypot(dx_initial, dy_initial, dz_initial) /
                             kMaxVelocity;
    // clang-format off
    x_hat_vecs[i] <<
    x_initial, dx_initial / dt_initial, 0.0f,
    y_initial, dy_initial / dt_initial, 0.0f,
    z_initial, dz_initial / dt_initial, 0.0f;
    // clang-format on
    // Initial Guess.
    xu_vecs[i].block<STATE_DIM, 1>(0, 0) = x_hat_vecs[i];
    xu_vecs[i - 1].block<CONTROL_DIM - 1, 1>(STATE_DIM, 0) =
        xu_vecs[i].block<STATE_DIM, 1>(0, 0) -
        xu_vecs[i - 1].block<STATE_DIM, 1>(0, 0);
    xu_vecs[i - 1](ALL_DIM - 1) = dt_initial;
  }
  // clang-format off
  x_hat_vecs[num_steps - 1] <<
  goal.x, 0.0f, 0.0f,
  goal.y, 0.0f, 0.0f,
  goal.z, 0.0f, 0.0f;
  // clang-format on
  xu_vecs[num_steps - 1].block<STATE_DIM, 1>(0, 0) = x_hat_vecs[num_steps - 1];
  xu_vecs[num_steps - 2].block<CONTROL_DIM - 1, 1>(STATE_DIM, 0) =
      xu_vecs[num_steps - 1].block<STATE_DIM, 1>(0, 0) -
      xu_vecs[num_steps - 2].block<STATE_DIM, 1>(0, 0);
  {
    const float dx_initial =
        xu_vecs[num_steps - 1](0) - xu_vecs[num_steps - 2](0);
    const float dy_initial =
        xu_vecs[num_steps - 1](3) - xu_vecs[num_steps - 2](3);
    const float dz_initial =
        xu_vecs[num_steps - 1](6) - xu_vecs[num_steps - 2](6);
    const float initial_dt = kInitialTimeScale *
                             std::hypot(dx_initial, dy_initial, dz_initial) /
                             kMaxVelocity;
    xu_vecs[num_steps - 2](ALL_DIM - 1) = initial_dt;
  }

  // Initial Guess Solution 2.
  // for (int i = 0; i < num_steps - 1; ++i) {
  //   // clang-format off
  //   x_hat_vecs[i] <<
  //   ilqr_path[i].x, 0.0f, 0.0f,
  //   ilqr_path[i].y, 0.0f, 0.0f,
  //   ilqr_path[i].z, 0.0f, 0.0f;
  //   // clang-format on
  //   // Initial Guess.
  //   xu_vecs[i].block<STATE_DIM, 1>(0, 0) = x_hat_vecs[i];
  //   const float dist = GetDistanceBetween(ilqr_path[i], ilqr_path[i + 1]);
  //   // clang-format off
  //   xu_vecs[i].block<CONTROL_DIM, 1>(STATE_DIM, 0) <<
  //   ilqr_path[i + 1].x - ilqr_path[i].x, 0.0f, 0.0f,
  //   ilqr_path[i + 1].y - ilqr_path[i].y, 0.0f, 0.0f,
  //   ilqr_path[i + 1].z - ilqr_path[i].z, 0.0f, 0.0f, kInitialTimeScale * dist
  //   / kMaxVelocity;
  //   // clang-format on
  // }
  // // clang-format off
  // x_hat_vecs[num_steps - 1] <<
  // ilqr_path[num_steps - 1].x, 0.0f, 0.0f,
  // ilqr_path[num_steps - 1].y, 0.0f, 0.0f,
  // ilqr_path[num_steps - 1].z, 0.0f, 0.0f;
  // xu_vecs[num_steps - 1].block<STATE_DIM, 1>(0, 0) = x_hat_vecs[num_steps -
  // 1]; xu_vecs[num_steps - 1].block<CONTROL_DIM, 1>(STATE_DIM, 0) << 0.0f,
  // 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f;
  // // clang-format on

  // Output the initial guess.
  for (int i = 0; i < num_steps; ++i) {
    std::cout << "xu_vecs[" << i << "]: " << xu_vecs[i].transpose()
              << std::endl;
  }

  // Calculate the coefficents.
  std::cout << "start ilqr optimization..." << std::endl;
  bool is_ilqr_success = false;
  float cost_sum = 0.0f;
  float last_cost_sum = cost_sum;
  float path_length = 0.0f;
  float last_path_length = path_length;
  float time_sum = 0.0f;
  float last_time_sum = time_sum;
  float reg_coeff = kRegularization;
  // Iteration Loop.
  for (int iter = 0; iter < kMaxIteration; ++iter) {
    Eigen::Matrix<float, STATE_DIM, STATE_DIM> V;
    Eigen::Matrix<float, STATE_DIM, 1> v;
    cost_sum = 0.0f;
    std::pair<float, float> delta_V(0.0f, 0.0f);
    // Backward Pass.
    // track.SetStartTime();
    std::vector<std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
                          Eigen::Matrix<float, ALL_DIM, 1>>>
        costs(num_steps,
              std::make_pair(Eigen::Matrix<float, ALL_DIM, ALL_DIM>::Zero(),
                             Eigen::Matrix<float, ALL_DIM, 1>::Zero()));
    // Cost can be calculated in parallel.
    // Update: Actually, in this case, it is not worthwhile to parallelize the
    // cost. Time is mainly spent in the calculation of V and v.
    costs[0] = GetTrajCost(xu_vecs[0], bubbles[0], radius[0], bubbles[0],
                           radius[0], kMaxVelocity, kMaxAcceleration, coeff[0]);
    cost_sum +=
        GetTrajRealCost(xu_vecs[0], bubbles[0], radius[0], bubbles[0],
                        radius[0], kMaxVelocity, kMaxAcceleration, coeff[0])
            .total_cost;
    F_mats[0] = GetTransition(xu_vecs[0]);
    costs[num_steps - 1] = GetTrajTermCost(xu_vecs[num_steps - 1], goal);
    cost_sum += GetTrajRealTermCost(xu_vecs[num_steps - 1], goal);
    F_mats[num_steps - 1] = GetTransition(xu_vecs[num_steps - 1]);
    for (int k = 1; k < num_steps - 1; ++k) {
      costs[k] =
          GetTrajCost(xu_vecs[k], bubbles[k - 1], radius[k - 1], bubbles[k],
                      radius[k], kMaxVelocity, kMaxAcceleration, coeff[k]);
      cost_sum +=
          GetTrajRealCost(xu_vecs[k], bubbles[k - 1], radius[k - 1], bubbles[k],
                          radius[k], kMaxVelocity, kMaxAcceleration, coeff[k])
              .total_cost;
      F_mats[k] = GetTransition(xu_vecs[k]);
    }
    // Calculate V/v and K/k.
    bool is_backward_pass_done = false;
    while (!is_backward_pass_done) {
      bool is_Quu_full_rank = true;
      for (int k = num_steps - 1; k >= 0; --k) {
        Eigen::Matrix<float, ALL_DIM, ALL_DIM> Q;
        Eigen::Matrix<float, ALL_DIM, 1> q;
        if (k == num_steps - 1) {
          // Terminal cost.
          const std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
                          Eigen::Matrix<float, ALL_DIM, 1>>
              cost = costs[k];
          Q = cost.first;
          q = cost.second;
        } else {
          const std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
                          Eigen::Matrix<float, ALL_DIM, 1>>
              cost = costs[k];
          const Eigen::Matrix<float, STATE_DIM, ALL_DIM> F = F_mats[k];
          Q = cost.first + F.transpose() * V * F;
          q = cost.second + F.transpose() * v;
        }
        const Eigen::Matrix<float, STATE_DIM, STATE_DIM> Qxx =
            Q.block<STATE_DIM, STATE_DIM>(0, 0);
        const Eigen::Matrix<float, STATE_DIM, CONTROL_DIM> Qxu =
            Q.block<STATE_DIM, CONTROL_DIM>(0, STATE_DIM);
        const Eigen::Matrix<float, CONTROL_DIM, STATE_DIM> Qux =
            Q.block<CONTROL_DIM, STATE_DIM>(STATE_DIM, 0);
        const Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> Quu =
            Q.block<CONTROL_DIM, CONTROL_DIM>(STATE_DIM, STATE_DIM);
        const Eigen::Matrix<float, STATE_DIM, 1> qx =
            q.block<STATE_DIM, 1>(0, 0);
        const Eigen::Matrix<float, CONTROL_DIM, 1> qu =
            q.block<CONTROL_DIM, 1>(STATE_DIM, 0);

        if (k == num_steps - 1) {
          V = Qxx;
          v = qx;
        } else {
          // Quu needs to be regularized.
          Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> Quu_reg = Quu;
          const Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM> reg =
              reg_coeff *
              Eigen::Matrix<float, CONTROL_DIM, CONTROL_DIM>::Identity();
          Quu_reg += reg;
          if (std::fabs(Quu_reg.determinant()) < 1e-3) {
            is_Quu_full_rank = false;
            break;
          }
          K_mats[k] = -Quu_reg.inverse() * Qux;
          k_vecs[k] = -Quu_reg.inverse() * qu;
          const Eigen::Matrix<float, CONTROL_DIM, STATE_DIM> K_mat = K_mats[k];
          const Eigen::Matrix<float, CONTROL_DIM, 1> k_vec = k_vecs[k];
          V = Qxx + Qxu * K_mat + K_mat.transpose() * Qux +
              K_mat.transpose() * Quu_reg * K_mat;
          v = qx + Qxu * k_vec + K_mat.transpose() * qu +
              K_mat.transpose() * Quu_reg * k_vec;
          delta_V.first += k_vec.transpose() * qu;
          delta_V.second += 0.5f * k_vec.transpose() * Quu_reg * k_vec;
        }
      }
      if (is_Quu_full_rank) {
        is_backward_pass_done = true;
      } else {
        reg_coeff *= kRegularizationScale;
        delta_V.first = 0.0f;
        delta_V.second = 0.0f;
      }
    }
    // track.OutputPassingTime("Backward Pass");

    float alpha = 1.0f;
    bool is_line_search_done = false;
    int line_search_iter = 0;
    // TODO: Parellel line search.
    // Update: It is a pity that it is not worthwhile to parallelize the line
    // search, too. The overhead of creating threads is too high.
    // track.SetStartTime();
    const std::vector<Eigen::Matrix<float, ALL_DIM, 1>> cur_xu_vecs(xu_vecs);
    while (!is_line_search_done && line_search_iter < kMaxLineSearchIter) {
      ++line_search_iter;
      // Forward Pass.
      float next_cost_sum = 0.0f;
      float state_cost = 0.0f;
      float time_cost = 0.0f;
      float smooth_cost = 0.0f;
      for (int k = 0; k < num_steps - 1; ++k) {
        const Eigen::Matrix<float, STATE_DIM, 1> x =
            cur_xu_vecs[k].block<STATE_DIM, 1>(0, 0);
        const Eigen::Matrix<float, CONTROL_DIM, 1> u =
            cur_xu_vecs[k].block<CONTROL_DIM, 1>(STATE_DIM, 0);
        xu_vecs[k].block<CONTROL_DIM, 1>(STATE_DIM, 0) =
            K_mats[k] * (x_hat_vecs[k] - x) + alpha * k_vecs[k] + u;
        // Naive clamp.
        xu_vecs[k](18) = std::max(xu_vecs[k](18), kMinTimeStep);
        xu_vecs[k].block<STATE_DIM, 1>(0, 0) = x_hat_vecs[k];
        x_hat_vecs[k + 1] = GetRealTransition(xu_vecs[k]);
        // Calculate the new cost.
        if (k == 0) {
          const RealCostOutput cost_output = GetTrajRealCost(
              xu_vecs[k], bubbles[k], radius[k], bubbles[k], radius[k],
              kMaxVelocity, kMaxAcceleration, coeff[k]);
          next_cost_sum += cost_output.total_cost;
          state_cost += cost_output.state_cost;
          time_cost += cost_output.time_cost;
          smooth_cost += cost_output.smooth_cost;
        } else {
          const RealCostOutput cost_output = GetTrajRealCost(
              xu_vecs[k], bubbles[k - 1], radius[k - 1], bubbles[k], radius[k],
              kMaxVelocity, kMaxAcceleration, coeff[k]);
          next_cost_sum += cost_output.total_cost;
          state_cost += cost_output.state_cost;
          time_cost += cost_output.time_cost;
          smooth_cost += cost_output.smooth_cost;
        }
      }
      xu_vecs[num_steps - 1].block<STATE_DIM, 1>(0, 0) =
          x_hat_vecs[num_steps - 1];
      next_cost_sum += GetTrajRealTermCost(xu_vecs[num_steps - 1], goal);
      // Check if J satisfy line search condition.
      const float ratio_decrease =
          (next_cost_sum - cost_sum) /
          (alpha * (delta_V.first + alpha * delta_V.second));
      if (ratio_decrease <= 1e-4 || ratio_decrease >= 10.0f) {
        alpha *= 0.5f;
      } else {
        is_line_search_done = true;
        cost_sum = next_cost_sum;
        std::cout << " state cost: " << state_cost << std::endl
                  << " time cost: " << time_cost << std::endl
                  << " smooth cost: " << smooth_cost << std::endl
                  << "";
        // Decrease the regularization coefficient.
        reg_coeff /= kRegularizationScale;
      }
    }
    // track.OutputPassingTime("Forward Pass");
    if (!is_line_search_done) {
      // Increase the regularization coefficient.
      xu_vecs = cur_xu_vecs;
      reg_coeff *= kRegularizationScale;
      std::cout << "Line search failed !" << std::endl;
      continue;
    }

    // Calculate the path length.
    path_length = 0.0f;
    time_sum = 0.0f;
    for (int k = 0; k < num_steps - 1; ++k) {
      const float dx = xu_vecs[k + 1](0) - xu_vecs[k](0);
      const float dy = xu_vecs[k + 1](3) - xu_vecs[k](3);
      const float dz = xu_vecs[k + 1](6) - xu_vecs[k](6);
      path_length += std::hypot(dx, dy, dz);
      time_sum += xu_vecs[k](18);
    }

    // Terminate condition.
    if (iter > 0 &&
        std::fabs(cost_sum - last_cost_sum) < kTrajConvergenceThreshold) {
      std::cout << "Convergence reached ! iter: " << iter
                << " line search iter: " << line_search_iter
                << " cost: " << cost_sum << " path_length: " << path_length
                << " time_sum: " << time_sum << std::endl;
      ilqr_traj.num_iter = iter;
      ilqr_traj.traj_length = path_length;
      ilqr_traj.total_time = time_sum;
      break;
    } else {
      last_path_length = path_length;
      last_cost_sum = cost_sum;
      last_time_sum = time_sum;
      std::cout << "iter: " << iter << " line search iter: " << line_search_iter
                << " cost: " << cost_sum << " path_length: " << path_length
                << " time_sum: " << time_sum << std::endl;
    }
  }

  // Output the trajectory.
  std::cout << std::fixed << std::setprecision(4);
  for (int i = 0; i < num_steps; ++i) {
    for (int j = 0; j < xu_vecs[i].rows(); ++j) {
      std::cout << std::setw(10) << xu_vecs[i](j); // 设置宽度为10
    }
    std::cout << std::endl;
  }
  ilqr_traj.traj = std::move(xu_vecs);
  return ilqr_traj;
}

Eigen::Matrix<float, STATE_DIM, ALL_DIM>
DynamicVoronoi3D::GetTransition(const Eigen::Matrix<float, ALL_DIM, 1> &xu) {
  Eigen::Matrix<float, STATE_DIM, ALL_DIM> F =
      Eigen::Matrix<float, STATE_DIM, ALL_DIM>::Zero();

  F.block<STATE_DIM, STATE_DIM>(0, 0) =
      Eigen::Matrix<float, STATE_DIM, STATE_DIM>::Identity();
  F.block<STATE_DIM, CONTROL_DIM - 1>(0, STATE_DIM) =
      Eigen::Matrix<float, STATE_DIM, CONTROL_DIM - 1>::Identity();
  return F;
}

Eigen::Matrix<float, STATE_DIM, 1> DynamicVoronoi3D::GetRealTransition(
    const Eigen::Matrix<float, ALL_DIM, 1> &xu) {
  const float px = xu(0);
  const float vx = xu(1);
  const float ax = xu(2);

  const float py = xu(3);
  const float vy = xu(4);
  const float ay = xu(5);

  const float pz = xu(6);
  const float vz = xu(7);
  const float az = xu(8);

  const float dpx = xu(9);
  const float dvx = xu(10);
  const float dax = xu(11);

  const float dpy = xu(12);
  const float dvy = xu(13);
  const float day = xu(14);

  const float dpz = xu(15);
  const float dvz = xu(16);
  const float daz = xu(17);

  const float dt = xu(18);

  // std::cout << "px: " << px << " py: " << py << " pz: " << pz << std::endl;
  // std::cout << "vx: " << vx << " vy: " << vy << " vz: " << vz << std::endl;
  // std::cout << "ax: " << ax << " ay: " << ay << " az: " << az << std::endl;
  // std::cout << "jx: " << jx << " jy: " << jy << " jz: " << jz << std::endl;
  // std::cout << "sx: " << sx << " sy: " << sy << " sz: " << sz << std::endl;
  // std::cout << "dt: " << dt << std::endl;

  Eigen::Matrix<float, STATE_DIM, 1> next_xu;
  // clang-format off
  next_xu <<
  px + dpx,
  vx + dvx,
  ax + dax,
  py + dpy,
  vy + dvy,
  ay + day,
  pz + dpz,
  vz + dvz,
  az + daz;
  // clang-format on
  return next_xu;
}

std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
          Eigen::Matrix<float, ALL_DIM, 1>>
DynamicVoronoi3D::GetTrajCost(const Eigen::Matrix<float, ALL_DIM, 1> &xu,
                              const IntPoint3D &bubble_1, const float radius_1,
                              const IntPoint3D &bubble_2, const float radius_2,
                              const float max_vel, const float max_acc,
                              const float coeff) {
  const float px = xu(0);
  const float vx = xu(1);
  const float ax = xu(2);

  const float py = xu(3);
  const float vy = xu(4);
  const float ay = xu(5);

  const float pz = xu(6);
  const float vz = xu(7);
  const float az = xu(8);

  const float dpx = xu(9);
  const float dvx = xu(10);
  const float dax = xu(11);

  const float dpy = xu(12);
  const float dvy = xu(13);
  const float day = xu(14);

  const float dpz = xu(15);
  const float dvz = xu(16);
  const float daz = xu(17);

  const float dt = xu(18);
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;
  const float dt_5 = dt_4 * dt;
  const float dt_6 = dt_5 * dt;
  const float dt_7 = dt_6 * dt;

  float dc_px = 0.0;
  float dc_py = 0.0;
  float dc_pz = 0.0;
  float ddc_px = 0.0;
  float ddc_py = 0.0;
  float ddc_pz = 0.0;
  float ddc_pxpy = 0.0;
  float ddc_pxpz = 0.0;
  float ddc_pypz = 0.0;

  float dc_vx = 0.0;
  float dc_vy = 0.0;
  float dc_vz = 0.0;
  float ddc_vx = 0.0;
  float ddc_vy = 0.0;
  float ddc_vz = 0.0;

  float dc_ax = 0.0;
  float dc_ay = 0.0;
  float dc_az = 0.0;
  float ddc_ax = 0.0;
  float ddc_ay = 0.0;
  float ddc_az = 0.0;

  // State constraints.
  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
      state_cost;
  // Constraints of bubble 1.
  const int dx_1 = px - bubble_1.x;
  const int dy_1 = py - bubble_1.y;
  const int dz_1 = pz - bubble_1.z;
  if (dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1 > radius_1 * radius_1) {
    const float rho = std::hypot(dx_1, dy_1, dz_1);
    const float rho_2 = rho * rho;
    const float rho_3 = rho_2 * rho;
    dc_px += kTrajBndWeight * dx_1 * (1.0f - radius_1 / rho);
    dc_py += kTrajBndWeight * dy_1 * (1.0f - radius_1 / rho);
    dc_pz += kTrajBndWeight * dz_1 * (1.0f - radius_1 / rho);
    ddc_px +=
        kTrajBndWeight * (1.0f + (dx_1 * dx_1 - rho_2) * radius_1 / rho_3);
    ddc_py +=
        kTrajBndWeight * (1.0f + (dy_1 * dy_1 - rho_2) * radius_1 / rho_3);
    ddc_pz +=
        kTrajBndWeight * (1.0f + (dz_1 * dz_1 - rho_2) * radius_1 / rho_3);
    ddc_pxpy += kTrajBndWeight * (radius_1 * dx_1 * dy_1 / rho_3);
    ddc_pxpz += kTrajBndWeight * (radius_1 * dx_1 * dz_1 / rho_3);
    ddc_pypz += kTrajBndWeight * (radius_1 * dy_1 * dz_1 / rho_3);
  }
  // Constraints of bubble 2.
  const int dx_2 = px - bubble_2.x;
  const int dy_2 = py - bubble_2.y;
  const int dz_2 = pz - bubble_2.z;
  if (dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2 > radius_2 * radius_2) {
    const float rho = std::hypot(dx_2, dy_2, dz_2);
    const float rho_2 = rho * rho;
    const float rho_3 = rho_2 * rho;
    dc_px += kTrajBndWeight * dx_2 * (1.0f - radius_2 / rho);
    dc_py += kTrajBndWeight * dy_2 * (1.0f - radius_2 / rho);
    dc_pz += kTrajBndWeight * dz_2 * (1.0f - radius_2 / rho);
    ddc_px +=
        kTrajBndWeight * (1.0f + (dx_2 * dx_2 - rho_2) * radius_2 / rho_3);
    ddc_py +=
        kTrajBndWeight * (1.0f + (dy_2 * dy_2 - rho_2) * radius_2 / rho_3);
    ddc_pz +=
        kTrajBndWeight * (1.0f + (dz_2 * dz_2 - rho_2) * radius_2 / rho_3);
    ddc_pxpy += kTrajBndWeight * (radius_2 * dx_2 * dy_2 / rho_3);
    ddc_pxpz += kTrajBndWeight * (radius_2 * dx_2 * dz_2 / rho_3);
    ddc_pypz += kTrajBndWeight * (radius_2 * dy_2 * dz_2 / rho_3);
  }
  // Velocity constraints.
  if (vx > max_vel) {
    dc_vx += kTrajBndWeight * (vx - max_vel);
    ddc_vx += kTrajBndWeight;
  } else if (vx < -max_vel) {
    dc_vx += kTrajBndWeight * (vx + max_vel);
    ddc_vx += kTrajBndWeight;
  }
  if (vy > max_vel) {
    dc_vy += kTrajBndWeight * (vy - max_vel);
    ddc_vy += kTrajBndWeight;
  } else if (vy < -max_vel) {
    dc_vy += kTrajBndWeight * (vy + max_vel);
    ddc_vy += kTrajBndWeight;
  }
  if (vz > max_vel) {
    dc_vz += kTrajBndWeight * (vz - max_vel);
    ddc_vz += kTrajBndWeight;
  } else if (vz < -max_vel) {
    dc_vz += kTrajBndWeight * (vz + max_vel);
    ddc_vz += kTrajBndWeight;
  }
  // Acceleration constraints.
  if (ax > max_acc) {
    dc_ax += kTrajBndWeight * (ax - max_acc);
    ddc_ax += kTrajBndWeight;
  } else if (ax < -max_acc) {
    dc_ax += kTrajBndWeight * (ax + max_acc);
    ddc_ax += kTrajBndWeight;
  }
  if (ay > max_acc) {
    dc_ay += kTrajBndWeight * (ay - max_acc);
    ddc_ay += kTrajBndWeight;
  } else if (ay < -max_acc) {
    dc_ay += kTrajBndWeight * (ay + max_acc);
    ddc_ay += kTrajBndWeight;
  }
  if (az > max_acc) {
    dc_az += kTrajBndWeight * (az - max_acc);
    ddc_az += kTrajBndWeight;
  } else if (az < -max_acc) {
    dc_az += kTrajBndWeight * (az + max_acc);
    ddc_az += kTrajBndWeight;
  }
  // clang-format off
  state_cost.first = Eigen::Matrix<float, ALL_DIM, ALL_DIM>::Zero();
  state_cost.first.block<STATE_DIM, STATE_DIM>(0, 0) <<
  ddc_px,   0.0f,   0.0f, ddc_pxpy,   0.0f,   0.0f, ddc_pxpz,   0.0f,   0.0f,
  0.0f,   ddc_vx,   0.0f,     0.0f,   0.0f,   0.0f,     0.0f,   0.0f,   0.0f,
  0.0f,     0.0f, ddc_ax,     0.0f,   0.0f,   0.0f,     0.0f,   0.0f,   0.0f,
  ddc_pxpy, 0.0f,   0.0f,   ddc_py,   0.0f,   0.0f, ddc_pypz,   0.0f,   0.0f,
  0.0f,     0.0f,   0.0f,     0.0f, ddc_vy,   0.0f,     0.0f,   0.0f,   0.0f,
  0.0f,     0.0f,   0.0f,     0.0f,   0.0f, ddc_ay,     0.0f,   0.0f,   0.0f,
  ddc_pxpz, 0.0f,   0.0f, ddc_pypz,   0.0f,   0.0f,   ddc_pz,   0.0f,   0.0f,
  0.0f,     0.0f,   0.0f,     0.0f,   0.0f,   0.0f,     0.0f, ddc_vz,   0.0f,
  0.0f,     0.0f,   0.0f,     0.0f,   0.0f,   0.0f,     0.0f,   0.0f, ddc_az;
  state_cost.second <<
  dc_px, dc_vx, dc_ax,
  dc_py, dc_vy, dc_ay,
  dc_pz, dc_vz, dc_az,
  0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f,
  0.0f;
  // clang-format on

  // Time cost.
  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
      time_cost;
  // Time cost.
  float dc_dt = kTimeWeight * (dt - kMinTimeStep);
  float ddc_dt = kTimeWeight;
  // Time constraint.
  // if (dt < kMinTimeStep) {
  //   dc_dt += kTimeBndWeight * (dt - kMinTimeStep);
  //   ddc_dt += kTimeBndWeight;
  // }
  time_cost.first = Eigen::Matrix<float, ALL_DIM, ALL_DIM>::Zero();
  time_cost.first(ALL_DIM - 1, ALL_DIM - 1) = ddc_dt;
  time_cost.second = Eigen::Matrix<float, ALL_DIM, 1>::Zero();
  time_cost.second(ALL_DIM - 1) = dc_dt;

  // Smoothness cost.
  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
      smooth_cost;
  Eigen::Vector3f a, da, v, dv, dp;
  a << ax, ay, az;
  da << dax, day, daz;
  v << vx, vy, vz;
  dv << dvx, dvy, dvz;
  dp << dpx, dpy, dpz;
  // clang-format off
  const float ddst_px = 0.0f;
  const float ddst_vx = (120.0f * (24.0f * dpx - 9.0f * dvx * dt - 18.0f * vx * dt + dax * dt_2)) / dt_5;
  const float ddst_ax = -(6.0f * (2.0f * ax * dt - 4.0f * dvx + dax * dt)) / dt_3;
  const float ddst_py = 0.0f;
  const float ddst_vy = (120.0f * (24.0f * dpy - 9.0f * dvy * dt - 18.0f * vy * dt + day * dt_2)) / dt_5;
  const float ddst_ay = -(6.0f * (2.0f * ay * dt - 4.0f * dvy + day * dt)) / dt_3;
  const float ddst_pz = 0.0f;
  const float ddst_vz = (120.0f * (24.0f * dpz - 9.0f * dvz * dt - 18.0f * vz * dt + daz * dt_2)) / dt_5; 
  const float ddst_az = -(6.0f * (2.0f * az * dt - 4.0f * dvz + daz * dt)) / dt_3;

  const float ddst_dpx = -(180.f * (20.0f * dpx - 8.0f * dvx * dt - 16.0f * vx * dt + dax * dt_2)) / dt_6;
  const float ddst_dvx = (24.0f * (60.0f * dpx - 24.0f * dvx * dt - 45.0f * vx * dt + ax * dt_2 + 3.0f * dax * dt_2)) / dt_5;
  const float ddst_dax = -(3.0f * (60.0f * dpx - 24.0f * dvx * dt - 40.0f * vx * dt + 2.0f * ax * dt_2 + 3.0f * dax * dt_2)) / dt_4;
  const float ddst_dpy = -(180.f * (20.0f * dpy - 8.0f * dvy * dt - 16.0f * vy * dt + day * dt_2)) / dt_6;
  const float ddst_dvy = (24.0f * (60.0f * dpy - 24.0f * dvy * dt - 45.0f * vy * dt + ay * dt_2 + 3.0f * day * dt_2)) / dt_5;
  const float ddst_day = -(3.0f * (60.0f * dpy - 24.0f * dvy * dt - 40.0f * vy * dt + 2.0f * ay * dt_2 + 3.0f * day * dt_2)) / dt_4;
  const float ddst_dpz = -(180.f * (20.0f * dpz - 8.0f * dvz * dt - 16.0f * vz * dt + daz * dt_2)) / dt_6;
  const float ddst_dvz = (24.0f * (60.0f * dpz - 24.0f * dvz * dt - 45.0f * vz * dt + az * dt_2 + 3.0f * daz * dt_2)) / dt_5;
  const float ddst_daz = -(3.0f * (60.0f * dpz - 24.0f * dvz * dt - 40.0f * vz * dt + 2.0f * az * dt_2 + 3.0f * daz * dt_2)) / dt_4;
  const float ddst_t = (3.0f / dt_7) * 
  ((4.0f * dt_4)  * a.transpose() * a +
  (4.0f * dt_4) * a.transpose() * da +
  (-24.0f * dt_3) * a.transpose() * dv +
  (3.0f * dt_4) * da.transpose() * da +
  (240.0f * dt_2) * da.transpose() * dp +
  (-72.0f * dt_3) * da.transpose() * dv +
  (-120.0f * dt_3) * da.transpose() * v +
  (3600.0f) * dp.transpose() * dp +
  (-2400.0f * dt) * dp.transpose() * dv +
  (-4800.0f * dt) * dp.transpose() * v +
  (384.0f * dt_2) * dv.transpose() * dv +
  (1440.0f * dt_2) * dv.transpose() * v +
  (1440.0f * dt_2) * v.transpose() * v)(0, 0);
  Eigen::Matrix3f dds_xx;
  dds_xx << 
  0.0f,          0.0f,       0.0f, 
  0.0f, 720.0f / dt_3,       0.0f, 
  0.0f,          0.0f, 12.0f / dt;
  Eigen::Matrix3f dds_uu;
  dds_uu << 
  720.0f / dt_5, -360.0f / dt_4,  60.0f / dt_3,
  -360.0f / dt_4, 192.0f / dt_3, -36.0f / dt_2,
  60.0f /dt_3,     -36.0f /dt_2,     9.0f / dt;
  Eigen::Matrix3f dds_xu;
  dds_xu << 
  0.0f,                    0.0f,          0.0f, 
  -720.0f / dt_4, 360.0f / dt_3, -60.0f / dt_2,
  0.0f,           -12.0f / dt_2,     6.0f / dt;
  smooth_cost.first = Eigen::Matrix<float, ALL_DIM, ALL_DIM>::Zero();
  smooth_cost.first.block<3, 3>(0, 0) = dds_xx;
  smooth_cost.first.block<3, 3>(3, 3) = dds_xx;
  smooth_cost.first.block<3, 3>(6, 6) = dds_xx;

  smooth_cost.first.block<3, 3>(9, 9) = dds_uu;
  smooth_cost.first.block<3, 3>(12, 12) = dds_uu;
  smooth_cost.first.block<3, 3>(15, 15) = dds_uu;

  smooth_cost.first(ALL_DIM - 1, ALL_DIM - 1) = ddst_t;

  smooth_cost.first.block<3, 3>(0, 9) = dds_xu;
  smooth_cost.first.block<3, 3>(3, 12) = dds_xu;
  smooth_cost.first.block<3, 3>(6, 15) = dds_xu;

  smooth_cost.first.block<CONTROL_DIM - 1, STATE_DIM>(STATE_DIM, 0) = 
    smooth_cost.first.block<STATE_DIM, CONTROL_DIM - 1>(0, STATE_DIM).transpose();

  smooth_cost.first.block<ALL_DIM - 1, 1>(0, ALL_DIM - 1) <<
  ddst_px, ddst_vx, ddst_ax,
  ddst_py, ddst_vy, ddst_ay,
  ddst_pz, ddst_vz, ddst_az,
  ddst_dpx, ddst_dvx, ddst_dax,
  ddst_dpy, ddst_dvy, ddst_day,
  ddst_dpz, ddst_dvz, ddst_daz;

  smooth_cost.first.block<1, ALL_DIM - 1>(ALL_DIM - 1, 0) = 
    smooth_cost.first.block<ALL_DIM - 1, 1>(0, ALL_DIM - 1).transpose();

  smooth_cost.second <<
  // Partial derivative of State.
  0.0f,
  -(60.0f * (12.0f * dpx - 6.0f * dvx * dt - 12.0f * vx * dt + dax * dt_2)) / dt_4,
  (6.0f * (2.0f * ax * dt - 2.0f * dvx + dax * dt)) / dt_2, 
  0.0f,
  -(60.0f * (12.0f * dpy - 6.0f * dvy * dt - 12.0f * vy * dt + day * dt_2)) / dt_4,
  (6.0f * (2.0f * ay * dt - 2.0f * dvy + day * dt)) / dt_2, 
  0.0f,
  -(60.0f * (12.0f * dpz - 6.0f * dvz * dt - 12.0f * vz * dt + daz * dt_2)) / dt_4,
  (6.0f * (2.0f * az * dt - 2.0f * dvz + daz * dt)) / dt_2,
  // Partial derivative delta state.
  (60.0f * (12.0f * dpx - 6.0f * dvx * dt - 12.0f * vx * dt + dax * dt_2)) / dt_5,
  -(12.0f * (30.0f * dpx - 16.0f * dvx * dt - 30.0f * vx * dt + ax * dt_2 + 3.0f * dax * dt_2)) / dt_4,
  (3.0f * (20.0f * dpx - 12.0f * dvx * dt - 20.0f * vx * dt + 2.0f * ax * dt_2 + 3.0f * dax * dt_2)) / dt_3,
  (60.0f * (12.0f * dpy - 6.0f * dvy * dt - 12.0f * vy * dt + day * dt_2)) / dt_5,
  -(12.0f * (30.0f * dpy - 16.0f * dvy * dt - 30.0f * vy * dt + ay * dt_2 + 3.0f * day * dt_2)) / dt_4,
  (3.0f * (20.0f * dpy - 12.0f * dvy * dt - 20.0f * vy * dt + 2.0f * ay * dt_2 + 3.0f * day * dt_2)) / dt_3,
  (60.0f * (12.0f * dpz - 6.0f * dvz * dt - 12.0f * vz * dt + daz * dt_2)) / dt_5,
  -(12.0f * (30.0f * dpz - 16.0f * dvz * dt - 30.0f * vz * dt + az * dt_2 + 3.0f * daz * dt_2)) / dt_4,
  (3.0f * (20.0f * dpz - 12.0f * dvz * dt - 20.0f * vz * dt + 2.0f * az * dt_2 + 3.0f * daz * dt_2)) / dt_3,
  // Partial derivative of time.
  (-3.0f) / (2.0f * dt_6) * 
  (4.0f * dt_4 * a.transpose() * a +
  4.0f * dt_4 * a.transpose() * da +
  -16.0f * dt_3 * a.transpose() * dv +
  3.0f * dt_4 * da.transpose() * da +
  120.0f * dt_2 * da.transpose() * dp +
  -48.0f * dt_3 * da.transpose() * dv +
  -80.0f * dt_3 * da.transpose() * v +
  1200.0f * dp.transpose() * dp +
  -960.0f * dt * dp.transpose() * dv +
  -1920.0f * dt * dp.transpose() * v +
  192.0f * dt_2 * dv.transpose() * dv +
  720.0f * dt_2 * dv.transpose() * v +
  720.0f * dt_2 * v.transpose() * v);
  // clang-format on
  // Results.
  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
      cost;
  cost.first =
      state_cost.first + time_cost.first + kSmoothWeight * smooth_cost.first;
  cost.second =
      state_cost.second + time_cost.second + kSmoothWeight * smooth_cost.second;
  return cost;
}

RealCostOutput DynamicVoronoi3D::GetTrajRealCost(
    const Eigen::Matrix<float, ALL_DIM, 1> &xu, const IntPoint3D &bubble_1,
    const float radius_1, const IntPoint3D &bubble_2, const float radius_2,
    const float max_vel, const float max_acc, const float coeff) {
  float state_cost = 0.0f;
  float time_cost = 0.0f;
  float smooth_cost = 0.0f;

  const float px = xu(0);
  const float vx = xu(1);
  const float ax = xu(2);

  const float py = xu(3);
  const float vy = xu(4);
  const float ay = xu(5);

  const float pz = xu(6);
  const float vz = xu(7);
  const float az = xu(8);

  const float dpx = xu(9);
  const float dvx = xu(10);
  const float dax = xu(11);

  const float dpy = xu(12);
  const float dvy = xu(13);
  const float day = xu(14);

  const float dpz = xu(15);
  const float dvz = xu(16);
  const float daz = xu(17);

  const float dt = xu(18);
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;
  const float dt_5 = dt_4 * dt;

  // Constraints of bubble 1.
  const int dx_1 = px - bubble_1.x;
  const int dy_1 = py - bubble_1.y;
  const int dz_1 = pz - bubble_1.z;
  if (dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1 > radius_1 * radius_1) {
    const float rho = std::sqrt(dx_1 * dx_1 + dy_1 * dy_1 + dz_1 * dz_1);
    state_cost += kTrajBndWeight * 0.5f * (rho - radius_1) * (rho - radius_1);
  }
  // Constraints of bubble 2.
  const int dx_2 = px - bubble_2.x;
  const int dy_2 = py - bubble_2.y;
  const int dz_2 = pz - bubble_2.z;
  if (dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2 > radius_2 * radius_2) {
    const float rho = std::sqrt(dx_2 * dx_2 + dy_2 * dy_2 + dz_2 * dz_2);
    state_cost += kTrajBndWeight * 0.5f * (rho - radius_2) * (rho - radius_2);
  }
  // Velocity constraints.
  if (vx > max_vel) {
    state_cost += kTrajBndWeight * 0.5f * (vx - max_vel) * (vx - max_vel);
  } else if (vx < -max_vel) {
    state_cost += kTrajBndWeight * 0.5f * (vx + max_vel) * (vx + max_vel);
  }
  if (vy > max_vel) {
    state_cost += kTrajBndWeight * 0.5f * (vy - max_vel) * (vy - max_vel);
  } else if (vy < -max_vel) {
    state_cost += kTrajBndWeight * 0.5f * (vy + max_vel) * (vy + max_vel);
  }
  if (vz > max_vel) {
    state_cost += kTrajBndWeight * 0.5f * (vz - max_vel) * (vz - max_vel);
  } else if (vz < -max_vel) {
    state_cost += kTrajBndWeight * 0.5f * (vz + max_vel) * (vz + max_vel);
  }
  // Acceleration constraints.
  if (ax > max_acc) {
    state_cost += kTrajBndWeight * 0.5f * (ax - max_acc) * (ax - max_acc);
  } else if (ax < -max_acc) {
    state_cost += kTrajBndWeight * 0.5f * (ax + max_acc) * (ax + max_acc);
  }
  if (ay > max_acc) {
    state_cost += kTrajBndWeight * 0.5f * (ay - max_acc) * (ay - max_acc);
  } else if (ay < -max_acc) {
    state_cost += kTrajBndWeight * 0.5f * (ay + max_acc) * (ay + max_acc);
  }
  if (az > max_acc) {
    state_cost += kTrajBndWeight * 0.5f * (az - max_acc) * (az - max_acc);
  } else if (az < -max_acc) {
    state_cost += kTrajBndWeight * 0.5f * (az + max_acc) * (az + max_acc);
  }

  // Time constraint.
  // if (dt < kMinTimeStep) {
  //   time_cost +=
  //       kTimeBndWeight * 0.5f * (dt - kMinTimeStep) * (dt - kMinTimeStep);
  // }
  // Time cost.
  time_cost += kTimeWeight * 0.5f * (dt - kMinTimeStep) * (dt - kMinTimeStep);

  // Smoothness cost.
  Eigen::Matrix3f A_inv, B;
  // clang-format off
  A_inv << 
  10.0f / dt_3, -4.0f / dt_2, 1.0f / (2.0f * dt), 
  -15.0f / dt_4, 7.0f / dt_3, -1.0f / dt_2,
  6.0f / dt_5, -3.0f / dt_4, 1.0f / (2.0f * dt_3);
  B <<
  0.0f,   dt, 0.5 * dt_2,
  0.0f, 0.0f,         dt,
  0.0f, 0.0f,       0.0f;
  // clang-format on
  Eigen::Vector3f state, dstate;
  Eigen::Vector3f coeff345;
  state << px, vx, ax;
  dstate << dpx, dvx, dax;
  coeff345 = A_inv * (dstate - B * state);
  const float Jx = GetSmoothCost(coeff345, dt);
  state << py, vy, ay;
  dstate << dpy, dvy, day;
  coeff345 = A_inv * (dstate - B * state);
  const float Jy = GetSmoothCost(coeff345, dt);
  state << pz, vz, az;
  dstate << dpz, dvz, daz;
  coeff345 = A_inv * (dstate - B * state);
  const float Jz = GetSmoothCost(coeff345, dt);
  smooth_cost += kSmoothWeight * (Jx + Jy + Jz);

  return RealCostOutput{
      .total_cost = state_cost + time_cost + smooth_cost,
      .state_cost = state_cost,
      .time_cost = time_cost,
      .smooth_cost = smooth_cost,
  };
}

float DynamicVoronoi3D::GetSmoothCost(const Eigen::Vector3f &coeff,
                                      const float dt) {
  const float a3 = coeff(0);
  const float a4 = coeff(1);
  const float a5 = coeff(2);
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;
  const float dt_5 = dt_4 * dt;
  const float J = 18.0f * a3 * a3 * dt +
                  dt_3 * (96.0f * a4 * a4 + 120.0f * a3 * a5) +
                  360.0f * a5 * a5 * dt_5 + 72.0f * a3 * a4 * dt_2 +
                  360.0f * a4 * a5 * dt_4;
  return J;
}

std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
          Eigen::Matrix<float, ALL_DIM, 1>>
DynamicVoronoi3D::GetTrajTermCost(const Eigen::Matrix<float, ALL_DIM, 1> &xu,
                                  const IntPoint3D &goal) {
  const float px = xu(0);
  const float vx = xu(1);
  const float ax = xu(2);

  const float py = xu(3);
  const float vy = xu(4);
  const float ay = xu(5);

  const float pz = xu(6);
  const float vz = xu(7);
  const float az = xu(8);

  std::pair<Eigen::Matrix<float, ALL_DIM, ALL_DIM>,
            Eigen::Matrix<float, ALL_DIM, 1>>
      cost;
  // clang-format off
  cost.first = Eigen::Matrix<float, ALL_DIM, ALL_DIM>::Zero();
  cost.first.block<STATE_DIM, STATE_DIM>(0, 0).diagonal().setConstant(kTrajTermWeight);
  cost.second <<
  kTrajTermWeight * (px - goal.x),
  kTrajTermWeight * vx,
  kTrajTermWeight * ax,
  kTrajTermWeight * (py - goal.y),
  kTrajTermWeight * vy,
  kTrajTermWeight * ay,
  kTrajTermWeight * (pz - goal.z),
  kTrajTermWeight * vz,
  kTrajTermWeight * az,
  0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 0.0f,
  0.0f;
  // clang-format on
  return cost;
}

float DynamicVoronoi3D::GetTrajRealTermCost(
    const Eigen::Matrix<float, ALL_DIM, 1> &xu, const IntPoint3D &goal) {
  const float px = xu(0);
  const float vx = xu(1);
  const float ax = xu(2);

  const float py = xu(3);
  const float vy = xu(4);
  const float ay = xu(5);

  const float pz = xu(6);
  const float vz = xu(7);
  const float az = xu(8);

  float real_cost = 0.0;
  const float dx = px - goal.x;
  const float dy = py - goal.y;
  const float dz = pz - goal.z;
  real_cost +=
      kTrajTermWeight * 0.5f * dx * dx + kTrajTermWeight * 0.5f * vx * vx +
      kTrajTermWeight * 0.5f * ax * ax + kTrajTermWeight * 0.5f * dy * dy +
      kTrajTermWeight * 0.5f * vy * vy + kTrajTermWeight * 0.5f * ay * ay +
      kTrajTermWeight * 0.5f * dz * dz + kTrajTermWeight * 0.5f * vz * vz +
      kTrajTermWeight * 0.5f * az * az;
  // std::cout << "Term cost: " << real_cost << std::endl;
  return real_cost;
}
