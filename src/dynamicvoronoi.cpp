#include "explorer/dynamicvoronoi.h"
#include "explorer/time_track.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>
#include <unordered_map>

namespace {
constexpr int kMaxIteration = 500;
constexpr float kConvergenceThreshold = 0.05f;
constexpr float kTermWeight = 1000.0;
constexpr float kBndWeight = 10000.0;
constexpr float kWeight = 100.0;
constexpr int kMaxLineSearchIter = 10;
constexpr int kDeadEndThreshold = 5;
} // namespace

VGraphNode::VGraphNode(const IntPoint &point) : point_(point) {}

void VGraphNode::RemoveEdge(const int dest_id) {
  edges_.erase(dest_id);
  return;
}

void VGraph::AddOneWayEdge(const IntPoint &src, const IntPoint &dest,
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

void VGraph::AddTwoWayEdge(const IntPoint &src, const IntPoint &dest,
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

bool VGraph::isNodeExist(const IntPoint &point) {
  return node_id_.find(point) != node_id_.end();
}

NodeProperty::NodeProperty(const AstarState state, const float g_score,
                           const float h_score, const int father_id)
    : state_(state), g_score_(g_score), h_score_(h_score),
      father_id_(father_id) {}

QueueNode::QueueNode(const IntPoint &point, const float f_score)
    : point_(point), f_score_(f_score) {}

DynamicVoronoi::DynamicVoronoi() {
  sqrt2 = sqrt(2.0);
  data = NULL;
  gridMap = NULL;
  alternativeDiagram = NULL;
  allocatedGridMap = false;
}

DynamicVoronoi::~DynamicVoronoi() {
  if (data) {
    for (int x = 0; x < sizeX; x++)
      delete[] data[x];
    delete[] data;
  }
  if (allocatedGridMap && gridMap) {
    for (int x = 0; x < sizeX; x++)
      delete[] gridMap[x];
    delete[] gridMap;
  }
}

void DynamicVoronoi::initializeEmpty(int _sizeX, int _sizeY, bool initGridMap) {
  if (data) {
    for (int x = 0; x < sizeX; x++)
      delete[] data[x];
    delete[] data;
    data = NULL;
  }
  if (alternativeDiagram) {
    for (int x = 0; x < sizeX; x++)
      delete[] alternativeDiagram[x];
    delete[] alternativeDiagram;
    alternativeDiagram = NULL;
  }
  if (initGridMap) {
    if (allocatedGridMap && gridMap) {
      for (int x = 0; x < sizeX; x++)
        delete[] gridMap[x];
      delete[] gridMap;
      gridMap = NULL;
      allocatedGridMap = false;
    }
  }

  sizeX = _sizeX;
  sizeY = _sizeY;
  data = new dataCell *[sizeX];
  for (int x = 0; x < sizeX; x++)
    data[x] = new dataCell[sizeY];

  if (initGridMap) {
    gridMap = new bool *[sizeX];
    for (int x = 0; x < sizeX; x++)
      gridMap[x] = new bool[sizeY];
    allocatedGridMap = true;
  }

  dataCell c;
  c.dist = INFINITY;
  c.sqdist = INT_MAX;
  c.obstX = invalidObstData;
  c.obstY = invalidObstData;
  c.voronoi = free;
  c.queueing = fwNotQueued;
  c.needsRaise = false;

  for (int x = 0; x < sizeX; x++)
    for (int y = 0; y < sizeY; y++)
      data[x][y] = c;

  if (initGridMap) {
    for (int x = 0; x < sizeX; x++)
      for (int y = 0; y < sizeY; y++)
        gridMap[x][y] = 0;
  }
}

void DynamicVoronoi::initializeMap(int _sizeX, int _sizeY, bool **_gridMap) {
  gridMap = _gridMap;
  initializeEmpty(_sizeX, _sizeY, false);

  for (int x = 0; x < sizeX; x++) {
    for (int y = 0; y < sizeY; y++) {
      if (gridMap[x][y]) {
        dataCell c = data[x][y];
        if (!isOccupied(x, y, c)) {

          bool isSurrounded = true;
          for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx;
            if (nx <= 0 || nx >= sizeX - 1)
              continue;
            for (int dy = -1; dy <= 1; dy++) {
              if (dx == 0 && dy == 0)
                continue;
              int ny = y + dy;
              if (ny <= 0 || ny >= sizeY - 1)
                continue;

              if (!gridMap[nx][ny]) {
                isSurrounded = false;
                break;
              }
            }
          }
          if (isSurrounded) {
            c.obstX = x;
            c.obstY = y;
            c.sqdist = 0;
            c.dist = 0;
            c.voronoi = occupied;
            c.queueing = fwProcessed;
            data[x][y] = c;
          } else
            setObstacle(x, y);
        }
      }
    }
  }
}

void DynamicVoronoi::occupyCell(int x, int y) {
  gridMap[x][y] = 1;
  setObstacle(x, y);
}
void DynamicVoronoi::clearCell(int x, int y) {
  gridMap[x][y] = 0;
  removeObstacle(x, y);
}

void DynamicVoronoi::setObstacle(int x, int y) {
  dataCell c = data[x][y];
  if (isOccupied(x, y, c))
    return;

  addList.push_back(INTPOINT(x, y));
  c.obstX = x;
  c.obstY = y;
  data[x][y] = c;
}

void DynamicVoronoi::removeObstacle(int x, int y) {
  dataCell c = data[x][y];
  if (isOccupied(x, y, c) == false)
    return;

  removeList.push_back(INTPOINT(x, y));
  c.obstX = invalidObstData;
  c.obstY = invalidObstData;
  c.queueing = bwQueued;
  data[x][y] = c;
}

void DynamicVoronoi::exchangeObstacles(std::vector<INTPOINT> &points) {

  for (unsigned int i = 0; i < lastObstacles.size(); i++) {
    int x = lastObstacles[i].x;
    int y = lastObstacles[i].y;

    bool v = gridMap[x][y];
    if (v)
      continue;
    removeObstacle(x, y);
  }

  lastObstacles.clear();

  for (unsigned int i = 0; i < points.size(); i++) {
    int x = points[i].x;
    int y = points[i].y;
    bool v = gridMap[x][y];
    if (v)
      continue;
    setObstacle(x, y);
    lastObstacles.push_back(points[i]);
  }
}

void DynamicVoronoi::update(bool updateRealDist) {

  commitAndColorize(updateRealDist);

  while (!open.empty()) {
    INTPOINT p = open.pop();
    int x = p.x;
    int y = p.y;
    dataCell c = data[x][y];

    if (c.queueing == fwProcessed)
      continue;

    if (c.needsRaise) {
      // RAISE
      for (int dx = -1; dx <= 1; dx++) {
        int nx = x + dx;
        if (nx <= 0 || nx >= sizeX - 1)
          continue;
        for (int dy = -1; dy <= 1; dy++) {
          if (dx == 0 && dy == 0)
            continue;
          int ny = y + dy;
          if (ny <= 0 || ny >= sizeY - 1)
            continue;
          dataCell nc = data[nx][ny];
          if (nc.obstX != invalidObstData && !nc.needsRaise) {
            if (!isOccupied(nc.obstX, nc.obstY, data[nc.obstX][nc.obstY])) {
              open.push(nc.sqdist, INTPOINT(nx, ny));
              nc.queueing = fwQueued;
              nc.needsRaise = true;
              nc.obstX = invalidObstData;
              nc.obstY = invalidObstData;
              if (updateRealDist)
                nc.dist = INFINITY;
              nc.sqdist = INT_MAX;
              data[nx][ny] = nc;
            } else {
              if (nc.queueing != fwQueued) {
                open.push(nc.sqdist, INTPOINT(nx, ny));
                nc.queueing = fwQueued;
                data[nx][ny] = nc;
              }
            }
          }
        }
      }
      c.needsRaise = false;
      c.queueing = bwProcessed;
      data[x][y] = c;
    } else if (c.obstX != invalidObstData &&
               isOccupied(c.obstX, c.obstY, data[c.obstX][c.obstY])) {

      // LOWER
      c.queueing = fwProcessed;
      c.voronoi = occupied;

      for (int dx = -1; dx <= 1; dx++) {
        int nx = x + dx;
        if (nx <= 0 || nx >= sizeX - 1)
          continue;
        for (int dy = -1; dy <= 1; dy++) {
          if (dx == 0 && dy == 0)
            continue;
          int ny = y + dy;
          if (ny <= 0 || ny >= sizeY - 1)
            continue;
          dataCell nc = data[nx][ny];
          if (!nc.needsRaise) {
            int distx = nx - c.obstX;
            int disty = ny - c.obstY;
            int newSqDistance = distx * distx + disty * disty;
            bool overwrite = (newSqDistance < nc.sqdist);
            if (!overwrite && newSqDistance == nc.sqdist) {
              if (nc.obstX == invalidObstData ||
                  isOccupied(nc.obstX, nc.obstY, data[nc.obstX][nc.obstY]) ==
                      false)
                overwrite = true;
            }
            if (overwrite) {
              open.push(newSqDistance, INTPOINT(nx, ny));
              nc.queueing = fwQueued;
              if (updateRealDist) {
                nc.dist = sqrt((double)newSqDistance);
              }
              nc.sqdist = newSqDistance;
              nc.obstX = c.obstX;
              nc.obstY = c.obstY;
            } else {
              checkVoro(x, y, nx, ny, c, nc);
            }
            data[nx][ny] = nc;
          }
        }
      }
    }
    data[x][y] = c;
  }
}

float DynamicVoronoi::getDistance(int x, int y) const {
  if ((x > 0) && (x < sizeX) && (y > 0) && (y < sizeY))
    return data[x][y].dist;
  else
    return INFINITY;
}

int DynamicVoronoi::getSquaredDistance(int x, int y) const {
  if ((x > 0) && (x < sizeX) && (y > 0) && (y < sizeY))
    return data[x][y].sqdist;
  else
    return INT_MAX;
}

bool DynamicVoronoi::isVoronoi(int x, int y) const {
  dataCell c = data[x][y];
  return (c.voronoi == free || c.voronoi == voronoiKeep);
}

bool DynamicVoronoi::isVoronoiAlternative(int x, int y) {
  int v = alternativeDiagram[x][y];
  return (v == free || v == voronoiKeep);
}

void DynamicVoronoi::commitAndColorize(bool updateRealDist) {
  // ADD NEW OBSTACLES
  for (unsigned int i = 0; i < addList.size(); i++) {
    INTPOINT p = addList[i];
    int x = p.x;
    int y = p.y;
    dataCell c = data[x][y];

    if (c.queueing != fwQueued) {
      if (updateRealDist)
        c.dist = 0;
      c.sqdist = 0;
      c.obstX = x;
      c.obstY = y;
      c.queueing = fwQueued;
      c.voronoi = occupied;
      data[x][y] = c;
      open.push(0, INTPOINT(x, y));
    }
  }

  // REMOVE OLD OBSTACLES
  for (unsigned int i = 0; i < removeList.size(); i++) {
    INTPOINT p = removeList[i];
    int x = p.x;
    int y = p.y;
    dataCell c = data[x][y];

    if (isOccupied(x, y, c) == true)
      continue; // obstacle was removed and reinserted
    open.push(0, INTPOINT(x, y));
    if (updateRealDist)
      c.dist = INFINITY;
    c.sqdist = INT_MAX;
    c.needsRaise = true;
    data[x][y] = c;
  }
  removeList.clear();
  addList.clear();
}

void DynamicVoronoi::checkVoro(int x, int y, int nx, int ny, dataCell &c,
                               dataCell &nc) {

  if ((c.sqdist > 1 || nc.sqdist > 1) && nc.obstX != invalidObstData) {
    if (abs(c.obstX - nc.obstX) > 10 || abs(c.obstY - nc.obstY) > 10) {
      // compute dist from x,y to obstacle of nx,ny
      int dxy_x = x - nc.obstX;
      int dxy_y = y - nc.obstY;
      int sqdxy = dxy_x * dxy_x + dxy_y * dxy_y;
      int stability_xy = sqdxy - c.sqdist;
      if (sqdxy - c.sqdist < 0)
        return;

      // compute dist from nx,ny to obstacle of x,y
      int dnxy_x = nx - c.obstX;
      int dnxy_y = ny - c.obstY;
      int sqdnxy = dnxy_x * dnxy_x + dnxy_y * dnxy_y;
      int stability_nxy = sqdnxy - nc.sqdist;
      if (sqdnxy - nc.sqdist < 0)
        return;

      // which cell is added to the Voronoi diagram?
      if (stability_xy <= stability_nxy && c.sqdist > 2) {
        if (c.voronoi != free) {
          c.voronoi = free;
          reviveVoroNeighbors(x, y);
          pruneQueue.push(INTPOINT(x, y));
        }
      }
      if (stability_nxy <= stability_xy && nc.sqdist > 2) {
        if (nc.voronoi != free) {
          nc.voronoi = free;
          reviveVoroNeighbors(nx, ny);
          pruneQueue.push(INTPOINT(nx, ny));
        }
      }
    }
  }
}

void DynamicVoronoi::reviveVoroNeighbors(int &x, int &y) {
  for (int dx = -1; dx <= 1; dx++) {
    int nx = x + dx;
    if (nx <= 0 || nx >= sizeX - 1)
      continue;
    for (int dy = -1; dy <= 1; dy++) {
      if (dx == 0 && dy == 0)
        continue;
      int ny = y + dy;
      if (ny <= 0 || ny >= sizeY - 1)
        continue;
      dataCell nc = data[nx][ny];
      if (nc.sqdist != INT_MAX && !nc.needsRaise &&
          (nc.voronoi == voronoiKeep || nc.voronoi == voronoiPrune)) {
        nc.voronoi = free;
        data[nx][ny] = nc;
        pruneQueue.push(INTPOINT(nx, ny));
      }
    }
  }
}

bool DynamicVoronoi::isOccupied(int x, int y) const {
  dataCell c = data[x][y];
  return (c.obstX == x && c.obstY == y);
}

bool DynamicVoronoi::isOccupied(int &x, int &y, dataCell &c) {
  return (c.obstX == x && c.obstY == y);
}

void DynamicVoronoi::visualize(const char *filename) {
  // write ppm files

  FILE *F = fopen(filename, "w");
  if (!F) {
    std::cerr << "could not open 'result.pgm' for writing!\n";
    return;
  }
  fprintf(F, "P6\n#\n");
  fprintf(F, "%d %d\n255\n", sizeX, sizeY);

  for (int y = sizeY - 1; y >= 0; y--) {
    for (int x = 0; x < sizeX; x++) {
      unsigned char c = 0;
      if (alternativeDiagram != NULL &&
          (alternativeDiagram[x][y] == free ||
           alternativeDiagram[x][y] == voronoiKeep)) {
        if (getNumVoronoiNeighborsAlternative(x, y) > 2) {
          fputc(0, F);
          fputc(255, F);
          fputc(0, F);
        } else {
          fputc(255, F);
          fputc(0, F);
          fputc(0, F);
        }
      } else if (isVoronoi(x, y)) {
        fputc(0, F);
        fputc(0, F);
        fputc(255, F);
      } else if (data[x][y].sqdist == 0) {
        fputc(0, F);
        fputc(0, F);
        fputc(0, F);
      } else {
        float f = 80 + (sqrt(data[x][y].sqdist) * 10);
        if (f > 255)
          f = 255;
        if (f < 0)
          f = 0;
        c = (unsigned char)f;
        fputc(c, F);
        fputc(c, F);
        fputc(c, F);
      }
    }
  }
  fclose(F);
}

void DynamicVoronoi::prune() {
  // filler
  while (!pruneQueue.empty()) {
    INTPOINT p = pruneQueue.front();
    pruneQueue.pop();
    int x = p.x;
    int y = p.y;

    if (data[x][y].voronoi == occupied)
      continue;
    if (data[x][y].voronoi == freeQueued)
      continue;

    data[x][y].voronoi = freeQueued;
    sortedPruneQueue.push(data[x][y].sqdist, p);

    /* tl t tr
       l c r
       bl b br */

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
        sortedPruneQueue.push(r.sqdist, INTPOINT(x + 1, y));
        data[x + 1][y] = r;
      }
    }
    if (x - 2 >= 0 && l.voronoi == occupied) {
      // fill to the left
      if (tl.voronoi != occupied && bl.voronoi != occupied &&
          data[x - 2][y].voronoi != occupied) {
        l.voronoi = freeQueued;
        sortedPruneQueue.push(l.sqdist, INTPOINT(x - 1, y));
        data[x - 1][y] = l;
      }
    }
    if (y + 2 < sizeY && t.voronoi == occupied) {
      // fill to the top
      if (tr.voronoi != occupied && tl.voronoi != occupied &&
          data[x][y + 2].voronoi != occupied) {
        t.voronoi = freeQueued;
        sortedPruneQueue.push(t.sqdist, INTPOINT(x, y + 1));
        data[x][y + 1] = t;
      }
    }
    if (y - 2 >= 0 && b.voronoi == occupied) {
      // fill to the bottom
      if (br.voronoi != occupied && bl.voronoi != occupied &&
          data[x][y - 2].voronoi != occupied) {
        b.voronoi = freeQueued;
        sortedPruneQueue.push(b.sqdist, INTPOINT(x, y - 1));
        data[x][y - 1] = b;
      }
    }
  }

  while (!sortedPruneQueue.empty()) {
    INTPOINT p = sortedPruneQueue.pop();
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
        INTPOINT p = pruneQueue.front();
        pruneQueue.pop();
        sortedPruneQueue.push(data[p.x][p.y].sqdist, p);
      }
    }
  }
  //  printf("match: %d\nnomat: %d\n", matchCount, noMatchCount);
}

void DynamicVoronoi::updateAlternativePrunedDiagram() {

  if (alternativeDiagram == NULL) {
    alternativeDiagram = new int *[sizeX];
    for (int x = 0; x < sizeX; x++) {
      alternativeDiagram[x] = new int[sizeY];
    }
  }

  std::queue<INTPOINT> end_cells;
  BucketPrioQueue<INTPOINT> sortedPruneQueue;
  for (int x = 1; x < sizeX - 1; x++) {
    for (int y = 1; y < sizeY - 1; y++) {
      dataCell &c = data[x][y];
      alternativeDiagram[x][y] = c.voronoi;
      if (c.voronoi <= free) {
        sortedPruneQueue.push(c.sqdist, INTPOINT(x, y));
        end_cells.push(INTPOINT(x, y));
      }
    }
  }

  for (int x = 1; x < sizeX - 1; x++) {
    for (int y = 1; y < sizeY - 1; y++) {
      if (getNumVoronoiNeighborsAlternative(x, y) >= 3) {
        alternativeDiagram[x][y] = voronoiKeep;
        sortedPruneQueue.push(data[x][y].sqdist, INTPOINT(x, y));
        end_cells.push(INTPOINT(x, y));
      }
    }
  }

  for (int x = 1; x < sizeX - 1; x++) {
    for (int y = 1; y < sizeY - 1; y++) {
      if (getNumVoronoiNeighborsAlternative(x, y) >= 3) {
        alternativeDiagram[x][y] = voronoiKeep;
        sortedPruneQueue.push(data[x][y].sqdist, INTPOINT(x, y));
        end_cells.push(INTPOINT(x, y));
      }
    }
  }

  while (!sortedPruneQueue.empty()) {
    INTPOINT p = sortedPruneQueue.pop();

    if (markerMatchAlternative(p.x, p.y)) {
      alternativeDiagram[p.x][p.y] = voronoiPrune;
    } else {
      alternativeDiagram[p.x][p.y] = voronoiKeep;
    }
  }

  // //delete worms
  while (!end_cells.empty()) {
    INTPOINT p = end_cells.front();
    end_cells.pop();

    if (isVoronoiAlternative(p.x, p.y) &&
        getNumVoronoiNeighborsAlternative(p.x, p.y) == 1) {
      alternativeDiagram[p.x][p.y] = voronoiPrune;

      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (!(dx || dy) || (dx && dy)) {
            continue;
          }
          int nx = p.x + dx;
          int ny = p.y + dy;
          if (nx < 0 || nx >= sizeX || ny < 0 || ny >= sizeY) {
            continue;
          }
          if (isVoronoiAlternative(nx, ny)) {
            if (getNumVoronoiNeighborsAlternative(nx, ny) == 1) {
              end_cells.push(INTPOINT(nx, ny));
            }
          }
        }
      }
    }
  }
}

bool DynamicVoronoi::markerMatchAlternative(int x, int y) {
  // prune if this returns true

  bool f[8];

  int nx, ny;
  int dx, dy;

  int i = 0;
  //  int obstacleCount=0;
  int voroCount = 0;
  for (dy = 1; dy >= -1; dy--) {
    ny = y + dy;
    for (dx = -1; dx <= 1; dx++) {
      if (dx || dy) {
        nx = x + dx;
        int v = alternativeDiagram[nx][ny];
        bool b = (v <= free && v != voronoiPrune);
        //	if (v==occupied) obstacleCount++;
        f[i] = b;
        if (v <= free && !(dx && dy))
          voroCount++;
        i++;
      }
    }
  }

  /*
   * 5 6 7
   * 3   4
   * 0 1 2
   */

  {
    // connected horizontal or vertically to only one cell
    if (voroCount == 1 && (f[1] || f[3] || f[4] || f[6])) {
      return false;
    }

    // 4-connected
    if ((!f[0] && f[1] && f[3]) || (!f[2] && f[1] && f[4]) ||
        (!f[5] && f[3] && f[6]) || (!f[7] && f[6] && f[4]))
      return false;

    if ((f[3] && f[4] && !f[1] && !f[6]) || (f[1] && f[6] && !f[3] && !f[4]))
      return false;
  }
  return true;
}

int DynamicVoronoi::getNumVoronoiNeighborsAlternative(int x, int y) const {
  int count = 0;
  for (int dx = -1; dx <= 1; dx++) {
    for (int dy = -1; dy <= 1; dy++) {
      if ((dx == 0 && dy == 0) || (dx != 0 && dy != 0)) {
        continue;
      }

      int nx = x + dx;
      int ny = y + dy;
      if (nx < 0 || nx >= sizeX || ny < 0 || ny >= sizeY) {
        continue;
      }
      if (alternativeDiagram[nx][ny] == free ||
          alternativeDiagram[nx][ny] == voronoiKeep) {
        count++;
      }
    }
  }
  return count;
}

DynamicVoronoi::markerMatchResult DynamicVoronoi::markerMatch(int x, int y) {
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

const std::vector<IntPoint> neighbor_offsets = {
    {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

std::vector<IntPoint> DynamicVoronoi::GetVoronoiNeighbors(const int x,
                                                          const int y) {
  std::vector<IntPoint> neighbors;
  const int num_neighbors = neighbor_offsets.size();
  for (int i = 0; i < num_neighbors; ++i) {
    const IntPoint &offset = neighbor_offsets[i];
    const int neighbor_x = x + offset.x;
    const int neighbor_y = y + offset.y;
    if (neighbor_x >= 0 && neighbor_x < sizeX && neighbor_y >= 0 &&
        neighbor_y < sizeY) {
      if (isVoronoi(neighbor_x, neighbor_y)) {
        neighbors.emplace_back(neighbor_x, neighbor_y);
      }
    }
  }
  return neighbors;
}

int DynamicVoronoi::GetNumVoronoiNeighbors(const int x, const int y) const {
  int num_voronoi_neighbors = 0;
  const int num_neighbors = neighbor_offsets.size();
  for (int i = 0; i < num_neighbors; ++i) {
    const IntPoint &offset = neighbor_offsets[i];
    const int neighbor_x = x + offset.x;
    const int neighbor_y = y + offset.y;
    if (neighbor_x >= 0 && neighbor_x < sizeX && neighbor_y >= 0 &&
        neighbor_y < sizeY) {
      if (isVoronoi(neighbor_x, neighbor_y)) {
        ++num_voronoi_neighbors;
      }
    }
  }
  return num_voronoi_neighbors;
}

float DynamicVoronoi::GetUnionVolume(const IntPoint &p1,
                                     const IntPoint &p2) const {
  return 0.0;
}

void DynamicVoronoi::ConstructSparseGraph() {
  // 0 represents in the queue, 1 represents visited.
  std::unordered_map<IntPoint, int, IntPointHash> is_visited;
  std::queue<IntPoint> cell_queue;
  // Traverse all cells and add unvisited voronoi cells to the queue.
  for (int x = 0; x < sizeX; ++x) {
    for (int y = 0; y < sizeY; ++y) {
      if (isVoronoi(x, y) &&
          is_visited.find(IntPoint(x, y)) == is_visited.end() &&
          GetNumVoronoiNeighbors(x, y) > 2) {
        cell_queue.emplace(x, y);
        while (!cell_queue.empty()) {
          const IntPoint current_cell = cell_queue.front();
          cell_queue.pop();
          is_visited[current_cell] = true;
          const std::vector<IntPoint> neighbors =
              GetVoronoiNeighbors(current_cell.x, current_cell.y);
          // Each neighbor indicates a direction of the edge.
          for (const IntPoint &neighbor : neighbors) {
            if (is_visited.find(neighbor) == is_visited.end()) {
              bool is_expandable = true;
              IntPoint check = neighbor;
              while (is_expandable) {
                std::vector<IntPoint> check_neighbors =
                    GetVoronoiNeighbors(check.x, check.y);
                // Find the key cell.
                const int num_check_nbrs = check_neighbors.size();
                if (num_check_nbrs > 2) {
                  is_expandable = false;
                } else if (num_check_nbrs == 2) {
                  is_visited[check] = true;
                  bool is_valid = false;
                  for (const IntPoint &check_nbr : check_neighbors) {
                    if (is_visited.find(check_nbr) == is_visited.end()) {
                      is_valid = true;
                      check = check_nbr;
                      break;
                    }
                  }
                  if (is_valid) {
                  } else {
                    is_expandable = false;
                  }
                } else {
                  is_visited[check] = true;
                  is_expandable = false;
                }
              }

              // Add the edge between the key cell and the current cell.
              if (GetNumVoronoiNeighbors(check.x, check.y) > 2 &&
                  is_visited.find(check) == is_visited.end()) {
                const float weight = std::hypot(current_cell.x - check.x,
                                                current_cell.y - check.y);
                cell_queue.emplace(check);
                graph_.AddTwoWayEdge(current_cell, check, weight);
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

void DynamicVoronoi::ConstructSparseGraphBK() {
  std::unordered_map<IntPoint, QueueState, IntPointHash> is_visited;
  std::queue<IntPoint> cell_queue;
  // Traverse all cells and add unvisited voronoi cells to the queue.
  for (int x = 0; x < sizeX; ++x) {
    for (int y = 0; y < sizeY; ++y) {
      if (isVoronoi(x, y) &&
          is_visited.find(IntPoint(x, y)) == is_visited.end()) {
        cell_queue.emplace(x, y);
        while (!cell_queue.empty()) {
          const IntPoint core = cell_queue.front();
          cell_queue.pop();
          is_visited[core] = kCellProcessed;
          const float obstacle_dist = getDistance(core.x, core.y);
          std::queue<IntPoint> bfs_queue;
          bfs_queue.emplace(core);
          while (!bfs_queue.empty()) {
            const IntPoint point = bfs_queue.front();
            bfs_queue.pop();
            is_visited[point] = kProcessed;
            const int p_to_core = GetDistanceBetween(core, point);
            if (p_to_core >= obstacle_dist) {
              // Add the edge.
              cell_queue.emplace(point);
              is_visited[point] = kCellQueue;
              if (getDistance(point.x, point.y) >= kDeadEndThreshold) {
                graph_.AddTwoWayEdge(core, point, p_to_core);
              }
            } else {
              // Expansion.
              const std::vector<IntPoint> nbrs =
                  GetVoronoiNeighbors(point.x, point.y);
              for (const IntPoint &nbr : nbrs) {
                if (is_visited.find(nbr) == is_visited.end()) {
                  bfs_queue.emplace(nbr);
                  is_visited[nbr] = kBfsQueue;
                } else if (is_visited[nbr] == kCellQueue ||
                           is_visited[nbr] == kCellProcessed) {
                  const float nbr_to_core = GetDistanceBetween(core, nbr);
                  if (getDistance(nbr.x, nbr.y) >= kDeadEndThreshold) {
                    graph_.AddTwoWayEdge(core, nbr, nbr_to_core);
                  }
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

std::vector<IntPoint> DynamicVoronoi::GetAstarPath(const IntPoint &start,
                                                   const IntPoint &goal) {
  TimeTrack track;
  // Add the start and goal nodes to the graph.
  const int num_nodes = graph_.nodes_.size();
  for (int i = 0; i < num_nodes; ++i) {
    const IntPoint point = graph_.nodes_[i].point_;
    const int obstacle_distance = getSquaredDistance(point.x, point.y);
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
  std::priority_queue<QueueNode, std::vector<QueueNode>, QueueNodeCmp> astar_q;
  std::unordered_map<IntPoint, NodeProperty, IntPointHash> node_properties;
  if (graph_.node_id_.find(start) == graph_.node_id_.end()) {
    std::cout << "Start node not found in graph !" << std::endl;
  } else {
    node_properties[start] = NodeProperty(NodeProperty::AstarState::kOpen, 0.0,
                                          GetHeuristic(start, goal), -1);
    astar_q.push(QueueNode(start, node_properties[start].g_score_ +
                                      node_properties[start].h_score_));
  }
  bool is_path_found = false;
  int count = 0;
  while (!astar_q.empty()) {
    ++count;
    // Selection.
    const QueueNode current_node = astar_q.top();
    astar_q.pop();
    // Check if the current node is the goal.
    if (current_node.point_ == goal) {
      is_path_found = true;
      break;
    }
    // Skip visited nodes due to the same
    if (node_properties[current_node.point_].state_ ==
        NodeProperty::AstarState::kClose) {
      continue;
    }
    node_properties[current_node.point_].state_ =
        NodeProperty::AstarState::kClose;
    // Expansion.
    const int current_node_id = graph_.node_id_[current_node.point_];
    const auto &edges = graph_.nodes_[current_node_id].edges_;
    for (const auto &edge : edges) {
      const IntPoint neighbor = graph_.nodes_[edge.first].point_;
      const float edge_weight = edge.second;
      const float g_score =
          node_properties[current_node.point_].g_score_ + edge_weight;
      if (node_properties.find(neighbor) == node_properties.end()) {
        const float h_score = GetHeuristic(neighbor, goal);
        node_properties[neighbor] = NodeProperty(
            NodeProperty::AstarState::kOpen, g_score, h_score, current_node_id);
        astar_q.push(QueueNode(neighbor, g_score + h_score));
      } else if (node_properties[neighbor].state_ ==
                 NodeProperty::AstarState::kOpen) {
        if (g_score < node_properties[neighbor].g_score_) {
          node_properties[neighbor].g_score_ = g_score;
          node_properties[neighbor].father_id_ = current_node_id;
          astar_q.push(QueueNode(neighbor,
                                 g_score + node_properties[neighbor].h_score_));
        }
      }
    }
  }
  track.OutputPassingTime("Run A* to find the path");
  std::cout << "A* count: " << count << std::endl;

  track.SetStartTime();
  std::vector<IntPoint> path;
  if (is_path_found) {
    std::cout << "Path found !" << std::endl;
    IntPoint waypoint = goal;
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
    std::cout << "No path found from " << start.x << "," << start.y << " to "
              << goal.x << "," << goal.y << std::endl;
  }
  track.OutputPassingTime("Output the path");
  // Remove the start and goal nodes from the graph.
  // Remove last two nodes in the graph.
  for (int i = 0; i < 2; ++i) {
    VGraphNode node = graph_.nodes_.back();
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
    std::cout << "Path length: " << len << std::endl;
  }
  return path;
}

std::vector<IntPoint>
DynamicVoronoi::GetiLQRPath(const std::vector<IntPoint> &path) {
  TimeTrack track;
  std::vector<IntPoint> ilqr_path;
  if (path.empty()) {
    return ilqr_path;
  }
  // Construct the constraints.
  const int num_bubbles = path.size() - 2;
  std::vector<IntPoint> bubbles(path.begin() + 1, path.end() - 1);
  std::vector<float> radius;
  radius.reserve(num_bubbles);
  for (int i = 0; i < num_bubbles; ++i) {
    radius.emplace_back(getDistance(bubbles[i].x, bubbles[i].y));
  }
  // iLQR Path Optimization.
  const int num_steps = path.size() - 1;
  Eigen::Matrix<float, 2, 4> F;
  // clang-format off
  F <<
  1.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 1.0f;
  // clang-format on
  std::vector<Eigen::Matrix2f> K_mats(num_steps, Eigen::Matrix2f::Zero());
  std::vector<Eigen::Vector2f> k_vecs(num_steps, Eigen::Vector2f::Zero());
  std::vector<Eigen::Vector4f> xu_vecs(num_steps, Eigen::Vector4f::Zero());
  std::vector<Eigen::Vector2f> x_hat_vecs(num_steps, Eigen::Vector2f::Zero());
  std::vector<Eigen::Vector2f> ov_center(num_bubbles - 1,
                                         Eigen::Vector2f::Zero());
  std::vector<float> coeff(num_steps, 0.0f);
  track.OutputPassingTime("Initialize");
  // Construct the initial guess.
  std::cout << "Construct the initial guess..." << std::endl;
  const IntPoint start = path.front();
  const IntPoint goal = path.back();
  xu_vecs[0] << start.x, start.y, 0.0f, 0.0f;
  x_hat_vecs[0] << start.x, start.y;
  for (int i = 1; i < num_steps - 1; ++i) {
    const float delta_c = GetDistanceBetween(bubbles[i - 1], bubbles[i]);
    const float dist =
        0.5f * (delta_c + (radius[i - 1] + radius[i]) *
                              (radius[i - 1] - radius[i]) / delta_c);
    const float x_initial =
        bubbles[i - 1].x + dist / delta_c * (bubbles[i].x - bubbles[i - 1].x);
    const float y_initial =
        bubbles[i - 1].y + dist / delta_c * (bubbles[i].y - bubbles[i - 1].y);
    xu_vecs[i].block<2, 1>(0, 0) << x_initial, y_initial;
    xu_vecs[i - 1].block<2, 1>(2, 0) =
        xu_vecs[i].block<2, 1>(0, 0) - xu_vecs[i - 1].block<2, 1>(0, 0);
    ov_center[i - 1] << x_initial, y_initial;
  }
  xu_vecs[num_steps - 1] << goal.x, goal.y, 0.0f, 0.0f;
  xu_vecs[num_steps - 2].block<2, 1>(2, 0) =
      xu_vecs[num_steps - 1].block<2, 1>(0, 0) -
      xu_vecs[num_steps - 2].block<2, 1>(0, 0);

  // Calculate the coefficents.
  coeff[0] = (ov_center[0] - Eigen::Vector2f(start.x, start.y)).norm();
  coeff[num_bubbles - 1] =
      (ov_center[num_bubbles - 2] - Eigen::Vector2f(goal.x, goal.y)).norm();
  for (int i = 0; i < num_bubbles - 2; ++i) {
    coeff[i + 1] = (ov_center[i + 1] - ov_center[i]).norm();
  }

  std::cout << "start ilqr optimization..." << std::endl;
  bool is_ilqr_success = false;
  float cost_sum = 0.0f;
  float last_cost_sum = cost_sum;
  float path_length = 0.0f;
  float last_path_length = path_length;
  // Iteration Loop.
  for (int iter = 0; iter < kMaxIteration; ++iter) {
    Eigen::Matrix2f V;
    Eigen::Vector2f v;
    cost_sum = 0.0f;
    std::pair<float, float> delta_V(0.0f, 0.0f);
    // Backward Pass.
    // track.SetStartTime();
    std::vector<std::pair<Eigen::Matrix4f, Eigen::Vector4f>> costs(
        num_steps,
        std::make_pair(Eigen::Matrix4f::Zero(), Eigen::Vector4f::Zero()));
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
      Eigen::Matrix4f Q;
      Eigen::Vector4f q;
      if (k == num_steps - 1) {
        // Terminal cost.
        const std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost = costs[k];
        Q = cost.first;
        q = cost.second;
      } else {
        const std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost = costs[k];
        Q = cost.first + F.transpose() * V * F;
        q = cost.second + F.transpose() * v;
      }
      const Eigen::Matrix2f Qxx = Q.block<2, 2>(0, 0);
      const Eigen::Matrix2f Qxu = Q.block<2, 2>(0, 2);
      const Eigen::Matrix2f Qux = Q.block<2, 2>(2, 0);
      const Eigen::Matrix2f Quu = Q.block<2, 2>(2, 2);
      const Eigen::Vector2f qx = q.block<2, 1>(0, 0);
      const Eigen::Vector2f qu = q.block<2, 1>(2, 0);
      K_mats[k] = -Quu.inverse() * Qux;
      k_vecs[k] = -Quu.inverse() * qu;
      const Eigen::Matrix2f K_mat = K_mats[k];
      const Eigen::Vector2f k_vec = k_vecs[k];
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
    const std::vector<Eigen::Vector4f> cur_xu_vecs(xu_vecs);
    while (!is_line_search_done && line_search_iter < kMaxLineSearchIter) {
      ++line_search_iter;
      // Forward Pass.
      float next_cost_sum = 0.0f;
      for (int k = 0; k < num_steps - 1; ++k) {
        const Eigen::Vector2f x = cur_xu_vecs[k].block<2, 1>(0, 0);
        const Eigen::Vector2f u = cur_xu_vecs[k].block<2, 1>(2, 0);
        xu_vecs[k].block<2, 1>(2, 0) =
            K_mats[k] * (x_hat_vecs[k] - x) + alpha * k_vecs[k] + u;
        xu_vecs[k].block<2, 1>(0, 0) = x_hat_vecs[k];
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
      xu_vecs[num_steps - 1].block<2, 1>(0, 0) = x_hat_vecs[num_steps - 1];
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
      path_length += xu_vecs[k].block<2, 1>(2, 0).norm();
    }
    // Terminate condition.
    if (iter > 0 &&
        std::fabs(path_length - last_path_length) < kConvergenceThreshold) {
      std::cout << "Convergence reached ! iter: " << iter
                << " cost: " << cost_sum << " path_length: " << path_length
                << std::endl;
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
    const Eigen::Vector2f x = xu_vecs[i].block<2, 1>(0, 0);
    ilqr_path.emplace_back(x(0), x(1));
  }
  return ilqr_path;
}

const VGraph &DynamicVoronoi::GetSparseGraph() const { return graph_; }

bool DynamicVoronoi::isInSparseGraph(const IntPoint &point) const {
  return graph_.node_id_.find(point) != graph_.node_id_.end();
}

float DynamicVoronoi::GetDistanceBetween(const IntPoint &p1,
                                         const IntPoint &p2) const {
  return std::hypot(p1.x - p2.x, p1.y - p2.y);
}

int DynamicVoronoi::GetSquaredDistanceBetween(const IntPoint &p1,
                                              const IntPoint &p2) const {
  const int dx = p1.x - p2.x;
  const int dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

// Calculate the heuristic value for A* search.
float DynamicVoronoi::GetHeuristic(const IntPoint &start,
                                   const IntPoint &goal) const {
  return GetDistanceBetween(start, goal);
}

std::pair<Eigen::Matrix4f, Eigen::Vector4f>
DynamicVoronoi::GetCost(const Eigen::Vector4f &xu, const IntPoint &bubble_1,
                        const float radius_1, const IntPoint &bubble_2,
                        const float radius_2, const float coeff) {
  float dcx = 0.0;
  float ddcx = 0.0;
  float dcy = 0.0;
  float ddcy = 0.0;
  // Constraints on bubble 1.
  const int dx_1 = xu(0) - bubble_1.x;
  const int dy_1 = xu(1) - bubble_1.y;
  if (dx_1 * dx_1 + dy_1 * dy_1 > radius_1 * radius_1) {
    dcx += kBndWeight * dx_1;
    dcy += kBndWeight * dy_1;
    ddcx += kBndWeight;
    ddcy += kBndWeight;
  }
  // Constraints on bubble 2.
  const int dx_2 = xu(0) - bubble_2.x;
  const int dy_2 = xu(1) - bubble_2.y;
  if (dx_2 * dx_2 + dy_2 * dy_2 > radius_2 * radius_2) {
    dcx += kBndWeight * dx_2;
    dcy += kBndWeight * dy_2;
    ddcx += kBndWeight;
    ddcy += kBndWeight;
  }
  std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost;
  // clang-format off
  cost.first << 
  ddcx, 0.0, 0.0, 0.0,
  0.0, ddcy, 0.0, 0.0,
  0.0, 0.0, kWeight / coeff, 0.0, 
  0.0, 0.0, 0.0, kWeight / coeff;
  cost.second <<
  dcx,
  dcy,
  kWeight / coeff * xu(2),
  kWeight / coeff * xu(3);
  // clang-format on
  return cost;
}

float DynamicVoronoi::GetRealCost(const Eigen::Vector4f &xu,
                                  const IntPoint &bubble_1,
                                  const float radius_1,
                                  const IntPoint &bubble_2,
                                  const float radius_2, const float coeff) {
  float real_cost = 0.0;
  real_cost += kWeight * 0.5 / coeff * xu(2) * xu(2) +
               kWeight * 0.5 / coeff * xu(3) * xu(3);
  // Constraints on bubble 1.
  const int dx_1 = xu(0) - bubble_1.x;
  const int dy_1 = xu(1) - bubble_1.y;
  if (dx_1 * dx_1 + dy_1 * dy_1 > radius_1 * radius_1) {
    real_cost += kBndWeight * 0.5 * dx_1 * dx_1 +
                 kBndWeight * 0.5 * dy_1 * dy_1 -
                 kBndWeight * 0.5 * radius_1 * radius_1;
  }
  // Constraints on bubble 2.
  const int dx_2 = xu(0) - bubble_2.x;
  const int dy_2 = xu(1) - bubble_2.y;
  if (dx_2 * dx_2 + dy_2 * dy_2 > radius_2 * radius_2) {
    real_cost += kBndWeight * 0.5 * dx_2 * dx_2 +
                 kBndWeight * 0.5 * dy_2 * dy_2 -
                 kBndWeight * 0.5 * radius_2 * radius_2;
  }
  return real_cost;
}

std::pair<Eigen::Matrix4f, Eigen::Vector4f>
DynamicVoronoi::GetTermCost(const Eigen::Vector4f &xu, const IntPoint &goal) {
  std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost;
  // clang-format off
  cost.first << 
  kTermWeight, 0.0, 0.0, 0.0,
  0.0, kTermWeight, 0.0, 0.0,
  0.0, 0.0, kWeight, 0.0,
  0.0, 0.0, 0.0, kWeight;
  cost.second << 
  kTermWeight * (xu(0) - goal.x),
  kTermWeight * (xu(1) - goal.y),
  kWeight * xu(2),
  kWeight * xu(3);
  // clang-format on
  return cost;
}

float DynamicVoronoi::GetRealTermCost(const Eigen::Vector4f &xu,
                                      const IntPoint &goal) {
  float real_cost = 0.0;
  const float dx = xu(0) - goal.x;
  const float dy = xu(1) - goal.y;
  real_cost += kTermWeight * 0.5 * dx * dx + kTermWeight * 0.5 * dy * dy;
  real_cost += kWeight * 0.5 * xu(2) * xu(2) + kWeight * 0.5 * xu(3) * xu(3);
  return real_cost;
}
