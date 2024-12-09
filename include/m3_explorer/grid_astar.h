#ifndef GRID_ASTAR_H
#define GRID_ASTAR_H
#include "m3_explorer/block.h"
#include <Eigen/Dense>
#include <octomap/octomap.h>
#include <ros/ros.h>
#include <vector>

class GridAstarNode {
public:
  int index_x_;
  int index_y_;
  int index_z_;
  float f_score_ = 0.0;
  std::shared_ptr<GridAstarNode> father_node_ = nullptr;

public:
  GridAstarNode(const int index_x, const int index_y, const int index_z);
};

// 用于优先队列
struct GridAstarNodeCmp {
  bool operator()(const std::shared_ptr<GridAstarNode> &lhs,
                  const std::shared_ptr<GridAstarNode> &rhs) const {
    return lhs->f_score_ > rhs->f_score_;
  }
};

struct DijkstraNodeCmp {
  bool operator()(const std::pair<int, float> &lhs,
                  const std::pair<int, float> &rhs) const {
    return lhs.second > rhs.second;
  }
};

class GridKey {
public:
  int index_x_;
  int index_y_;
  int index_z_;

public:
  GridKey(const int index_x, const int index_y, const int index_z)
      : index_x_(index_x), index_y_(index_y), index_z_(index_z){};
  bool operator==(const GridKey &key) const {
    return index_x_ == key.index_x_ && index_y_ == key.index_y_ &&
           index_z_ == key.index_z_;
  }
};

// 用于unordered_map
struct GridHash {
  size_t operator()(const GridKey &key) const {
    // x max_range: [-50, 50], 10 bit
    // y max_range: [-50, 50], 10 bit
    // z max_range: [0, 3], 5 bit
    // key = x  y   z
    // bit = 24-15   14-5    4-0
    const int x0 = key.index_x_ << 15;
    const int y0 = key.index_y_ << 5;
    const int z0 = key.index_z_ << 0;
    const int hash_value = x0 | y0 | z0;
    return std::hash<int>()(hash_value);
  }
};

class GridInfo {
public:
  enum class AstarState { kNull = 0, kOpen, kClose };
  AstarState state_ = AstarState::kNull;
  float g_score_ = 0.0;
  float h_score_ = 0.0;
};

class KeyBlock {
public:
  int x_ = 0;
  // ID of Block3D
  int block_id_ = 0;
  Block2D block_;

public:
  KeyBlock() = default;
  KeyBlock(const KeyBlock &rhs) = default;
  KeyBlock(const int x, const int block_id, const Block2D &block)
      : x_(x), block_id_(block_id), block_(block){};
};

class GraphEdge {
public:
  int dest_id_ = 0;
  float weight_ = 0.0;
  GraphEdge() = default;
  GraphEdge(const int dest_id, const float weight)
      : dest_id_(dest_id), weight_(weight){};
};

class GraphNode {
public:
  KeyBlock key_block_;
  std::vector<GraphEdge> edges_;
  GraphNode() = default;
  GraphNode(const KeyBlock &key_block) : key_block_(key_block){};
};

class GraphTable {
public:
  std::vector<GraphNode> nodes_;
  void AddNewEdge(const KeyBlock &src_key_block, const KeyBlock &dest_key_block,
                  const float weight = 1.0);
  void AddEdgeBetweenExistingNode(const int src_id, const int dest_id,
                                  const float weight);
  void UpdateEdgesInSameBlock();
};

class GridAstar {
public:
  enum class GridState { kFree = 0, kUnknown, kOcc };

private:
  float min_x_ = -50.0;
  float max_x_ = 50.0;
  float min_y_ = -50.0;
  float max_y_ = 50.0;
  float min_z_ = 0.0;
  float max_z_ = 2.5;
  float resolution_ = 0.1;
  std::vector<std::vector<std::vector<GridState>>> grid_map_;
  std::vector<std::vector<std::vector<RangeVoxel>>> merge_map_;
  std::vector<std::vector<Block2D>> merge_map_2d_;
  std::vector<Block3D> merge_map_3d_;
  std::vector<std::shared_ptr<GridAstarNode>> path_;
  std::vector<int> block_path_;
  std::vector<std::vector<float>> ilqr_path_;
  // Return the set of merged_voxels.
  std::vector<Block2D>
  Merge2DVoxelAlongY(const std::vector<std::vector<RangeVoxel>> &yz_voxels);
  std::vector<Block3D>
  Merge3DVoxelAlongX(const std::vector<std::vector<Block2D>> &xyz_voxels);
  GraphTable graph_table_;

public:
  const std::vector<std::vector<std::vector<GridState>>> &grid_map() const;
  const std::vector<std::vector<std::vector<RangeVoxel>>> &merge_map() const;
  const std::vector<std::vector<Block2D>> &merge_map_2d() const;
  const std::vector<Block3D> &merge_map_3d() const;
  const std::vector<std::shared_ptr<GridAstarNode>> &path() const;
  const std::vector<int> &block_path() const;
  const std::vector<std::vector<float>> &ilqr_path() const;
  const GraphTable &graph_table() const;

  GridAstar(const float min_x, const float max_x, const float min_y,
            const float max_y, const float min_z, const float max_z,
            const float resolution = 0.1);
  void UpdateFromMap(const octomap::OcTree *ocmap,
                     const octomap::point3d &bbx_min,
                     const octomap::point3d &bbx_max);
  void MergeMap();
  void Merge2DVoxelAlongYUnitTest();
  void MergeMap2D();
  void Merge3DVoxelAlongXUnitTest();
  void MergeMap3D();
  float AstarPathDistance(const Eigen::Vector3f &start_p,
                          const Eigen::Vector3f &end_p);
  float BlockPathDistance(const Eigen::Vector3f &start_p,
                          const Eigen::Vector3f &end_p);
  float BlockPathRefine(const std::vector<int> &block_path,
                        const Eigen::Vector3f &start_p,
                        const Eigen::Vector3f &end_p);

  std::pair<Eigen::Matrix4f, Eigen::Vector4f>
  GetCost(const Eigen::Vector4f &xu, const float y_lb, const float y_ub,
          const float z_lb, const float z_ub);
  float GetRealCost(const Eigen::Vector4f &xu, const float y_lb,
                    const float y_ub, const float z_lb, const float z_ub);
  std::pair<Eigen::Matrix4f, Eigen::Vector4f>
  GetTermCost(const Eigen::Vector4f &xu, const float target_y,
              const float target_z);
  float GetRealTermCost(const Eigen::Vector4f &xu, const float target_y,
                        const float target_z);
  inline float CalHeurScore(const std::shared_ptr<GridAstarNode> &node,
                            const Eigen::Vector3f &end_p);
};

#endif
