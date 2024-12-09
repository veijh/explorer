#include "m3_explorer/grid_astar.h"
#include "m3_explorer/time_track.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace {
constexpr float kFreeThreshold = 0.3;
constexpr float kOccThreshold = 0.7;
constexpr int kMapZSize = 8;
constexpr int kMapYZSize = 128;
constexpr int kMapXYZSize = 512;
constexpr int kMergeBuffer = 5;
constexpr int kFilterMinX = 4;
constexpr int kFilterMinY = 4;
constexpr int kFilterMinZ = 4;
constexpr int kConnectivityMinY = 4;
constexpr int kConnectivityMinZ = 4;
constexpr int kMaxIteration = 100;
constexpr float kConvergenceThreshold = 0.01;
constexpr float kTermWeight = 100.0;
constexpr float kBndWeight = 100.0;
constexpr float kWeight = 0.2;
constexpr int kMaxLineSearchIter = 10;
} // namespace

const std::vector<Eigen::Vector3i> expand_offset = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};

void GraphTable::AddNewEdge(const KeyBlock &src_key_block,
                            const KeyBlock &dest_key_block,
                            const float weight) {
  int src_id = nodes_.size();
  int dest_id = nodes_.size() + 1;
  nodes_.emplace_back(src_key_block);
  nodes_.emplace_back(dest_key_block);
  // Add edge between src and dest.
  nodes_[src_id].edges_.emplace_back(dest_id, weight);
  nodes_[dest_id].edges_.emplace_back(src_id, weight);
}

void GraphTable::AddEdgeBetweenExistingNode(const int src_id, const int dest_id,
                                            const float weight) {
  nodes_[src_id].edges_.emplace_back(dest_id, weight);
}

void GraphTable::UpdateEdgesInSameBlock() {
  std::unordered_map<int, std::shared_ptr<std::vector<int>>> block_group;
  const int num_nodes = nodes_.size();
  // Group nodes by block_id.
  for (int i = 0; i < num_nodes; ++i) {
    if (block_group.find(nodes_[i].key_block_.block_id_) == block_group.end()) {
      block_group[nodes_[i].key_block_.block_id_] =
          std::make_shared<std::vector<int>>();
      block_group[nodes_[i].key_block_.block_id_]->emplace_back(i);
    } else {
      block_group[nodes_[i].key_block_.block_id_]->emplace_back(i);
    }
  }
  // Update edges between nodes in the same block.
  for (auto it = block_group.begin(); it != block_group.end(); ++it) {
    const int group_size = it->second->size();
    for (int i = 0; i < group_size; ++i) {
      for (int j = 0; j < group_size; ++j) {
        if (i != j) {
          const int src_index = it->second->at(i);
          const int dest_index = it->second->at(j);
          const KeyBlock &src_key_block = nodes_[src_index].key_block_;
          const KeyBlock &dest_key_block = nodes_[dest_index].key_block_;
          const float delta_x = src_key_block.x_ - dest_key_block.x_;
          const float weight = src_key_block.block_.GetRoughDistance(
              dest_key_block.block_, delta_x);
          AddEdgeBetweenExistingNode(src_index, dest_index, weight);
        }
      }
    }
  }
}

GridAstarNode::GridAstarNode(const int index_x, const int index_y,
                             const int index_z)
    : index_x_(index_x), index_y_(index_y), index_z_(index_z) {}

GridAstar::GridAstar(const float min_x, const float max_x, const float min_y,
                     const float max_y, const float min_z, const float max_z,
                     const float resolution)
    : min_x_(min_x), max_x_(max_x), min_y_(min_y), max_y_(max_y), min_z_(min_z),
      max_z_(max_z), resolution_(resolution) {
  const int num_x_grid =
      static_cast<int>(std::ceil((max_x_ - min_x_) / resolution_));
  const int num_y_grid =
      static_cast<int>(std::ceil((max_y_ - min_y_) / resolution_));
  const int num_z_grid =
      static_cast<int>(std::ceil((max_z_ - min_z_) / resolution_));
  // grid_map_[i][j][k] indicates the occupancy of node in
  // min_x_ + i * resolution -> min_x_ + (i + 1) * resolution
  // min_y_ + j * resolution -> min_y_ + (j + 1) * resolution
  // min_z_ + k * resolution -> min_z_ + (k + 1) * resolution
  grid_map_.resize(
      num_x_grid,
      std::vector<std::vector<GridState>>(
          num_y_grid, std::vector<GridState>(num_z_grid, GridState::kUnknown)));
}

const std::vector<std::vector<std::vector<GridAstar::GridState>>> &
GridAstar::grid_map() const {
  return grid_map_;
}

const std::vector<std::vector<std::vector<RangeVoxel>>> &
GridAstar::merge_map() const {
  return merge_map_;
}

const std::vector<std::vector<Block2D>> &GridAstar::merge_map_2d() const {
  return merge_map_2d_;
}

const std::vector<Block3D> &GridAstar::merge_map_3d() const {
  return merge_map_3d_;
}

const std::vector<std::shared_ptr<GridAstarNode>> &GridAstar::path() const {
  return path_;
}

const std::vector<int> &GridAstar::block_path() const { return block_path_; }

const GraphTable &GridAstar::graph_table() const { return graph_table_; }

const std::vector<std::vector<float>> &GridAstar::ilqr_path() const {
  return ilqr_path_;
}

void GridAstar::UpdateFromMap(const octomap::OcTree *ocmap,
                              const octomap::point3d &bbx_min,
                              const octomap::point3d &bbx_max) {
  if (ocmap == nullptr)
    return;

  const int num_x_grid = grid_map_.size();
  const int num_y_grid = grid_map_[0].size();
  const int num_z_grid = grid_map_[0][0].size();
  for (octomap::OcTree::leaf_bbx_iterator
           it = ocmap->begin_leafs_bbx(bbx_min, bbx_max),
           end = ocmap->end_leafs_bbx();
       it != end; ++it) {
    float occ_probility = it->getOccupancy();
    // Do not update the unknown node.
    if (occ_probility >= kFreeThreshold && occ_probility <= kOccThreshold) {
      continue;
    }

    GridState grid_state =
        occ_probility < (kFreeThreshold + kOccThreshold) * 0.5
            ? GridState::kFree
            : GridState::kOcc;
    const float size = static_cast<float>(it.getSize());
    const octomap::point3d center = it.getCoordinate();
    const float node_min_x = center.x() - size * 0.5;
    const float node_max_x = center.x() + size * 0.5;
    const float node_min_y = center.y() - size * 0.5;
    const float node_max_y = center.y() + size * 0.5;
    const float node_min_z = center.z() - size * 0.5;
    const float node_max_z = center.z() + size * 0.5;

    int min_x_index =
        static_cast<int>(std::floor((node_min_x - min_x_) / resolution_));
    int max_x_index =
        static_cast<int>(std::ceil((node_max_x - min_x_) / resolution_));
    int min_y_index =
        static_cast<int>(std::floor((node_min_y - min_y_) / resolution_));
    int max_y_index =
        static_cast<int>(std::ceil((node_max_y - min_y_) / resolution_));
    int min_z_index =
        static_cast<int>(std::floor((node_min_z - min_z_) / resolution_));
    int max_z_index =
        static_cast<int>(std::ceil((node_max_z - min_z_) / resolution_));

    min_x_index = std::clamp(min_x_index, 0, num_x_grid - 1);
    max_x_index = std::clamp(max_x_index, 0, num_x_grid - 1);
    min_y_index = std::clamp(min_y_index, 0, num_y_grid - 1);
    max_y_index = std::clamp(max_y_index, 0, num_y_grid - 1);
    min_z_index = std::clamp(min_z_index, 0, num_z_grid - 1);
    max_z_index = std::clamp(max_z_index, 0, num_z_grid - 1);

    for (int i = min_x_index; i <= max_x_index; ++i) {
      for (int j = min_y_index; j <= max_y_index; ++j) {
        for (int k = min_z_index; k <= max_z_index; ++k) {
          grid_map_[i][j][k] = grid_state;
        }
      }
    }
  }
}

void GridAstar::MergeMap() {
  const int num_x_grid = grid_map_.size();
  const int num_y_grid = grid_map_[0].size();
  const int num_z_grid = grid_map_[0][0].size();
  merge_map_.resize(num_x_grid,
                    std::vector<std::vector<RangeVoxel>>(num_y_grid));
  int total_num = 0;
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      int min = 0;
      int max = num_z_grid - 1;
      int state = 0;
      merge_map_[i][j].clear();
      merge_map_[i][j].reserve(kMapZSize);
      for (int k = 0; k < num_z_grid; ++k) {
        switch (state) {
        case 0:
          if (grid_map_[i][j][k] == GridState::kFree) {
            min = k;
            state = 1;
          }
          break;
        case 1:
          if (k == num_z_grid - 1 || grid_map_[i][j][k] != GridState::kFree) {
            max = grid_map_[i][j][k] != GridState::kFree ? k - 1 : k;
            // Filter narrow range.
            if (max - min + 1 > kFilterMinZ) {
              merge_map_[i][j].emplace_back(min, max);
              ++total_num;
            }
            state = 0;
          }
          break;
        }
      }
    }
  }
  std::cout << "Voxel1D total voxel num: " << total_num << std::endl;
}

std::vector<Block2D> GridAstar::Merge2DVoxelAlongY(
    const std::vector<std::vector<RangeVoxel>> &yz_voxels) {
  const int num_y_voxels = yz_voxels.size();
  std::set<Block2DWrapper> unmerged_voxels;
  std::vector<Block2D> merge_results;
  merge_results.reserve(kMapYZSize);
  for (int i = 0; i < num_y_voxels; ++i) {
    std::set<Block2DWrapper> merged_voxels;
    const int num_z_voxels = yz_voxels[i].size();
    for (int j = 0; j < num_z_voxels; ++j) {
      const RangeVoxel new_z_voxel = yz_voxels[i][j];
      // Find the range to merge in unmerged_voxels.
      const int voxel_min_lb = new_z_voxel.min_ - kMergeBuffer + 1;
      const int voxel_max_lb = new_z_voxel.min_ + kMergeBuffer - 1;
      // Binary search.
      auto range_lower_it = std::lower_bound(
          unmerged_voxels.begin(), unmerged_voxels.end(),
          Block2DWrapper(RangeVoxel(voxel_min_lb, new_z_voxel.max_)));
      auto range_upper_it = std::upper_bound(
          unmerged_voxels.begin(), unmerged_voxels.end(),
          Block2DWrapper(RangeVoxel(voxel_max_lb, new_z_voxel.max_)));
      if (std::distance(range_lower_it, range_upper_it) >= 1) {
        const int voxel_min_ub = new_z_voxel.max_ - kMergeBuffer + 1;
        const int voxel_max_ub = new_z_voxel.max_ + kMergeBuffer - 1;
        const int plug_max = range_lower_it->plug_.max_;
        if (voxel_min_ub <= plug_max && plug_max <= voxel_max_ub) {
          // Merge and add to the result.
          Block2DWrapper new_voxel_wrapper = *range_lower_it;
          new_voxel_wrapper.plug_ = new_z_voxel;
          new_voxel_wrapper.block_2d_.EmplaceRangeBack(new_z_voxel);
          merged_voxels.insert(new_voxel_wrapper);
          // Erase the range.
          unmerged_voxels.erase(range_lower_it);
        } else {
          const Block2D new_voxel(i, new_z_voxel);
          const Block2DWrapper new_voxel_wrapper(new_voxel, new_z_voxel);
          merged_voxels.insert(new_voxel_wrapper);
        }
      } else {
        const Block2D new_voxel(i, new_z_voxel);
        const Block2DWrapper new_voxel_wrapper(new_voxel, new_z_voxel);
        merged_voxels.insert(new_voxel_wrapper);
      }
    }
    // Add all item of unmerged_voxels to result.
    for (auto &item : unmerged_voxels) {
      // Filter small voxels.
      if (item.block_2d_.y_max_ - item.block_2d_.y_min_ + 1 <= kFilterMinY) {
        continue;
      }
      merge_results.emplace_back(item.block_2d_);
    }
    unmerged_voxels = merged_voxels;
  }
  // Add all item of unmerged_voxels to result.
  for (auto &item : unmerged_voxels) {
    // Filter small voxels.
    if (item.block_2d_.y_max_ - item.block_2d_.y_min_ + 1 <= kFilterMinY) {
      continue;
    }
    merge_results.emplace_back(item.block_2d_);
  }
  return merge_results;
}

void GridAstar::Merge2DVoxelAlongYUnitTest() {
  std::vector<std::vector<RangeVoxel>> yz_voxels;
  yz_voxels.resize(7);
  yz_voxels[0].emplace_back(RangeVoxel(0, 2));
  yz_voxels[0].emplace_back(RangeVoxel(5, 6));
  yz_voxels[1].emplace_back(RangeVoxel(0, 1));
  yz_voxels[1].emplace_back(RangeVoxel(4, 6));
  yz_voxels[2].emplace_back(RangeVoxel(0, 1));
  yz_voxels[2].emplace_back(RangeVoxel(3, 6));
  yz_voxels[4].emplace_back(RangeVoxel(0, 1));
  yz_voxels[4].emplace_back(RangeVoxel(3, 3));
  yz_voxels[4].emplace_back(RangeVoxel(5, 6));
  yz_voxels[5].emplace_back(RangeVoxel(0, 1));
  yz_voxels[5].emplace_back(RangeVoxel(3, 6));
  yz_voxels[6].emplace_back(RangeVoxel(0, 6));
  std::vector<Block2D> result = Merge2DVoxelAlongY(yz_voxels);
  std::cout << "[Result Size]: " << result.size() << std::endl;
  for (auto item : result) {
    std::cout << "[Item]: (" << item.y_min_ << ", " << item.y_max_ << "), ("
              << item.z_min_ << ", " << item.z_max_ << ")" << std::endl;
  }
}

void GridAstar::MergeMap2D() {
  auto cmp = [](const Block2D &lhs, const Block2D &rhs) {
    return lhs.y_min_ != rhs.y_min_ ? lhs.y_min_ < rhs.y_min_
                                    : lhs.z_min_ < rhs.z_min_;
  };
  const int num_x_voxel = merge_map_.size();
  merge_map_2d_.resize(num_x_voxel);
  int total_num = 0;
  for (int i = 0; i < num_x_voxel; ++i) {
    merge_map_2d_[i] = Merge2DVoxelAlongY(merge_map_[i]);
    std::sort(merge_map_2d_[i].begin(), merge_map_2d_[i].end(), cmp);
    total_num += merge_map_2d_[i].size();
  }
  std::cout << "total_num: " << total_num << std::endl;
}

std::vector<Block3D> GridAstar::Merge3DVoxelAlongX(
    const std::vector<std::vector<Block2D>> &xyz_voxels) {
  const int num_x_voxels = xyz_voxels.size();
  int block_id = 0;
  GraphTable graph_table;
  std::vector<Block3D> merge_results;
  merge_results.reserve(kMapXYZSize);
  std::set<Block3DWrapper> unmerged_voxels;
  for (int i = 0; i < num_x_voxels; ++i) {
    std::set<Block3DWrapper> merged_voxels;
    const int num_yz_voxels = xyz_voxels[i].size();
    for (int j = 0; j < num_yz_voxels; ++j) {
      const Block2D new_yz_voxel = xyz_voxels[i][j];
      // Find the range to merge in unmerged_voxels.
      const int voxel_y_min_lb = new_yz_voxel.y_min_ - kMergeBuffer + 1;
      const int voxel_y_min_ub = new_yz_voxel.y_min_ + kMergeBuffer - 1;
      const int voxel_z_min_lb = new_yz_voxel.z_min_ - kMergeBuffer + 1;
      const int voxel_z_min_ub = new_yz_voxel.z_min_ + kMergeBuffer - 1;
      const int voxel_y_max_lb = new_yz_voxel.y_max_ - kMergeBuffer + 1;
      const int voxel_y_max_ub = new_yz_voxel.y_max_ + kMergeBuffer - 1;
      const int voxel_z_max_lb = new_yz_voxel.z_max_ - kMergeBuffer + 1;
      const int voxel_z_max_ub = new_yz_voxel.z_max_ + kMergeBuffer - 1;

      bool is_merged = false;
      bool is_connected = false;
      for (auto it = unmerged_voxels.begin(); it != unmerged_voxels.end();) {
        const int plug_y_min = it->plug_.y_min_;
        const int plug_z_min = it->plug_.z_min_;
        const int plug_y_max = it->plug_.y_max_;
        const int plug_z_max = it->plug_.z_max_;
        if (voxel_y_min_lb <= plug_y_min && plug_y_min <= voxel_y_min_ub &&
            voxel_z_min_lb <= plug_z_min && plug_z_min <= voxel_z_min_ub &&
            voxel_y_max_lb <= plug_y_max && plug_y_max <= voxel_y_max_ub &&
            voxel_z_max_lb <= plug_z_max && plug_z_max <= voxel_z_max_ub) {
          // Merge and add to the result.
          Block3DWrapper new_voxel_wrapper = *it;
          new_voxel_wrapper.plug_ = new_yz_voxel;
          new_voxel_wrapper.block_3d_.EmplaceBlockBack(new_yz_voxel);
          merged_voxels.insert(new_voxel_wrapper);
          // Erase the range.
          unmerged_voxels.erase(it++);
          is_merged = true;
          break;
        }
        const std::vector<Block2D> overlap_blocks =
            new_yz_voxel.GetOverlap(it->plug_);
        for (const Block2D &overlap_block : overlap_blocks) {
          const int overlap_y_min = overlap_block.y_min_;
          const int overlap_z_min = overlap_block.z_min_;
          const int overlap_y_max = overlap_block.y_max_;
          const int overlap_z_max = overlap_block.z_max_;
          if (overlap_y_max - overlap_y_min + 1 >= kConnectivityMinY &&
              overlap_z_max - overlap_z_min + 1 >= kConnectivityMinZ) {
            // If two voxels are partially overlapped, add a new edge to the
            // graph.
            const int key_block_x = i;
            const KeyBlock last_key_block(
                key_block_x - 1, it->block_3d_.block_id_, overlap_block);
            const KeyBlock key_block(key_block_x, block_id, overlap_block);
            graph_table.AddNewEdge(last_key_block, key_block);
            break;
          }
        }
        it++;
      }
      if (!is_merged) {
        const Block3D new_voxel(i, new_yz_voxel);
        const Block3DWrapper new_voxel_wrapper(new_voxel, new_yz_voxel,
                                               block_id);
        // Update block id of next voxel.
        ++block_id;
        merged_voxels.insert(new_voxel_wrapper);
      }
    }
    // Add all item of unmerged_voxels to result.
    for (auto &item : unmerged_voxels) {
      merge_results.emplace_back(item.block_3d_);
    }
    unmerged_voxels = merged_voxels;
  }
  // Add all item of unmerged_voxels to result.
  for (auto &item : unmerged_voxels) {
    merge_results.emplace_back(item.block_3d_);
  }
  // Update topology of the graph.
  graph_table.UpdateEdgesInSameBlock();
  graph_table_ = graph_table;
  return merge_results;
}

void GridAstar::Merge3DVoxelAlongXUnitTest() {
  std::vector<std::vector<Block2D>> xyz_voxels;
  xyz_voxels.resize(7);
  xyz_voxels[0].emplace_back(Block2D(0, 4, 0, 4));
  xyz_voxels[0].emplace_back(Block2D(5, 10, 0, 5));
  xyz_voxels[1].emplace_back(Block2D(0, 4, 0, 4));
  xyz_voxels[1].emplace_back(Block2D(5, 10, 0, 4));
  xyz_voxels[2].emplace_back(Block2D(0, 4, 0, 4));
  xyz_voxels[2].emplace_back(Block2D(5, 10, 0, 3));
  xyz_voxels[4].emplace_back(Block2D(0, 4, 0, 4));
  xyz_voxels[5].emplace_back(Block2D(0, 4, 0, 4));
  xyz_voxels[6].emplace_back(Block2D(0, 4, 0, 4));
  std::vector<Block3D> result = Merge3DVoxelAlongX(xyz_voxels);
  std::cout << "[Result Size]: " << result.size() << std::endl;
  for (auto item : result) {
    std::cout << "[Item]: (" << item.x_min_ << ", " << item.x_max_ << "), ("
              << item.y_min_ << ", " << item.y_max_ << "), (" << item.z_min_
              << ", " << item.z_max_ << ")" << std::endl;
  }
}

void GridAstar::MergeMap3D() {
  merge_map_3d_ = Merge3DVoxelAlongX(merge_map_2d_);
  std::cout << "total_num: " << merge_map_3d_.size() << std::endl;
}

float GridAstar::AstarPathDistance(const Eigen::Vector3f &start_p,
                                   const Eigen::Vector3f &end_p) {
  TimeTrack tracker;
  const int num_x_grid = grid_map_.size();
  const int num_y_grid = grid_map_[0].size();
  const int num_z_grid = grid_map_[0][0].size();

  std::priority_queue<std::shared_ptr<GridAstarNode>,
                      std::vector<std::shared_ptr<GridAstarNode>>,
                      GridAstarNodeCmp>
      astar_q;

  std::unordered_map<GridKey, GridInfo, GridHash> grid_info;

  bool is_path_found = false;
  int count = 0;

  int index_start_x =
      static_cast<int>(std::floor((start_p.x() - min_x_) / resolution_));
  int index_start_y =
      static_cast<int>(std::floor((start_p.y() - min_y_) / resolution_));
  int index_start_z =
      static_cast<int>(std::floor((start_p.z() - min_z_) / resolution_));
  int index_end_x =
      static_cast<int>(std::floor((end_p.x() - min_x_) / resolution_));
  int index_end_y =
      static_cast<int>(std::floor((end_p.y() - min_y_) / resolution_));
  int index_end_z =
      static_cast<int>(std::floor((end_p.z() - min_z_) / resolution_));

  std::shared_ptr<GridAstarNode> astar_start = std::make_shared<GridAstarNode>(
      index_start_x, index_start_y, index_start_z);
  astar_start->father_node_ = nullptr;
  const GridKey start_key(index_start_x, index_start_y, index_start_z);
  grid_info[start_key].g_score_ = 0.0;
  grid_info[start_key].h_score_ = CalHeurScore(astar_start, end_p);
  astar_start->f_score_ =
      grid_info[start_key].g_score_ + grid_info[start_key].h_score_;
  grid_info[start_key].state_ = GridInfo::AstarState::kOpen;
  astar_q.push(astar_start);

  std::shared_ptr<GridAstarNode> astar_end = nullptr;
  tracker.OutputPassingTime("Astar Init");
  tracker.SetStartTime();

  while (!astar_q.empty()) {
    ++count;
    std::shared_ptr<GridAstarNode> cur_node = astar_q.top();
    astar_q.pop();

    const int cur_index_x = cur_node->index_x_;
    const int cur_index_y = cur_node->index_y_;
    const int cur_index_z = cur_node->index_z_;
    const GridKey cur_key(cur_index_x, cur_index_y, cur_index_z);

    // Skip the same node due to the update of g_score.
    if (grid_info[cur_key].state_ == GridInfo::AstarState::kClose) {
      continue;
    }
    // set node to closed
    grid_info[cur_key].state_ = GridInfo::AstarState::kClose;
    if (cur_node->index_x_ == index_end_x &&
        cur_node->index_y_ == index_end_y &&
        cur_node->index_z_ == index_end_z) {
      is_path_found = true;
      astar_end = cur_node;
      break;
    }

    // expand neighbor nodes
    for (auto &offset : expand_offset) {
      int next_index_x = offset.x() + cur_node->index_x_;
      int next_index_y = offset.y() + cur_node->index_y_;
      int next_index_z = offset.z() + cur_node->index_z_;
      const GridKey next_key(next_index_x, next_index_y, next_index_z);
      // skip nodes that is out of range
      if (next_index_x < 0 || next_index_y < 0 || next_index_z < 0) {
        continue;
      }
      if (next_index_x >= num_x_grid || next_index_y >= num_y_grid ||
          next_index_z >= num_z_grid) {
        continue;
      }
      // only expand free nodes
      if (grid_map_[next_index_x][next_index_y][next_index_z] ==
              GridAstar::GridState::kFree &&
          grid_info[next_key].state_ != GridInfo::AstarState::kClose) {
        // Calculate GScore.
        const float new_g_score = grid_info[cur_key].g_score_ + resolution_;
        // Next node has not being visited.
        if (grid_info[next_key].state_ == GridInfo::AstarState::kNull) {
          std::shared_ptr<GridAstarNode> next_node =
              std::make_shared<GridAstarNode>(next_index_x, next_index_y,
                                              next_index_z);
          next_node->father_node_ = cur_node;
          grid_info[next_key].g_score_ = new_g_score;
          grid_info[next_key].h_score_ = CalHeurScore(next_node, end_p);
          // Calculate Fscore.
          next_node->f_score_ = new_g_score + grid_info[next_key].h_score_;
          grid_info[next_key].state_ = GridInfo::AstarState::kOpen;
          astar_q.push(next_node);
        } else {
          // Next node has being visited.
          std::shared_ptr<GridAstarNode> next_node =
              std::make_shared<GridAstarNode>(next_index_x, next_index_y,
                                              next_index_z);
          if (new_g_score < grid_info[next_key].g_score_) {
            next_node->father_node_ = cur_node;
            // Update Gscore.
            grid_info[next_key].g_score_ = new_g_score;
            const float h_score = grid_info[next_key].h_score_;
            // Update Fscore.
            next_node->f_score_ = new_g_score + h_score;
            astar_q.push(next_node);
          }
        }
      }
    }
  }
  tracker.OutputPassingTime("Astar Done");
  tracker.SetStartTime();

  if (is_path_found) {
    float distance = 0.0;
    path_.clear();
    // Insert last node.
    path_.emplace_back(astar_end);
    std::shared_ptr<GridAstarNode> last_path_node = astar_end;
    std::shared_ptr<GridAstarNode> path_node = astar_end->father_node_;
    while (path_node != nullptr) {
      const int delta_x = last_path_node->index_x_ - path_node->index_x_;
      const int delta_y = last_path_node->index_y_ - path_node->index_y_;
      const int delta_z = last_path_node->index_z_ - path_node->index_z_;
      // distance += resolution_ *
      //             (std::abs(delta_x) + std::abs(delta_y) +
      //             std::abs(delta_z));
      distance += resolution_ * std::hypot(delta_x, delta_y, delta_z);
      path_.emplace_back(path_node);
      last_path_node = path_node;
      path_node = path_node->father_node_;
    }
    reverse(path_.begin(), path_.end());
    std::cout << "[Astar] waypoint generated!! waypoint num: " << path_.size()
              << ", select node num: " << count << std::endl;
    tracker.OutputPassingTime("Astar Output");
    return distance;
  } else {
    std::cout << "[WARNING] no path !! from " << std::endl
              << start_p << std::endl
              << "to " << std::endl
              << end_p << std::endl;
    return (end_p - start_p).norm();
  }
}

float GridAstar::BlockPathDistance(const Eigen::Vector3f &start_p,
                                   const Eigen::Vector3f &end_p) {
  block_path_.clear();
  int index_start_x =
      static_cast<int>(std::floor((start_p.x() - min_x_) / resolution_));
  int index_start_y =
      static_cast<int>(std::floor((start_p.y() - min_y_) / resolution_));
  int index_start_z =
      static_cast<int>(std::floor((start_p.z() - min_z_) / resolution_));
  int index_end_x =
      static_cast<int>(std::floor((end_p.x() - min_x_) / resolution_));
  int index_end_y =
      static_cast<int>(std::floor((end_p.y() - min_y_) / resolution_));
  int index_end_z =
      static_cast<int>(std::floor((end_p.z() - min_z_) / resolution_));
  // Find the block that contains the start point.
  int num_blocks = merge_map_3d_.size();
  bool is_start_locked = false;
  bool is_end_locked = false;
  int start_block_index = -1;
  int end_block_index = -1;
  for (int i = 0; i < num_blocks; ++i) {
    if (is_start_locked && is_end_locked) {
      break;
    }
    if (!is_start_locked && merge_map_3d_[i].IsInBlock(
                                index_start_x, index_start_y, index_start_z)) {
      start_block_index = merge_map_3d_[i].block_id_;
      is_start_locked = true;
    }
    if (!is_end_locked &&
        merge_map_3d_[i].IsInBlock(index_end_x, index_end_y, index_end_z)) {
      end_block_index = merge_map_3d_[i].block_id_;
      is_end_locked = true;
    }
  }
  if (start_block_index == end_block_index) {
    // TODO: Calculate the distance between two points in the same block.
    return (end_p - start_p).norm();
  } else {
    // Dijkstra algorithm to find the path between two blocks.
    const int nums_node = graph_table_.nodes_.size();
    std::vector<int> father_nodes(nums_node, -1);
    int end_father = -1;
    std::vector<int> is_visited(nums_node, 0);
    int is_end_visited = 0;
    std::vector<float> distance(nums_node, 0.0);
    float end_distance = 0.0;
    std::priority_queue<std::pair<int, float>,
                        std::vector<std::pair<int, float>>, DijkstraNodeCmp>
        dijkstra_q;
    // Add the key block of start block to the queue.
    for (int i = 0; i < nums_node; ++i) {
      const KeyBlock key_block = graph_table_.nodes_[i].key_block_;
      if (key_block.block_id_ == start_block_index) {
        // Compute the rough distance between the start point and the key block.
        const float delta_x = index_start_x - key_block.x_;
        const float delta_y = index_start_y - 0.5 * (key_block.block_.y_min_ +
                                                     key_block.block_.y_max_);
        const float delta_z = index_start_z - 0.5 * (key_block.block_.z_min_ +
                                                     key_block.block_.z_max_);
        const float edge_weight = std::hypot(delta_x, delta_y, delta_z);
        dijkstra_q.emplace(i, edge_weight);
        is_visited[i] = 1;
        father_nodes[i] = -1;
        distance[i] = edge_weight;
      }
    }
    bool is_path_found = false;
    while (!dijkstra_q.empty()) {
      // Select the node with the smallest distance.
      std::pair<int, float> cur_node = dijkstra_q.top();
      dijkstra_q.pop();
      // Check if the node is the end block.
      if (cur_node.first == -1) {
        is_path_found = true;
        break;
      }
      // Skip nodes that is in Closed state.
      if (is_visited[cur_node.first] == 2) {
        continue;
      }
      // Set the node to Closed state.
      is_visited[cur_node.first] = 2;
      const GraphNode graph_node = graph_table_.nodes_[cur_node.first];
      // Check if the node is in the end block.
      if (graph_node.key_block_.block_id_ == end_block_index) {
        const float delta_x = index_end_x - graph_node.key_block_.x_;
        const float delta_y =
            index_end_y - 0.5 * (graph_node.key_block_.block_.y_min_ +
                                 graph_node.key_block_.block_.y_max_);
        const float delta_z =
            index_end_z - 0.5 * (graph_node.key_block_.block_.z_min_ +
                                 graph_node.key_block_.block_.z_max_);
        const float edge_weight =
            std::hypot(delta_x, delta_y, delta_z) + cur_node.second;
        if (is_end_visited == 0 ||
            (is_end_visited == 1 && edge_weight < end_distance)) {
          end_father = cur_node.first;
          is_end_visited = 1;
          end_distance = edge_weight;
          dijkstra_q.emplace(-1, edge_weight);
        }
      }
      // Expand the node.
      const int num_edges = graph_node.edges_.size();
      for (int i = 0; i < num_edges; ++i) {
        const int neighbor_index = graph_node.edges_[i].dest_id_;
        const float edge_weight =
            graph_node.edges_[i].weight_ + cur_node.second;
        if (is_visited[neighbor_index] == 0 ||
            (is_visited[neighbor_index] == 1 &&
             edge_weight < distance[neighbor_index])) {
          father_nodes[neighbor_index] = cur_node.first;
          is_visited[neighbor_index] = 1;
          distance[neighbor_index] = edge_weight;
          dijkstra_q.emplace(neighbor_index, edge_weight);
        }
      }
    }
    if (is_path_found) {
      std::vector<int> path_blocks;
      int cur_node_index = end_father;
      while (cur_node_index != -1) {
        path_blocks.emplace_back(cur_node_index);
        cur_node_index = father_nodes[cur_node_index];
      }
      std::cout << "[Block Astar] waypoint generated!! waypoint num: "
                << path_blocks.size() << std::endl;
      std::reverse(path_blocks.begin(), path_blocks.end());
      block_path_ = path_blocks;
      return end_distance;
    } else {
      std::cout << "[WARNING] no path !! from " << std::endl
                << start_p << std::endl
                << "to " << std::endl
                << end_p << std::endl;
      return (end_p - start_p).norm();
    }
  }
}

float GridAstar::BlockPathRefine(const std::vector<int> &block_path,
                                 const Eigen::Vector3f &start_p,
                                 const Eigen::Vector3f &end_p) {
  if (block_path.empty()) {
    return (end_p - start_p).norm();
  }
  // Determine the 3D index of the start point and end point.
  int index_start_x =
      static_cast<int>(std::floor((start_p.x() - min_x_) / resolution_));
  int index_start_y =
      static_cast<int>(std::floor((start_p.y() - min_y_) / resolution_));
  int index_start_z =
      static_cast<int>(std::floor((start_p.z() - min_z_) / resolution_));
  int index_end_x =
      static_cast<int>(std::floor((end_p.x() - min_x_) / resolution_));
  int index_end_y =
      static_cast<int>(std::floor((end_p.y() - min_y_) / resolution_));
  int index_end_z =
      static_cast<int>(std::floor((end_p.z() - min_z_) / resolution_));
  // Determine the index of block in the vector according to the block_id.
  std::unordered_map<int, int> block_id_map;
  for (int i = 0; i < merge_map_3d_.size(); ++i) {
    if (block_id_map.find(merge_map_3d_[i].block_id_) == block_id_map.end()) {
      block_id_map[merge_map_3d_[i].block_id_] = i;
    } else {
      std::cout << "Error: block_id_map has duplicate key." << std::endl;
      std::cout << "block_id: " << merge_map_3d_[i].block_id_ << std::endl;
    }
  }
  // Construct the constraints of keyframes.
  std::vector<Block2D> key_frames;
  key_frames.reserve(1000);
  std::vector<int> key_frame_x;
  key_frame_x.reserve(1000);
  std::vector<int> key_frames_index;
  key_frames_index.reserve(1000);
  // Segment 0
  {
    const int node_id = block_path.front();
    const KeyBlock &block2d = graph_table_.nodes_[node_id].key_block_;
    const int block_x = block2d.x_;
    const int x_step =
        index_start_x == block_x ? 0 : (index_start_x < block_x ? 1 : -1);
    const Block3D &block3d = merge_map_3d_[block_id_map[block2d.block_id_]];
    const int x_min = block3d.x_min_;
    for (int i = index_start_x; i != block_x; i += x_step) {
      key_frames.emplace_back(block3d.blocks_[i - x_min]);
      key_frame_x.emplace_back(i);
    }
    key_frames.emplace_back(block2d.block_);
    key_frame_x.emplace_back(block_x);
    key_frames_index.emplace_back(0);
    key_frames_index.emplace_back(key_frames.size() - 1);
  }
  // Segment 1 to n-1
  for (int i = 1; i < block_path.size(); ++i) {
    const int last_node_id = block_path[i - 1];
    const KeyBlock &last_block2d = graph_table_.nodes_[last_node_id].key_block_;
    const int node_id = block_path[i];
    const KeyBlock &block2d = graph_table_.nodes_[node_id].key_block_;
    const int last_block_x = last_block2d.x_;
    const int block_x = block2d.x_;
    const int x_step =
        last_block_x == block_x ? 0 : (last_block_x < block_x ? 1 : -1);
    const Block3D &block3d = merge_map_3d_[block_id_map[block2d.block_id_]];
    const int x_min = block3d.x_min_;
    for (int i = last_block_x + x_step; i != block_x; i += x_step) {
      key_frames.emplace_back(block3d.blocks_[i - x_min]);
      key_frame_x.emplace_back(i);
    }
    key_frames.emplace_back(block2d.block_);
    key_frame_x.emplace_back(block_x);
    key_frames_index.emplace_back(key_frames.size() - 1);
  }
  // Segment n
  {
    const int node_id = block_path.back();
    const KeyBlock &block2d = graph_table_.nodes_[node_id].key_block_;
    const int block_x = block2d.x_;
    const int x_step =
        block_x == index_end_x ? 0 : (block_x < index_end_x ? 1 : -1);
    const Block3D &block3d = merge_map_3d_[block_id_map[block2d.block_id_]];
    const int x_min = block3d.x_min_;
    for (int i = block_x + x_step; i != index_end_x; i += x_step) {
      key_frames.emplace_back(block3d.blocks_[i - x_min]);
      key_frame_x.emplace_back(i);
    }
    key_frames_index.emplace_back(key_frames.size() - 1);
  }
  // iLQR Path Optimization.
  const int num_key_frames = key_frames.size();
  const int num_constraints = key_frames_index.size();
  Eigen::Matrix<float, 2, 4> F;
  // clang-format off
  F <<
  1.0, 0.0, 1.0, 0.0,
  0.0, 1.0, 0.0, 1.0;
  // clang-format on
  std::vector<Eigen::Matrix2f> K_mats;
  K_mats.reserve(num_key_frames);
  K_mats.resize(num_constraints, Eigen::Matrix2f::Zero());
  std::vector<Eigen::Vector2f> k_vecs(num_key_frames, Eigen::Vector2f::Zero());
  k_vecs.reserve(num_key_frames);
  k_vecs.resize(num_constraints, Eigen::Vector2f::Zero());
  std::vector<Eigen::Vector4f> xu_vecs(num_key_frames, Eigen::Vector4f::Zero());
  xu_vecs.reserve(num_key_frames);
  xu_vecs.resize(num_constraints, Eigen::Vector4f::Zero());
  std::vector<Eigen::Vector2f> x_hat_vecs(num_key_frames,
                                          Eigen::Vector2f::Zero());
  x_hat_vecs.reserve(num_key_frames);
  x_hat_vecs.resize(num_constraints, Eigen::Vector2f::Zero());
  // Construct the initial guess.
  std::cout << "Construct the initial guess..." << std::endl;
  xu_vecs[0] << index_start_y, index_start_z, 0.0, 0.0;
  x_hat_vecs[0] << index_start_y, index_start_z;
  for (int i = 1; i < num_constraints; ++i) {
    const int index = key_frames_index[i];
    const int y_lb = key_frames[index].y_min_;
    const int y_ub = key_frames[index].y_max_;
    const int y_initial = (y_lb + y_ub) / 2;
    RangeVoxel z_range;
    key_frames[index].GetRangeAtY(y_initial, &z_range);
    const int z_initial = (z_range.min_ + z_range.max_) / 2;
    xu_vecs[i].block<2, 1>(0, 0) << y_initial, z_initial;
    xu_vecs[i - 1].block<2, 1>(2, 0) =
        xu_vecs[i].block<2, 1>(0, 0) - xu_vecs[i - 1].block<2, 1>(0, 0);
  }
  std::cout << "start ilqr optimization..." << std::endl;
  bool is_ilqr_success = false;
  float cost_sum = 0.0;
  float last_cost_sum = cost_sum;
  float path_length = 0.0;
  float last_path_length = path_length;
  while (!is_ilqr_success) {
    // Iteration Loop.
    for (int iter = 0; iter < kMaxIteration; ++iter) {
      Eigen::Matrix2f V;
      Eigen::Vector2f v;
      cost_sum = 0.0;
      std::pair<float, float> delta_V(0.0, 0.0);
      // Backward Pass.
      for (int k = num_constraints - 1; k >= 0; --k) {
        Eigen::Matrix4f Q;
        Eigen::Vector4f q;
        // Cost can be calculated in parallel.
        if (k == num_constraints - 1) {
          const std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost =
              GetTermCost(xu_vecs[k], index_end_y, index_end_z);
          cost_sum += GetRealTermCost(xu_vecs[k], index_end_y, index_end_z);
          Q = cost.first;
          q = cost.second;
        } else {
          const int real_index = key_frames_index[k];
          const int y_lb = key_frames[real_index].y_min_;
          const int y_ub = key_frames[real_index].y_max_;
          const int y_id =
              std::clamp(static_cast<int>(xu_vecs[k](0)), y_lb, y_ub);
          RangeVoxel z_range;
          key_frames[real_index].GetRangeAtY(y_id, &z_range);
          const std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost =
              GetCost(xu_vecs[k], y_lb, y_ub, z_range.min_, z_range.max_);
          cost_sum +=
              GetRealCost(xu_vecs[k], y_lb, y_ub, z_range.min_, z_range.max_);
          Q = cost.first + F.transpose() * V * F;
          q = cost.second + F.transpose() * v;
        }
        const Eigen::Matrix2f Qxx = Q.block<2, 2>(0, 0);
        const Eigen::Matrix2f Qxu = Q.block<2, 2>(0, 2);
        const Eigen::Matrix2f Qux = Q.block<2, 2>(2, 0);
        const Eigen::Matrix2f Quu = Q.block<2, 2>(2, 2);
        const Eigen::Vector2f qx = q.block<2, 1>(0, 0);
        const Eigen::Vector2f qu = q.block<2, 1>(2, 0);
        const Eigen::Matrix2f K_mat = K_mats[k];
        const Eigen::Vector2f k_vec = k_vecs[k];
        K_mats[k] = -Quu.inverse() * Qux;
        k_vecs[k] = -Quu.inverse() * qu;
        V = Qxx + Qxu * K_mat + K_mat.transpose() * Qux +
            K_mat.transpose() * Quu * K_mat;
        v = qx + Qxu * k_vec + K_mat.transpose() * qu +
            K_mat.transpose() * Quu * k_vec;
        delta_V.first += k_vecs[k].transpose() * qu;
        delta_V.second += 0.5 * k_vecs[k].transpose() * Quu * k_vecs[k];
      }

      float alpha = 1.0;
      bool is_line_search_done = false;
      int line_search_iter = 0;
      // TODO: Parellel line search.
      const std::vector<Eigen::Vector4f> cur_xu_vecs(xu_vecs);
      while (!is_line_search_done && line_search_iter < kMaxLineSearchIter) {
        ++line_search_iter;
        // Forward Pass.
        float next_cost_sum = 0.0;
        for (int k = 0; k < num_constraints - 1; ++k) {
          const Eigen::Vector2f x = cur_xu_vecs[k].block<2, 1>(0, 0);
          const Eigen::Vector2f u = cur_xu_vecs[k].block<2, 1>(2, 0);
          xu_vecs[k].block<2, 1>(2, 0) =
              K_mats[k] * (x_hat_vecs[k] - x) + alpha * k_vecs[k] + u;
          xu_vecs[k].block<2, 1>(0, 0) = x_hat_vecs[k];
          x_hat_vecs[k + 1] = F * xu_vecs[k];
          // Calculate the new cost.
          const int real_index = key_frames_index[k];
          const int y_lb = key_frames[real_index].y_min_;
          const int y_ub = key_frames[real_index].y_max_;
          const int y_id =
              std::clamp(static_cast<int>(xu_vecs[k](0)), y_lb, y_ub);
          RangeVoxel z_range;
          key_frames[real_index].GetRangeAtY(y_id, &z_range);
          next_cost_sum +=
              GetRealCost(xu_vecs[k], y_lb, y_ub, z_range.min_, z_range.max_);
        }
        xu_vecs[num_constraints - 1].block<2, 1>(0, 0) =
            x_hat_vecs[num_constraints - 1];
        next_cost_sum += GetRealTermCost(xu_vecs[num_constraints - 1],
                                         index_end_y, index_end_z);
        // Check if J satisfy line search condition.
        const float ratio_decrease =
            (next_cost_sum - cost_sum) /
            (alpha * (delta_V.first + alpha * delta_V.second));
        if (line_search_iter < kMaxLineSearchIter &&
            (ratio_decrease <= 1e-4 || ratio_decrease >= 10)) {
          alpha *= 0.5;
        } else {
          is_line_search_done = true;
        }
      }

      // Calculate the path length.
      path_length = 0.0;
      for (int k = 0; k < num_constraints - 1; ++k) {
        const float delta_x = key_frame_x[key_frames_index[k + 1]] -
                              key_frame_x[key_frames_index[k]];
        path_length += std::hypot(xu_vecs[k].block<2, 1>(2, 0).norm(), delta_x);
      }
      // Terminate condition.
      if (iter > 0 &&
          std::fabs(path_length - last_path_length) < kConvergenceThreshold) {
        break;
      } else {
        last_path_length = path_length;
      }
    }
    // Check if the path is feasible.
    is_ilqr_success = true;
    // Add extra constraints.
  }

  ilqr_path_.clear();
  ilqr_path_.reserve(num_constraints);
  for (int i = 0; i < num_constraints; ++i) {
    float x = key_frame_x[key_frames_index[i]];
    float y = xu_vecs[i](0);
    float z = xu_vecs[i](1);
    ilqr_path_.emplace_back();
    ilqr_path_.back() = {x, y, z};
  }
  return path_length;
}

std::pair<Eigen::Matrix4f, Eigen::Vector4f>
GridAstar::GetCost(const Eigen::Vector4f &xu, const float y_lb,
                   const float y_ub, const float z_lb, const float z_ub) {
  float dcy = 0.0;
  float ddcy = 0.0;
  float dcz = 0.0;
  float ddcz = 0.0;
  // Constraints on y.
  if (xu(0) > y_ub) {
    dcy = kBndWeight * (xu(0) - y_ub);
    ddcy = kBndWeight;
  } else if (xu(0) < y_lb) {
    dcy = kBndWeight * (xu(0) - y_lb);
    ddcy = kBndWeight;
  }
  // Constraints on z.
  if (xu(1) > z_ub) {
    dcz = kBndWeight * (xu(1) - z_ub);
    ddcz = kBndWeight;
  } else if (xu(1) < z_lb) {
    dcz = kBndWeight * (xu(1) - z_lb);
    ddcz = kBndWeight;
  }
  std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost;
  // clang-format off
  cost.first << 
  ddcy, 0.0, 0.0, 0.0,
  0.0, ddcz, 0.0, 0.0,
  0.0, 0.0, kWeight, 0.0, 
  0.0, 0.0, 0.0, kWeight;
  cost.second <<
  dcy,
  dcz,
  kWeight * xu(2),
  kWeight * xu(3);
  // clang-format on
  return cost;
}

float GridAstar::GetRealCost(const Eigen::Vector4f &xu, const float y_lb,
                             const float y_ub, const float z_lb,
                             const float z_ub) {
  float real_cost = 0.0;
  real_cost += kWeight * 0.5 * xu(2) * xu(2) + kWeight * 0.5 * xu(3) * xu(3);
  if (xu(0) > y_ub) {
    const float delta_y_square = (xu(0) - y_ub) * (xu(0) - y_ub);
    real_cost += kBndWeight * 0.5 * delta_y_square;
  } else if (xu(0) < y_lb) {
    const float delta_y_square = (xu(0) - y_lb) * (xu(0) - y_lb);
    real_cost += kBndWeight * 0.5 * delta_y_square;
  }
  if (xu(1) > z_ub) {
    const float delta_z_square = (xu(1) - z_ub) * (xu(1) - z_ub);
    real_cost += kBndWeight * 0.5 * delta_z_square;
  } else if (xu(1) < z_lb) {
    const float delta_z_square = (xu(1) - z_lb) * (xu(1) - z_lb);
    real_cost += kBndWeight * 0.5 * delta_z_square;
  }
  return real_cost;
}

std::pair<Eigen::Matrix4f, Eigen::Vector4f>
GridAstar::GetTermCost(const Eigen::Vector4f &xu, const float target_y,
                       const float target_z) {
  std::pair<Eigen::Matrix4f, Eigen::Vector4f> cost;
  // clang-format off
  cost.first << 
  kTermWeight, 0.0, 0.0, 0.0,
  0.0, kTermWeight, 0.0, 0.0,
  0.0, 0.0, kWeight, 0.0,
  0.0, 0.0, 0.0, kWeight;
  cost.second << 
  kTermWeight * (xu(0) - target_y),
  kTermWeight * (xu(1) - target_z),
  kWeight * xu(2),
  kWeight * xu(3);
  // clang-format on
  return cost;
}

float GridAstar::GetRealTermCost(const Eigen::Vector4f &xu,
                                 const float target_y, const float target_z) {
  float real_cost = 0.0;
  real_cost += kTermWeight * 0.5 * (xu(0) - target_y) * (xu(0) - target_y) +
               kTermWeight * 0.5 * (xu(1) - target_z) * (xu(1) - target_z);
  real_cost += kWeight * 0.5 * xu(2) * xu(2) + kWeight * 0.5 * xu(3) * xu(3);
  return real_cost;
}

inline float GridAstar::CalHeurScore(const std::shared_ptr<GridAstarNode> &node,
                                     const Eigen::Vector3f &end_p) {
  float delta_x = end_p.x() - (node->index_x_ * resolution_ + min_x_);
  float delta_y = end_p.y() - (node->index_y_ * resolution_ + min_y_);
  float delta_z = end_p.z() - (node->index_z_ * resolution_ + min_z_);
  return std::abs(delta_x) + std::abs(delta_y) + 10.0 * std::abs(delta_z);
}
