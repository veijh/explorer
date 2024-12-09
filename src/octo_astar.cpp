#include "m3_explorer/octo_astar.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <geometry_msgs/PoseStamped.h>
#include <map>
#include <memory>
#include <queue>
#include <stack>
#include <string>
#include <unordered_map>
#include <chrono>

const std::vector<Eigen::Vector3f> center_offset = {
    {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {1.0, 1.0, -1.0},
    {-1.0, -1.0, 1.0},  {1.0, -1.0, 1.0},  {-1.0, 1.0, 1.0},  {1.0, 1.0, 1.0}};

const std::vector<Eigen::Vector3f> expand_offset = {
    {-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0},  {0.0, -1.0, 0.0},
    {0.0, 1.0, 0.0},  {0.0, 0.0, -1.0}, {0.0, 0.0, 1.0}};

const std::vector<std::vector<int>> adj_child = {{1, 3, 5, 7}, {0, 2, 4, 6},
                                                 {2, 3, 6, 7}, {0, 1, 4, 5},
                                                 {4, 5, 6, 7}, {0, 1, 2, 3}};

OctoAstar::OctoAstar(const octomap::OcTree *ocmap) : ocmap_(ocmap){
  root_node_ = std::make_shared<OctoNode>(ocmap_->getRoot(), Eigen::Vector3f(0.0, 0.0, 0.0), 0.1 * 65536, nullptr);
}

std::shared_ptr<OctoNode> OctoAstar::search_octonode(const Eigen::Vector3f &p) {
  std::shared_ptr<OctoNode> octo_node = root_node_;
  while (ocmap_->nodeHasChildren(octo_node->node_)) {
    Eigen::Vector3f node_center = octo_node->center_;
    int child_id = 1 * (p.x() > node_center.x()) +
                   2 * (p.y() > node_center.y()) +
                   4 * (p.z() > node_center.z());
    if (ocmap_->nodeChildExists(octo_node->node_, child_id)) {
      if (octo_node->child_node_[child_id] == nullptr) {
        octo_node->child_node_[child_id] = std::make_shared<OctoNode>(
            ocmap_->getNodeChild(octo_node->node_, child_id),
            node_center + 0.5 * octo_node->half_size_ * center_offset[child_id],
            octo_node->half_size_, octo_node);
      }
      octo_node = octo_node->child_node_[child_id];
    } else {
      return nullptr;
    }
  }
  return octo_node;
}

float OctoAstar::astar_path_distance(const Eigen::Vector3f &start_p,
                                     const Eigen::Vector3f &end_p) {

  const int expand_size = expand_offset.size();

  // search the node at start_p from top to bottom
  std::shared_ptr<OctoNode> octo_node = search_octonode(start_p);
  if(octo_node == nullptr) {
    return 999.0;
  }

  std::priority_queue<std::shared_ptr<OctoAstarNode>,
                      std::vector<std::shared_ptr<OctoAstarNode>>,
                      OctoAstarNodeCmp>
      astar_q;

  bool is_path_found = false;
  int count = 0;

  OctoAstarNodeKey astar_root_key(octo_node->center_);
  std::shared_ptr<OctoAstarNode> astar_root = std::make_shared<OctoAstarNode>(octo_node->center_);
  astar_root->octo_node_ = octo_node;
  astar_root->father_node_ = nullptr;
  astar_root->g_score_ = 0.0;
  astar_root->h_score_ = calc_h_score(astar_root->position_, end_p);
  astar_root->f_score_ = astar_root->g_score_ + astar_root->h_score_;
  astar_root->state = 0;
  node_at_key.emplace(astar_root_key, astar_root);
  astar_q.push(astar_root);
  std::shared_ptr<OctoAstarNode> end_astar_node = nullptr;

  while (!astar_q.empty()) {
    ++count;
    std::shared_ptr<OctoAstarNode> astar_node = astar_q.top();
    std::shared_ptr<OctoNode> cur_octo_node = astar_node->octo_node_;
    astar_q.pop();

    // skip the same node due to the update of g_score
    if (astar_node->state == 1) {
      continue;
    }
    // set node to closed
    astar_node->state = 1;
    if (cur_octo_node->is_in_bbx(end_p)) {
      is_path_found = true;
      end_astar_node = astar_node;
      break;
    }

    // expand in six directions
    for (int i = 0; i < expand_size; ++i) {
      Eigen::Vector3f next_pos = astar_node->position_ + cur_octo_node->size_ * expand_offset[i];

      // search for free adjacent leaf octo node
      // go from bottom to top until bbx contains next_pos
      std::shared_ptr<OctoNode> next_node = cur_octo_node->father_node_;
      while (!next_node->is_in_bbx(next_pos)) {
        next_node = next_node->father_node_;
      }
      bool is_adj_node_null = false;
      // go from top to bottom until reaching specified depth
      while (ocmap_->nodeHasChildren(next_node->node_) &&
             next_node->size_ > cur_octo_node->size_) {
        Eigen::Vector3f node_center = next_node->center_;
        int child_id = 1 * (next_pos.x() > node_center.x()) +
                       2 * (next_pos.y() > node_center.y()) +
                       4 * (next_pos.z() > node_center.z());
        if (ocmap_->nodeChildExists(next_node->node_, child_id)) {
          if (next_node->child_node_[child_id] == nullptr) {
            next_node->child_node_[child_id] = std::make_shared<OctoNode>(
                ocmap_->getNodeChild(next_node->node_, child_id),
                node_center +
                    0.5 * next_node->half_size_ * center_offset[child_id],
                next_node->half_size_, next_node);
          }
          next_node = next_node->child_node_[child_id];
        } else {
          is_adj_node_null = true;
          break;
        }
      }

      if (is_adj_node_null) {
        continue;
      }

      // big node to small node, may add multi nodes
      if (ocmap_->nodeHasChildren(next_node->node_)) {
        // bfs
        std::queue<std::shared_ptr<OctoNode>> bfs_q;
        bfs_q.push(next_node);
        while (!bfs_q.empty()) {
          std::shared_ptr<OctoNode> bfs_node = bfs_q.front();
          bfs_q.pop();
          
          if (!ocmap_->nodeHasChildren(bfs_node->node_)) {
            if(!ocmap_->isNodeOccupied(bfs_node->node_)){
              add_node_to_q(bfs_node, astar_node, end_p, astar_q);
            }
            continue;
          }

          if(bfs_node->size_ < 0.2) continue;

          for (int index = 0; index < 4; ++index) {
            if (ocmap_->nodeChildExists(bfs_node->node_, adj_child[i][index])) {
              octomap::OcTreeNode *childe_node =
                  ocmap_->getNodeChild(bfs_node->node_, adj_child[i][index]);
              if (ocmap_->isNodeOccupied(childe_node)) {
                continue;
              }
              std::shared_ptr<OctoNode> bfs_nbr_node =
                  std::make_shared<OctoNode>(
                      childe_node,
                      bfs_node->center_ +
                          0.5 * bfs_node->half_size_ *
                              center_offset[adj_child[i][index]],
                      bfs_node->half_size_, bfs_node);
              bfs_q.push(bfs_nbr_node);
            }
          }
        }
      } else {
        // small node to big node, only add one node, shift to the center of free leaf node
        if(!ocmap_->isNodeOccupied(next_node->node_)){
          add_node_to_q(next_node, astar_node, end_p, astar_q);
        }
      }
    }
  }

  if (is_path_found) {
    float distance = 0.0;
    path_.clear();
    // add accurate end point
    path_.emplace_back(std::make_shared<OctoAstarNode>(end_p));
    std::shared_ptr<OctoAstarNode> last_path_node = end_astar_node;
    distance += (end_astar_node->position_ - end_p).norm();
    path_.emplace_back(end_astar_node);
    std::shared_ptr<OctoAstarNode> path_node = end_astar_node->father_node_;
    while (path_node != nullptr) {
      distance += (path_node->position_ - last_path_node->position_).norm();
      path_.emplace_back(path_node);
      last_path_node = path_node;
      path_node = path_node->father_node_;
    }
    distance += (last_path_node->position_ - start_p).norm();
    path_.emplace_back(std::make_shared<OctoAstarNode>(start_p));
    reverse(path_.begin(), path_.end());
    std::cout << "[Astar] waypoint generated!! waypoint num: " << path_.size()
              << ", select node num: " << count << std::endl;
    return distance;
  } else {
    std::cout << "[WARNING] no path !! from " << std::endl
              << start_p << std::endl
              << "to " << std::endl
              << end_p << std::endl;
    return (end_p - start_p).norm();
  }
}

inline float OctoAstar::calc_h_score(const Eigen::Vector3f &start_p,
                              const Eigen::Vector3f &end_p) {
  // return (end_p - start_p).norm();
  // return (end_p - start_p).lpNorm<1>();
  Eigen::Vector3f delta = end_p - start_p;
  return abs(delta.x()) + abs(delta.y()) + 10.0 * abs(delta.z());
}

bool OctoAstar::is_path_valid(const Eigen::Vector3f &cur_pos,
                              const Eigen::Vector3f &next_pos) {
  // octomap::point3d bbx_min(std::min(cur_pos.x(), next_pos.x()) - 0.2,
  //                          std::min(cur_pos.y(), next_pos.y()) - 0.2,
  //                          std::min(cur_pos.z(), next_pos.z()) - 0.1);
  // octomap::point3d bbx_max(std::max(cur_pos.x(), next_pos.x()) + 0.2,
  //                          std::max(cur_pos.y(), next_pos.y()) + 0.2,
  //                          std::max(cur_pos.z(), next_pos.z()) + 0.1);

  // for (octomap::OcTree::leaf_bbx_iterator
  //          it = ocmap->begin_leafs_bbx(bbx_min, bbx_max),
  //          end = ocmap->end_leafs_bbx();
  //      it != end; ++it) {
  //   if (ocmap->isNodeOccupied(*it)) {
  //     return false;
  //   }
  // }
  // return true;
  octomap::OcTreeNode *oc_node = ocmap_->search(next_pos.x(), next_pos.y(), next_pos.z());
  if (oc_node == nullptr) {
    // cout << "unknown" << endl;
    return false;
  }
  if (oc_node != nullptr && ocmap_->isNodeOccupied(oc_node)) {
    // cout << "occ" << endl;
    return false;
  }
  return true;
}

inline bool OctoAstar::add_node_to_q(
    const std::shared_ptr<OctoNode> &next_node,
    const std::shared_ptr<OctoAstarNode> &cur_node,
    const Eigen::Vector3f &end_p,
    std::priority_queue<std::shared_ptr<OctoAstarNode>,
                        std::vector<std::shared_ptr<OctoAstarNode>>,
                        OctoAstarNodeCmp> &astar_q) {
  // check next node is valid
  Eigen::Vector3f next_pos = next_node->center_;
  if (next_pos.z() > max_z_ || next_pos.z() < min_z_) {
    return false;
  }

  OctoAstarNodeKey astar_key(next_pos);
  const float delta_g = (next_pos - cur_node->position_).norm();

  // check if node is in open/closed list
  if (node_at_key.find(astar_key) != node_at_key.end()) {
    std::shared_ptr<OctoAstarNode> next_astar_node = node_at_key[astar_key];
    if (next_astar_node->state == 1) {
      return false;
    }
    // next node is in open list, and the distance through the current node to
    // next node is farther.
    if (next_astar_node->state == 0 &&
        cur_node->g_score_ + delta_g > next_astar_node->g_score_) {
      return false;
    }
    // update father node and g/f_score
    next_astar_node->father_node_ = cur_node;
    next_astar_node->g_score_ = cur_node->g_score_ + delta_g;
    next_astar_node->f_score_ =
        next_astar_node->g_score_ + next_astar_node->h_score_;
    astar_q.push(next_astar_node);
  } else {
    std::shared_ptr<OctoAstarNode> next_astar_node =
        std::make_shared<OctoAstarNode>(next_pos);
    next_astar_node->octo_node_ = next_node;
    next_astar_node->father_node_ = cur_node;
    next_astar_node->state = 0;
    next_astar_node->g_score_ = cur_node->g_score_ + delta_g;
    next_astar_node->h_score_ = calc_h_score(next_pos, end_p);
    next_astar_node->f_score_ =
        next_astar_node->g_score_ + next_astar_node->h_score_;
    node_at_key[astar_key] = next_astar_node;
    astar_q.push(next_astar_node);
  }
  return true;
}
