#include "m3_explorer/astar.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <geometry_msgs/PoseStamped.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

float Astar::astar_path_distance(const octomap::OcTree *ocmap,
                                 const Eigen::Vector3f &start_p,
                                 const Eigen::Vector3f &end_p) {
  vector<Eigen::Vector3f> expand_offset = {{0.2, 0.0, 0.0}, {-0.2, 0.0, 0.0},
                                           {0.0, 0.2, 0.0}, {0.0, -0.2, 0.0},
                                           {0.0, 0.0, 0.2}, {0.0, 0.0, -0.2}};
  const int expand_size = expand_offset.size();

  priority_queue<AstarNode, vector<AstarNode>, AstarNodeCmp> astar_q;
  vector<AstarNode> closed_list;
  closed_list.reserve(99999);
  // size of closed_list, maybe faster than closed_list.size()
  int count = 0;

  // state: 0 -> open; 1 -> closed
  unordered_map<AstarNode, int, AstarNodeHash> node_state;
  unordered_map<AstarNode, float, AstarNodeHash> node_g_score;

  // map<AstarNode, int, AstarMapCmp> node_state;
  // map<AstarNode, float, AstarMapCmp> node_g_score;

  bool is_path_found = false;

  AstarNode root(start_p);
  root.father_id_ = -1;
  root.g_score_ = 0.0;
  root.h_score_ = calc_h_score(root.position_, end_p);
  root.f_score_ = root.g_score_ + root.h_score_;
  astar_q.push(root);
  node_state[root] = 0;
  node_g_score[root] = 0.0;

  while (!astar_q.empty()) {
    // cin.get();
    // cout << "top node: " << endl;
    // selection
    AstarNode node = astar_q.top();
    astar_q.pop();
    // add node to closed list
    // g值更新导致节点重复
    if (node_state[node] == 1) {
      continue;
    }
    // closed_list[count] = node;
    closed_list.emplace_back(node);
    node_state[node] = 1;

    // cout << (node.position_) << ", " << node.f_score_ << endl << endl;

    if ((node.position_ - end_p).norm() < 0.2) {
      is_path_found = true;
      break;
    }

    // expansion
    Eigen::Vector3f next_pos;

    for (int i = 0; i < expand_size; ++i) {
      // cin.get();
      next_pos = node.position_ + expand_offset[i];
      // cout << next_pos << endl;
      // check next node is valid
      if (next_pos.z() > max_z_ || next_pos.z() < min_z_) {
        continue;
      }

      bool is_next_node_valid = is_path_valid(ocmap, node.position_, next_pos);
      if (!is_next_node_valid) {
        continue;
      }

      AstarNode next_node(next_pos);
      // check if node is in open/closed list
      if (node_state.find(next_node) == node_state.end()) {
        node_state[next_node] = 0;
      } else {
        if (node_state[next_node] == 0 &&
            node.g_score_ + 0.2 > node_g_score[next_node]) {
          continue;
        }
      }
      next_node.father_id_ = count;
      next_node.h_score_ = calc_h_score(next_node.position_, end_p);
      next_node.g_score_ = node.g_score_ + 0.2;
      node_g_score[next_node] = next_node.g_score_;
      next_node.f_score_ = next_node.g_score_ + next_node.h_score_;
      astar_q.push(next_node);
    }

    count++;
  }

  if (is_path_found) {
    float distance = 0.0;
    path_.clear();
    // add accurate end point
    path_.emplace_back(end_p);
    path_.emplace_back(closed_list[count]);

    Eigen::Vector3f last = closed_list[count].position_;
    distance += (last - end_p).norm();

    int id = closed_list[count].father_id_;
    while (id != -1) {
      distance += (closed_list[id].position_ - last).norm();
      last = closed_list[id].position_;

      path_.emplace_back(closed_list[id]);
      id = closed_list[id].father_id_;
    }
    reverse(path_.begin(), path_.end());
    cout << "[Astar] waypoint generated!! waypoint num: " << path_.size()
         << ", select node num: " << count << endl;
    return distance;
  } else {
    cout << "[WARNING] no path !! from " << endl
         << start_p << endl
         << "to " << endl
         << end_p << endl;
    return (end_p - start_p).norm();
  }
}

float Astar::calc_h_score(const Eigen::Vector3f &start_p,
                          const Eigen::Vector3f &end_p) {
  // return (end_p - start_p).norm();
  // return (end_p - start_p).lpNorm<1>();
  Eigen::Vector3f delta = end_p - start_p;
  return abs(delta.x()) + abs(delta.y()) + 10.0 * abs(delta.z());
}

bool Astar::is_path_valid(const octomap::OcTree *ocmap,
                          const Eigen::Vector3f &cur_pos,
                          const Eigen::Vector3f &next_pos) {
  octomap::OcTreeNode *oc_node = ocmap->search(next_pos.x(), next_pos.y(), next_pos.z());
  if (oc_node == nullptr) {
    // cout << "unknown" << endl;
    return false;
  }

  if (oc_node != nullptr && ocmap->isNodeOccupied(oc_node)) {
    // cout << "occ" << endl;
    return false;
  }

  return true;
}
