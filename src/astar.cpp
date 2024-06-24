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
  vector<Eigen::Vector3f> expand_offset = {
      {0.2, 0.0, 0.0}, {-0.2, 0.0, 0.0}, {0.0, 0.2, 0.0},   {0.0, -0.2, 0.0},
      {0.0, 0.0, 0.2}, {0.0, 0.0, -0.2}};

  priority_queue<AstarNode, vector<AstarNode>, AstarNodeCmp> astar_q;
  vector<AstarNode> closed_list;
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
    // 可以考虑将priority_queue替换为set，因为g值更新导致节点重复
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

    for (int i = 0; i < expand_offset.size(); ++i) {
      // cin.get();
      next_pos = node.position_ + expand_offset[i];
      // cout << next_pos << endl;
      // check next node is valid
      bool is_next_node_valid = is_path_valid(ocmap, node.position_, next_pos);

      if (!is_next_node_valid) {
        // cout << "not valid " << i << endl;
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
    path.clear();

    float end_yaw = atan2(end_p.y() - closed_list[count].position.y(),
                          end_p.x() - closed_list[count].position.x());
    // add accurate end point
    PathNode end(end_p, end_yaw);
    path.push_back(end);

    int id = closed_list[count].father_id;
    path.push_back(closed_list[count]);
    while (id != -1) {
      path.push_back(closed_list[id]);
      id = closed_list[id].father_id;
    }
    reverse(path.begin(), path.end());
    cout << "[Hastar] waypoint generated!! waypoint num: " << path.size()
         << endl;
    return ;
  } else {
    cout << "[WARNING] no path !! from " << endl << start_p << endl << "to " << endl << end_p << endl;
    return (end_p - start_p).norm();
  }  
}

float Astar::calc_h_score(const Eigen::Vector3f &start_p,
                          const Eigen::Vector3f &end_p) {
  return (end_p - start_p).norm();
}

bool Astar::is_path_valid(const octomap::OcTree *ocmap,
                          const Eigen::Vector3f &cur_pos,
                          const Eigen::Vector3f &next_pos) {
  octomap::point3d next_pos_check(next_pos.x(), next_pos.y(), next_pos.z());
  octomap::OcTreeNode *oc_node = ocmap->search(next_pos_check);
  if (oc_node == nullptr){
    // cout << "unknown" << endl;
    return false;
  }

  if (oc_node != nullptr && ocmap->isNodeOccupied(oc_node)) {
    // cout << "occ" << endl;
    return false;
  }

  return true;
}
