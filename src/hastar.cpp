#include "m3_explorer/hastar.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <geometry_msgs/PoseStamped.h>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>

const float MAX_VEL = 0.5;
const float MAX_ACC = 1.0;

bool Hastar::search_path(const octomap::OcTree *ocmap,
                         const Eigen::Vector3f &start_p,
                         const Eigen::Vector3f &end_p, const float &yaw) {
  vector<float> omega = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0};

  priority_queue<PathNode, vector<PathNode>, NodeCmp> hastar_q;
  vector<PathNode> closed_list;
  // size of closed_list, maybe faster than closed_list.size()
  int count = 0;
  // state: 0 -> open; 1 -> closed

  unordered_map<PathNode, int, NodeHash> node_state;
  unordered_map<PathNode, float, NodeHash> node_g_score;

  // map<PathNode, int, MapCmp> node_state;
  // map<PathNode, float, MapCmp> node_g_score;

  PathNode root(start_p, yaw);
  root.father_id = -1;
  root.g_score = 0.0;
  root.h_score = calc_h_score(ocmap, root.position, end_p);
  root.f_score = root.g_score + root.h_score;
  hastar_q.push(root);
  node_state[root] = 0;
  node_g_score[root] = 0.0;
  bool is_path_found = false;

  while (!hastar_q.empty()) {
    // selection
    PathNode node = hastar_q.top();
    hastar_q.pop();
    // add node to closed list
    // 可以考虑将priority_queue替换为set，因为g值更新导致节点重复
    if (node_state[node] == 1) {
      continue;
    }
    // closed_list[count] = node;
    closed_list.emplace_back(node);
    node_state[node] = 1;

    // cout << (node.position - end_p).norm() << endl << endl;
    // cout << (node.position) << node.vel << endl << endl;

    // 终点处应当约束速度为0,此处可以用庞特里亚金求解
    if ((node.position - end_p).norm() < 0.2) {
      // if((node.position - end_p).norm() < 0.2){
      is_path_found = true;
      cout << "[Hastar] find_path !!!" << endl;
      break;
    }

    // expansion
    Eigen::Vector3f next_pos;
    float next_yaw;
    for (int i = 0; i < omega.size(); ++i) {
      // 保证yaw在[-pi, pi]之间
      next_yaw =
          atan2(sin(node.yaw + omega[i] * tau), cos(node.yaw + omega[i] * tau));
      if (omega[i] * tau == 0) {
        Eigen::Vector3f start_vel = {MAX_VEL * cos(node.yaw),
                                     MAX_VEL * sin(node.yaw), 0};
        next_pos = node.position + start_vel * tau;
      } else {
        float rad = MAX_VEL / omega[i];
        float std_x = rad * sin(omega[i] * tau);
        float std_y = rad * (1 - cos(omega[i] * tau));
        Eigen::Vector3f offset(cos(node.yaw) * std_x - sin(node.yaw) * std_y,
                               sin(node.yaw) * std_x + cos(node.yaw) * std_y,
                               0.0);
        next_pos = node.position + offset;
      }

      // check next node is valid
      bool is_next_node_valid = is_path_valid(ocmap, node.position, next_pos);

      if (!is_next_node_valid) {
        continue;
      }

      PathNode next_node(next_pos, next_yaw);
      // check if node is in open/closed list
      if (node_state.find(next_node) == node_state.end()) {
        node_state[next_node] = 0;
      } else {
        if (node_state[next_node] == 0 &&
            node.g_score + tau > node_g_score[next_node]) {
          continue;
        }
      }
      next_node.father_id = count;
      next_node.father_yaw_offset = omega[i] * tau;
      next_node.h_score = calc_h_score(ocmap, next_node.position, end_p);
      next_node.g_score = node.g_score + tau;
      node_g_score[next_node] = next_node.g_score;
      next_node.f_score = next_node.g_score + next_node.h_score;
      hastar_q.push(next_node);
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
    trajectory_generate(yaw);
    return true;
  } else {
    cout << "[Hastar] no path" << endl;
    return false;
  }
}

float Hastar::calc_h_score(const octomap::OcTree *ocmap,
                           const Eigen::Vector3f &start_p,
                           const Eigen::Vector3f &end_p) {
  return (end_p - start_p).norm() / MAX_VEL;
}

bool Hastar::is_path_valid(const octomap::OcTree *ocmap,
                           const Eigen::Vector3f &cur_pos,
                           const Eigen::Vector3f &next_pos) {
  octomap::point3d next_pos_check(next_pos.x(), next_pos.y(), next_pos.z());
  octomap::OcTreeNode *oc_node = ocmap->search(next_pos_check);
  if (oc_node == nullptr)
    return false;

  float bbx_x0 = min(cur_pos.x(), next_pos.x()) - 0.4;
  float bbx_x1 = max(cur_pos.x(), next_pos.x()) + 0.4;
  float bbx_y0 = min(cur_pos.y(), next_pos.y()) - 0.4;
  float bbx_y1 = max(cur_pos.y(), next_pos.y()) + 0.4;
  float bbx_z0 = min(cur_pos.z(), next_pos.z()) - 0.1;
  float bbx_z1 = max(cur_pos.z(), next_pos.z()) + 0.1;

  for (float check_x = bbx_x0; check_x <= bbx_x1; check_x += 0.1) {
    for (float check_y = bbx_y0; check_y <= bbx_y1; check_y += 0.1) {
      for (float check_z = bbx_z0; check_z <= bbx_z1; check_z += 0.1) {
        octomap::point3d check(check_x, check_y, check_z);
        octomap::OcTreeNode *oc_node = ocmap->search(check);
        if (oc_node != nullptr && ocmap->isNodeOccupied(oc_node)) {
          return false;
        }
      }
    }
  }
  return true;
}

// bool Hastar::is_path_valid(const octomap::OcTree* ocmap, const
// Eigen::Vector3f& cur_pos, const Eigen::Vector3f& next_pos)
// {
//     return true;
// }

bool Hastar::trajectory_generate(const float &yaw) {
  traj.clear();
  if (path.size() > 2) {
    for (int i = 0; i < path.size() - 2; i++) {
      for (float time = 0.0; time < tau; time += traj_sample) {
        Traj traj_point;
        traj_point.yaw =
            path[i].yaw + path[i + 1].father_yaw_offset * time / tau;
        traj_point.vel << MAX_VEL * cos(traj_point.yaw),
            MAX_VEL * sin(traj_point.yaw), 0.0;
        if (path[i + 1].father_yaw_offset == 0) {
          Eigen::Vector3f start_vel = {MAX_VEL * cos(path[i].yaw),
                                       MAX_VEL * sin(path[i].yaw), 0.0};
          traj_point.pos = path[i].position + start_vel * time;
          traj_point.acc = Eigen::Vector3f::Zero();
          traj_point.yaw_rate = 0.0;
        } else {
          float delta_yaw = path[i + 1].father_yaw_offset * time / tau;
          float rad = MAX_VEL * time / delta_yaw;
          float std_x = rad * sin(delta_yaw);
          float std_y = rad * (1 - cos(delta_yaw));
          Eigen::Vector3f offset(
              cos(path[i].yaw) * std_x - sin(path[i].yaw) * std_y,
              sin(path[i].yaw) * std_x + cos(path[i].yaw) * std_y, 0.0);
          traj_point.pos = path[i].position + offset;
          float new_yaw = traj_point.yaw + M_PI / 2.0;
          traj_point.acc << MAX_VEL * MAX_VEL / rad * cos(new_yaw),
              MAX_VEL * MAX_VEL / rad * sin(new_yaw), 0.0;
          traj_point.yaw_rate = path[i + 1].father_yaw_offset / tau;
        }
        traj.push_back(traj_point);
      }
    }
    Traj traj_point;
    traj_point.acc = Eigen::Vector3f::Zero();
    traj_point.vel = Eigen::Vector3f::Zero();
    traj_point.pos = path.back().position;
    // keep yaw constant
    traj_point.yaw = traj.back().yaw;
    traj.push_back(traj_point);
    cout << "[Hastar] Traj generate OK!! traj point num: " << traj.size()
         << endl;
  } else if (path.size() == 2) {
    Traj traj_point;
    traj_point.acc = Eigen::Vector3f::Zero();
    traj_point.vel = Eigen::Vector3f::Zero();
    traj_point.pos = path.back().position;
    // keep yaw constant
    traj_point.yaw = yaw;
    traj.push_back(traj_point);
  } else {
    return false;
  }
  return true;
}
