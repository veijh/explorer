#include "m3_explorer/path_planning.h"

#include <Eigen/Dense>
#include <fstream>
#include <iostream>

geometry_msgs::PoseArray
atsp_path(const geometry_msgs::PointStamped &current_pose,
          const geometry_msgs::PoseArray &view_points,
          ros::ServiceClient &lkh_client, const string &problem_path) {
  geometry_msgs::PoseArray result;
  string par = problem_path + "/single.par";
  string atsp = problem_path + "/single.atsp";
  string tour = problem_path + "/single.tour";
  // create par file for lkh
  std::ofstream par_file(par);
  if (!par_file.is_open()) {
    std::cerr << "Failed to open par file." << std::endl;
    return result;
  }
  par_file << "PROBLEM_FILE = " << atsp << endl;
  par_file << "TOUR_FILE = " << tour << endl;
  par_file.close();

  // create atsp file for lkh
  std::ofstream atsp_file(atsp);
  if (!atsp_file.is_open()) {
    std::cerr << "Failed to open atsp file." << std::endl;
    return result;
  }
  atsp_file << "NAME : EXPLORATION" << endl;
  atsp_file << "TYPE : ATSP" << endl;
  atsp_file << "DIMENSION : " << view_points.poses.size() + 1 << endl;
  atsp_file << "EDGE_WEIGHT_TYPE: EXPLICIT" << endl;
  atsp_file << "EDGE_WEIGHT_FORMAT: FULL_MATRIX" << endl;
  atsp_file << "EDGE_WEIGHT_SECTION" << endl;
  // weight of edge from i to j
  geometry_msgs::PoseArray all_node(view_points);
  geometry_msgs::Pose start_node;
  start_node.orientation.w = 1;
  start_node.orientation.x = 0;
  start_node.orientation.y = 0;
  start_node.orientation.z = 0;
  start_node.position.x = current_pose.point.x;
  start_node.position.y = current_pose.point.y;
  start_node.position.z = current_pose.point.z;
  all_node.poses.push_back(start_node);

  cout << "writing file !!" << endl;
  for (int i = 0; i < all_node.poses.size(); i++) {
    for (int j = 0; j < all_node.poses.size(); j++) {
      if (j == all_node.poses.size() - 1 || i == j) {
        atsp_file << 0 << " ";
      } else {
        Eigen::Vector2f edge_w(
            all_node.poses[i].position.x - all_node.poses[j].position.x,
            all_node.poses[i].position.y - all_node.poses[j].position.y);
        atsp_file << (int)(edge_w.norm() * 1000.0) << " ";
      }
    }
    atsp_file << endl;
  }
  atsp_file.close();

  // call service
  lkh_ros::Solve srv;
  srv.request.problem_file = par;
  if (lkh_client.call(srv)) {
    cout << "[INFO] atsp solved" << endl;
  } else {
    cout << "[ERROR] atsp solving error!!" << endl;
  }
  // read result && return

  std::ifstream tour_file(tour);
  if (!tour_file.is_open()) {
    std::cerr << "Failed to open tour file." << std::endl;
    return result;
  }

  std::string line;
  while (std::getline(tour_file, line)) {
    if (line == "TOUR_SECTION") {
      break;
    }
  }

  vector<int> node_seq;
  while (std::getline(tour_file, line)) {
    int num = stoi(line);
    if (num == -1) {
      break;
    } else {
      node_seq.push_back(num);
    }
  }
  tour_file.close();

  // check whether the result is reasonable
  if (node_seq.size() != all_node.poses.size()) {
    std::cerr << "the result is not reasonable" << std::endl;
    return result;
  }

  result.header.frame_id = "map";
  vector<int>::iterator it =
      find(node_seq.begin(), node_seq.end(), all_node.poses.size());

  for (int i = 0; i < node_seq.size(); i++) {
    int id = *it - 1;
    result.poses.push_back(all_node.poses[id]);
    if (it + 1 == node_seq.end()) {
      it = node_seq.begin();
    } else {
      it++;
    }
  }
  return result;
}

geometry_msgs::PoseArray
amtsp_path(const octomap::OcTree *ocmap, const geometry_msgs::PointStamped &current_pose,
           const vector<geometry_msgs::PointStamped> &other_poses,
           const geometry_msgs::PoseArray &view_points,
           ros::ServiceClient &lkh_client, const string &problem_path) {
  geometry_msgs::PoseArray result;
  string par = problem_path + "/multi.par";
  string atsp = problem_path + "/multi.atsp";
  string tour = problem_path + "/multi.tour";
  string mtsp_solution = problem_path + "/mtsp_solution.tour";

  const int num_salesman = 1 + other_poses.size();
  // create par file for lkh
  std::ofstream par_file(par);
  if (!par_file.is_open()) {
    std::cerr << "Failed to open par file." << std::endl;
    return result;
  }
  par_file << "SPECIAL" << endl;
  par_file << "PROBLEM_FILE = " << atsp << endl;
  par_file << "TOUR_FILE = " << tour << endl;
  par_file << "MTSP_OBJECTIVE = MINMAX" << endl;
  par_file << "MTSP_SOLUTION_FILE = " << mtsp_solution << endl;
  par_file << "SALESMEN = " << num_salesman << endl;
  par_file << "MTSP_MIN_SIZE = 0" << endl;
  par_file.close();

  // create atsp file for lkh
  std::ofstream atsp_file(atsp);
  if (!atsp_file.is_open()) {
    std::cerr << "Failed to open atsp file." << std::endl;
    return result;
  }
  atsp_file << "NAME : EXPLORATION" << endl;
  atsp_file << "TYPE : ATSP" << endl;
  // view point + uavs + virtual depot
  atsp_file << "DIMENSION : " << view_points.poses.size() + num_salesman + 1
            << endl;
  atsp_file << "EDGE_WEIGHT_TYPE: EXPLICIT" << endl;
  atsp_file << "EDGE_WEIGHT_FORMAT: FULL_MATRIX" << endl;
  atsp_file << "EDGE_WEIGHT_SECTION" << endl;

  // weight of edge from i to j
  geometry_msgs::PoseArray all_node;
  all_node.header = view_points.header;
  geometry_msgs::Pose new_node;
  new_node.orientation.w = 1;
  new_node.orientation.x = 0;
  new_node.orientation.y = 0;
  new_node.orientation.z = 0;

  // virtual depot
  all_node.poses.emplace_back(new_node);

  // self
  new_node.position.x = current_pose.point.x;
  new_node.position.y = current_pose.point.y;
  new_node.position.z = current_pose.point.z;
  all_node.poses.emplace_back(new_node);

  // other
  const int num_other_nodes = other_poses.size();
  for (int i = 0; i < num_other_nodes; ++i) {
    new_node.position.x = other_poses[i].point.x;
    new_node.position.y = other_poses[i].point.y;
    new_node.position.z = other_poses[i].point.z;
    all_node.poses.emplace_back(new_node);
  }

  // view points
  all_node.poses.insert(all_node.poses.end(), view_points.poses.begin(),
                        view_points.poses.end());

  const int num_all_node = all_node.poses.size();

  cout << "writing file !!" << endl;
  for (int i = 0; i < num_all_node; i++) {
    for (int j = 0; j < num_all_node; j++) {
      if (i == 0) {
        if (j <= num_salesman) {
          atsp_file << 0 << " ";
        } else {
          atsp_file << 999999 << " ";
        }
      } else if (j == 0) {
        atsp_file << 0 << " ";
      } else if (i == j) {
        atsp_file << 0 << " ";
      } else {
        // Eigen::Vector3f start_p(all_node.poses[i].position.x, all_node.poses[i].position.y, all_node.poses[i].position.z);
        // Eigen::Vector3f end_p(all_node.poses[j].position.x, all_node.poses[j].position.y, all_node.poses[j].position.z);
        // Astar astar;
        // double edge_w = astar.astar_path_distance(ocmap, start_p, end_p);
        double edge_w = hypot(all_node.poses[i].position.x - all_node.poses[j].position.x, all_node.poses[i].position.y - all_node.poses[j].position.y);
        atsp_file << (int)(edge_w * 100.0) << " ";
      }
    }
    atsp_file << endl;
  }
  atsp_file.close();

  // call service
  lkh_ros::Solve srv;
  srv.request.problem_file = par;
  if (lkh_client.call(srv)) {
    cout << "[INFO] atsp solved" << endl;
  } else {
    cout << "[ERROR] atsp solving error!!" << endl;
  }
  // read result && return

  std::ifstream tour_file(tour);
  if (!tour_file.is_open()) {
    std::cerr << "Failed to open tour file." << std::endl;
    return result;
  }

  std::string line;
  while (std::getline(tour_file, line)) {
    if (line == "TOUR_SECTION") {
      break;
    }
  }

  vector<int> node_seq;
  while (std::getline(tour_file, line)) {
    int num = stoi(line);
    if (num == -1) {
      break;
    } else {
      node_seq.push_back(num);
    }
  }
  tour_file.close();

  // check whether the result is reasonable
  if (node_seq.size() != num_all_node + num_salesman - 1) {
    std::cerr << "the result is not reasonable" << std::endl;
    return result;
  }

  result.header.frame_id = "map";
  vector<int>::iterator it = find(node_seq.begin(), node_seq.end(), 2);

  for(; it != node_seq.end(); ++it) {
    int id = *it - 1;
    if(id >= num_all_node) {
      break;
    }
    else {
      result.poses.push_back(all_node.poses[id]);
    }
  }
  return result;
}
