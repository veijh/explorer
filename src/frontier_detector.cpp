#include "m3_explorer/frontier_detector.h"

#include <Eigen/Dense>
#include <cmath>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <queue>
#include <sys/time.h>
#include <tf2/LinearMath/Quaternion.h>
#include <unordered_map>
#include <utility>
#include <visualization_msgs/MarkerArray.h>

void frontier_detect(set<QuadMesh> &frontiers, octomap::OcTree *ocmap,
                     const geometry_msgs::PointStamped &cur_pose,
                     const double &sensor_range) {
  // check old frontier
  if (!frontiers.empty() && ocmap != nullptr) {
    auto it = frontiers.begin();
    while (it != frontiers.end()) {
      // if (abs(it->center.x() - cur_pose.point.x) > sensor_range) {
      //   it++;
      //   continue;
      // }
      // if (abs(it->center.y() - cur_pose.point.y) > sensor_range) {
      //   it++;
      //   continue;
      // }

      if (ocmap->search(it->center) != nullptr) {
        it = frontiers.erase(it);
      } else {
        it++;
      }
    }
  }

  // add new frontier
  // check whether the point is frontier
  octomap::point3d_collection nbr_dir = {{1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0},
                                         {0.0, 1.0, 0.0}, {0.0, -1.0, 0.0},
                                         {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0}};
  octomap::point3d_collection offset = {
      {0.0, 1.0, 1.0}, {0.0, -1.0, 1.0}, {0.0, -1.0, -1.0}, {0.0, 1.0, -1.0},
      {1.0, 0.0, 1.0}, {-1.0, 0.0, 1.0}, {-1.0, 0.0, -1.0}, {1.0, 0.0, -1.0},
      {1.0, 1.0, 0.0}, {-1.0, 1.0, 0.0}, {-1.0, -1.0, 0.0}, {1.0, -1.0, 0.0}};

  octomap::point3d check_bbx_min(cur_pose.point.x - sensor_range,
                                 cur_pose.point.y - sensor_range,
                                 max(1.0, cur_pose.point.z - sensor_range));
  octomap::point3d check_bbx_max(cur_pose.point.x + sensor_range,
                                 cur_pose.point.y + sensor_range,
                                 min(2.0, cur_pose.point.z + sensor_range));

  if (ocmap == nullptr) {
    cout << "[ERROR] the ptr of octomap is null" << endl;
  } else {
    for (octomap::OcTree::leaf_bbx_iterator
             it = ocmap->begin_leafs_bbx(check_bbx_min, check_bbx_max),
             end = ocmap->end_leafs_bbx();
         it != end; ++it) {
      double size = it.getSize();
      int depth = it.getDepth();
      octomap::point3d center = it.getCoordinate();

      // no need to check obstacles' neighbors
      // no need to check the interior of obstacle
      if (is_next_to_obstacle(ocmap, center, 0.3, 0.8)) {
        continue;
      }

      // check 6 faces
      for (int i = 0; i < nbr_dir.size(); i++) {
        octomap::point3d nbr_point = center + nbr_dir[i] * size;
        octomap::OcTreeNode *nbr_node = ocmap->search(nbr_point, depth);
        QuadMesh mesh;
        if (nbr_node == nullptr) {
          mesh.center = center + nbr_dir[i] * (size / 2.0);
          mesh.normal = nbr_dir[i];
          mesh.size = size;
          frontiers.insert(mesh);
        } else {
          if (ocmap->nodeHasChildren(nbr_node)) {
            // bfs search unknown voxel
            queue<pair<octomap::point3d, int>> bfs_queue;
            octomap::point3d surface =
                center +
                nbr_dir[i] * (size / 2.0 + ocmap->getResolution() / 2.0);
            bfs_queue.push(make_pair(surface, depth));

            while (!bfs_queue.empty()) {
              octomap::point3d point = bfs_queue.front().first;
              int point_depth = bfs_queue.front().second;
              double point_size = size * pow(0.5, point_depth - depth);
              bfs_queue.pop();

              octomap::OcTreeNode *point_node =
                  ocmap->search(point, point_depth);
              if (point_node != nullptr) {
                if (ocmap->nodeHasChildren(point_node)) {
                  // add 4 points into queue
                  for (int offset_i = 0; offset_i < 4; offset_i++) {
                    double child_size = point_size / 2.0;
                    octomap::point3d child =
                        point +
                        offset[offset_i + 4 * (i / 2)] * (child_size / 2.0);
                    bfs_queue.push(make_pair(child, point_depth + 1));
                  }
                }
              } else {
                mesh.center = point;
                mesh.normal = nbr_dir[i];
                mesh.size = point_size;
                frontiers.insert(mesh);
              }
            }
          }
        }
      }
    }
  }

  // remove frontiers near obstacle
  if (!frontiers.empty() && ocmap != nullptr) {
    auto it = frontiers.begin();
    while (it != frontiers.end()) {
      if (abs(it->center.x() - cur_pose.point.x) > sensor_range) {
        it++;
        continue;
      }
      if (abs(it->center.y() - cur_pose.point.y) > sensor_range) {
        it++;
        continue;
      }
      if (is_next_to_obstacle(ocmap, it->center, 0.4, 0.7)) {
        it = frontiers.erase(it);
      } else {
        it++;
      }
    }
  }
}

bool is_next_to_obstacle(octomap::OcTree *ocmap, const octomap::point3d &point,
                         const double &check_box_size, const double &occ_trs) {
  octomap::point3d check_point(point);
  for (double x_offset = -check_box_size / 2.0;
       x_offset <= check_box_size / 2.0; x_offset += 0.05) {
    for (double y_offset = -check_box_size / 2.0;
         y_offset <= check_box_size / 2.0; y_offset += 0.05) {
      check_point.x() = point.x() + x_offset;
      check_point.y() = point.y() + y_offset;
      octomap::OcTreeNode *result = ocmap->search(check_point);
      if (result != nullptr && result->getOccupancy() > occ_trs) {
        return true;
      }
    }
  }
  return false;
}

void frontier_visualize(set<QuadMesh> &frontiers, const double &mesh_thickness,
                        ros::Publisher &frontier_maker_array_pub) {
  // marker template
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  std_msgs::ColorRGBA color;
  color.r = 0.0;
  color.g = 1.0;
  color.b = 0.0;
  color.a = 1.0;
  marker.color = color;
  marker.type = visualization_msgs::Marker::CUBE_LIST;

  // CUBE_LIST requires that all elements in the cube list have the same scale.
  vector<vector<visualization_msgs::Marker>> all_marker(3);
  if (!frontiers.empty()) {
    int total_type = 0;
    // mesh_size -> the type of
    vector<unordered_map<double, int>> marker_type(3);
    auto it = frontiers.begin();
    for (int i = 0; i < frontiers.size(); i++) {
      geometry_msgs::Vector3 scale;
      int flag = -1;
      if (it->normal.x() > 0.5 || it->normal.x() < -0.5) {
        flag = 0;
        scale.x = mesh_thickness;
        scale.y = it->size;
        scale.z = it->size;
      }
      if (it->normal.y() > 0.5 || it->normal.y() < -0.5) {
        flag = 1;
        scale.x = it->size;
        scale.y = mesh_thickness;
        scale.z = it->size;
      }
      if (it->normal.z() > 0.5 || it->normal.z() < -0.5) {
        flag = 2;
        scale.x = it->size;
        scale.y = it->size;
        scale.z = mesh_thickness;
      }

      if (flag > -1) {
        if (marker_type[flag].find(it->size) == marker_type[flag].end()) {
          marker_type[flag][it->size] = marker_type[flag].size();
          all_marker[flag].resize(marker_type[flag].size(), marker);
          all_marker[flag][marker_type[flag][it->size]].id =
              marker_type[flag][it->size] + flag * 100;
          all_marker[flag][marker_type[flag][it->size]].scale = scale;
        }
        geometry_msgs::Point point_pose;
        point_pose.x = it->center.x();
        point_pose.y = it->center.y();
        point_pose.z = it->center.z();
        all_marker[flag][marker_type[flag][it->size]].points.push_back(
            point_pose);
      }
      it++;
    }
  }

  visualization_msgs::MarkerArray frontier_maker_array;

  for (auto &i : all_marker) {
    for (auto &j : i) {
      frontier_maker_array.markers.push_back(j);
    }
  }

  frontier_maker_array_pub.publish(frontier_maker_array);
}

void frontier_normal_visualize(set<QuadMesh> &frontiers,
                               ros::Publisher &frontier_normal_pub) {
  geometry_msgs::PoseArray frontier_normal_array;
  frontier_normal_array.header.frame_id = "map";
  geometry_msgs::Pose normal_pose;

  if (!frontiers.empty()) {
    auto it = frontiers.begin();
    for (int i = 0; i < frontiers.size(); i++) {
      normal_pose.position.x = it->center.x();
      normal_pose.position.y = it->center.y();
      normal_pose.position.z = it->center.z();

      tf2::Quaternion q;
      if (it->normal.x() > 0.5) {
        q.setRPY(0, 0, 0);
      }
      if (it->normal.x() < -0.5) {
        q.setRPY(0, 0, M_PI);
      }
      if (it->normal.y() > 0.5) {
        q.setRPY(0, 0, M_PI / 2);
      }
      if (it->normal.y() < -0.5) {
        q.setRPY(0, 0, -M_PI / 2);
      }
      if (it->normal.z() > 0.5) {
        q.setRPY(0, -M_PI / 2, 0);
      }
      if (it->normal.z() < -0.5) {
        q.setRPY(0, M_PI / 2, 0);
      }
      normal_pose.orientation.w = q.w();
      normal_pose.orientation.x = q.x();
      normal_pose.orientation.y = q.y();
      normal_pose.orientation.z = q.z();
      frontier_normal_array.poses.push_back(normal_pose);
      it++;
    }
  }

  frontier_normal_pub.publish(frontier_normal_array);
}
