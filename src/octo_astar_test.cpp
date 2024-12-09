#include "m3_explorer/octo_astar.h"
#include <Eigen/Dense>
#include <chrono>
#include <geometry_msgs/Point.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <random>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

octomap::OcTree *ocmap = nullptr;
void octomap_cb(const octomap_msgs::Octomap::ConstPtr &msg) {
  delete ocmap;
  ocmap = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "octo_astar_test");
  ros::NodeHandle nh("");

  // get octomap
  ros::Subscriber octomap_sub =
      nh.subscribe<octomap_msgs::Octomap>("/merged_map", 1, octomap_cb);

  // visualize waypoint
  ros::Publisher wp_pub =
      nh.advertise<visualization_msgs::Marker>("/waypoint", 10);

  // visualize waypoint
  ros::Publisher node_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/leafnode", 10);

  visualization_msgs::Marker waypoint;
  waypoint.header.frame_id = "map";
  waypoint.header.stamp = ros::Time::now();

  waypoint.ns = "line_strip";
  waypoint.action = visualization_msgs::Marker::ADD;
  waypoint.scale.x = 0.1;
  waypoint.pose.orientation.w = 1.0;
  waypoint.id = 0;
  waypoint.type = visualization_msgs::Marker::LINE_STRIP;
  waypoint.color.r = 1.0;
  waypoint.color.g = 1.0;
  waypoint.color.b = 1.0;
  waypoint.color.a = 1.0;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> xy(-10.0, 10.0);
  std::uniform_real_distribution<float> z(1.0, 2.0);

  ros::Rate rate(0.5);

  visualization_msgs::MarkerArray marker_array;

  while (ros::ok()) {
    rate.sleep();
    ros::spinOnce();
    if (ocmap == nullptr)
      continue;

    // astar test
    // 设定起点终点
    Eigen::Vector3f start_pt = {xy(gen), xy(gen), z(gen)};
    Eigen::Vector3f end_pt = {xy(gen), xy(gen), z(gen)};
    // Eigen::Vector3f start_pt = {0.0, -8.0, 1.5};
    // Eigen::Vector3f end_pt = {8.0, 0.0, 1.5};
    octomap::OcTreeNode *start_node =
        ocmap->search(start_pt.x(), start_pt.y(), start_pt.z());
    octomap::OcTreeNode *end_node =
        ocmap->search(end_pt.x(), end_pt.y(), end_pt.z());
    if (!start_node || !end_node)
      continue;
    if (ocmap->isNodeOccupied(start_node) || ocmap->isNodeOccupied(end_node))
      continue;

    // A*寻路，并统计时间
    OctoAstar octo_astar(ocmap);
    auto start_time = std::chrono::system_clock::now();
    octo_astar.astar_path_distance(start_pt, end_pt);
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end_time - start_time;
    std::cout << "[astar search] :" << elapsed.count() << " ms" << std::endl;
    // 可视化轨迹
    waypoint.points.clear();
    const int wp_num = octo_astar.path_.size();
    for (int i = 0; i < wp_num; ++i) {
      const Eigen::Vector3f pos = octo_astar.path_[i]->position_;
      geometry_msgs::Point wp_pos;
      wp_pos.x = pos.x();
      wp_pos.y = pos.y();
      wp_pos.z = pos.z();
      waypoint.points.emplace_back(wp_pos);
    }
    wp_pub.publish(waypoint);
  }
}

// octomap::point3d bx_min(-10.0, -10.0, 0.2);
// octomap::point3d bx_max(10.0, 10.0, 0.4);
// int count = 0;
// marker_array.markers.clear();
// for(octomap::OcTree::leaf_bbx_iterator it = ocmap->begin_leafs_bbx(bx_min,
// bx_max), end=ocmap->end_leafs_bbx(); it != end; it++){
//   visualization_msgs::Marker box;
//   box.header.frame_id = "map";
//   box.header.stamp = ros::Time::now();

//   box.action = visualization_msgs::Marker::ADD;
//   box.scale.x = it.getSize()*0.8;
//   box.scale.y = it.getSize()*0.8;
//   box.scale.z = it.getSize()*0.8;
//   box.pose.orientation.w = 1.0;
//   box.id = count;
//   box.type = visualization_msgs::Marker::CUBE;
//   box.color.r = 0.0;
//   box.color.g = 0.5;
//   box.color.b = 0.0;
//   box.color.a = 1.0;
//   box.pose.position.x = it.getCoordinate().x();
//   box.pose.position.y = it.getCoordinate().y();
//   box.pose.position.z = it.getCoordinate().z();
//   marker_array.markers.emplace_back(box);
//   ++count;
// }
// node_pub.publish(marker_array);
