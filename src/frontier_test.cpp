#include "lkh_ros/Solve.h"
#include "m3_explorer/Traj.h"
#include "m3_explorer/frontier_cluster.h"
#include "m3_explorer/frontier_detector.h"
#include "m3_explorer/hastar.h"
#include "m3_explorer/path_planning.h"
#include <Eigen/Dense>
#include <cmath>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <nav_msgs/Odometry.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <queue>
#include <random>
#include <ros/ros.h>
#include <set>
#include <std_msgs/Float32MultiArray.h>
#include <string>
#include <sys/time.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_map>
#include <utility>
#include <vector>
#include <visualization_msgs/MarkerArray.h>

tf2_ros::Buffer tf_buffer;

using namespace std;

octomap::OcTree* ocmap = nullptr;
void octomap_cb(const octomap_msgs::Octomap::ConstPtr &msg) {
  delete ocmap;
  ocmap = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "m3_explorer");
  ros::NodeHandle nh("");

  double resolution = 0.1, sensor_range = 5.0;

  tf2_ros::TransformListener tf_listener(tf_buffer);
  ros::Duration(1.0).sleep(); // 等待tf2变换树准备好
  ros::Rate rate(5);

  // frontier voxels display
  ros::Publisher frontier_maker_array_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/uav0/frontier", 10);
  // frontier normal display
  ros::Publisher frontier_normal_pub =
      nh.advertise<geometry_msgs::PoseArray>("/uav0/frontier_normal", 10);
  // cluster center and its normal display
  ros::Publisher cluster_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/uav0/cluster", 10);
  // distinguish diffrent clusters display
  ros::Publisher cluster_vis_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/uav0/cluster_vis", 10);
  // view point display
  ros::Publisher view_point_pub =
      nh.advertise<geometry_msgs::PoseArray>("/uav0/view_point", 10);
  // get octomap
  ros::Subscriber uav0_octomap_sub = nh.subscribe<octomap_msgs::Octomap>("/uav0/octomap_binary", 1, octomap_cb);

  // marker template
  visualization_msgs::Marker marker;
  marker.header.frame_id = "map";
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  std_msgs::ColorRGBA color;
  color.r = 1.0;
  color.g = 0.0;
  color.b = 0.0;
  color.a = 1.0;
  marker.color = color;
  marker.type = visualization_msgs::Marker::CUBE_LIST;

  // frontiers
  set<QuadMesh> frontiers;

  while (ros::ok()) {
    ros::spinOnce();

    // lookup transform of 3 axises
    geometry_msgs::PointStamped cam_o_in_cam;
    cam_o_in_cam.header.frame_id = "uav0_camera_depth_frame";
    cam_o_in_cam.point.x = 0.0;
    cam_o_in_cam.point.y = 0.0;
    cam_o_in_cam.point.z = 0.0;

    geometry_msgs::PointStamped cam_o_in_map;

    try {
      // 使用lookupTransform函数查询坐标变换
      tf_buffer.transform(cam_o_in_cam, cam_o_in_map, "map");
    } catch (tf2::TransformException &ex) {
      ROS_ERROR("Failed to transform point: %s", ex.what());
    }

    // input = octomap; current pose; FOV; max range; region bbx
    // output = a set containing all frontier voxels

    ros::Time current_time = ros::Time::now();

    frontier_detect(frontiers, ocmap, cam_o_in_map, sensor_range);

    ros::Duration elapsed_time = ros::Time::now() - current_time;
    cout << "[frontier detect]: " << elapsed_time.toSec() * 1000.0
         << " ms, ";
    cout << "[voxel num]: " << frontiers.size() << endl;
    frontier_visualize(frontiers, 0.1, frontier_maker_array_pub);
    frontier_normal_visualize(frontiers, frontier_normal_pub);

    current_time = ros::Time::now();

    vector<Cluster> cluster_vec;
    geometry_msgs::PoseArray vp_array;

    if (!frontiers.empty()){
      cluster_vec = dbscan_cluster(frontiers, 0.4, 8, 8, cluster_vis_pub);
      vp_array = view_point_generate(cluster_vec, ocmap);
      cout << "[frontier cluster]: "
            << (ros::Time::now() - current_time).toSec() * 1000.0 << " ms"
            << endl;
      
      cluster_visualize(cluster_vec, cluster_pub);
      view_point_pub.publish(vp_array);
    }

    rate.sleep();
  }
  return 0;
}
