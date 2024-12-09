#include "lkh_ros/Solve.h"
#include "m3_explorer/Traj.h"
#include "m3_explorer/frontier_cluster.h"
#include "m3_explorer/frontier_detector.h"
#include "m3_explorer/hastar.h"
#include "m3_explorer/path_planning.h"
#include "m3_explorer/time_track.hpp"
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

enum CTRL_STATE { VOID = 0, POS_CTRL, YAW_CTRL };
enum PLAN_FSM { WAIT = 0, PLAN, EXEC };

Hastar planning;
PLAN_FSM state = PLAN_FSM::PLAN;
tf2_ros::Buffer tf_buffer;

using namespace std;

int self_id = 0;

octomap::OcTree *ocmap = nullptr;
void octomap_cb(const octomap_msgs::Octomap::ConstPtr &msg) {
  delete ocmap;
  ocmap = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
}

float cur_yaw = 0.0;
// get UAV yaw from odom. note: it can merge to pos_cb
void base_link_cb(const nav_msgs::Odometry::ConstPtr &msg) {
  tf2::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  double roll, pitch, yaw;
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
  cur_yaw = (float)yaw;
}

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr &msg) { current_state = *msg; }

geometry_msgs::PoseStamped cur_pos;
void pos_cb(const geometry_msgs::PoseStamped::ConstPtr &msg) { cur_pos = *msg; }

// switch to OFFBOARD && takeoff to desired height
void offboard_takeoff(ros::NodeHandle &nh, const double &height) {
  ros::Subscriber state_sub =
      nh.subscribe<mavros_msgs::State>("mavros/state", 10, state_cb);
  ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::PoseStamped>(
      "mavros/local_position/pose", 10, pos_cb);
  ros::Publisher local_pos_pub = nh.advertise<geometry_msgs::PoseStamped>(
      "mavros/setpoint_position/local", 10);
  ros::ServiceClient arming_client =
      nh.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
  ros::ServiceClient set_mode_client =
      nh.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

  // the setpoint publishing rate MUST be faster than 2Hz
  ros::Rate rate(20.0);

  // wait for FCU connection
  while (ros::ok() && !current_state.connected) {
    ros::spinOnce();
    rate.sleep();
  }
  ROS_INFO("mavros connected");

  // get takeoff position
  ros::Time start_time = ros::Time::now();
  int count = 0;
  geometry_msgs::PoseStamped takeoff_pos;
  ROS_INFO("reading takeoff position");
  while (ros::Time::now() - start_time < ros::Duration(2.0)) {
    ros::spinOnce();
    count++;
    takeoff_pos.pose.position.x += cur_pos.pose.position.x;
    takeoff_pos.pose.position.y += cur_pos.pose.position.y;
    takeoff_pos.pose.position.z += cur_pos.pose.position.z;
    rate.sleep();
  }
  takeoff_pos.pose.position.x /= count;
  takeoff_pos.pose.position.y /= count;
  takeoff_pos.pose.position.z /= count;
  ROS_INFO("takeoff position: %.2lf, %.2lf, %.2lf", takeoff_pos.pose.position.x,
           takeoff_pos.pose.position.y, takeoff_pos.pose.position.z);

  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = takeoff_pos.pose.position.x;
  pose.pose.position.y = takeoff_pos.pose.position.y;
  pose.pose.position.z = takeoff_pos.pose.position.z + height;

  // send a few setpoints before starting
  for (int i = 5; ros::ok() && i > 0; --i) {
    local_pos_pub.publish(pose);
    ros::spinOnce();
    rate.sleep();
  }

  mavros_msgs::SetMode offb_set_mode;
  offb_set_mode.request.custom_mode = "OFFBOARD";

  mavros_msgs::CommandBool arm_cmd;
  arm_cmd.request.value = true;

  ros::Time last_request = ros::Time::now();

  while (abs(cur_pos.pose.position.z - pose.pose.position.z) > 0.1) {
    if (current_state.mode != "OFFBOARD" &&
        (ros::Time::now() - last_request > ros::Duration(1.0))) {
      if (set_mode_client.call(offb_set_mode) &&
          offb_set_mode.response.mode_sent) {
        ROS_INFO("Offboard enabled");
      }
      last_request = ros::Time::now();
    } else {
      if (!current_state.armed &&
          (ros::Time::now() - last_request > ros::Duration(1.0))) {
        if (arming_client.call(arm_cmd) && arm_cmd.response.success) {
          ROS_INFO("Vehicle armed");
        }
        last_request = ros::Time::now();
      }
    }

    local_pos_pub.publish(pose);

    ros::spinOnce();
    rate.sleep();
  }

  ROS_INFO("takeoff done");
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "m3_explorer");
  ros::NodeHandle nh("");

  double resolution = 0.1, sensor_range = 50.0;

  tf2_ros::TransformListener tf_listener(tf_buffer);
  ros::Duration(1.0).sleep(); // 等待tf2变换树准备好
  ros::Rate rate(5);

  // frontier voxels display
  ros::Publisher frontier_maker_array_pub =
      nh.advertise<visualization_msgs::MarkerArray>("frontier", 10);
  // frontier normal display
  ros::Publisher frontier_normal_pub =
      nh.advertise<geometry_msgs::PoseArray>("frontier_normal", 10);
  // cluster center and its normal display
  ros::Publisher cluster_pub =
      nh.advertise<visualization_msgs::MarkerArray>("cluster", 10);
  // distinguish diffrent clusters display
  ros::Publisher cluster_vis_pub =
      nh.advertise<visualization_msgs::MarkerArray>("cluster_vis", 10);
  // view point display
  ros::Publisher view_point_pub =
      nh.advertise<geometry_msgs::PoseArray>("view_point", 10);
  // path display
  ros::Publisher history_path_pub =
      nh.advertise<visualization_msgs::Marker>("history_path", 10);
  // traj pub
  ros::Publisher traj_pub = nh.advertise<m3_explorer::Traj>("traj", 10);

  // lkh client
  ros::ServiceClient lkh_client = nh.serviceClient<lkh_ros::Solve>("lkh_solve");
  string problem_path;
  if (!nh.getParam("Problem_Path", problem_path)) {
    rate.sleep();
  }

  nh.getParam("ID", self_id);
  cout << "[INFO] uav ID = " << self_id << endl;

  // get octomap
  ros::Subscriber uav0_octomap_sub =
      nh.subscribe<octomap_msgs::Octomap>("/merged_map", 1, octomap_cb);
  // get current pose
  ros::Subscriber base_link_sub = nh.subscribe<nav_msgs::Odometry>(
      "ground_truth/base_link", 1, base_link_cb);

  // planner input: target pose
  ros::Publisher goal_pub =
      nh.advertise<geometry_msgs::PoseStamped>("goal", 10);
  ros::Publisher ctrl_state_pub =
      nh.advertise<std_msgs::Float32MultiArray>("ctrl_state", 10);

  int ctrl_state = CTRL_STATE::POS_CTRL;
  bool goal_exec = false;
  int id_exec = 0;
  vector<geometry_msgs::PoseStamped> goal_list;

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

  visualization_msgs::Marker history_path(marker);
  history_path.color.r = 0.0;
  history_path.color.g = 0.0;
  history_path.color.b = 1.0;
  history_path.color.a = 1.0;
  history_path.scale.x = 0.1;
  history_path.scale.y = 0.1;
  history_path.scale.z = 0.1;
  history_path.type = visualization_msgs::Marker::POINTS;

  // frontiers
  set<QuadMesh> frontiers;

  // switch to offboard mode && takeoff to desired height
  offboard_takeoff(nh, 1.5);

  ros::Time explore_start = ros::Time::now();

  geometry_msgs::PoseArray explore_path;
  int path_id = 0;

  while (ros::ok()) {
    ros::spinOnce();

    // lookup transform of 3 axises
    geometry_msgs::PointStamped cam_o_in_cam;
    cam_o_in_cam.header.frame_id =
        "uav" + to_string(self_id) + "_camera_depth_frame";
    cam_o_in_cam.point.x = 0.0;
    cam_o_in_cam.point.y = 0.0;
    cam_o_in_cam.point.z = 0.0;

    geometry_msgs::PointStamped cam_o_in_map;

    try {
      // 使用lookupTransform函数查询坐标变换
      tf_buffer.transform(cam_o_in_cam, cam_o_in_map, "map");
      history_path.points.push_back(cam_o_in_map.point);
      history_path_pub.publish(history_path);
    } catch (tf2::TransformException &ex) {
      ROS_ERROR("Failed to transform point: %s", ex.what());
    }

    vector<geometry_msgs::PointStamped> other_uav_poses;
    geometry_msgs::PointStamped other_uav(cam_o_in_cam);
    geometry_msgs::PointStamped other_uav_in_map;
    for (int i = 0; i < 3; ++i) {
      if (i != self_id) {
        other_uav.header.frame_id =
            "uav" + to_string(i) + "_camera_depth_frame";
        try {
          // 使用lookupTransform函数查询坐标变换
          tf_buffer.transform(other_uav, other_uav_in_map, "map");
        } catch (tf2::TransformException &ex) {
          ROS_ERROR("Failed to transform point: %s", ex.what());
        }
        other_uav_poses.emplace_back(other_uav_in_map);
      }
    }

    //// input = octomap; current pose; FOV; max range; region bbx
    //// output = a set containing all frontier voxels

    TimeTrack tracker;

    frontier_detect(frontiers, ocmap, cam_o_in_map, sensor_range);
    frontier_visualize(frontiers, 0.02, frontier_maker_array_pub);
    frontier_normal_visualize(frontiers, frontier_normal_pub);

    tracker.OutputPassingTime("Frontier Detect");
    cout << "[voxel num]: " << frontiers.size() << endl;

    vector<Cluster> cluster_vec;
    geometry_msgs::PoseArray vp_array;

    if (!frontiers.empty() && goal_exec == false) {
      tracker.SetStartTime();
      //// frontier clustering
      //// input = the set containing all frontier voxels
      //// output = a vector containing all frontier clusters
      // cluster_vec = k_mean_cluster(frontiers);
      cluster_vec = dbscan_cluster(frontiers, 0.4, 8, 8, cluster_vis_pub);
      cluster_visualize(cluster_vec, cluster_pub);
      tracker.OutputPassingTime("Frontier Cluster");

      tracker.SetStartTime();
      //// generate view point
      //// input = a vector containing all frontier clusters
      //// output = a vector containing poses of all view points
      vp_array = view_point_generate(cluster_vec, ocmap);
      view_point_pub.publish(vp_array);
      tracker.OutputPassingTime("Viewpoint Gen");

      tracker.SetStartTime();
      //// path planning
      //// input = current pose of uav && a vector containing poses of all view
      /// points / output = a vector containing the sequence of waypoints
      explore_path.poses.clear();
      if (vp_array.poses.size() > 2) {
        explore_path = amtsp_path(ocmap, cam_o_in_map, other_uav_poses,
                                  vp_array, lkh_client, problem_path);
        // remove start point: cam_o_in_map
        explore_path.poses.erase(explore_path.poses.begin());
      } else if (!vp_array.poses.empty()) {
        explore_path.poses.push_back(vp_array.poses[0]);
      } else {
        cout << "no space to explore !!" << endl;
      }
      goal_exec = true;
      tracker.OutputPassingTime("Tour Planning");
    }

    if (goal_exec) {
      // receding horizon
      if (path_id >= explore_path.poses.size() || path_id >= 1) {
        path_id = 0;
        goal_exec = false;
      } else {
        cout << "path_id : " << path_id << "/" << explore_path.poses.size() - 1
             << endl;
        Eigen::Vector2f delta(
            cam_o_in_map.point.x - explore_path.poses[path_id].position.x,
            cam_o_in_map.point.y - explore_path.poses[path_id].position.y);
        cout << "distance to target = " << delta.norm() << " m" << endl;
        octomap::point3d target_point(explore_path.poses[path_id].position.x,
                                      explore_path.poses[path_id].position.y,
                                      1.5);

        // calculate target yaw
        tf2::Quaternion q(explore_path.poses[path_id].orientation.x,
                          explore_path.poses[path_id].orientation.y,
                          explore_path.poses[path_id].orientation.z,
                          explore_path.poses[path_id].orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
        float target_yaw = (float)yaw;

        if (is_next_to_obstacle(ocmap, target_point, 0.8, 0.8)) {
          path_id++;
          state = PLAN_FSM::PLAN;
        } else {

          switch (state) {
          case PLAN_FSM::PLAN: {
            cout << "Searching Path" << endl;
            tracker.SetStartTime();
            // Hybrid A* search path
            bool is_planned = planning.search_path(
                ocmap,
                Eigen::Vector3f(cam_o_in_map.point.x, cam_o_in_map.point.y,
                                1.5),
                Eigen::Vector3f(explore_path.poses[path_id].position.x,
                                explore_path.poses[path_id].position.y, 1.5),
                cur_yaw);
            tracker.OutputPassingTime("Hybrid A*");

            tracker.SetStartTime();
            // re-search
            if (is_planned == false) {
              is_planned = planning.search_path(
                  ocmap,
                  Eigen::Vector3f(cam_o_in_map.point.x - 0.4 * cos(cur_yaw),
                                  cam_o_in_map.point.y - 0.4 * sin(cur_yaw),
                                  1.5),
                  Eigen::Vector3f(explore_path.poses[path_id].position.x,
                                  explore_path.poses[path_id].position.y, 1.5),
                  cur_yaw);
            }
            tracker.OutputPassingTime("Hybrid A* Replan");

            if (is_planned) {
              // send traj
              m3_explorer::Traj send_traj;
              mavros_msgs::PositionTarget target_pose;
              target_pose.header.frame_id = "map";
              target_pose.header.stamp = ros::Time::now();
              target_pose.coordinate_frame =
                  mavros_msgs::PositionTarget::FRAME_LOCAL_NED; // actually
                                                                // redundant
              target_pose.type_mask =
                  mavros_msgs::PositionTarget::IGNORE_YAW_RATE; // actually
                                                                // redundant
              for (int i = 0; i < planning.traj.size(); i++) {
                target_pose.position.x = planning.traj[i].pos.x();
                target_pose.position.y = planning.traj[i].pos.y();
                target_pose.position.z = planning.traj[i].pos.z();

                target_pose.velocity.x = planning.traj[i].vel.x();
                target_pose.velocity.y = planning.traj[i].vel.y();
                target_pose.velocity.z = planning.traj[i].vel.z();

                target_pose.acceleration_or_force.x = planning.traj[i].acc.x();
                target_pose.acceleration_or_force.y = planning.traj[i].acc.y();
                target_pose.acceleration_or_force.z = planning.traj[i].acc.z();

                target_pose.yaw = planning.traj[i].yaw;
                target_pose.yaw_rate = planning.traj[i].yaw_rate;
                send_traj.traj.push_back(target_pose);
              }

              // set target yaw at end_p
              target_pose.yaw = target_yaw;
              send_traj.traj.push_back(target_pose);

              traj_pub.publish(send_traj);
              state = PLAN_FSM::EXEC;
            }
            break;
          }

          case PLAN_FSM::EXEC: {
            if (delta.norm() < 0.2 &&
                abs(target_yaw - cur_yaw) < 5.0 * M_PI / 180.0) {
              path_id++;
              state = PLAN_FSM::PLAN;
            }
            break;
          }
          }
        }
      }
    }

    rate.sleep();
  }
  return 0;
}
