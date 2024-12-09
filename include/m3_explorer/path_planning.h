#ifndef PATH_PLANNING_H
#define PATH_PLANNING_H
#include "lkh_ros/Solve.h"
#include "m3_explorer/astar.h"
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>
#include <string>
#include <vector>
#include <octomap/octomap.h>

using namespace std;
geometry_msgs::PoseArray
atsp_path(const geometry_msgs::PointStamped &current_pose,
          const geometry_msgs::PoseArray &view_points,
          ros::ServiceClient &lkh_client, const string &problem_path);
geometry_msgs::PoseArray
amtsp_path(const octomap::OcTree *ocmap, const geometry_msgs::PointStamped &current_pose,
            const vector<geometry_msgs::PointStamped> &other_poses,
          const geometry_msgs::PoseArray &view_points,
          ros::ServiceClient &lkh_client, const string &problem_path);
#endif
