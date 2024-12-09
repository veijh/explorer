#include "m3_explorer/Traj.h"
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <mavros_msgs/PositionTarget.h>
#include <ros/ros.h>

using namespace std;

ros::Publisher local_pos_pub;
ros::Publisher sp_pub;
int traj_id = 0;

m3_explorer::Traj traj;
void traj_cb(const m3_explorer::Traj::ConstPtr &msg) {
  traj = *msg;
  traj_id = 0;
}

mavros_msgs::PositionTarget target_pose;
geometry_msgs::PoseStamped sp;

void cmdCallback(const ros::TimerEvent &e) {

  if (traj_id < traj.traj.size() && traj_id >= 0) {
    target_pose.header.stamp = sp.header.stamp = ros::Time::now();
    target_pose.position = sp.pose.position = traj.traj[traj_id].position;
    target_pose.velocity = traj.traj[traj_id].velocity;
    target_pose.acceleration_or_force =
        traj.traj[traj_id].acceleration_or_force;
    target_pose.yaw = traj.traj[traj_id].yaw;
    traj_id++;
  }

  local_pos_pub.publish(target_pose);
  sp_pub.publish(sp);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "mavros_ctrl");
  ros::NodeHandle nh("");

  // input: traj
  ros::Subscriber traj_sub =
      nh.subscribe<m3_explorer::Traj>("traj", 10, traj_cb);

  // output: cmd
  local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>(
      "mavros/setpoint_raw/local", 10);
  sp_pub = nh.advertise<geometry_msgs::PoseStamped>("mavros/setpoint", 10);

  sp.header.frame_id = "map";
  sp.pose.orientation.w = 1.0;
  sp.pose.position.z = 1.5;

  target_pose.header.frame_id = "map";
  target_pose.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  target_pose.type_mask = 0b100000000000;
  target_pose.position.z = 1.5;
  target_pose.position.x = NAN;
  target_pose.position.y = NAN;

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.05), cmdCallback);

  ros::Duration(1.0).sleep();

  ros::spin();

  return 0;
}