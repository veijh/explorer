#include <ros/ros.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <geometry_msgs/PointStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <Eigen/Dense>
#include "fstream"

using namespace std;

std::ofstream outputFile("/src/new_output.txt");

octomap::OcTree* ocmap;
void octomap_cb(const octomap_msgs::Octomap::ConstPtr& msg)
{
  // free memory for old map
  delete ocmap;
  ocmap = dynamic_cast<octomap::OcTree*>(msgToMap(*msg));
}

ros::Publisher marker0_pub;
ros::Publisher marker1_pub;
ros::Publisher marker2_pub;
visualization_msgs::Marker line0_strip;
visualization_msgs::Marker line1_strip;
visualization_msgs::Marker line2_strip;
nav_msgs::Odometry uav0_odom;
nav_msgs::Odometry uav1_odom;
nav_msgs::Odometry uav2_odom;

nav_msgs::Odometry last_uav0_odom;
nav_msgs::Odometry last_uav1_odom;
nav_msgs::Odometry last_uav2_odom;

bool init0 = false, init1 = false, init2 = false;

void odom0_cb(const nav_msgs::Odometry::ConstPtr& msg)
{
  if(!init0) {
    init0 = true;
    last_uav0_odom = *msg;
  }
  uav0_odom = *msg;
  geometry_msgs::Point p;
  p.x = uav0_odom.pose.pose.position.x;
  p.y = uav0_odom.pose.pose.position.y;
  p.z = uav0_odom.pose.pose.position.z;
  line0_strip.points.push_back(p);
  marker0_pub.publish(line0_strip);
}

void odom1_cb(const nav_msgs::Odometry::ConstPtr& msg)
{
  if(!init1) {
    init1 = true;
    last_uav1_odom = *msg;
  }
  uav1_odom = *msg;
  geometry_msgs::Point p;
  p.x = uav1_odom.pose.pose.position.x;
  p.y = uav1_odom.pose.pose.position.y;
  p.z = uav1_odom.pose.pose.position.z;
  line1_strip.points.push_back(p);
  marker1_pub.publish(line1_strip);
}

void odom2_cb(const nav_msgs::Odometry::ConstPtr& msg)
{
  if(!init2) {
    init2 = true;
    last_uav2_odom = *msg;
  }
  uav2_odom = *msg;
  geometry_msgs::Point p;
  p.x = uav2_odom.pose.pose.position.x;
  p.y = uav2_odom.pose.pose.position.y;
  p.z = uav2_odom.pose.pose.position.z;
  line2_strip.points.push_back(p);
  marker2_pub.publish(line2_strip);
}

float dis0 = 0.0, dis1 = 0.0, dis2 = 0.0;

void timerCallback(const ros::TimerEvent&)
{
  if(ocmap == nullptr) return;

  int count = 0;
  for(double x = -10.0; x <= 10.0; x += 0.1){
    for(double y = -10.0; y <= 10.0; y += 0.1){
      for(double z = 0.0; z <= 2.5; z += 0.1){
        if(ocmap->search(octomap::point3d(x,y,z)) != nullptr) {
          ++count;
        }
      }
    }
  }
  
  Eigen::Vector3f p0(uav0_odom.pose.pose.position.x, uav0_odom.pose.pose.position.y, uav0_odom.pose.pose.position.z);
  Eigen::Vector3f last_p0(last_uav0_odom.pose.pose.position.x, last_uav0_odom.pose.pose.position.y, last_uav0_odom.pose.pose.position.z);
  Eigen::Vector3f p1(uav1_odom.pose.pose.position.x, uav1_odom.pose.pose.position.y, uav1_odom.pose.pose.position.z);
  Eigen::Vector3f last_p1(last_uav1_odom.pose.pose.position.x, last_uav1_odom.pose.pose.position.y, last_uav1_odom.pose.pose.position.z);
  Eigen::Vector3f p2(uav2_odom.pose.pose.position.x, uav2_odom.pose.pose.position.y, uav2_odom.pose.pose.position.z);
  Eigen::Vector3f last_p2(last_uav2_odom.pose.pose.position.x, last_uav2_odom.pose.pose.position.y, last_uav2_odom.pose.pose.position.z);

  dis0 += (p0 - last_p0).norm();
  dis1 += (p1 - last_p1).norm();
  dis2 += (p2 - last_p2).norm();

  Eigen::Vector3f v0(uav0_odom.twist.twist.linear.x, uav0_odom.twist.twist.linear.y, uav0_odom.twist.twist.linear.z);
  Eigen::Vector3f v1(uav1_odom.twist.twist.linear.x, uav1_odom.twist.twist.linear.y, uav1_odom.twist.twist.linear.z);
  Eigen::Vector3f v2(uav2_odom.twist.twist.linear.x, uav2_odom.twist.twist.linear.y, uav2_odom.twist.twist.linear.z);
  outputFile << count << ", " << v0.norm() << ", " << v1.norm() << ", " << v2.norm() << 
  ", " << dis0 << ", " << dis1 << ", " << dis2 << endl;

  last_uav0_odom = uav0_odom;
  last_uav1_odom = uav1_odom;
  last_uav2_odom = uav2_odom;
}


int main(int argc, char** argv){
  ros::init(argc, argv, "analyze");
  ros::NodeHandle nh("");
  // get octomap
  ros::Subscriber octomap_sub = nh.subscribe<octomap_msgs::Octomap>("/merged_map", 1, octomap_cb);
  ros::Subscriber odom0_sub = nh.subscribe<nav_msgs::Odometry>("/uav0/mavros/local_position/odom", 1, odom0_cb);
  ros::Subscriber odom1_sub = nh.subscribe<nav_msgs::Odometry>("/uav1/mavros/local_position/odom", 1, odom1_cb);
  ros::Subscriber odom2_sub = nh.subscribe<nav_msgs::Odometry>("/uav2/mavros/local_position/odom", 1, odom2_cb);

  marker0_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker0", 10);
  marker1_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker1", 10);
  marker2_pub = nh.advertise<visualization_msgs::Marker>("visualization_marker2", 10);

  ros::Timer timer = nh.createTimer(ros::Duration(1.0), timerCallback);

  line0_strip.header.frame_id = "map"; // 坐标系
  line0_strip.header.stamp = ros::Time::now();
  line0_strip.ns = "line0_strip";
  line0_strip.action = visualization_msgs::Marker::ADD;
  line0_strip.pose.orientation.w = 1.0;
  line0_strip.id = 0;
  line0_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line0_strip.scale.x = 0.1; // 线宽
  line0_strip.color.r = 1.0; // 红色
  line0_strip.color.a = 1.0; // 不透明度

  line1_strip.header.frame_id = "map"; // 坐标系
  line1_strip.header.stamp = ros::Time::now();
  line1_strip.ns = "line1_strip";
  line1_strip.action = visualization_msgs::Marker::ADD;
  line1_strip.pose.orientation.w = 1.0;
  line1_strip.id = 0;
  line1_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line1_strip.scale.x = 0.1; // 线宽
  line1_strip.color.r = 1.0; // 红色
  line1_strip.color.b = 1.0; // 红色
  line1_strip.color.a = 1.0; // 不透明度

  line2_strip.header.frame_id = "map"; // 坐标系
  line2_strip.header.stamp = ros::Time::now();
  line2_strip.ns = "line2_strip";
  line2_strip.action = visualization_msgs::Marker::ADD;
  line2_strip.pose.orientation.w = 1.0;
  line2_strip.id = 0;
  line2_strip.type = visualization_msgs::Marker::LINE_STRIP;
  line2_strip.scale.x = 0.1; // 线宽
  line2_strip.color.b = 1.0; // 红色
  line2_strip.color.a = 1.0; // 不透明度


  if (!outputFile.is_open()) { // 检查文件是否成功打开
    std::cerr << "Failed to open the file!" << std::endl;
    return 1;
  }

  ros::spin();

  return 0;
}
