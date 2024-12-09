#ifndef FRONTIER_CLUSTER_H
#define FRONTIER_CLUSTER_H

#include "m3_explorer/QuadMesh.h"
#include <Eigen/Dense>
#include <geometry_msgs/PoseArray.h>
#include <octomap/octomap.h>
#include <ros/ros.h>
#include <set>
#include <vector>

#define MAX_ITERATION (1000)

struct Cluster {
  Eigen::Vector3f center;
  Eigen::Vector3f normal;
};

using namespace std;

//// frontier clustering
//// input = the set containing all frontier voxels
//// output = a vector containing all frontier clusters
vector<Cluster> k_mean_cluster(set<QuadMesh> &frontiers);
void cluster_visualize(vector<Cluster> &cluster_vec,
                       ros::Publisher &cluster_pub);
vector<Cluster> dbscan_cluster(set<QuadMesh> &frontiers, const float &eps,
                               const int &min_pts, const int &min_cluster_pts,
                               ros::Publisher &cluster_vis_pub);

//// generate view point
//// input = a vector containing all frontier clusters
//// output = a vector containing poses of all view points
geometry_msgs::PoseArray view_point_generate(vector<Cluster> &cluster_vec,
                                             octomap::OcTree *ocmap);
#endif
