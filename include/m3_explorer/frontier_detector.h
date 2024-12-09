#ifndef FRONTIER_DETECTOR_H
#define FRONTIER_DETECTOR_H
#include "m3_explorer/QuadMesh.h"
#include <geometry_msgs/PointStamped.h>
#include <octomap/octomap.h>
#include <ros/ros.h>
#include <set>
#include <vector>

using namespace std;

// detect frontier:
// input = octomap; current pose; FOV; max range; region bbx
// output = a set containing all frontier voxels
void frontier_detect(set<QuadMesh> &frontiers, octomap::OcTree *ocmap,
                     const geometry_msgs::PointStamped &cur_pose,
                     const double &sensor_range);
bool is_next_to_obstacle(octomap::OcTree *ocmap, const octomap::point3d &point,
                         const double &check_box_size, const double &occ_trs);
void frontier_visualize(set<QuadMesh> &frontiers, const double &mesh_thickness,
                        ros::Publisher &frontier_maker_array_pub);
void frontier_normal_visualize(set<QuadMesh> &frontiers,
                               ros::Publisher &frontier_normal_pub);

#endif
