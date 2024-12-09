#include "m3_explorer/grid_astar.h"
#include "m3_explorer/time_track.hpp"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <ros/ros.h>
#include <yaml-cpp/yaml.h>

octomap::OcTree *ocmap = nullptr;
void octomap_cb(const octomap_msgs::Octomap::ConstPtr &msg) {
  delete ocmap;
  ocmap = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "create_map_test");
  ros::NodeHandle nh("");

  // get octomap
  ros::Subscriber octomap_sub =
      nh.subscribe<octomap_msgs::Octomap>("/merged_map", 1, octomap_cb);

  ros::Rate rate(0.1);

  const float min_x = -10.5;
  const float max_x = 10.5;
  const float min_y = -10.5;
  const float max_y = 10.5;
  const float min_z = -0.1;
  const float max_z = 2.0;
  const float resolution = 0.1;
  GridAstar grid_astar(min_x, max_x, min_y, max_y, min_z, max_z, resolution);

  while (ros::ok()) {
    rate.sleep();
    ros::spinOnce();
    if (ocmap == nullptr) {
      continue;
    }
    octomap::point3d bx_min(min_x, min_y, min_z);
    octomap::point3d bx_max(max_x, max_y, max_z);
    TimeTrack track;
    grid_astar.UpdateFromMap(ocmap, bx_min, bx_max);
    track.OutputPassingTime("Update Map");

    const std::vector<std::vector<std::vector<GridAstar::GridState>>>
        &grid_map = grid_astar.grid_map();

    const int x_size = grid_map.size();
    const int y_size = grid_map[0].size();
    const int z_size = grid_map[0][0].size();

    // Set start & goal
    std::vector<double> start{7.5, 7.5, 1.5};
    std::vector<double> goal{-7.5, -7.5, 1.5};
    // Create a map
    std::vector<double> origin{min_x, min_y, min_z}; // set origin
    std::vector<int> dim{x_size, y_size,
                         z_size}; // set the number of cells in each dimension
    std::vector<int> data; // occupancy data, the subscript follows: id = x +
                           // dim.x * y + dim.x * dim.y * z;
    data.resize(dim[0] * dim[1] * dim[2],
                0); // initialize as free map, free cell has 0 occupancy

    for (int i = 0; i < x_size; ++i) {
      for (int j = 0; j < y_size; ++j) {
        for (int k = 0; k < z_size; ++k) {
          if (grid_map[i][j][k] != GridAstar::GridState::kFree) {
            data[i + dim[0] * j + dim[0] * dim[1] * k] = 1;
          }
        }
      }
    }

    YAML::Emitter out;
    out << YAML::BeginSeq;
    // Encode start coordinate
    out << YAML::BeginMap;
    out << YAML::Key << "start" << YAML::Value << YAML::Flow << start;
    out << YAML::EndMap;
    // Encode goal coordinate
    out << YAML::BeginMap;
    out << YAML::Key << "goal" << YAML::Value << YAML::Flow << goal;
    out << YAML::EndMap;
    // Encode origin coordinate
    out << YAML::BeginMap;
    out << YAML::Key << "origin" << YAML::Value << YAML::Flow << origin;
    out << YAML::EndMap;
    // Encode dimension as number of cells
    out << YAML::BeginMap;
    out << YAML::Key << "dim" << YAML::Value << YAML::Flow << dim;
    out << YAML::EndMap;
    // Encode resolution
    out << YAML::BeginMap;
    out << YAML::Key << "resolution" << YAML::Value << resolution;
    out << YAML::EndMap;
    // Encode occupancy
    out << YAML::BeginMap;
    out << YAML::Key << "data" << YAML::Value << YAML::Flow << data;
    out << YAML::EndMap;

    out << YAML::EndSeq;
    std::cout << "Here is the example map:\n" << out.c_str() << std::endl;

    std::ofstream file;
    file.open("/src/catkin_ws/src/m3_explorer/forest.yaml");
    file << out.c_str();
    file.close();
  }
  return 0;
}
