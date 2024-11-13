#include "m3_explorer/time_track.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

int main(int argc, char **argv) {
  ros::init(argc, argv, "create_s3dis_map_test");
  ros::NodeHandle nh("");

  ros::Rate rate(1);

  const float min_x = -40;
  const float max_x = 40;
  const float min_y = -40;
  const float max_y = 40;
  const float min_z = -0.5;
  const float max_z = 5.0;
  const float resolution = 0.1;

  const int num_x_grid =
      static_cast<int>(std::ceil((max_x - min_x) / resolution));
  const int num_y_grid =
      static_cast<int>(std::ceil((max_y - min_y) / resolution));
  const int num_z_grid =
      static_cast<int>(std::ceil((max_z - min_z) / resolution));

  std::vector<std::vector<std::vector<char>>> grid(
      num_x_grid, std::vector<std::vector<char>>(
                      num_y_grid, std::vector<char>(num_z_grid, 0)));

  std::vector<std::vector<std::vector<float>>> r_grid(
      num_x_grid, std::vector<std::vector<float>>(
                      num_y_grid, std::vector<float>(num_z_grid, 0)));

  std::vector<std::vector<std::vector<float>>> g_grid(
      num_x_grid, std::vector<std::vector<float>>(
                      num_y_grid, std::vector<float>(num_z_grid, 0)));

  std::vector<std::vector<std::vector<float>>> b_grid(
      num_x_grid, std::vector<std::vector<float>>(
                      num_y_grid, std::vector<float>(num_z_grid, 0)));

  const int floor_z_lb =
      static_cast<int>(std::ceil((-0.2 - min_z) / resolution));
  const int floor_z_ub =
      static_cast<int>(std::ceil((0.0 - min_z) / resolution));

  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      for (int k = floor_z_lb; k <= floor_z_ub; ++k) {
        grid[i][j][k] = 1;
        r_grid[i][j][k] = 128.0;
        g_grid[i][j][k] = 128.0;
        b_grid[i][j][k] = 128.0;
      }
    }
  }

  std::filesystem::path dir = "/src/Stanford3dDataset_v1.2/Area_5";
  std::vector<std::string> file_names;
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    file_names.emplace_back(entry.path().filename());
    std::cout << "Found " << entry.path().filename() << std::endl;
  }
  std::sort(file_names.begin(), file_names.end());
  for (const auto &file_name : file_names) {
    std::cout << "Reading " << file_name << std::endl;
    std::string file_path =
        dir.string() + "/" + file_name + "/" + file_name + ".txt";
    FILE *file = fopen(file_path.c_str(), "r");
    if (file == nullptr) {
      std::cout << "Failed to open " << file_path << std::endl;
      continue;
    }
    float x, y, z;
    float r, g, b;
    int count = 0;
    while (fscanf(file, "%f%f%f%f%f%f", &x, &y, &z, &r, &g, &b) == 6) {
      int x_grid = static_cast<int>(std::floor((x - min_x) / resolution));
      int y_grid = static_cast<int>(std::floor((y - min_y) / resolution));
      int z_grid = static_cast<int>(std::floor((z - min_z) / resolution));
      if (x_grid < 0 || x_grid >= num_x_grid || y_grid < 0 ||
          y_grid >= num_y_grid || z_grid < 0 || z_grid >= num_z_grid) {
        continue;
      }
      grid[x_grid][y_grid][z_grid] = 1;
      r_grid[x_grid][y_grid][z_grid] = r;
      g_grid[x_grid][y_grid][z_grid] = g;
      b_grid[x_grid][y_grid][z_grid] = b;
    }
    fclose(file);
    std::cout << "Finish reading " << file_name << std::endl;
  }

  visualization_msgs::Marker cube_list;
  cube_list.header.frame_id = "map";
  cube_list.header.stamp = ros::Time::now();
  cube_list.ns = "cube_list";
  cube_list.action = visualization_msgs::Marker::ADD;
  cube_list.pose.orientation.w = 1.0;
  cube_list.id = 0;
  cube_list.type = visualization_msgs::Marker::CUBE_LIST;
  cube_list.scale.x = resolution * 0.95;
  cube_list.scale.y = resolution * 0.95;
  cube_list.scale.z = resolution * 0.95;

  cube_list.points.clear();
  cube_list.colors.clear();
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      for (int k = 0; k < num_z_grid; ++k) {
        if (grid[i][j][k] == 1 && min_z + k * resolution < 2.8) {
          geometry_msgs::Point grid_pos;
          grid_pos.x = min_x + i * resolution + 0.5 * resolution;
          grid_pos.y = min_y + j * resolution + 0.5 * resolution;
          grid_pos.z = min_z + k * resolution + 0.5 * resolution;
          cube_list.points.emplace_back(grid_pos);
          cube_list.colors.emplace_back();
          cube_list.colors.back().a = 1.0;
          cube_list.colors.back().r = r_grid[i][j][k] / 255.0;
          cube_list.colors.back().g = g_grid[i][j][k] / 255.0;
          cube_list.colors.back().b = b_grid[i][j][k] / 255.0;
        }
      }
    }
  }

  ros::Publisher map_pub = nh.advertise<visualization_msgs::Marker>("/map", 10);

  while (ros::ok()) {
    rate.sleep();
    ros::spinOnce();
    map_pub.publish(cube_list);
  }

  return 0;
}
