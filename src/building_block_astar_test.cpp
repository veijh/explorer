#include "m3_explorer/grid_astar.h"
#include "m3_explorer/time_track.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <random>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

namespace {
const float min_x = -40.0;
const float max_x = 40.0;
const float min_y = -40.0;
const float max_y = 40.0;
const float resolution = 0.1;
const int num_floors = 3;
const int floor_height = 30;
const int floor_thickness = 3;
const int num_office = 10;
const int wall_thickness = 5;
const int door_width = 15;
const int num_obstacle = 150;
const int num_stair = 2;
const int stair_length = 50;
const int stair_width = 50;
} // namespace

class Box2D {
public:
  int center_x_;
  int center_y_;
  float heading_;
  int length_;
  int width_;
  int min_x_;
  int max_x_;
  int min_y_;
  int max_y_;

public:
  Box2D(const int center_x, const int center_y, const float heading,
        const int length, const int width);
  ~Box2D();
  bool isNear(const int x, const int y, const int delta) const;
  bool isInside(const int x, const int y) const;
};

Box2D::Box2D(const int center_x, const int center_y, const float heading,
             const int length, const int width)
    : center_x_(center_x), center_y_(center_y), heading_(heading),
      length_(length), width_(width) {
  min_x_ = center_x_;
  min_y_ = center_y_;
  max_x_ = center_x_;
  max_y_ = center_y_;
  const float half_length = length_ * 0.5;
  const float half_width = width_ * 0.5;
  const std::vector<float> signs = {-1.0, 1.0};
  for (const float sign_x : signs) {
    for (const float sign_y : signs) {
      const int transformed_x = cos(heading_) * sign_x * half_length -
                                sin(heading_) * sign_y * half_width + center_x_;
      const int transformed_y = sin(heading_) * sign_x * half_length +
                                cos(heading_) * sign_y * half_width + center_y_;
      min_x_ = std::min(min_x_, transformed_x);
      max_x_ = std::max(max_x_, transformed_x);
      min_y_ = std::min(min_y_, transformed_y);
      max_y_ = std::max(max_y_, transformed_y);
    }
  }
}

Box2D::~Box2D(){};

bool Box2D::isNear(const int x, const int y, const int delta) const {
  const int shifted_x = x - center_x_;
  const int shifted_y = y - center_y_;
  const int transformed_x =
      cos(heading_) * shifted_x + sin(heading_) * shifted_y;
  const int transformed_y =
      -sin(heading_) * shifted_x + cos(heading_) * shifted_y;
  const int abs_x = std::fabs(transformed_x);
  const int abs_y = std::fabs(transformed_y);
  bool is_near_x = (std::fabs(abs_x - length_ * 0.5) < delta) &&
                   (abs_y < 0.5 * width_ + delta);
  bool is_near_y = (std::fabs(abs_y - width_ * 0.5) < delta) &&
                   (abs_x < 0.5 * length_ + delta);
  return is_near_x || is_near_y;
}

bool Box2D::isInside(const int x, const int y) const {
  const int shifted_x = x - center_x_;
  const int shifted_y = y - center_y_;
  const int transformed_x =
      cos(heading_) * shifted_x + sin(heading_) * shifted_y;
  const int transformed_y =
      -sin(heading_) * shifted_x + cos(heading_) * shifted_y;
  return (std::fabs(transformed_x) < length_ * 0.5 &&
          std::fabs(transformed_y) < width_ * 0.5);
}

std::vector<std::vector<std::vector<float>>> r_grid;
std::vector<std::vector<std::vector<float>>> g_grid;
std::vector<std::vector<std::vector<float>>> b_grid;
std::vector<std::vector<std::vector<float>>> a_grid;

void create_floor(
    std::vector<std::vector<std::vector<GridAstar::GridState>>> &grid,
    int floor_z_lb, int floor_z_ub) {
  const int num_x_grid = grid.size();
  const int num_y_grid = grid[0].size();
  const int num_z_grid = grid[0][0].size();
  if (floor_z_lb < 0 || floor_z_lb >= num_z_grid || floor_z_ub < 0 ||
      floor_z_ub >= num_z_grid) {
    std::cout << "Invalid floor z range" << std::endl;
    return;
  }
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      for (int k = floor_z_lb; k <= floor_z_ub; ++k) {
        grid[i][j][k] = GridAstar::GridState::kOcc;
        r_grid[i][j][k] = 0.5;
        g_grid[i][j][k] = 0.5;
        b_grid[i][j][k] = 0.5;
        a_grid[i][j][k] = 1.0;
      }
    }
  }
}

void create_office(
    std::vector<std::vector<std::vector<GridAstar::GridState>>> &grid,
    int floor_z, int ceiling_z, const Box2D &box, int wall_thickness) {
  const int num_x_grid = grid.size();
  const int num_y_grid = grid[0].size();
  const int num_z_grid = grid[0][0].size();
  // create walls
  for (int x = std::max(0, box.min_x_ - wall_thickness);
       x <= std::min(num_x_grid - 1, box.max_x_ + wall_thickness); ++x) {
    for (int y = std::max(0, box.min_y_ - wall_thickness);
         y <= std::min(num_y_grid - 1, box.max_y_ + wall_thickness); ++y) {
      if (box.isInside(x, y)) {
        for (int z = floor_z; z <= ceiling_z; ++z) {
          grid[x][y][z] = GridAstar::GridState::kFree;
        }
      } else if (box.isNear(x, y, wall_thickness)) {
        for (int z = floor_z; z <= ceiling_z; ++z) {
          grid[x][y][z] = GridAstar::GridState::kOcc;
          r_grid[x][y][z] = 1.0;
          g_grid[x][y][z] = 1.0;
          b_grid[x][y][z] = 1.0;
          a_grid[x][y][z] = 1.0;
        }
      }
    }
  }
  // create doors
  const int door_center_x =
      box.center_x_ +
      cos(box.heading_) * (0.5 * box.length_ + 0.5 * wall_thickness) -
      sin(box.heading_) * 0;
  const int door_center_y =
      box.center_y_ +
      sin(box.heading_) * (0.5 * box.length_ + 0.5 * wall_thickness) +
      cos(box.heading_) * 0;
  Box2D door_box(door_center_x, door_center_y, box.heading_, wall_thickness + 2,
                 door_width);
  for (int x = std::max(0, door_box.min_x_);
       x <= std::min(num_y_grid - 1, door_box.max_x_); ++x) {
    for (int y = std::max(0, door_box.min_y_);
         y <= std::min(num_y_grid - 1, door_box.max_y_); ++y) {
      if (door_box.isInside(x, y)) {
        for (int z = floor_z; z <= ceiling_z; ++z) {
          grid[x][y][z] = GridAstar::GridState::kFree;
        }
      }
    }
  }
  // const int backup_door_center_x =
  //     box.center_x_ + cos(box.heading_) * 0 -
  //     sin(box.heading_) * (0.5 * box.width_ + 0.5 * wall_thickness);
  // const int backup_door_center_y =
  //     box.center_y_ + sin(box.heading_) * 0 +
  //     cos(box.heading_) * (0.5 * box.width_ + 0.5 * wall_thickness);
  // Box2D backup_door_box(backup_door_center_x, backup_door_center_y,
  //                       box.heading_ + M_PI_2, wall_thickness + 2,
  //                       door_width);
  // for (int x = std::max(0, backup_door_box.min_x_);
  //      x <= std::min(num_y_grid - 1, backup_door_box.max_x_); ++x) {
  //   for (int y = std::max(0, backup_door_box.min_y_);
  //        y <= std::min(num_y_grid - 1, backup_door_box.max_y_); ++y) {
  //     if (backup_door_box.isInside(x, y)) {
  //       for (int z = floor_z; z <= ceiling_z; ++z) {
  //         grid[x][y][z] = GridAstar::GridState::kFree;
  //       }
  //     }
  //   }
  // }
}

void create_obstacle(
    std::vector<std::vector<std::vector<GridAstar::GridState>>> &grid,
    int obstacle_z_lb, int obstacle_z_ub, const Box2D &box) {
  const int num_x_grid = grid.size();
  const int num_y_grid = grid[0].size();
  const int num_z_grid = grid[0][0].size();
  // create cube with random size and position
  for (int x = std::max(0, box.min_x_);
       x <= std::min(num_x_grid - 1, box.max_x_); ++x) {
    for (int y = std::max(0, box.min_y_);
         y <= std::min(num_y_grid - 1, box.max_y_); ++y) {
      if (box.isInside(x, y)) {
        for (int z = obstacle_z_lb; z < obstacle_z_ub; ++z) {
          grid[x][y][z] = GridAstar::GridState::kOcc;
          r_grid[x][y][z] = 0.5;
          g_grid[x][y][z] = 0.25;
          b_grid[x][y][z] = 0.0;
          a_grid[x][y][z] = 1.0;
        }
      }
    }
  }
}

void create_stair(
    std::vector<std::vector<std::vector<GridAstar::GridState>>> &grid,
    int stair_z_lb, int stair_z_ub, const Box2D &box) {
  const int num_x_grid = grid.size();
  const int num_y_grid = grid[0].size();
  const int num_z_grid = grid[0][0].size();
  // remove floor
  for (int x = std::max(0, box.min_x_);
       x <= std::min(num_x_grid - 1, box.max_x_); ++x) {
    for (int y = std::max(0, box.min_y_);
         y <= std::min(num_y_grid - 1, box.max_y_); ++y) {
      if (box.isInside(x, y)) {
        for (int z = stair_z_lb; z < stair_z_ub; ++z) {
          grid[x][y][z] = GridAstar::GridState::kFree;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "building_block_astar_test");
  ros::NodeHandle nh("");

  ros::Rate rate(1);

  const float min_z = 0.0;
  const float max_z = (floor_height * num_floors + 1) * resolution;

  const int num_x_grid =
      static_cast<int>(std::ceil((max_x - min_x) / resolution));
  const int num_y_grid =
      static_cast<int>(std::ceil((max_y - min_y) / resolution));
  const int num_z_grid =
      static_cast<int>(std::ceil((max_z - min_z) / resolution));

  std::vector<std::vector<std::vector<GridAstar::GridState>>> grid(
      num_x_grid,
      std::vector<std::vector<GridAstar::GridState>>(
          num_y_grid, std::vector<GridAstar::GridState>(
                          num_z_grid, GridAstar::GridState::kFree)));

  r_grid.resize(num_x_grid, std::vector<std::vector<float>>(
                                num_y_grid, std::vector<float>(num_z_grid, 0)));

  g_grid.resize(num_x_grid, std::vector<std::vector<float>>(
                                num_y_grid, std::vector<float>(num_z_grid, 0)));

  b_grid.resize(num_x_grid, std::vector<std::vector<float>>(
                                num_y_grid, std::vector<float>(num_z_grid, 0)));

  a_grid.resize(num_x_grid, std::vector<std::vector<float>>(
                                num_y_grid, std::vector<float>(num_z_grid, 0)));

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> random_x(0, num_x_grid);
  std::uniform_int_distribution<> random_y(0, num_y_grid);
  std::uniform_int_distribution<> random_office_length(50, 100);
  std::uniform_int_distribution<> random_office_width(50, 100);
  std::uniform_real_distribution<> random_heading(0, M_PI);
  std::uniform_int_distribution<> random_obstacle_length(3, 100);
  std::uniform_int_distribution<> random_obstacle_width(3, 10);
  std::uniform_int_distribution<> random_obstacle_height(3, floor_height);

  // create multiple floors
  for (int floor_id = 0; floor_id < num_floors; ++floor_id) {
    create_floor(grid, floor_id * floor_height,
                 floor_id * floor_height + floor_thickness);
    const int floor_surface = floor_id * floor_height + floor_thickness + 1;
    // create random offices
    for (int office_id = 0; office_id < num_office; ++office_id) {
      create_office(grid, floor_surface,
                    floor_surface + floor_height - floor_thickness,
                    Box2D(random_x(gen), random_y(gen), random_heading(gen),
                          random_office_length(gen), random_office_width(gen)),
                    wall_thickness);
    }
    // create obstacles
    for (int obstacle_id = 0; obstacle_id < num_obstacle; ++obstacle_id) {
      create_obstacle(grid, floor_surface, floor_surface + floor_height,
                      Box2D(random_x(gen), random_y(gen), random_heading(gen),
                            random_obstacle_length(gen),
                            random_obstacle_width(gen)));
    }
    // create stairs
    for (int stair_id = 0; stair_id < num_stair; ++stair_id) {
      create_stair(
          grid, floor_id * floor_height, floor_surface,
          Box2D(random_x(gen), random_y(gen), 0.0, stair_length, stair_width));
    }
  }

  visualization_msgs::Marker cube_list0;
  cube_list0.header.frame_id = "map";
  cube_list0.header.stamp = ros::Time::now();
  cube_list0.ns = "cube_list0";
  cube_list0.action = visualization_msgs::Marker::ADD;
  cube_list0.pose.orientation.w = 1.0;
  cube_list0.id = 0;
  cube_list0.type = visualization_msgs::Marker::CUBE_LIST;
  cube_list0.scale.x = resolution * 0.95;
  cube_list0.scale.y = resolution * 0.95;
  cube_list0.scale.z = resolution * 0.95;

  cube_list0.points.clear();
  cube_list0.colors.clear();
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      for (int k = 0; k < floor_height; ++k) {
        if (grid[i][j][k] == GridAstar::GridState::kOcc) {
          geometry_msgs::Point grid_pos;
          grid_pos.x = min_x + i * resolution + 0.5 * resolution;
          grid_pos.y = min_y + j * resolution + 0.5 * resolution;
          grid_pos.z = min_z + k * resolution + 0.5 * resolution;
          cube_list0.points.emplace_back(grid_pos);
          cube_list0.colors.emplace_back();
          cube_list0.colors.back().a = a_grid[i][j][k];
          cube_list0.colors.back().r = r_grid[i][j][k];
          cube_list0.colors.back().g = g_grid[i][j][k];
          cube_list0.colors.back().b = b_grid[i][j][k];
        }
      }
    }
  }

  visualization_msgs::Marker cube_list1;
  cube_list1.header.frame_id = "map";
  cube_list1.header.stamp = ros::Time::now();
  cube_list1.ns = "cube_list1";
  cube_list1.action = visualization_msgs::Marker::ADD;
  cube_list1.pose.orientation.w = 1.0;
  cube_list1.id = 0;
  cube_list1.type = visualization_msgs::Marker::CUBE_LIST;
  cube_list1.scale.x = resolution * 0.95;
  cube_list1.scale.y = resolution * 0.95;
  cube_list1.scale.z = resolution * 0.95;

  if (num_floors > 1) {
    cube_list1.points.clear();
    cube_list1.colors.clear();
    for (int i = 0; i < num_x_grid; ++i) {
      for (int j = 0; j < num_y_grid; ++j) {
        for (int k = floor_height; k < 2 * floor_height; ++k) {
          if (grid[i][j][k] == GridAstar::GridState::kOcc) {
            geometry_msgs::Point grid_pos;
            grid_pos.x = min_x + i * resolution + 0.5 * resolution;
            grid_pos.y = min_y + j * resolution + 0.5 * resolution;
            grid_pos.z = min_z + k * resolution + 0.5 * resolution;
            cube_list1.points.emplace_back(grid_pos);
            cube_list1.colors.emplace_back();
            cube_list1.colors.back().a = a_grid[i][j][k];
            cube_list1.colors.back().r = r_grid[i][j][k];
            cube_list1.colors.back().g = g_grid[i][j][k];
            cube_list1.colors.back().b = b_grid[i][j][k];
          }
        }
      }
    }
  }

  visualization_msgs::Marker cube_list2;
  cube_list2.header.frame_id = "map";
  cube_list2.header.stamp = ros::Time::now();
  cube_list2.ns = "cube_list2";
  cube_list2.action = visualization_msgs::Marker::ADD;
  cube_list2.pose.orientation.w = 1.0;
  cube_list2.id = 0;
  cube_list2.type = visualization_msgs::Marker::CUBE_LIST;
  cube_list2.scale.x = resolution * 0.95;
  cube_list2.scale.y = resolution * 0.95;
  cube_list2.scale.z = resolution * 0.95;

  if (num_floors > 2) {
    cube_list2.points.clear();
    cube_list2.colors.clear();
    for (int i = 0; i < num_x_grid; ++i) {
      for (int j = 0; j < num_y_grid; ++j) {
        for (int k = 2 * floor_height; k < 3 * floor_height; ++k) {
          if (grid[i][j][k] == GridAstar::GridState::kOcc) {
            geometry_msgs::Point grid_pos;
            grid_pos.x = min_x + i * resolution + 0.5 * resolution;
            grid_pos.y = min_y + j * resolution + 0.5 * resolution;
            grid_pos.z = min_z + k * resolution + 0.5 * resolution;
            cube_list2.points.emplace_back(grid_pos);
            cube_list2.colors.emplace_back();
            cube_list2.colors.back().a = a_grid[i][j][k];
            cube_list2.colors.back().r = r_grid[i][j][k];
            cube_list2.colors.back().g = g_grid[i][j][k];
            cube_list2.colors.back().b = b_grid[i][j][k];
          }
        }
      }
    }
  }

  ros::Publisher map0_pub =
      nh.advertise<visualization_msgs::Marker>("/map0", 10);
  ros::Publisher map1_pub =
      nh.advertise<visualization_msgs::Marker>("/map1", 10);
  ros::Publisher map2_pub =
      nh.advertise<visualization_msgs::Marker>("/map2", 10);

  // visualize waypoint
  ros::Publisher wp_pub =
      nh.advertise<visualization_msgs::Marker>("/waypoint", 10);

  // visualize voxel
  ros::Publisher voxel_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/voxel", 10);

  // visualize voxel2D
  ros::Publisher voxel2d_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/voxel2d", 10);

  // visualize voxel3D
  ros::Publisher voxel3d_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/voxel3d", 10);

  // visualize topo point
  ros::Publisher topo_pub =
      nh.advertise<visualization_msgs::Marker>("/topo_point", 10);

  // visualize topo point
  ros::Publisher block_path_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/block_path", 10);

  // visualize ilqr_waypoint
  ros::Publisher ilqr_pub =
      nh.advertise<visualization_msgs::Marker>("/ilqr_wp", 10);

  GridAstar grid_astar(min_x, max_x, min_y, max_y, min_z, max_z, resolution,
                       grid);

  visualization_msgs::Marker waypoint;
  waypoint.header.frame_id = "map";
  waypoint.header.stamp = ros::Time::now();
  waypoint.ns = "line_list";
  waypoint.action = visualization_msgs::Marker::ADD;
  waypoint.scale.x = 0.1;
  waypoint.pose.orientation.w = 1.0;
  waypoint.id = 0;
  waypoint.type = visualization_msgs::Marker::LINE_STRIP;

  visualization_msgs::MarkerArray voxels;
  visualization_msgs::Marker voxel;
  voxel.header.frame_id = "map";
  voxel.header.stamp = ros::Time::now();
  voxel.ns = "voxel";
  voxel.action = visualization_msgs::Marker::ADD;
  voxel.pose.orientation.w = 1.0;
  voxel.type = visualization_msgs::Marker::CUBE;
  voxel.scale.x = 0.06;
  voxel.color.r = 0.5;
  voxel.color.g = 1.0;
  voxel.color.a = 1.0;

  visualization_msgs::Marker topo_list;
  topo_list.header.frame_id = "map";
  topo_list.header.stamp = ros::Time::now();
  topo_list.ns = "topo_list";
  topo_list.action = visualization_msgs::Marker::ADD;
  topo_list.pose.orientation.w = 1.0;
  topo_list.id = 0;
  topo_list.type = visualization_msgs::Marker::LINE_LIST;
  topo_list.scale.x = 0.1;

  visualization_msgs::MarkerArray blocks;
  visualization_msgs::Marker block;
  block.header.frame_id = "map";
  block.header.stamp = ros::Time::now();
  block.ns = "block";
  block.action = visualization_msgs::Marker::ADD;
  block.pose.orientation.w = 1.0;
  block.type = visualization_msgs::Marker::CUBE;
  block.color.r = 0.2;
  block.color.g = 0.2;
  block.color.g = 0.2;
  block.color.a = 0.8;

  std::uniform_real_distribution<float> xy(min_x, max_x);
  std::uniform_real_distribution<float> z(1.0, 15.0);

  std::uniform_real_distribution<float> distrib(0.0, 1.0);

  int rolling_x = 0;
  int count = 0;

  while (ros::ok()) {
    rate.sleep();
    ros::spinOnce();
    // std::getchar();
    if (count == 1000) {
      break;
    } else {
      ++count;
    }
    map0_pub.publish(cube_list0);
    map1_pub.publish(cube_list1);
    map2_pub.publish(cube_list2);

    TimeTrack track;
    track.SetStartTime();
    grid_astar.MergeMap();
    track.OutputPassingTime("Merge Map");

    track.SetStartTime();
    grid_astar.MergeMap2D();
    track.OutputPassingTime("Merge Map2D");

    track.SetStartTime();
    grid_astar.MergeMap3D();
    track.OutputPassingTime("Merge Map3D");

    track.SetStartTime();
    const std::vector<std::vector<std::vector<GridAstar::GridState>>>
        &grid_map = grid_astar.grid_map();

    const int x_size = grid_map.size();
    const int y_size = grid_map[0].size();
    const int z_size = grid_map[0][0].size();

    track.SetStartTime();
    voxels.markers.clear();
    rolling_x = (rolling_x + 1) % x_size;
    const std::vector<std::vector<std::vector<RangeVoxel>>> &merge_map =
        grid_astar.merge_map();
    int id = 0;
    const int num_x = merge_map.size();
    const int num_y = merge_map[0].size();
    for (int i = rolling_x; i < rolling_x + 1; ++i) {
      for (int j = 0; j < num_y; ++j) {
        const int num_z = merge_map[i][j].size();
        for (int k = 0; k < num_z; ++k) {
          const RangeVoxel range_voxel = merge_map[i][j][k];
          voxel.id = id;
          voxel.scale.x = resolution * 0.8;
          voxel.scale.y = resolution * 0.8;
          voxel.scale.z =
              resolution * (range_voxel.max_ - range_voxel.min_ + 1);
          voxel.pose.position.x = min_x + i * resolution + 0.5 * resolution;
          voxel.pose.position.y = min_y + j * resolution + 0.5 * resolution;
          voxel.pose.position.z =
              min_z + 0.5 * (range_voxel.min_ + range_voxel.max_) * resolution +
              0.5 * resolution;
          ++id;
          voxels.markers.emplace_back(voxel);
        }
      }
    }
    voxel_pub.publish(voxels);
    track.OutputPassingTime("Visualize Voxel");

    track.SetStartTime();
    voxels.markers.clear();
    const std::vector<std::vector<Block2D>> &merge_map_2d =
        grid_astar.merge_map_2d();
    int voxel2d_id = 0;
    const int num_x_voxel2d = merge_map_2d.size();
    for (int i = rolling_x; i < rolling_x + 1; ++i) {
      const int num_voxel2d = merge_map_2d[i].size();
      for (int j = 0; j < num_voxel2d; ++j) {
        const Block2D range_voxel_2d = merge_map_2d[i][j];
        voxel.id = voxel2d_id;
        voxel.color.a = 1.0;
        voxel.color.r = distrib(gen);
        voxel.color.g = distrib(gen);
        voxel.color.b = distrib(gen);
        voxel.scale.x = resolution * 0.8;
        voxel.scale.y =
            resolution * (range_voxel_2d.y_max_ - range_voxel_2d.y_min_ + 1);
        voxel.scale.z =
            resolution * (range_voxel_2d.z_max_ - range_voxel_2d.z_min_ + 1);
        voxel.pose.position.x = min_x + i * resolution + 0.5 * resolution;
        voxel.pose.position.y =
            min_y +
            0.5 * (range_voxel_2d.y_min_ + range_voxel_2d.y_max_) * resolution +
            0.5 * resolution;
        voxel.pose.position.z =
            min_z +
            0.5 * (range_voxel_2d.z_min_ + range_voxel_2d.z_max_) * resolution +
            0.5 * resolution;
        ++voxel2d_id;
        voxels.markers.emplace_back(voxel);
      }
    }
    voxel2d_pub.publish(voxels);
    track.OutputPassingTime("Visualize Voxel2D");

    track.SetStartTime();
    voxels.markers.clear();
    const std::vector<Block3D> &merge_map_3d = grid_astar.merge_map_3d();
    int voxel3d_id = 0;
    const int num_voxel3d = merge_map_3d.size();
    for (int i = 0; i < num_voxel3d; ++i) {
      const Block3D range_voxel_3d = merge_map_3d[i];
      voxel.id = voxel3d_id;
      voxel.color.a = 1.0;
      voxel.color.r = distrib(gen);
      voxel.color.g = distrib(gen);
      voxel.color.b = distrib(gen);
      voxel.scale.x =
          resolution * (range_voxel_3d.x_max_ - range_voxel_3d.x_min_ + 1);
      voxel.scale.y =
          resolution * (range_voxel_3d.y_max_ - range_voxel_3d.y_min_ + 1);
      voxel.scale.z =
          resolution * (range_voxel_3d.z_max_ - range_voxel_3d.z_min_ + 1);
      voxel.pose.position.x =
          min_x +
          0.5 * (range_voxel_3d.x_min_ + range_voxel_3d.x_max_) * resolution +
          0.5 * resolution;
      voxel.pose.position.y =
          min_y +
          0.5 * (range_voxel_3d.y_min_ + range_voxel_3d.y_max_) * resolution +
          0.5 * resolution;
      voxel.pose.position.z =
          min_z +
          0.5 * (range_voxel_3d.z_min_ + range_voxel_3d.z_max_) * resolution +
          0.5 * resolution;
      ++voxel3d_id;
      voxels.markers.emplace_back(voxel);
    }
    voxel3d_pub.publish(voxels);
    track.OutputPassingTime("Visualize Voxel3D");

    track.SetStartTime();
    topo_list.points.clear();
    topo_list.colors.clear();
    const GraphTable &graph_table = grid_astar.graph_table();
    const int num_node = graph_table.nodes_.size();
    for (int i = 0; i < num_node; ++i) {
      const KeyBlock &key_block = graph_table.nodes_[i].key_block_;
      const int num_ranges = key_block.block_.ranges_.size();
      const int block_x = key_block.x_;
      const float r = distrib(gen);
      const float g = distrib(gen);
      const float b = distrib(gen);
      const float a = 1.0;
      for (int j = 0; j < num_ranges; ++j) {
        const RangeVoxel &range = key_block.block_.ranges_[j];
        const int range_y = key_block.block_.y_min_ + j;
        // Add start point.
        topo_list.points.emplace_back();
        topo_list.points.back().x =
            min_x + block_x * resolution + 0.5 * resolution;
        topo_list.points.back().y =
            min_y + range_y * resolution + 0.5 * resolution;
        topo_list.points.back().z =
            min_z + range.min_ * resolution + 0.5 * resolution;
        topo_list.colors.emplace_back();
        topo_list.colors.back().r = r;
        topo_list.colors.back().g = g;
        topo_list.colors.back().b = b;
        topo_list.colors.back().a = a;
        // Add end point.
        topo_list.points.emplace_back();
        topo_list.points.back().x =
            min_x + block_x * resolution + 0.5 * resolution;
        topo_list.points.back().y =
            min_y + range_y * resolution + 0.5 * resolution;
        topo_list.points.back().z =
            min_z + range.max_ * resolution + 0.5 * resolution;
        topo_list.colors.emplace_back();
        topo_list.colors.back().r = r;
        topo_list.colors.back().g = g;
        topo_list.colors.back().b = b;
        topo_list.colors.back().a = a;
      }
    }
    topo_pub.publish(topo_list);
    track.OutputPassingTime("Visualize Key Blocks");

    // 设定起点终点
    // Eigen::Vector3f start_pt = {xy(gen), xy(gen), z(gen)};
    // Eigen::Vector3f end_pt = {xy(gen), xy(gen), z(gen)};
    Eigen::Vector3f start_pt = {-30.0, -30.0, 0.5 * floor_height * resolution};
    Eigen::Vector3f end_pt = {30.0, 30.0, 2.5 * floor_height * resolution};
    // Eigen::Vector3f start_pt = {-30.0, -30.0, 2.5};
    // Eigen::Vector3f end_pt = {30.0, 30.0, 12.5};

    // 检查是否起点、终点是否合法
    int index_start_x =
        static_cast<int>(std::floor((start_pt.x() - min_x) / resolution));
    int index_start_y =
        static_cast<int>(std::floor((start_pt.y() - min_y) / resolution));
    int index_start_z =
        static_cast<int>(std::floor((start_pt.z() - min_z) / resolution));
    int index_end_x =
        static_cast<int>(std::floor((end_pt.x() - min_x) / resolution));
    int index_end_y =
        static_cast<int>(std::floor((end_pt.y() - min_y) / resolution));
    int index_end_z =
        static_cast<int>(std::floor((end_pt.z() - min_z) / resolution));
    if (grid_map[index_start_x][index_start_y][index_start_z] ==
            GridAstar::GridState::kOcc ||
        grid_map[index_end_x][index_end_y][index_end_z] ==
            GridAstar::GridState::kOcc) {
      continue;
    }

    // Djikstra寻路，并统计时间
    track.SetStartTime();
    grid_astar.BlockPathDistance(start_pt, end_pt);
    track.OutputPassingTime("--Djikstra Search Total--");

    // 可视化
    blocks.markers.clear();
    const std::vector<int> &block_path = grid_astar.block_path();
    std::unordered_map<int, int> block_id_map;
    for (int i = 0; i < merge_map_3d.size(); ++i) {
      if (block_id_map.find(merge_map_3d[i].block_id_) == block_id_map.end()) {
        block_id_map[merge_map_3d[i].block_id_] = i;
      } else {
        std::cout << "Error: block_id_map has duplicate key." << std::endl;
        std::cout << "block_id: " << merge_map_3d[i].block_id_ << std::endl;
      }
    }
    std::vector<int> block_path_id;
    for (int i = 0; i < block_path.size(); ++i) {
      const int node_id = block_path[i];
      const int block_id = graph_table.nodes_[node_id].key_block_.block_id_;
      block_path_id.emplace_back(block_id_map[block_id]);
    }
    std::unique(block_path_id.begin(), block_path_id.end());
    const int num_block_path_id = block_path_id.size();
    int block_3d_id = 0;
    for (int i = 0; i < num_block_path_id; ++i) {
      const Block3D block_3d = merge_map_3d[block_path_id[i]];
      block.id = block_3d_id;
      block.color.a = 0.5;
      block.color.r = distrib(gen);
      block.color.g = distrib(gen);
      block.color.b = distrib(gen);
      block.scale.x = resolution * (block_3d.x_max_ - block_3d.x_min_ + 1);
      block.scale.y = resolution * (block_3d.y_max_ - block_3d.y_min_ + 1);
      block.scale.z = resolution * (block_3d.z_max_ - block_3d.z_min_ + 1);
      block.pose.position.x =
          min_x + 0.5 * (block_3d.x_min_ + block_3d.x_max_) * resolution +
          0.5 * resolution;
      block.pose.position.y =
          min_y + 0.5 * (block_3d.y_min_ + block_3d.y_max_) * resolution +
          0.5 * resolution;
      block.pose.position.z =
          min_z + 0.5 * (block_3d.z_min_ + block_3d.z_max_) * resolution +
          0.5 * resolution;
      ++block_3d_id;
      blocks.markers.emplace_back(block);
    }
    for (int i = num_block_path_id; i < 100; ++i) {
      block.id = block_3d_id;
      block.color.a = 0.0;
      ++block_3d_id;
      blocks.markers.emplace_back(block);
    }
    block_path_pub.publish(blocks);

    track.SetStartTime();
    std::cout << "Refine Block Path Distance: "
              << grid_astar.BlockPathRefine(grid_astar.block_path(), start_pt,
                                            end_pt)
              << std::endl;
    track.OutputPassingTime("Block Path Refine");

    // 可视化轨迹
    // waypoint.points.clear();
    waypoint.color.r = 1.0;
    waypoint.color.g = 0.2;
    waypoint.color.b = 0.2;
    waypoint.color.a = 1.0;
    const std::vector<std::vector<float>> &ilqr_path = grid_astar.ilqr_path();
    const int ilqr_wp_num = ilqr_path.size();
    for (int i = 0; i < ilqr_wp_num; ++i) {
      geometry_msgs::Point wp_pos;
      wp_pos.x = min_x + ilqr_path[i][0] * resolution;
      wp_pos.y = min_y + ilqr_path[i][1] * resolution;
      wp_pos.z = min_z + ilqr_path[i][2] * resolution;
      waypoint.points.emplace_back(wp_pos);
      if (i != 0 && i != ilqr_wp_num - 1) {
        waypoint.points.emplace_back(wp_pos);
      }
    }
    ilqr_pub.publish(waypoint);

    // A*寻路，并统计时间
    track.SetStartTime();
    grid_astar.AstarPathDistance(start_pt, end_pt);
    track.OutputPassingTime("--Astar Search Total--");
    // 可视化轨迹
    waypoint.points.clear();
    waypoint.color.r = 0.0;
    waypoint.color.g = 0.0;
    waypoint.color.b = 1.0;
    waypoint.color.a = 1.0;
    const std::vector<std::shared_ptr<GridAstarNode>> &path = grid_astar.path();
    const int wp_num = path.size();
    for (int i = 0; i < wp_num; ++i) {
      geometry_msgs::Point wp_pos;
      wp_pos.x = min_x + path[i]->index_x_ * resolution;
      wp_pos.y = min_y + path[i]->index_y_ * resolution;
      wp_pos.z = min_z + path[i]->index_z_ * resolution;
      waypoint.points.emplace_back(wp_pos);
    }
    wp_pub.publish(waypoint);
  }
}
