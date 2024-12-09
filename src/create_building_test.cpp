#include "explorer/time_track.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

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

void create_floor(std::vector<std::vector<std::vector<char>>> &grid,
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
        if (grid[i][j][k] == -1) {
          grid[i][j][k] = 1;
          r_grid[i][j][k] = 0.5;
          g_grid[i][j][k] = 0.5;
          b_grid[i][j][k] = 0.5;
          a_grid[i][j][k] = 0.8;
        }
      }
    }
  }
}

void create_office(std::vector<std::vector<std::vector<char>>> &grid,
                   int floor_z, int ceiling_z, const Box2D &box,
                   int wall_thickness) {
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
          grid[x][y][z] = 0;
        }
      } else if (box.isNear(x, y, wall_thickness)) {
        for (int z = floor_z; z <= ceiling_z; ++z) {
          grid[x][y][z] = 1;
          r_grid[x][y][z] = 1.0;
          g_grid[x][y][z] = 1.0;
          b_grid[x][y][z] = 1.0;
          a_grid[x][y][z] = 1.0;
        }
      }
    }
  }
  // create doors
  const int door_width = 10;
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
          grid[x][y][z] = 0;
        }
      }
    }
  }
}

void create_obstacle(std::vector<std::vector<std::vector<char>>> &grid,
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
          grid[x][y][z] = 1;
          r_grid[x][y][z] = 0.5;
          g_grid[x][y][z] = 0.25;
          b_grid[x][y][z] = 0.0;
          a_grid[x][y][z] = 1.0;
        }
      }
    }
  }
}

void create_stair(std::vector<std::vector<std::vector<char>>> &grid,
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
          grid[x][y][z] = 0;
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "create_s3dis_map_test");
  ros::NodeHandle nh("");

  ros::Rate rate(1);

  const float min_x = -40;
  const float max_x = 40;
  const float min_y = -40;
  const float max_y = 40;
  const float min_z = -0.5;
  const float max_z = 20.0;
  const float resolution = 0.1;

  const int num_x_grid =
      static_cast<int>(std::ceil((max_x - min_x) / resolution));
  const int num_y_grid =
      static_cast<int>(std::ceil((max_y - min_y) / resolution));
  const int num_z_grid =
      static_cast<int>(std::ceil((max_z - min_z) / resolution));

  std::vector<std::vector<std::vector<char>>> grid(
      num_x_grid, std::vector<std::vector<char>>(
                      num_y_grid, std::vector<char>(num_z_grid, -1)));

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
  std::uniform_int_distribution<> random_office_length(50, 150);
  std::uniform_int_distribution<> random_office_width(50, 150);
  std::uniform_real_distribution<> random_heading(0, M_PI);
  std::uniform_int_distribution<> random_obstacle_length(3, 50);
  std::uniform_int_distribution<> random_obstacle_width(3, 10);
  std::uniform_int_distribution<> random_obstacle_height(3, 20);

  const int num_floors = 3;
  const int floor_height = 50;
  const int floor_thickness = 3;
  // create multiple floors
  for (int floor_id = 0; floor_id < num_floors; ++floor_id) {
    create_floor(grid, floor_id * floor_height,
                 floor_id * floor_height + floor_thickness);
    const int floor_surface = floor_id * floor_height + floor_thickness + 1;
    // create random offices
    const int num_office = 30;
    const int wall_thickness = 5;
    for (int office_id = 0; office_id < num_office; ++office_id) {
      create_office(grid, floor_surface,
                    floor_surface + floor_height - floor_thickness,
                    Box2D(random_x(gen), random_y(gen), random_heading(gen),
                          random_office_length(gen), random_office_width(gen)),
                    wall_thickness);
    }
    // create obstacles
    const int num_obstacle = 350;
    for (int obstacle_id = 0; obstacle_id < num_obstacle; ++obstacle_id) {
      create_obstacle(
          grid, floor_surface, floor_surface + random_obstacle_height(gen),
          Box2D(random_x(gen), random_y(gen), random_heading(gen),
                random_obstacle_length(gen), random_obstacle_width(gen)));
    }
    // create stairs
    const int num_stair = 2;
    for (int stair_id = 0; stair_id < num_stair; ++stair_id) {
      create_stair(
          grid, floor_id * floor_height, floor_surface,
          Box2D(random_x(gen), random_y(gen), random_heading(gen), 20, 20));
    }
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
        if (grid[i][j][k] == 1) {
          geometry_msgs::Point grid_pos;
          grid_pos.x = min_x + i * resolution + 0.5 * resolution;
          grid_pos.y = min_y + j * resolution + 0.5 * resolution;
          grid_pos.z = min_z + k * resolution + 0.5 * resolution;
          cube_list.points.emplace_back(grid_pos);
          cube_list.colors.emplace_back();
          cube_list.colors.back().a = a_grid[i][j][k];
          cube_list.colors.back().r = r_grid[i][j][k];
          cube_list.colors.back().g = g_grid[i][j][k];
          cube_list.colors.back().b = b_grid[i][j][k];
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
