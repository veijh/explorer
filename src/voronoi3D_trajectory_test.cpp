#include "explorer/dynamicvoronoi3D.h"
#include "explorer/grid_astar.h"
#include "explorer/time_track.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <iomanip>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <random>
#include <ros/ros.h>
#include <string.h>
#include <unordered_set>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define LOG_OUTPUT (false)
#define ASTAR (false)

namespace {
const float min_x = -25.0;
const float max_x = 25.0;
const float min_y = -25.0;
const float max_y = 25.0;
const float resolution = 0.1;
const int num_floors = 3;
const int floor_height = 30;
const int floor_thickness = 3;
const int num_office = 5;
const int wall_thickness = 5;
const int door_width = 20;
const int num_obstacle = 50;
const int num_stair = 2;
const int stair_length = 50;
const int stair_width = 50;
} // namespace

std::string getTimestamp() {
  // 获取当前时间
  std::time_t now = std::time(nullptr);
  std::tm *timeInfo = std::localtime(&now);

  // 将时间格式化为字符串：年_月_日_时_分_秒
  std::ostringstream timestamp;
  timestamp << std::put_time(timeInfo, "%Y-%m-%d_%H-%M-%S");

  return timestamp.str();
}

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

void publish_odometry(const ros::Publisher &odom_pub,
                      const std::vector<nav_msgs::Odometry> &traj,
                      const float sample_time) {
  const int num_samples = traj.size();
  ros::Rate rate(1.0f / sample_time);
  for (int i = 0; i < num_samples; ++i) {
    nav_msgs::Odometry odom;
    odom.header.stamp = ros::Time::now();
    // 设置机器人的位置（坐标）和朝向（姿态）
    odom.header.frame_id = "map";
    odom.pose = traj[i].pose;
    odom.twist = traj[i].twist;
    // 发布消息
    odom_pub.publish(odom);
    rate.sleep();
  }
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "voronoi3D_test");
  ros::NodeHandle nh("");

  ros::Rate rate(1);

  // 输出文件
  std::string fileName = "output_" + getTimestamp() + ".txt";
  std::ofstream outFile(fileName);
  if (LOG_OUTPUT) {
    if (!outFile.is_open()) {
      return -1;
    }
  }

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
  std::uniform_int_distribution<> random_office_length(30, 60);
  std::uniform_int_distribution<> random_office_width(30, 60);
  std::uniform_real_distribution<> random_heading(0, M_PI);
  std::uniform_int_distribution<> random_obstacle_length(3, 50);
  std::uniform_int_distribution<> random_obstacle_width(3, 10);
  std::uniform_int_distribution<> random_obstacle_height(3, floor_height * 0.6);

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
      const int obstacle_height = random_obstacle_height(gen);
      create_obstacle(grid, floor_surface, floor_surface + obstacle_height,
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

  GridAstar grid_astar(min_x, max_x, min_y, max_y, min_z, max_z, resolution,
                       grid);

  // Visualize the grids on different layers.
  std::vector<visualization_msgs::Marker> cube_lists(num_floors);
  for (int floor_iter = 0; floor_iter < num_floors; ++floor_iter) {
    cube_lists[floor_iter].header.frame_id = "map";
    cube_lists[floor_iter].header.stamp = ros::Time::now();
    cube_lists[floor_iter].ns = "cube_list" + std::to_string(floor_iter);
    cube_lists[floor_iter].action = visualization_msgs::Marker::ADD;
    cube_lists[floor_iter].pose.orientation.w = 1.0;
    cube_lists[floor_iter].id = 0;
    cube_lists[floor_iter].type = visualization_msgs::Marker::CUBE_LIST;
    cube_lists[floor_iter].scale.x = resolution * 0.95;
    cube_lists[floor_iter].scale.y = resolution * 0.95;
    cube_lists[floor_iter].scale.z = resolution * 0.95;

    cube_lists[floor_iter].points.clear();
    cube_lists[floor_iter].colors.clear();
    for (int i = 0; i < num_x_grid; ++i) {
      for (int j = 0; j < num_y_grid; ++j) {
        for (int k = floor_iter * floor_height;
             k < (floor_iter + 1) * floor_height; ++k) {
          if (grid[i][j][k] == GridAstar::GridState::kOcc) {
            geometry_msgs::Point grid_pos;
            grid_pos.x = min_x + i * resolution + 0.5 * resolution;
            grid_pos.y = min_y + j * resolution + 0.5 * resolution;
            grid_pos.z = min_z + k * resolution + 0.5 * resolution;
            cube_lists[floor_iter].points.emplace_back(grid_pos);
            cube_lists[floor_iter].colors.emplace_back();
            cube_lists[floor_iter].colors.back().a = a_grid[i][j][k];
            cube_lists[floor_iter].colors.back().r = r_grid[i][j][k];
            cube_lists[floor_iter].colors.back().g = g_grid[i][j][k];
            cube_lists[floor_iter].colors.back().b = b_grid[i][j][k];
          }
        }
      }
    }
  }

  std::vector<nav_msgs::Odometry> odom_msgs;

  visualization_msgs::Marker gvd_points;
  gvd_points.header.frame_id = "map";
  gvd_points.header.stamp = ros::Time::now();
  gvd_points.ns = "gvd_points";
  gvd_points.action = visualization_msgs::Marker::ADD;
  gvd_points.pose.orientation.w = 1.0;
  gvd_points.id = 0;
  gvd_points.type = visualization_msgs::Marker::CUBE_LIST;
  gvd_points.scale.x = resolution * 0.5;
  gvd_points.scale.y = resolution * 0.5;
  gvd_points.scale.z = resolution * 0.5;
  gvd_points.color.a = 1.0;
  gvd_points.color.r = 0.2;
  gvd_points.color.g = 1.0;
  gvd_points.color.b = 0.2;

  // create the voronoi object and initialize it with the map
  bool ***grid_map_3d = new bool **[num_x_grid];
  for (int i = 0; i < num_x_grid; ++i) {
    grid_map_3d[i] = new bool *[num_y_grid];
    for (int j = 0; j < num_y_grid; ++j) {
      grid_map_3d[i][j] = new bool[num_z_grid];
      for (int k = 0; k < num_z_grid; ++k) {
        if (i == 0 || i == num_x_grid - 1 || j == 0 || j == num_y_grid - 1 ||
            k == 0 || k == num_z_grid - 1) {
          grid_map_3d[i][j][k] = true;
          continue;
        }
        if (grid[i][j][k] == GridAstar::GridState::kOcc) {
          grid_map_3d[i][j][k] = true;
        } else {
          grid_map_3d[i][j][k] = false;
        }
      }
    }
  }
  DynamicVoronoi3D voronoi;
  voronoi.initializeMap(num_x_grid, num_y_grid, num_z_grid, grid_map_3d);
  TimeTrack track;
  voronoi.update(); // update distance map and Voronoi diagram
  if (LOG_OUTPUT) {
    outFile << "Preprocess time, " << track.OutputPassingTime("Update")
            << std::endl;
  }
  int num_voronoi_cells = 0;
  int num_free_cells = 0;
  int num_key_voronoi_cells = 0;
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      for (int k = 0; k < num_z_grid; ++k) {
        if (voronoi.isVoronoi(i, j, k) == true) {
          ++num_voronoi_cells;
          // if (k < floor_height) {
          gvd_points.points.emplace_back();
          gvd_points.points.back().x =
              min_x + i * resolution + 0.5 * resolution;
          gvd_points.points.back().y =
              min_y + j * resolution + 0.5 * resolution;
          gvd_points.points.back().z =
              min_z + k * resolution + 0.5 * resolution;
          //}
        }
        if (voronoi.isOccupied(i, j, k) == false) {
          ++num_free_cells;
        }
      }
    }
  }
  std::cerr << "Generated " << num_voronoi_cells << " Voronoi cells. "
            << num_free_cells << " free cells.\n";

  track.SetStartTime();
  voronoi.ConstructSparseGraphBK();
  if (LOG_OUTPUT) {
    outFile << "SSSC Graph, " << track.OutputPassingTime("ConstructSparseGraph")
            << std::endl;
  }
  const auto &graph = voronoi.GetSparseGraph();

  if (LOG_OUTPUT) {
    outFile << "Free cells, " << num_free_cells << std::endl;
    outFile << "Voronoi cells, " << num_voronoi_cells << std::endl;
    outFile << "Num Sparse Graph Nodes, " << graph.nodes_.size() << std::endl;
  }

  visualization_msgs::Marker connectivity;
  connectivity.header.frame_id = "map";
  connectivity.header.stamp = ros::Time::now();
  connectivity.ns = "connectivity";
  connectivity.action = visualization_msgs::Marker::ADD;
  connectivity.pose.orientation.w = 1.0;
  connectivity.id = 0;
  connectivity.type = visualization_msgs::Marker::LINE_LIST;
  connectivity.scale.x = 0.05;
  connectivity.color.a = 1.0;
  connectivity.color.r = 1.0;
  connectivity.color.g = 0.0;
  connectivity.color.b = 0.0;

  const int num_nodes = graph.nodes_.size();
  for (int i = 0; i < num_nodes; ++i) {
    for (const auto &edge : graph.nodes_[i].edges_) {
      const IntPoint3D src_point = graph.nodes_[i].point_;
      const IntPoint3D dst_point = graph.nodes_[edge.first].point_;
      connectivity.points.emplace_back();
      connectivity.points.back().x =
          min_x + src_point.x * resolution + 0.5 * resolution;
      connectivity.points.back().y =
          min_y + src_point.y * resolution + 0.5 * resolution;
      connectivity.points.back().z =
          min_z + src_point.z * resolution + 0.5 * resolution;
      connectivity.points.emplace_back();
      connectivity.points.back().x =
          min_x + dst_point.x * resolution + 0.5 * resolution;
      connectivity.points.back().y =
          min_y + dst_point.y * resolution + 0.5 * resolution;
      connectivity.points.back().z =
          min_z + dst_point.z * resolution + 0.5 * resolution;
    }
  }

  visualization_msgs::Marker path_marker;
  path_marker.header.frame_id = "map";
  path_marker.header.stamp = ros::Time::now();
  path_marker.ns = "path";
  path_marker.action = visualization_msgs::Marker::ADD;
  path_marker.pose.orientation.w = 1.0;
  path_marker.id = 0;
  path_marker.type = visualization_msgs::Marker::LINE_STRIP;
  path_marker.scale.x = 0.2;
  path_marker.color.a = 1.0;
  path_marker.color.r = 0.0;
  path_marker.color.g = 1.0;
  path_marker.color.b = 0.0;

  visualization_msgs::Marker ilqr_path_marker;
  ilqr_path_marker.header.frame_id = "map";
  ilqr_path_marker.header.stamp = ros::Time::now();
  ilqr_path_marker.ns = "ilqr_path";
  ilqr_path_marker.action = visualization_msgs::Marker::ADD;
  ilqr_path_marker.pose.orientation.w = 1.0;
  ilqr_path_marker.id = 0;
  ilqr_path_marker.type = visualization_msgs::Marker::LINE_STRIP;
  ilqr_path_marker.scale.x = 0.2;
  ilqr_path_marker.color.a = 1.0;
  ilqr_path_marker.color.r = 0.0;
  ilqr_path_marker.color.g = 0.0;
  ilqr_path_marker.color.b = 1.0;

  visualization_msgs::MarkerArray corridors;
  visualization_msgs::Marker corridor;
  corridor.header.frame_id = "map";
  corridor.header.stamp = ros::Time::now();
  corridor.ns = "corridor";
  corridor.action = visualization_msgs::Marker::ADD;
  corridor.pose.orientation.w = 1.0;
  corridor.id = 0;
  corridor.type = visualization_msgs::Marker::SPHERE;
  corridor.color.a = 0.6;
  corridor.color.r = 0.0;
  corridor.color.g = 1.0;
  corridor.color.b = 0.0;

  visualization_msgs::Marker waypoint;
  waypoint.header.frame_id = "map";
  waypoint.header.stamp = ros::Time::now();
  waypoint.ns = "line_strip";
  waypoint.action = visualization_msgs::Marker::ADD;
  waypoint.scale.x = 0.1;
  waypoint.pose.orientation.w = 1.0;
  waypoint.id = 0;
  waypoint.type = visualization_msgs::Marker::LINE_STRIP;

  visualization_msgs::Marker trajectory;
  trajectory.header.frame_id = "map";
  trajectory.header.stamp = ros::Time::now();
  trajectory.ns = "trajectory";
  trajectory.action = visualization_msgs::Marker::ADD;
  trajectory.scale.x = 0.1;
  trajectory.pose.orientation.w = 1.0;
  trajectory.id = 0;
  trajectory.type = visualization_msgs::Marker::LINE_STRIP;

  std::vector<ros::Publisher> maps_pub(num_floors);
  for (int floor_iter = 0; floor_iter < num_floors; ++floor_iter) {
    maps_pub[floor_iter] = nh.advertise<visualization_msgs::Marker>(
        "/map" + std::to_string(floor_iter), 10);
  }
  ros::Publisher gvd_pub = nh.advertise<visualization_msgs::Marker>("/gvd", 10);
  ros::Publisher connect_pub =
      nh.advertise<visualization_msgs::Marker>("/connect", 10);
  ros::Publisher path_pub =
      nh.advertise<visualization_msgs::Marker>("/path", 10);
  ros::Publisher ilqr_path_pub =
      nh.advertise<visualization_msgs::Marker>("/ilqr_path", 10);
  ros::Publisher corridor_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/corridor", 10);
  ros::Publisher astar_pub =
      nh.advertise<visualization_msgs::Marker>("/astar_wp", 10);
  ros::Publisher traj_pub =
      nh.advertise<visualization_msgs::Marker>("/ilqr_traj", 10);

  // 设定起点
  const Eigen::Vector3f start_pt = {-20.0, -20.0, 1.0};
  const int start_index_x = (start_pt.x() - min_x) / resolution;
  const int start_index_y = (start_pt.y() - min_y) / resolution;
  const int start_index_z = (start_pt.z() - min_z) / resolution;
  if (LOG_OUTPUT) {
    outFile << "Start, " << start_pt.x() << ", " << start_pt.y() << ", "
            << start_pt.z() << std::endl;
  }
  if (grid[start_index_x][start_index_y][start_index_z] ==
      GridAstar::GridState::kOcc) {
    return 0;
  }

  // 设定终点
  std::vector<Eigen::Vector3f> end_pts;
  for (float end_x = -20.0; end_x <= 20.0 + 1e-2; end_x += 4.0) {
    for (float end_y = -20.0; end_y <= 20.0 + 1e-2; end_y += 4.0) {
      for (float end_z = 4.5; end_z <= 9.0; end_z += 3.0) {
        const int end_index_x = (end_x - min_x) / resolution;
        const int end_index_y = (end_y - min_y) / resolution;
        const int end_index_z = (end_z - min_z) / resolution;
        if (grid[end_index_x][end_index_y][end_index_z] ==
            GridAstar::GridState::kOcc) {
          continue;
        }
        end_pts.emplace_back(Eigen::Vector3f(end_x, end_y, end_z));
      }
    }
  }
  const int end_num = end_pts.size();
  int count = 0;

  for (int floor_iter = 0; floor_iter < num_floors; ++floor_iter) {
    maps_pub[floor_iter].publish(cube_lists[floor_iter]);
  }
  gvd_pub.publish(gvd_points);
  connect_pub.publish(connectivity);

  if (count >= end_num) {
    return 0;
  }
  // Eigen::Vector3f end_pt = end_pts[count];
  Eigen::Vector3f end_pt = {20.0, 20.0, 1.5};
  ++count;

  const int end_index_x = (end_pt.x() - min_x) / resolution;
  const int end_index_y = (end_pt.y() - min_y) / resolution;
  const int end_index_z = (end_pt.z() - min_z) / resolution;
  if (LOG_OUTPUT) {
    outFile << "End, " << end_pt.x() << ", " << end_pt.y() << ", " << end_pt.z()
            << std::endl;
  }
  const IntPoint3D start_point(start_index_x, start_index_y, start_index_z);
  const IntPoint3D goal_point(end_index_x, end_index_y, end_index_z);
  track.SetStartTime();
  AstarOutput astar_output = voronoi.GetAstarPath(start_point, goal_point);
  if (LOG_OUTPUT) {
    outFile << "Rough Path Time, " << track.OutputPassingTime("GetAstarPath")
            << std::endl;
  }
  // Skip failed paths.
  if (astar_output.success == false) {
    return 0;
  } else {
    if (LOG_OUTPUT) {
      outFile << "Rough Num Exp, " << astar_output.num_expansions << std::endl;
      outFile << "Rough Path Length, " << astar_output.path_length << std::endl;
    }
  }

  const std::vector<IntPoint3D> &path = astar_output.path;
  const int num_path_points = path.size();
  path_marker.points.clear();
  corridor.id = 0;
  for (int i = 0; i < num_path_points; ++i) {
    path_marker.points.emplace_back();
    path_marker.points.back().x =
        min_x + path[i].x * resolution + 0.5 * resolution;
    path_marker.points.back().y =
        min_y + path[i].y * resolution + 0.5 * resolution;
    path_marker.points.back().z =
        min_z + path[i].z * resolution + 0.5 * resolution;
    corridor.id = corridor.id + 1;
    corridor.pose.position.x =
        min_x + path[i].x * resolution + 0.5 * resolution;
    corridor.pose.position.y =
        min_y + path[i].y * resolution + 0.5 * resolution;
    corridor.pose.position.z =
        min_z + path[i].z * resolution + 0.5 * resolution;
    const float distance = voronoi.getDistance(path[i].x, path[i].y, path[i].z);
    corridor.scale.x = 2 * distance * resolution;
    corridor.scale.y = 2 * distance * resolution;
    corridor.scale.z = 2 * distance * resolution;
    corridors.markers.emplace_back(corridor);
  }
  // Postprocess the path.
  if (!path.empty()) {
    path_pub.publish(path_marker);
    corridor_pub.publish(corridors);

    // Use iLQR to refine the path.
    track.SetStartTime();
    iLQROutput ilqr_output = voronoi.GetiLQRPath(path);
    track.OutputPassingTime("GetiLQRPath");
    if (LOG_OUTPUT) {
      outFile << "iLQR Path Time, " << track.OutputPassingTime("GetiLQRPath")
              << std::endl;
      outFile << "iLQR Num Iter, " << ilqr_output.num_iter << std::endl;
      outFile << "iLQR Path Length, " << ilqr_output.path_length << std::endl;
    }
    const std::vector<IntPoint3D> &ilqr_path = ilqr_output.path;
    ilqr_path_marker.points.clear();
    const int num_ilqr_path_points = ilqr_path.size();
    for (int i = 0; i < num_ilqr_path_points; ++i) {
      ilqr_path_marker.points.emplace_back();
      ilqr_path_marker.points.back().x =
          min_x + ilqr_path[i].x * resolution + 0.5 * resolution;
      ilqr_path_marker.points.back().y =
          min_y + ilqr_path[i].y * resolution + 0.5 * resolution;
      ilqr_path_marker.points.back().z =
          min_z + ilqr_path[i].z * resolution + 0.5 * resolution;
    }
    ilqr_path_pub.publish(ilqr_path_marker);

    // Trajectory generation.
    track.SetStartTime();
    iLQRTrajectory ilqr_traj = voronoi.GetiLQRTrajectory(path, ilqr_path);
    track.OutputPassingTime("GetiLQRTrajectory");

    // Visualize the trajectory.
    trajectory.points.clear();
    trajectory.colors.clear();
    const auto traj = ilqr_traj.traj;
    const int num_traj_points = traj.size();
    for (int points_iter = 0; points_iter < num_traj_points - 1;
         ++points_iter) {
      const float dt = traj[points_iter](18);
      const float dt_2 = dt * dt;
      const float dt_3 = dt_2 * dt;
      const float dt_4 = dt_3 * dt;
      const float dt_5 = dt_4 * dt;
      // clang-format off
        Eigen::Matrix<float, 6, 6> A;
        A <<
        1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 0.0f,
        1.0f, dt, dt_2, dt_3, dt_4, dt_5,
        0.0f, 1.0f, 2.0f * dt, 3.0f * dt_2, 4.0f * dt_3, 5.0f * dt_4,
        0.0f, 0.0f, 2.0f, 6.0f * dt, 12.0f * dt_2, 20.0f * dt_3;
        Eigen::Matrix<float, 6, 1> state_x;
        Eigen::Matrix<float, 6, 1> state_y;
        Eigen::Matrix<float, 6, 1> state_z;
        const Eigen::Matrix<float, 19, 1> state = traj[points_iter];
        const Eigen::Matrix<float, 19, 1> next_state = traj[points_iter+1];
        state_x << state(0), state(1), state(2), next_state(0), next_state(1), next_state(2);
        state_y << state(3), state(4), state(5), next_state(3), next_state(4), next_state(5);
        state_z << state(6), state(7), state(8), next_state(6), next_state(7), next_state(8);
      // clang-format on
      const Eigen::Matrix<float, 6, 1> ax_coeff = A.inverse() * state_x;
      const Eigen::Matrix<float, 6, 1> ay_coeff = A.inverse() * state_y;
      const Eigen::Matrix<float, 6, 1> az_coeff = A.inverse() * state_z;

      for (float t = 0.0f; t <= dt; t += 0.05f) {
        const float t_2 = t * t;
        const float t_3 = t_2 * t;
        const float t_4 = t_3 * t;
        const float t_5 = t_4 * t;
        Eigen::Matrix<float, 1, 6> p_coeff;
        Eigen::Matrix<float, 1, 6> v_coeff;
        p_coeff << 1, t, t_2, t_3, t_4, t_5;
        v_coeff << 0, 1, 2 * t, 3 * t_2, 4 * t_3, 5 * t_4;
        // Calculate the position.
        const float pxt = p_coeff * ax_coeff;
        const float pyt = p_coeff * ay_coeff;
        const float pzt = p_coeff * az_coeff;

        // Calculate the velocity.
        const float vxt = v_coeff * ax_coeff;
        const float vyt = v_coeff * ay_coeff;
        const float vzt = v_coeff * az_coeff;
        const float vel = std::hypot(vxt, vyt, vzt);

        trajectory.points.emplace_back();
        trajectory.points.back().x =
            min_x + pxt * resolution + 0.5 * resolution;
        trajectory.points.back().y =
            min_y + pyt * resolution + 0.5 * resolution;
        trajectory.points.back().z =
            min_z + pzt * resolution + 0.5 * resolution;
        trajectory.colors.emplace_back();
        trajectory.colors.back().r = vel / (15.0f);
        trajectory.colors.back().g = 0.0;
        trajectory.colors.back().b = 1 - vel / (15.0f);
        trajectory.colors.back().a = 1.0;
        odom_msgs.emplace_back();
        odom_msgs.back().header.frame_id = "map";
        odom_msgs.back().pose.pose.position.x = trajectory.points.back().x;
        odom_msgs.back().pose.pose.position.y = trajectory.points.back().y;
        odom_msgs.back().pose.pose.position.z = trajectory.points.back().z;
        odom_msgs.back().pose.pose.orientation.w = 1.0;
        odom_msgs.back().twist.twist.linear.x = vxt * resolution;
        odom_msgs.back().twist.twist.linear.y = vyt * resolution;
        odom_msgs.back().twist.twist.linear.z = vzt * resolution;
      }
    }
    traj_pub.publish(trajectory);
  }

  ros::Publisher odom_pub = nh.advertise<nav_msgs::Odometry>("/odom", 10);

  // A*寻路，并统计时间
  if (ASTAR) {
    track.SetStartTime();
    GridAstarOutput grid_astar_output =
        grid_astar.AstarPathDistance(start_pt, end_pt);
    if (LOG_OUTPUT) {
      outFile << "Grid Astar Time,"
              << track.OutputPassingTime("--Astar Search Total--") << std::endl;
      outFile << "Grid Astar Num Exp," << grid_astar_output.num_expansions
              << std::endl;
      outFile << "Grid Astar Path Length," << grid_astar_output.path_length
              << std::endl;
    }
    // 可视化轨迹
    waypoint.points.clear();
    waypoint.color.r = 1.0;
    waypoint.color.g = 1.0;
    waypoint.color.b = 1.0;
    waypoint.color.a = 1.0;
    const std::vector<std::shared_ptr<GridAstarNode>> &astar_path =
        grid_astar.path();
    const int wp_num = astar_path.size();
    for (int i = 0; i < wp_num; ++i) {
      geometry_msgs::Point wp_pos;
      wp_pos.x = min_x + astar_path[i]->index_x_ * resolution;
      wp_pos.y = min_y + astar_path[i]->index_y_ * resolution;
      wp_pos.z = min_z + astar_path[i]->index_z_ * resolution;
      waypoint.points.emplace_back(wp_pos);
    }
    astar_pub.publish(waypoint);
  }

  while (ros::ok()) {
    rate.sleep();
    ros::spinOnce();
    for (int floor_iter = 0; floor_iter < num_floors; ++floor_iter) {
      maps_pub[floor_iter].publish(cube_lists[floor_iter]);
    }
    traj_pub.publish(trajectory);
    publish_odometry(odom_pub, odom_msgs, 0.05);
  }
}
