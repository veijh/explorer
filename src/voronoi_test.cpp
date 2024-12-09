#include "explorer/dynamicvoronoi.h"
#include "explorer/grid_astar.h"
#include "explorer/time_track.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <geometry_msgs/Point.h>
#include <iostream>
#include <random>
#include <ros/ros.h>
#include <string.h>
#include <unordered_set>
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
const int num_office = 5;
const int wall_thickness = 5;
const int door_width = 15;
const int num_obstacle = 50;
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

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "voronoi_test");
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

  visualization_msgs::MarkerArray cyliner_list;
  visualization_msgs::Marker cyliner;
  cyliner.header.frame_id = "map";
  cyliner.header.stamp = ros::Time::now();
  cyliner.ns = "cyliner";
  cyliner.action = visualization_msgs::Marker::ADD;
  cyliner.pose.orientation.w = 1.0;
  cyliner.id = 0;
  cyliner.type = visualization_msgs::Marker::CYLINDER;
  cyliner.color.a = 0.95;
  cyliner.color.r = 0.0;
  cyliner.color.g = 1.0;
  cyliner.color.b = 0.0;

  bool doPrune = false;
  bool doPruneAlternative = false;

  // create the voronoi object and initialize it with the map
  bool **grid_map_2d = new bool *[num_x_grid];
  for (int i = 0; i < num_x_grid; ++i) {
    grid_map_2d[i] = new bool[num_y_grid];
    for (int j = 0; j < num_y_grid; ++j) {
      if (i == 0 || i == num_x_grid - 1 || j == 0 || j == num_y_grid - 1) {
        grid_map_2d[i][j] = true;
        continue;
      }
      if (grid[i][j][25] == GridAstar::GridState::kOcc) {
        grid_map_2d[i][j] = true;
      } else {
        grid_map_2d[i][j] = false;
      }
    }
  }
  DynamicVoronoi voronoi;
  voronoi.initializeMap(num_x_grid, num_y_grid, grid_map_2d);
  TimeTrack track;
  voronoi.update(); // update distance map and Voronoi diagram
  track.OutputPassingTime("Update");
  int num_voronoi_cells = 0;
  int num_free_cells = 0;
  int num_key_voronoi_cells = 0;
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      if (voronoi.isVoronoi(i, j) == true) {
        ++num_voronoi_cells;
      }
      if (voronoi.isOccupied(i, j) == false) {
        ++num_free_cells;
      }
    }
  }
  std::cerr << "Generated " << num_voronoi_cells << " Voronoi cells. "
            << num_free_cells << " free cells. " << num_key_voronoi_cells
            << " key cells.\n";
  voronoi.visualize("initial.ppm");

  track.SetStartTime();
  voronoi.prune(); // prune the Voronoi
  track.OutputPassingTime("Prune");
  num_voronoi_cells = 0;
  num_free_cells = 0;
  num_key_voronoi_cells = 0;
  std::unordered_set<IntPoint, IntPointHash> key_cells_map;
  for (int i = 0; i < num_x_grid; ++i) {
    for (int j = 0; j < num_y_grid; ++j) {
      if (voronoi.isVoronoi(i, j) == true) {
        ++num_voronoi_cells;
        std::vector<IntPoint> neighbors = voronoi.GetVoronoiNeighbors(i, j);
        if (neighbors.size() > 2) {
          ++num_key_voronoi_cells;
          key_cells_map.insert(IntPoint(i, j));
          cyliner.pose.position.x = min_x + i * resolution + 0.5 * resolution;
          cyliner.pose.position.y = min_y + j * resolution + 0.5 * resolution;
          cyliner.pose.position.z = min_z + 25 * resolution + 0.5 * resolution;
          const float distance = voronoi.getDistance(i, j);
          cyliner.scale.x = 2 * distance * resolution;
          cyliner.scale.y = 2 * distance * resolution;
          cyliner.scale.z = 1.0 * resolution;
          cyliner.id = cyliner.id + 1;
          cyliner_list.markers.emplace_back(cyliner);
        }
      }
      if (voronoi.isOccupied(i, j) == false) {
        ++num_free_cells;
      }
    }
  }
  std::cerr << "Pruned " << num_voronoi_cells << " Voronoi cells. "
            << num_free_cells << " free cells. " << num_key_voronoi_cells
            << " key cells.\n";
  voronoi.visualize("pruned.ppm");

  track.SetStartTime();
  voronoi.ConstructSparseGraphBK();
  track.OutputPassingTime("ConstructSparseGraph");
  const VGraph &graph = voronoi.GetSparseGraph();

  visualization_msgs::Marker connectivity;
  connectivity.header.frame_id = "map";
  connectivity.header.stamp = ros::Time::now();
  connectivity.ns = "connectivity";
  connectivity.action = visualization_msgs::Marker::ADD;
  connectivity.pose.orientation.w = 1.0;
  connectivity.id = 0;
  connectivity.type = visualization_msgs::Marker::LINE_LIST;
  connectivity.scale.x = 0.1;
  connectivity.color.a = 1.0;
  connectivity.color.r = 1.0;
  connectivity.color.g = 0.0;
  connectivity.color.b = 0.0;

  const int num_nodes = graph.nodes_.size();
  for (int i = 0; i < num_nodes; ++i) {
    for (const auto &edge : graph.nodes_[i].edges_) {
      const IntPoint src_point = graph.nodes_[i].point_;
      const IntPoint dst_point = graph.nodes_[edge.first].point_;
      connectivity.points.emplace_back();
      connectivity.points.back().x =
          min_x + src_point.x * resolution + 0.5 * resolution;
      connectivity.points.back().y =
          min_y + src_point.y * resolution + 0.5 * resolution;
      connectivity.points.back().z = min_z + 25 * resolution + 0.5 * resolution;
      connectivity.points.emplace_back();
      connectivity.points.back().x =
          min_x + dst_point.x * resolution + 0.5 * resolution;
      connectivity.points.back().y =
          min_y + dst_point.y * resolution + 0.5 * resolution;
      connectivity.points.back().z = min_z + 25 * resolution + 0.5 * resolution;
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

  ros::Publisher app_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/app", 10);
  ros::Publisher connect_pub =
      nh.advertise<visualization_msgs::Marker>("/connect", 10);
  ros::Publisher path_pub =
      nh.advertise<visualization_msgs::Marker>("/path", 10);
  ros::Publisher ilqr_path_pub =
      nh.advertise<visualization_msgs::Marker>("/ilqr_path", 10);
  ros::Publisher corridor_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/corridor", 10);
  ros::Publisher union_pub =
      nh.advertise<visualization_msgs::MarkerArray>("/union", 10);

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
  corridor.type = visualization_msgs::Marker::CYLINDER;
  corridor.color.a = 0.6;
  corridor.color.r = 0.0;
  corridor.color.g = 1.0;
  corridor.color.b = 0.0;

  visualization_msgs::MarkerArray unions;
  visualization_msgs::Marker element;
  element.header.frame_id = "map";
  element.header.stamp = ros::Time::now();
  element.ns = "element";
  element.action = visualization_msgs::Marker::ADD;
  element.pose.orientation.w = 1.0;
  element.id = 0;
  element.type = visualization_msgs::Marker::CYLINDER;
  element.color.a = 0.6;
  element.color.r = 0.0;
  element.color.g = 1.0;
  element.color.b = 0.0;
  for (const auto &node : graph.nodes_) {
    element.id = element.id + 1;
    element.pose.position.x =
        min_x + node.point_.x * resolution + 0.5 * resolution;
    element.pose.position.y =
        min_y + node.point_.y * resolution + 0.5 * resolution;
    element.pose.position.z = min_z + 20 * resolution + 0.5 * resolution;
    const float distance = voronoi.getDistance(node.point_.x, node.point_.y);
    element.scale.x = 2 * distance * resolution;
    element.scale.y = 2 * distance * resolution;
    element.scale.z = 1.0 * resolution;
    unions.markers.emplace_back(element);
  }

  while (ros::ok()) {
    rate.sleep();
    ros::spinOnce();
    map0_pub.publish(cube_list0);
    map1_pub.publish(cube_list1);
    map2_pub.publish(cube_list2);
    app_pub.publish(cyliner_list);
    connect_pub.publish(connectivity);
    union_pub.publish(unions);

    track.SetStartTime();
    std::vector<IntPoint> path =
        voronoi.GetAstarPath(IntPoint(30, 30), IntPoint(600, 600));
    track.OutputPassingTime("GetAstarPath");

    corridors.markers.clear();
    corridor.id = 0;
    const int num_path_points = path.size();
    path_marker.points.clear();
    for (int i = 0; i < num_path_points; ++i) {
      path_marker.points.emplace_back();
      path_marker.points.back().x =
          min_x + path[i].x * resolution + 0.5 * resolution;
      path_marker.points.back().y =
          min_y + path[i].y * resolution + 0.5 * resolution;
      path_marker.points.back().z = min_z + 25 * resolution + 0.5 * resolution;
      corridor.id = corridor.id + 1;
      corridor.pose.position.x =
          min_x + path[i].x * resolution + 0.5 * resolution;
      corridor.pose.position.y =
          min_y + path[i].y * resolution + 0.5 * resolution;
      corridor.pose.position.z = min_z + 20 * resolution + 0.5 * resolution;
      const float distance = voronoi.getDistance(path[i].x, path[i].y);
      corridor.scale.x = 2 * distance * resolution;
      corridor.scale.y = 2 * distance * resolution;
      corridor.scale.z = 1.0 * resolution;
      corridors.markers.emplace_back(corridor);
    }
    if (!path.empty()) {
      path_pub.publish(path_marker);
      corridor_pub.publish(corridors);
      track.SetStartTime();
      std::vector<IntPoint> ilqr_path = voronoi.GetiLQRPath(path);
      track.OutputPassingTime("GetiLQRPath");

      ilqr_path_marker.points.clear();
      const int num_ilqr_path_points = ilqr_path.size();
      std::cout << "iLQR path: " << num_ilqr_path_points << std::endl;
      for (int i = 0; i < num_ilqr_path_points; ++i) {
        ilqr_path_marker.points.emplace_back();
        ilqr_path_marker.points.back().x =
            min_x + ilqr_path[i].x * resolution + 0.5 * resolution;
        ilqr_path_marker.points.back().y =
            min_y + ilqr_path[i].y * resolution + 0.5 * resolution;
        ilqr_path_marker.points.back().z =
            min_z + 25 * resolution + 0.5 * resolution;
      }
      ilqr_path_pub.publish(ilqr_path_marker);
    }
  }
}
