#include "m3_explorer/grid_astar.h"
#include "m3_explorer/time_track.hpp"
#include <Eigen/Dense>
#include <geometry_msgs/Point.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <random>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

octomap::OcTree *ocmap = nullptr;
void octomap_cb(const octomap_msgs::Octomap::ConstPtr &msg) {
  delete ocmap;
  ocmap = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "grid_astar_test");
  ros::NodeHandle nh("");

  // get octomap
  ros::Subscriber octomap_sub =
      nh.subscribe<octomap_msgs::Octomap>("/merged_map", 1, octomap_cb);

  // visualize waypoint
  ros::Publisher wp_pub =
      nh.advertise<visualization_msgs::Marker>("/waypoint", 10);

  // visualize map
  ros::Publisher node_pub =
      nh.advertise<visualization_msgs::Marker>("/leafnode", 10);

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

  ros::Rate rate(0.1);

  visualization_msgs::Marker cube_list;
  cube_list.header.frame_id = "map";
  cube_list.header.stamp = ros::Time::now();
  cube_list.ns = "cube_list";
  cube_list.action = visualization_msgs::Marker::ADD;
  cube_list.pose.orientation.w = 1.0;
  cube_list.id = 0;
  cube_list.type = visualization_msgs::Marker::CUBE_LIST;
  cube_list.scale.x = 0.09;
  cube_list.scale.y = 0.09;
  cube_list.scale.z = 0.09;

  visualization_msgs::Marker waypoint;
  waypoint.header.frame_id = "map";
  waypoint.header.stamp = ros::Time::now();
  waypoint.ns = "line_strip";
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

  const float min_x = -10.5;
  const float max_x = 10.5;
  const float min_y = -10.5;
  const float max_y = 10.5;
  const float min_z = -0.1;
  const float max_z = 2.0;
  const float resolution = 0.1;
  GridAstar grid_astar(min_x, max_x, min_y, max_y, min_z, max_z, resolution);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> xy(min_x, max_x);
  std::uniform_real_distribution<float> z(1.0, 2.0);

  std::uniform_real_distribution<float> distrib(0.0, 1.0);

  int rolling_x = 0;
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
    cube_list.points.clear();
    cube_list.colors.clear();
    for (int i = 0; i < x_size; ++i) {
      for (int j = 0; j < y_size; ++j) {
        for (int k = 0; k < z_size; ++k) {
          if (grid_map[i][j][k] == GridAstar::GridState::kOcc) {
            geometry_msgs::Point grid_pos;
            grid_pos.x = min_x + i * resolution + 0.5 * resolution;
            grid_pos.y = min_y + j * resolution + 0.5 * resolution;
            grid_pos.z = min_z + k * resolution + 0.5 * resolution;
            cube_list.points.emplace_back(grid_pos);
            cube_list.colors.emplace_back();
            cube_list.colors.back().a = 1.0;
            cube_list.colors.back().g = 0.8;
          }
        }
      }
    }
    node_pub.publish(cube_list);
    track.OutputPassingTime("Visualize Map");

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
    Eigen::Vector3f start_pt = {7.5, 7.5, 1.5};
    Eigen::Vector3f end_pt = {-7.5, -7.5, 1.5};

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
    waypoint.points.clear();
    waypoint.color.r = 0.0;
    waypoint.color.g = 0.4;
    waypoint.color.b = 0.4;
    waypoint.color.a = 1.0;
    const std::vector<std::vector<float>> &ilqr_path = grid_astar.ilqr_path();
    const int ilqr_wp_num = ilqr_path.size();
    for (int i = 0; i < ilqr_wp_num; ++i) {
      geometry_msgs::Point wp_pos;
      wp_pos.x = min_x + ilqr_path[i][0] * resolution;
      wp_pos.y = min_y + ilqr_path[i][1] * resolution;
      wp_pos.z = min_z + ilqr_path[i][2] * resolution;
      waypoint.points.emplace_back(wp_pos);
    }
    ilqr_pub.publish(waypoint);

    // A*寻路，并统计时间
    track.SetStartTime();
    std::cout << "Astar Search Distance: "
              << grid_astar.AstarPathDistance(start_pt, end_pt) << std::endl;
    track.OutputPassingTime("--Astar Search Total--");
    // 可视化轨迹
    waypoint.points.clear();
    waypoint.color.r = 1.0;
    waypoint.color.g = 1.0;
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

// octomap::point3d bx_min(-10.0, -10.0, 0.2);
// octomap::point3d bx_max(10.0, 10.0, 0.4);
// int count = 0;
// marker_array.markers.clear();
// for(octomap::OcTree::leaf_bbx_iterator it = ocmap->begin_leafs_bbx(bx_min,
// bx_max), end=ocmap->end_leafs_bbx(); it != end; it++){
//   visualization_msgs::Marker box;
//   box.header.frame_id = "map";
//   box.header.stamp = ros::Time::now();

//   box.action = visualization_msgs::Marker::ADD;
//   box.scale.x = it.getSize()*0.8;
//   box.scale.y = it.getSize()*0.8;
//   box.scale.z = it.getSize()*0.8;
//   box.pose.orientation.w = 1.0;
//   box.id = count;
//   box.type = visualization_msgs::Marker::CUBE;
//   box.color.r = 0.0;
//   box.color.g = 0.5;
//   box.color.b = 0.0;
//   box.color.a = 1.0;
//   box.pose.position.x = it.getCoordinate().x();
//   box.pose.position.y = it.getCoordinate().y();
//   box.pose.position.z = it.getCoordinate().z();
//   marker_array.markers.emplace_back(box);
//   ++count;
// }
// node_pub.publish(marker_array);
