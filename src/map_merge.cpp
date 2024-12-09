#include <geometry_msgs/PointStamped.h>
#include <octomap/octomap.h>
#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/conversions.h>
#include <ros/ros.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <vector>

using namespace std;
tf2_ros::Buffer tf_buffer;
const int uav_num = 3;

void merge_octomap(octomap::OcTree *const from, octomap::OcTree *const to,
                   const int &id) {
  if (from == nullptr || to == nullptr)
    return;
  // Expand tree2 so we search all nodes
  from->expand();

  geometry_msgs::PointStamped cam_o_in_cam;
  cam_o_in_cam.header.frame_id = "uav" + to_string(id) + "_camera_depth_frame";
  cam_o_in_cam.point.x = 0.0;
  cam_o_in_cam.point.y = 0.0;
  cam_o_in_cam.point.z = 0.0;

  geometry_msgs::PointStamped cam_o_in_map;
  try {
    // 使用lookupTransform函数查询坐标变换
    tf_buffer.transform(cam_o_in_cam, cam_o_in_map, "map");
  } catch (tf2::TransformException &ex) {
    ROS_ERROR("Failed to transform point: %s", ex.what());
  }

  octomap::point3d check_bbx_min(cam_o_in_map.point.x - 5.0,
                                 cam_o_in_map.point.y - 5.0,
                                 max(-0.1, cam_o_in_map.point.z - 5.0));
  octomap::point3d check_bbx_max(cam_o_in_map.point.x + 5.0,
                                 cam_o_in_map.point.y + 5.0,
                                 min(3.0, cam_o_in_map.point.z + 5.0));

  // traverse nodes in tree2 to add them to tree1
  for (octomap::OcTree::leaf_bbx_iterator
           it = from->begin_leafs_bbx(check_bbx_min, check_bbx_max),
           end = from->end_leafs_bbx();
       it != end; ++it) {

    // find if the current node maps a point in map1
    octomap::point3d point = it.getCoordinate();
    octomap::OcTreeNode *nodeIn1 = to->search(point);
    if (nodeIn1 != NULL) {
      // Add the probability of tree2 node to the found node
      octomap::OcTreeKey nodeKey = to->coordToKey(point);
      to->updateNode(nodeKey, it->getLogOdds());
    } else {
      // Create a new node and set the probability from tree2
      octomap::OcTreeNode *newNode = to->updateNode(point, true);
      newNode->setLogOdds(it->getLogOdds());
    }
  }

  // clear occupancy of all UAVs
  for (int i = 0; i < uav_num; ++i) {
    geometry_msgs::PointStamped uav_o;
    uav_o.header.frame_id = "uav" + to_string(i) + "_base_link";
    uav_o.point.x = 0.0;
    uav_o.point.y = 0.0;
    uav_o.point.z = 0.0;

    geometry_msgs::PointStamped uav_in_map;
    try {
      // 使用lookupTransform函数查询坐标变换
      tf_buffer.transform(uav_o, uav_in_map, "map");
    } catch (tf2::TransformException &ex) {
      ROS_ERROR("Failed to transform point: %s", ex.what());
    }
    for (double x_offset = -0.4; x_offset <= 0.4; x_offset += 0.05) {
      for (double y_offset = -0.4; y_offset <= 0.4; y_offset += 0.05) {
        for (double z_offset = -0.2; z_offset <= 0.2; z_offset += 0.05) {
          octomap::OcTreeNode *nodeIn1 = to->search(
              uav_in_map.point.x + x_offset, uav_in_map.point.y + y_offset,
              uav_in_map.point.z + z_offset);
          if (nodeIn1 != NULL) {
            to->updateNode(uav_in_map.point.x + x_offset, uav_in_map.point.y + y_offset,
              uav_in_map.point.z + z_offset, false);
          }
        }
      }
    }
  }
}

class MapMerge {
private:
  int uav_id_;

public:
  static octomap::OcTree *ocmap;
  MapMerge(int uav_id) : uav_id_(uav_id){};
  void operator()(const octomap_msgs::Octomap::ConstPtr &msg) {
    // octomap init
    if (ocmap == nullptr) {
      ocmap = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
      return;
    }
    octomap::OcTree *new_map = dynamic_cast<octomap::OcTree *>(msgToMap(*msg));
    merge_octomap(new_map, ocmap, uav_id_);
    delete new_map;
  }
};

octomap::OcTree *MapMerge::ocmap = nullptr;

int main(int argc, char **argv) {
  ros::init(argc, argv, "map_merge");
  ros::NodeHandle nh("");

  tf2_ros::TransformListener tf_listener(tf_buffer);
  ros::Duration(1.0).sleep(); // 等待tf2变换树准备好
  ros::Rate rate(10);

  ros::Publisher octomap_pub =
      nh.advertise<octomap_msgs::Octomap>("/merged_map", 10);
  vector<ros::Subscriber> octomap_sub(uav_num);
  vector<MapMerge> octomap_cb;

  for (int i = 0; i < uav_num; ++i) {
    octomap_cb.emplace_back(i);
    octomap_sub[i] = nh.subscribe<octomap_msgs::Octomap>(
        "/uav" + to_string(i) + "/octomap_full", 1, octomap_cb[i]);
  }

  ROS_INFO("Map Merge has started !!");

  while (ros::ok()) {
    if (MapMerge::ocmap != nullptr) {
      octomap_msgs::Octomap merged_map;
      merged_map.header.frame_id = "map";
      merged_map.header.stamp = ros::Time::now();
      octomap_msgs::fullMapToMsg(*MapMerge::ocmap, merged_map);
      octomap_pub.publish(merged_map);
    }
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}