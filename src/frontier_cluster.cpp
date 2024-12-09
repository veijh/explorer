#include "m3_explorer/frontier_cluster.h"

#include "m3_explorer/frontier_detector.h"
#include "m3_explorer/kd_tree.h"
#include <geometry_msgs/Pose.h>
#include <random>
#include <tf2/LinearMath/Quaternion.h>
#include <unordered_map>
#include <visualization_msgs/MarkerArray.h>

vector<Cluster> k_mean_cluster(set<QuadMesh> &frontiers) {
  // dynamic cluster num
  const int num_frontiers = frontiers.size();
  int cluster_num = num_frontiers / 500;

  // convert set to matrix
  Eigen::Matrix3Xf frontier_center_mat(3, num_frontiers);
  Eigen::Matrix3Xf frontier_normal_mat(3, num_frontiers);
  Eigen::RowVectorXf frontier_size_mat(num_frontiers);
  set<QuadMesh>::iterator it = frontiers.begin();
  for (int i = 0; i < num_frontiers; i++) {
    frontier_center_mat.col(i) << it->center.x(), it->center.y(),
        it->center.z();
    frontier_normal_mat.col(i) << it->normal.x(), it->normal.y(),
        it->normal.z();
    frontier_size_mat.col(i) << it->size;
    it++;
  }

  int min = 0, max = num_frontiers - 1;
  random_device seed;      // 硬件生成随机数种子
  ranlux48 engine(seed()); // 利用种子生成随机数引擎
  uniform_int_distribution<int> distrib(min,
                                        max); // 设置随机数范围，并为均匀分布

  // initial clusters' center
  vector<int> initial_idx;
  while (initial_idx.size() < cluster_num) {
    int random = distrib(engine);
    if (find(initial_idx.begin(), initial_idx.end(), random) ==
        initial_idx.end()) {
      initial_idx.push_back(random);
    }
  }
  Eigen::Matrix3Xf cluster_center(3, cluster_num);

  for (int i = 0; i < initial_idx.size(); i++) {
    cluster_center.col(i) = frontier_center_mat.col(initial_idx[i]);
  }

  Eigen::MatrixXf distance_to_center(cluster_num, num_frontiers);
  for (int iter_count = 0; iter_count < MAX_ITERATION; iter_count++) {
    // calculate the distance from each point to all centers
    for (int i = 0; i < cluster_num; i++) {
      distance_to_center.row(i) =
          (cluster_center.col(i).replicate(1, frontier_center_mat.cols()) -
           frontier_center_mat)
              .colwise()
              .norm();
    }

    vector<double> cluster_weight(cluster_num, 0);
    Eigen::Matrix3Xf new_center(3, cluster_num);
    new_center.setZero();
    for (int i = 0; i < distance_to_center.cols(); i++) {
      int idx = 0;
      // cluster that frontiers belong to
      distance_to_center.col(i).minCoeff(&idx);
      double weight = frontier_size_mat(i) * frontier_size_mat(i);
      cluster_weight[idx] += weight;
      new_center.col(idx) += weight * frontier_center_mat.col(i);
    }

    for (int i = 0; i < new_center.cols(); i++) {
      new_center.col(i) /= cluster_weight[i];
    }

    float delta = (new_center - cluster_center).colwise().norm().maxCoeff();
    if (delta < 1e-3) {
      cout << "break after iteration " << iter_count << endl;
      break;
    } else {
      cluster_center = new_center;
    }
  }

  // calculate cluster normal
  Eigen::Matrix3Xf cluster_normal(3, cluster_num);
  cluster_normal.setZero();
  vector<double> cluster_weight(cluster_num, 0);
  for (int i = 0; i < distance_to_center.cols(); i++) {
    int idx = 0;
    distance_to_center.col(i).minCoeff(&idx);
    double weight = frontier_size_mat(i) * frontier_size_mat(i);
    cluster_weight[idx] += weight;
    cluster_normal.col(idx) += weight * frontier_normal_mat.col(i);
  }

  cluster_normal = cluster_normal.colwise().normalized();

  vector<Cluster> cluster_vec;
  Cluster cluster_result;
  for (int i = 0; i < cluster_num; i++) {
    cluster_result.center = cluster_center.col(i);
    cluster_result.normal = cluster_normal.col(i);
    cluster_vec.push_back(cluster_result);
  }
  return cluster_vec;
}

void cluster_visualize(vector<Cluster> &cluster_vec,
                       ros::Publisher &cluster_pub) {
  visualization_msgs::MarkerArray marker_cluster;

  // visualize cluster center
  visualization_msgs::Marker marker_cluster_center;
  marker_cluster_center.id = 0;
  marker_cluster_center.header.frame_id = "map";
  marker_cluster_center.pose.orientation.w = 1.0;
  marker_cluster_center.pose.orientation.x = 0.0;
  marker_cluster_center.pose.orientation.y = 0.0;
  marker_cluster_center.pose.orientation.z = 0.0;
  marker_cluster_center.type = visualization_msgs::Marker::SPHERE_LIST;
  marker_cluster_center.color.a = 1;
  marker_cluster_center.color.r = 0;
  marker_cluster_center.color.g = 1;
  marker_cluster_center.color.b = 0;
  marker_cluster_center.scale.x = 0.2;
  marker_cluster_center.scale.y = 0.2;
  marker_cluster_center.scale.z = 0.2;

  for (int i = 0; i < cluster_vec.size(); i++) {
    geometry_msgs::Point point;
    point.x = cluster_vec[i].center.x();
    point.y = cluster_vec[i].center.y();
    point.z = cluster_vec[i].center.z();
    marker_cluster_center.points.push_back(point);
  }
  marker_cluster.markers.push_back(marker_cluster_center);

  // visualize normal vector of cluster center
  visualization_msgs::Marker marker_cluster_normal(marker_cluster_center);
  marker_cluster_normal.type = visualization_msgs::Marker::ARROW;
  marker_cluster_normal.color.a = 1;
  marker_cluster_normal.color.r = 0;
  marker_cluster_normal.color.g = 1;
  marker_cluster_normal.color.b = 0;
  marker_cluster_normal.scale.x = 0.08;
  marker_cluster_normal.scale.y = 0.15;
  marker_cluster_normal.scale.z = 0.15;

  double arrow_len = 0.5;
  for (int i = 0; i < cluster_vec.size(); i++) {
    marker_cluster_normal.points.clear();
    marker_cluster_normal.id = 200 + i;
    geometry_msgs::Point point;
    point.x = cluster_vec[i].center.x();
    point.y = cluster_vec[i].center.y();
    point.z = cluster_vec[i].center.z();
    marker_cluster_normal.points.push_back(point);
    point.x = cluster_vec[i].center.x() + cluster_vec[i].normal.x() * arrow_len;
    point.y = cluster_vec[i].center.y() + cluster_vec[i].normal.y() * arrow_len;
    point.z = cluster_vec[i].center.z() + cluster_vec[i].normal.z() * arrow_len;
    marker_cluster_normal.points.push_back(point);
    marker_cluster.markers.push_back(marker_cluster_normal);
  }
  cluster_pub.publish(marker_cluster);
}

vector<Cluster> dbscan_cluster(set<QuadMesh> &frontiers, const float &eps,
                               const int &min_pts, const int &min_cluster_pts,
                               ros::Publisher &cluster_vis_pub) {

  visualization_msgs::MarkerArray all_frontier;

  visualization_msgs::Marker same_type_frontier;
  same_type_frontier.header.frame_id = "map";
  same_type_frontier.id = 0;
  same_type_frontier.type = visualization_msgs::Marker::CUBE_LIST;
  same_type_frontier.pose.orientation.w = 1.0;
  same_type_frontier.pose.orientation.x = 0.0;
  same_type_frontier.pose.orientation.y = 0.0;
  same_type_frontier.pose.orientation.z = 0.0;
  same_type_frontier.scale.x = 0.02;
  same_type_frontier.scale.y = 0.02;
  same_type_frontier.scale.z = 0.02;

  random_device seed;      // 硬件生成随机数种子
  ranlux48 engine(seed()); // 利用种子生成随机数引擎
  uniform_real_distribution<float> distrib(0.0,
                                           1.0); // 设置随机数范围，并为均匀分布

  vector<Cluster> clusters;

  vector<Point> point_array;
  for (auto it = frontiers.begin(); it != frontiers.end(); it++) {
    Eigen::Vector3f normal = {it->normal.x(), it->normal.y(), it->normal.z()};
    Point p(it->center.x(), it->center.y(), it->center.z(), normal);
    point_array.push_back(p);
  }

  // build KD tree from vector
  KdTree *frontier_kd_tree = new KdTree;
  frontier_kd_tree->BuildTree(point_array, 0, point_array.size());

  unordered_map<KdTree *, bool> is_node_visited;
  queue<KdTree *> bfs_q;
  bfs_q.push(frontier_kd_tree);

  while (!bfs_q.empty()) {

    KdTree *node = bfs_q.front();
    if (node->leftTree)
      bfs_q.push(node->leftTree);
    if (node->rightTree)
      bfs_q.push(node->rightTree);
    bfs_q.pop();

    if (is_node_visited[node] == false) {
      int count = 0;
      queue<KdTree *> cluster_q;
      cluster_q.push(node);

      Cluster cluster_candidate;
      cluster_candidate.center.setZero();
      cluster_candidate.normal.setZero();
      while (!cluster_q.empty()) {

        KdTree *cluster_node = cluster_q.front();
        cluster_q.pop();
        is_node_visited[cluster_node] = true;
        priority_queue<pair<float, KdTree *>, vector<pair<float, KdTree *>>,
                       KdTree::CustomCompare>
            nbr_queue;
        // kd tree search nearest point within eps
        GetEpsNbrPoint(frontier_kd_tree, cluster_node->point, eps, nbr_queue);
        // check this point is core point or not
        // if it is core point: add all neighbor point to queue
        if (nbr_queue.size() > min_pts) {
          count++;
          cluster_candidate.center +=
              Eigen::Vector3f(cluster_node->point.x, cluster_node->point.y,
                              cluster_node->point.z);
          // cout << "node normal = " << cluster_node->point.normal.x() << ", "
          // << cluster_node->point.normal.y() << ", " <<
          // cluster_node->point.normal.z() << endl;
          cluster_candidate.normal += cluster_node->point.normal;

          // vis
          geometry_msgs::Point fc;
          fc.x = cluster_node->point.x;
          fc.y = cluster_node->point.y;
          fc.z = cluster_node->point.z;
          same_type_frontier.points.push_back(fc);

          while (!nbr_queue.empty()) {
            if (is_node_visited[nbr_queue.top().second] == false) {
              cluster_q.push(nbr_queue.top().second);
              is_node_visited[nbr_queue.top().second] = true;
            }
            nbr_queue.pop();
          }
        }
        if (count > 500) {
          break;
        }
      }

      if (count >= min_cluster_pts) {
        cluster_candidate.center /= count;
        cluster_candidate.normal /= count;
        // cout << "count = " << count << ", norm of normal = " <<
        // cluster_candidate.normal.norm() << ", (" <<
        // cluster_candidate.normal.x() << ", " << cluster_candidate.normal.y()
        // << ", " << cluster_candidate.normal.z() << ")" << endl;
        cluster_candidate.normal = cluster_candidate.normal.normalized();
        // cout << "(" << cluster_candidate.normal.x() << ", " <<
        // cluster_candidate.normal.y() << ", " << cluster_candidate.normal.z()
        // << ")" << endl;
        clusters.push_back(cluster_candidate);

        // frontier cluster vis
        same_type_frontier.color.a = 0.5;
        same_type_frontier.color.r = distrib(engine);
        same_type_frontier.color.g = distrib(engine);
        same_type_frontier.color.b = distrib(engine);
        all_frontier.markers.push_back(same_type_frontier);
        same_type_frontier.points.clear();
        same_type_frontier.id++;
      }
    }
  }
  cluster_vis_pub.publish(all_frontier);
  cout << "dbscan cluster num :" << clusters.size() << endl;
  return clusters;
}

geometry_msgs::PoseArray view_point_generate(vector<Cluster> &cluster_vec,
                                             octomap::OcTree *ocmap) {
  geometry_msgs::PoseArray view_point_array;
  view_point_array.header.frame_id = "map";

  for (int i = 0; i < cluster_vec.size(); i++) {
    octomap::point3d center(cluster_vec[i].center.x(),
                            cluster_vec[i].center.y(),
                            cluster_vec[i].center.z());
    octomap::point3d normal(cluster_vec[i].normal.x(),
                            cluster_vec[i].normal.y(), 0);
    for (double offset = 0.8; offset < 2.5; offset += 0.1) {
      octomap::point3d view_point = center - normal.normalized() * offset;
      // view point has to be reachable
      octomap::OcTreeNode *view_node = ocmap->search(view_point);
      if (view_node == nullptr || view_node->getOccupancy() > 0.6) {
        break;
      }
      // there is no obstacle near the view point
      if (is_next_to_obstacle(ocmap, view_point, 0.8, 0.8)) {
        continue;
      } else {
        geometry_msgs::Pose view_point_pose;
        view_point_pose.position.x = view_point.x();
        view_point_pose.position.y = view_point.y();
        view_point_pose.position.z = view_point.z();
        tf2::Quaternion quaternion;
        quaternion.setRPY(0.0, 0.0, atan2(normal.y(), normal.x()));
        view_point_pose.orientation.w = quaternion.w();
        view_point_pose.orientation.x = quaternion.x();
        view_point_pose.orientation.y = quaternion.y();
        view_point_pose.orientation.z = quaternion.z();
        view_point_array.poses.push_back(view_point_pose);
        break;
      }
    }
  }
  return view_point_array;
}
