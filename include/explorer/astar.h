#ifndef ASTAR_H
#define ASTAR_H
#include <Eigen/Dense>
#include <octomap/octomap.h>
#include <ros/ros.h>
#include <vector>

using namespace std;

class AstarNode {
public:
  Eigen::Vector3f position_;
  int father_id_;
  float f_score_, g_score_, h_score_;

  AstarNode() { position_ = Eigen::Vector3f::Zero(); }
  AstarNode(const Eigen::Vector3f &position) { position_ = position; }
  // 用于unordered_map
  bool operator==(const AstarNode &n) const {
    int x0 = static_cast<int>(round(position_.x() / 0.1));
    int y0 = static_cast<int>(round(position_.y() / 0.1));
    int z0 = static_cast<int>(round(position_.z() / 0.1));

    int x1 = static_cast<int>(round(n.position_.x() / 0.1));
    int y1 = static_cast<int>(round(n.position_.y() / 0.1));
    int z1 = static_cast<int>(round(n.position_.z() / 0.1));

    return x0 == x1 && y0 == y1 && z0 == z1;
  }
};

// 用于优先队列
struct AstarNodeCmp {
  bool operator()(const AstarNode &lhs, const AstarNode &rhs) {
    return lhs.f_score_ > rhs.f_score_;
  }
};

// 用于map
struct AstarMapCmp {
  bool operator()(const AstarNode &lhs, const AstarNode &rhs) {
    int x0 = static_cast<int>(round(lhs.position_.x() / 0.1));
    int y0 = static_cast<int>(round(lhs.position_.y() / 0.1));
    int z0 = static_cast<int>(round(lhs.position_.z() / 0.1));

    int x1 = static_cast<int>(round(rhs.position_.x() / 0.1));
    int y1 = static_cast<int>(round(rhs.position_.y() / 0.1));
    int z1 = static_cast<int>(round(rhs.position_.z() / 0.1));

    if (x0 != x1)
      return x0 < x1;
    if (y0 != y1)
      return y0 < y1;
    return z0 < z1;
  }
};

// 用于unordered_map
struct AstarNodeHash {
  size_t operator()(const AstarNode &node) const {
    // x max_range: [-50, 50], 10 bit
    // y max_range: [-50, 50], 10 bit
    // z max_range: [0, 3], 5 bit
    // key = x  y   z
    // bit = 24-15   14-5    4-0
    int x0 = static_cast<int>(round(node.position_.x() / 0.1)) & 0x3FF;
    x0 <<= 15;
    int y0 = static_cast<int>(round(node.position_.y() / 0.1)) & 0x3FF;
    y0 <<= 5;
    int z0 = static_cast<int>(round(node.position_.z() / 0.1)) & 0x1F;
    z0 <<= 0;

    int key = x0 | y0 | z0;
    return hash<int>()(key);
  }
};

class Astar {
private:
  float max_z_ = 2.5;
  float min_z_ = 0.0;
public:
  std::vector<AstarNode> path_;
  float astar_path_distance(const octomap::OcTree *ocmap,
                            const Eigen::Vector3f &start_p,
                            const Eigen::Vector3f &end_p);
  float calc_h_score(const Eigen::Vector3f &start_p,
                     const Eigen::Vector3f &end_p);
  bool is_path_valid(const octomap::OcTree *ocmap,
                     const Eigen::Vector3f &cur_pos,
                     const Eigen::Vector3f &next_pos);
};

#endif
