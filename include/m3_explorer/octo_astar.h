#ifndef OCTO_ASTAR_H
#define OCTO_ASTAR_H
#include <Eigen/Dense>
#include <memory>
#include <octomap/octomap.h>
#include <queue>
#include <ros/ros.h>
#include <unordered_map>
#include <vector>

// wrapper of octomap node
class OctoNode {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  octomap::OcTreeNode *node_ = nullptr;
  Eigen::Vector3f center_;
  Eigen::Vector3f bbx_min_;
  Eigen::Vector3f bbx_max_;
  float size_ = 0.0;
  float half_size_ = 0.0;
  std::shared_ptr<OctoNode> father_node_ = nullptr;
  std::vector<std::shared_ptr<OctoNode>> child_node_;

  OctoNode(octomap::OcTreeNode *node, const Eigen::Vector3f &center, float size,
           std::shared_ptr<OctoNode> father_node)
      : node_(node), center_(center), size_(size), half_size_(0.5 * size),
        father_node_(father_node) {
    Eigen::Vector3f offset(half_size_, half_size_, half_size_);
    bbx_min_ = center_ - offset;
    bbx_max_ = center_ + offset;
    child_node_.resize(8, nullptr);
  }

  bool is_in_bbx(const Eigen::Vector3f &p) {
    bool is_x_in_bbx = (bbx_min_.x() <= p.x()) && (p.x() <= bbx_max_.x());
    bool is_y_in_bbx = (bbx_min_.y() <= p.y()) && (p.y() <= bbx_max_.y());
    bool is_z_in_bbx = (bbx_min_.z() <= p.z()) && (p.z() <= bbx_max_.z());
    return is_x_in_bbx && is_y_in_bbx && is_z_in_bbx;
  }
};

class OctoAstarNodeKey {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3f position_;

  OctoAstarNodeKey(const Eigen::Vector3f &position) : position_(position) {}
  // 用于unordered_map
  bool operator==(const OctoAstarNodeKey &n) const {
    int x0 = static_cast<int>(round(position_.x() / 0.1));
    int y0 = static_cast<int>(round(position_.y() / 0.1));
    int z0 = static_cast<int>(round(position_.z() / 0.1));

    int x1 = static_cast<int>(round(n.position_.x() / 0.1));
    int y1 = static_cast<int>(round(n.position_.y() / 0.1));
    int z1 = static_cast<int>(round(n.position_.z() / 0.1));

    return x0 == x1 && y0 == y1 && z0 == z1;
  }
};

// 用于unordered_map
struct OctoAstarNodeHash {
  size_t operator()(const OctoAstarNodeKey &key) const {
    // x max_range: [-50, 50], 10 bit
    // y max_range: [-50, 50], 10 bit
    // z max_range: [0, 3], 5 bit
    // key = x  y   z
    // bit = 24-15   14-5    4-0
    int x0 = static_cast<int>(round(key.position_.x() / 0.1)) & 0x3FF;
    x0 <<= 15;
    int y0 = static_cast<int>(round(key.position_.y() / 0.1)) & 0x3FF;
    y0 <<= 5;
    int z0 = static_cast<int>(round(key.position_.z() / 0.1)) & 0x1F;
    z0 <<= 0;

    int hash_value = x0 | y0 | z0;
    // return static_cast<size_t>(key);
    return std::hash<int>()(hash_value);
  }
};

class OctoAstarNode {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3f position_;
  std::shared_ptr<OctoNode> octo_node_ = nullptr;
  std::shared_ptr<OctoAstarNode> father_node_ = nullptr;
  float f_score_ = 0.0;
  float g_score_ = 0.0;
  float h_score_ = 0.0;
  // state: 0 -> open; 1 -> closed
  int state = -1;

  OctoAstarNode(const Eigen::Vector3f &position) : position_(position) {}
};

// 用于优先队列
struct OctoAstarNodeCmp {
  bool operator()(const std::shared_ptr<OctoAstarNode> &lhs,
                  const std::shared_ptr<OctoAstarNode> &rhs) {
    return lhs->f_score_ > rhs->f_score_;
  }
};

class OctoAstar {
private:
  float max_z_ = 2.0;
  float min_z_ = 1.0;
  const octomap::OcTree *ocmap_ = nullptr;
  std::shared_ptr<OctoNode> root_node_ = nullptr;
  std::unordered_map<OctoAstarNodeKey, std::shared_ptr<OctoAstarNode>,
                     OctoAstarNodeHash>
      node_at_key;

public:
  std::vector<std::shared_ptr<OctoAstarNode>> path_;

public:
  OctoAstar(const octomap::OcTree *ocmap);
  float astar_path_distance(const Eigen::Vector3f &start_p,
                            const Eigen::Vector3f &end_p);
  // search the node at p from top to bottom
  std::shared_ptr<OctoNode> search_octonode(const Eigen::Vector3f &p);
  inline float calc_h_score(const Eigen::Vector3f &start_p,
                            const Eigen::Vector3f &end_p);
  bool is_path_valid(const Eigen::Vector3f &cur_pos,
                     const Eigen::Vector3f &next_pos);
  inline bool
  add_node_to_q(const std::shared_ptr<OctoNode> &next_node,
                const std::shared_ptr<OctoAstarNode> &cur_node,
                const Eigen::Vector3f &end_p,
                std::priority_queue<std::shared_ptr<OctoAstarNode>,
                                    std::vector<std::shared_ptr<OctoAstarNode>>,
                                    OctoAstarNodeCmp> &astar_q);
};

#endif
