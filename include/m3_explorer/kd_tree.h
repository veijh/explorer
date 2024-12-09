#ifndef KD_TREE_H
#define KD_TREE_H
#include <Eigen/Dense>

#include <queue>
#include <vector>
using namespace std;

class Point {
public:
  Point() {}
  Point(float _x, float _y, float _z, Eigen::Vector3f _normal)
      : x(_x), y(_y), z(_z), normal(_normal) {}
  Point(const Point &other) {
    x = other.x;
    y = other.y;
    z = other.z;
    normal = other.normal;
  }
  float x;
  float y;
  float z;
  Eigen::Vector3f normal;
  friend bool XSortFunc(Point &a, Point &b);
  friend bool YSortFunc(Point &a, Point &b);
  friend bool ZSortFunc(Point &a, Point &b);
  Point operator=(const Point &other);
  bool operator==(const Point &other);
  bool operator!=(const Point &other);
};

class KdTree {
public:
  enum SPLIT_DIM { SPLIT_NULL, SPLIT_X, SPLIT_Y, SPLIT_Z };
  struct CustomCompare {
    bool operator()(const std::pair<float, KdTree *> &lhs,
                    const std::pair<float, KdTree *> &rhs) const {
      return lhs.first < rhs.first;
    }
  };

  KdTree() {
    splitD = SPLIT_NULL;
    splitV = 0;
    leftTree = nullptr, rightTree = nullptr;
    point.normal.setZero();
  }
  ~KdTree();
  KdTree(Point point, KdTree *leftTree, KdTree *rightTree, SPLIT_DIM splitD,
         float splitV) {
    this->point = point;
    this->leftTree = leftTree;
    this->rightTree = rightTree;
    this->splitD = splitD;
    this->splitV = splitV;
  }
  Point point;
  SPLIT_DIM splitD;
  float splitV;
  KdTree *leftTree;
  KdTree *rightTree;
  bool isLeftTreeNode(const Point &point);
  bool isRightTreeNode(const Point &point);
  void BuildTree(vector<Point> &pointArray, int l, int r);
  void Insert(Point point);
  void Delete(KdTree *parent, Point point);
  KdTree *GetMaxXTreeNode();
  KdTree *GetMinXTreeNode();
  KdTree *GetMaxYTreeNode();
  KdTree *GetMinYTreeNode();
  KdTree *GetMaxZTreeNode();
  KdTree *GetMinZTreeNode();
  friend void GetNearestPoint(KdTree *node, Point point, float &minDis,
                              Point &tarPoint);
  friend void GetNearestPoint_Rec(KdTree *node, Point point, float &minDis,
                                  Point &tarPoint);
  friend void GetEpsNbrPoint(
      KdTree *node, Point point, const float &eps,
      priority_queue<pair<float, KdTree *>, vector<pair<float, KdTree *>>,
                     CustomCompare> &nbr_queue);
};

#endif
