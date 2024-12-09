#include "m3_explorer/kd_tree.h"

#include <algorithm>
#include <cmath>

Point Point::operator=(const Point &other) {
  this->x = other.x;
  this->y = other.y;
  this->z = other.z;
  this->normal = other.normal;
  return *this;
}

bool Point::operator==(const Point &other) {
  return abs(x - other.x) < 1e-5 && abs(y - other.y) < 1e-5 &&
         abs(z - other.z) < 1e-5 && (normal - other.normal).norm() < 1e-5;
}

bool Point::operator!=(const Point &other) {
  return abs(x - other.x) > 1e-5 || abs(y - other.y) > 1e-5 ||
         abs(z - other.z) > 1e-5 || (normal - other.normal).norm() > 1e-5;
}

bool XSortFunc(Point &a, Point &b) { return a.x < b.x; }

bool YSortFunc(Point &a, Point &b) { return a.y < b.y; }

bool ZSortFunc(Point &a, Point &b) { return a.z < b.z; }

KdTree *KdTree::GetMaxXTreeNode() {
  if (!leftTree && !rightTree)
    return this;
  if (leftTree && !rightTree) {
    return point.x > leftTree->point.x ? this : leftTree;
  }

  if (rightTree && !leftTree) {
    return point.x > rightTree->point.x ? this : rightTree;
  }

  KdTree *leftMaxNode = leftTree->GetMaxXTreeNode();
  KdTree *rightMaxNode = rightTree->GetMaxXTreeNode();

  return leftMaxNode->point.x > rightMaxNode->point.x ? leftMaxNode
                                                      : rightMaxNode;
}

KdTree *KdTree::GetMinXTreeNode() {
  if (!leftTree && !rightTree)
    return this;
  if (leftTree && !rightTree) {
    return point.x < leftTree->point.x ? this : leftTree;
  }
  if (rightTree && !leftTree) {
    return point.x < rightTree->point.x ? this : rightTree;
  }

  KdTree *leftMinNode = leftTree->GetMinXTreeNode();
  KdTree *rightMinNode = rightTree->GetMinXTreeNode();

  return leftMinNode->point.x < rightMinNode->point.x ? leftMinNode
                                                      : rightMinNode;
}

KdTree *KdTree::GetMaxYTreeNode() {
  if (!leftTree && !rightTree)
    return this;
  if (leftTree && !rightTree) {
    return point.y > leftTree->point.y ? this : leftTree;
  }
  if (rightTree && !leftTree) {
    return point.y > rightTree->point.y ? this : rightTree;
  }

  KdTree *leftMaxNode = leftTree->GetMaxYTreeNode();
  KdTree *rightMaxNode = rightTree->GetMaxYTreeNode();

  return leftMaxNode->point.y > rightMaxNode->point.y ? leftMaxNode
                                                      : rightMaxNode;
}

KdTree *KdTree::GetMinYTreeNode() {
  if (!leftTree && !rightTree)
    return this;
  if (leftTree && !rightTree) {
    return point.y < leftTree->point.y ? this : leftTree;
  }

  if (rightTree && !leftTree) {
    return point.y < rightTree->point.y ? this : rightTree;
  }
  KdTree *leftMinNode = leftTree->GetMinYTreeNode();
  KdTree *rightMinNode = rightTree->GetMinYTreeNode();

  return leftMinNode->point.y < rightMinNode->point.y ? leftMinNode
                                                      : rightMinNode;
}

KdTree *KdTree::GetMaxZTreeNode() {
  if (!leftTree && !rightTree)
    return this;
  if (leftTree && !rightTree) {
    return point.z > leftTree->point.z ? this : leftTree;
  }

  if (rightTree && !leftTree) {
    return point.z > rightTree->point.z ? this : rightTree;
  }
  KdTree *leftMaxNode = leftTree->GetMaxZTreeNode();
  KdTree *rightMaxNode = rightTree->GetMaxZTreeNode();

  return leftMaxNode->point.z > rightMaxNode->point.z ? leftMaxNode
                                                      : rightMaxNode;
}

KdTree *KdTree::GetMinZTreeNode() {
  if (!leftTree && !rightTree)
    return this;
  if (leftTree && !rightTree) {
    return point.z < leftTree->point.z ? this : leftTree;
  }

  if (rightTree && !leftTree) {
    return point.z < rightTree->point.z ? this : rightTree;
  }
  KdTree *leftMinNode = leftTree->GetMinZTreeNode();
  KdTree *rightMinNode = rightTree->GetMinZTreeNode();

  return leftMinNode->point.z < rightMinNode->point.z ? leftMinNode
                                                      : rightMinNode;
}

bool KdTree::isLeftTreeNode(const Point &point) {
  switch (splitD) {
  case SPLIT_X:
    if (splitV < this->point.x)
      return true;
    break;

  case SPLIT_Y:
    if (splitV < this->point.y)
      return true;
    break;

  case SPLIT_Z:
    if (splitV < this->point.z)
      return true;
    break;

  default:
    break;
  }
  return false;
}

bool KdTree::isRightTreeNode(const Point &point) {
  switch (splitD) {
  case SPLIT_X:
    if (splitV > this->point.x)
      return true;
    break;

  case SPLIT_Y:
    if (splitV > this->point.y)
      return true;
    break;

  case SPLIT_Z:
    if (splitV > this->point.z)
      return true;
    break;

  default:
    break;
  }
  return false;
}

void KdTree::BuildTree(vector<Point> &pointArray, int l, int r) {
  float sdx = 0.0, sdy = 0.0, sdz = 0.0;
  float sumx = 0.0, sumy = 0.0, sumz = 0.0;
  for (int i = l; i < r; ++i) {
    sumx += pointArray[i].x;
    sumy += pointArray[i].y;
    sumz += pointArray[i].z;
  }
  float avex = sumx / (r - l), avey = sumy / (r - l), avez = sumz / (r - l);
  for (int i = l; i < r; ++i) {
    sdx += pow(pointArray[i].x - avex, 2);
    sdy += pow(pointArray[i].y - avey, 2);
    sdz += pow(pointArray[i].z - avez, 2);
  }
  int m = (l + r) / 2;
  if (sdx > sdy && sdx > sdz) {
    sort(pointArray.begin() + l, pointArray.begin() + r, XSortFunc);
    splitD = SPLIT_X;
    splitV = pointArray[m].x;
  } else if (sdy > sdx && sdy > sdz) {
    sort(pointArray.begin() + l, pointArray.begin() + r, YSortFunc);
    splitD = SPLIT_Y;
    splitV = pointArray[m].y;
  } else {
    sort(pointArray.begin() + l, pointArray.begin() + r, ZSortFunc);
    splitD = SPLIT_Z;
    splitV = pointArray[m].z;
  }

  point = pointArray[m];

  if (m > l) {
    leftTree = new KdTree();
    leftTree->BuildTree(pointArray, l, m);
  }
  if (m < r - 1) {
    rightTree = new KdTree();
    rightTree->BuildTree(pointArray, m + 1, r);
  }
}

void KdTree::Insert(Point point) {

  if (!leftTree && rightTree) {
    if (isLeftTreeNode(point)) {
      leftTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      return;
    }
  }

  if (!rightTree && leftTree) {
    if (isRightTreeNode(point)) {
      rightTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      return;
    }
  }

  if (!leftTree && !rightTree) {
    float delta_x = abs(point.x - this->point.x);
    float delta_y = abs(point.y - this->point.y);
    float delta_z = abs(point.z - this->point.z);

    if (delta_x > delta_y && delta_x > delta_z) {
      splitD = SPLIT_X;
      splitV = point.x;
      if (point.x > this->point.x) {
        rightTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      } else {
        leftTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      }
    } else if (delta_y > delta_x && delta_y > delta_z) {
      splitD = SPLIT_Y;
      splitV = point.y;
      if (point.y > this->point.y) {
        rightTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      } else {
        leftTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      }
    } else {
      splitD = SPLIT_Z;
      splitV = point.z;
      if (point.z > this->point.z) {
        rightTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      } else {
        leftTree = new KdTree(point, nullptr, nullptr, SPLIT_NULL, 0);
      }
    }

    return;
  }

  switch (splitD) {
  case SPLIT_X:
    if (point.x < splitV) {
      leftTree->Insert(point);
    } else {
      rightTree->Insert(point);
    }
    break;

  case SPLIT_Y:
    if (point.y < splitV) {
      leftTree->Insert(point);
    } else {
      rightTree->Insert(point);
    }
    break;

  case SPLIT_Z:
    if (point.z < splitV) {
      leftTree->Insert(point);
    } else {
      rightTree->Insert(point);
    }
    break;

  default:
    break;
  }
}

void KdTree::Delete(KdTree *parent, Point point) {
  if (this->point == point) {
    if (!leftTree && !rightTree) {
      if (parent->leftTree && parent->leftTree->point == this->point) {
        parent->leftTree = nullptr;
        this->leftTree = nullptr;
        this->rightTree = nullptr;
        delete this;
        return;
      } else {
        parent->rightTree = nullptr;
        this->leftTree = nullptr;
        this->rightTree = nullptr;
        delete this;
        return;
      }
    } else if (leftTree) {
      KdTree *tmpTree;
      if (splitD == SPLIT_X) {
        tmpTree = leftTree->GetMaxXTreeNode();
        this->splitV = tmpTree->point.x;
      } else if (splitD == SPLIT_Y) {
        tmpTree = leftTree->GetMaxYTreeNode();
        this->splitV = tmpTree->point.y;
      } else if (splitD == SPLIT_Y) {
        tmpTree = leftTree->GetMaxZTreeNode();
        this->splitV = tmpTree->point.z;
      }
      this->point = tmpTree->point;
      leftTree->Delete(this, tmpTree->point);
    } else if (!leftTree && rightTree) {
      KdTree *tmpTree;
      if (splitD == SPLIT_X) {
        tmpTree = rightTree->GetMinXTreeNode();
        this->splitV = tmpTree->point.x;
      } else if (splitD == SPLIT_Y) {
        tmpTree = rightTree->GetMinYTreeNode();
        this->splitV = tmpTree->point.y;
      } else {
        tmpTree = rightTree->GetMinZTreeNode();
        this->splitV = tmpTree->point.z;
      }
      this->point = tmpTree->point;
      rightTree->Delete(this, tmpTree->point);
    }
  }
  if (leftTree && ((splitD == SPLIT_X && point.x < splitV) ||
                   (splitD == SPLIT_Y && point.y < splitV) ||
                   (splitD == SPLIT_Z && point.z < splitV))) {
    leftTree->Delete(this, point);
  }
  if (rightTree && ((splitD == SPLIT_X && point.x > splitV) ||
                    (splitD == SPLIT_Y && point.y > splitV) ||
                    (splitD == SPLIT_Z && point.z > splitV))) {
    rightTree->Delete(this, point);
  }
}

KdTree::~KdTree() {
  if (this->leftTree || this->rightTree) {
    // bfs delete
    queue<KdTree *> childe_node;
    if (this->leftTree) {
      childe_node.push(this->leftTree);
    }
    if (this->rightTree) {
      childe_node.push(this->rightTree);
    }
    while (!childe_node.empty()) {
      KdTree *node = childe_node.front();
      if (node->leftTree) {
        childe_node.push(node->leftTree);
      }
      if (node->rightTree) {
        childe_node.push(node->rightTree);
      }
      childe_node.pop();

      node->leftTree = nullptr;
      node->rightTree = nullptr;
      delete node;
    }
  }
}

void GetNearestPoint(KdTree *node, Point point, float &minDis,
                     Point &tarPoint) {
  if (node == nullptr) {
    return;
  }
  minDis = pow(point.x - node->point.x, 2) + pow(point.y - node->point.y, 2) +
           pow(point.z - node->point.z, 2);
  tarPoint = node->point;
  GetNearestPoint_Rec(node, point, minDis, tarPoint);
}

void GetNearestPoint_Rec(KdTree *node, Point point, float &minDis,
                         Point &tarPoint) {
  if (node == nullptr) {
    return;
  }
  float dis = pow(point.x - node->point.x, 2) +
              pow(point.y - node->point.y, 2) + pow(point.z - node->point.z, 2);
  if (dis < minDis) {
    minDis = dis;
    tarPoint = node->point;
  }

  float dis_to_split = dis;
  bool go_left = false;
  switch (node->splitD) {
  case KdTree::SPLIT_X:
    if (point.x < node->point.x) {
      go_left = true;
    }
    dis_to_split = pow(point.x - node->point.x, 2);
    break;

  case KdTree::SPLIT_Y:
    if (point.y < node->point.y) {
      go_left = true;
    }
    dis_to_split = pow(point.y - node->point.y, 2);
    break;

  case KdTree::SPLIT_Z:
    if (point.z < node->point.z) {
      go_left = true;
    }
    dis_to_split = pow(point.z - node->point.z, 2);
    break;

  default:
    break;
  }

  if (go_left) {
    GetNearestPoint_Rec(node->leftTree, point, minDis, tarPoint);
    if (dis_to_split < minDis) {
      GetNearestPoint_Rec(node->rightTree, point, minDis, tarPoint);
    }
  } else {
    GetNearestPoint_Rec(node->rightTree, point, minDis, tarPoint);
    if (dis_to_split < minDis) {
      GetNearestPoint_Rec(node->leftTree, point, minDis, tarPoint);
    }
  }
}

void GetEpsNbrPoint(
    KdTree *node, Point point, const float &eps,
    priority_queue<pair<float, KdTree *>, vector<pair<float, KdTree *>>,
                   KdTree::CustomCompare> &nbr_queue) {
  if (node == nullptr) {
    return;
  }
  float dis = pow(point.x - node->point.x, 2) +
              pow(point.y - node->point.y, 2) + pow(point.z - node->point.z, 2);
  if (dis < eps * eps) {
    nbr_queue.push(make_pair(dis, node));
  }

  float dis_to_split = dis;
  bool go_left = false;
  switch (node->splitD) {
  case KdTree::SPLIT_X:
    if (point.x < node->point.x) {
      go_left = true;
    }
    dis_to_split = pow(point.x - node->point.x, 2);
    break;

  case KdTree::SPLIT_Y:
    if (point.y < node->point.y) {
      go_left = true;
    }
    dis_to_split = pow(point.y - node->point.y, 2);
    break;

  case KdTree::SPLIT_Z:
    if (point.z < node->point.z) {
      go_left = true;
    }
    dis_to_split = pow(point.z - node->point.z, 2);
    break;

  default:
    break;
  }

  if (go_left) {
    GetEpsNbrPoint(node->leftTree, point, eps, nbr_queue);
    if (dis_to_split < eps * eps) {
      GetEpsNbrPoint(node->rightTree, point, eps, nbr_queue);
    }
  } else {
    GetEpsNbrPoint(node->rightTree, point, eps, nbr_queue);
    if (dis_to_split < eps * eps) {
      GetEpsNbrPoint(node->leftTree, point, eps, nbr_queue);
    }
  }
}
