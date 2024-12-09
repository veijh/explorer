#ifndef BLOCK_H
#define BLOCK_H
#include <memory>
#include <vector>

class RangeVoxel {
public:
  int min_ = 0;
  int max_ = 0;

public:
  RangeVoxel() = default;
  RangeVoxel(const RangeVoxel &rhs) = default;
  RangeVoxel(const int min, const int max) : min_(min), max_(max){};
};

class Block2D {
public:
  // Bounding box of the block in yz plane.
  int y_min_ = 0;
  int y_max_ = 0;
  int z_min_ = 0;
  int z_max_ = 0;
  // Ranges in Block2D.
  std::vector<RangeVoxel> ranges_;

public:
  Block2D() = default;
  Block2D(const Block2D &rhs) = default;
  Block2D(const int y, const RangeVoxel &range);
  Block2D(const int y_min, const int y_max, const int z_min, const int z_max);
  bool EmplaceRangeBack(const RangeVoxel &range);
  bool IsInBlock(const int y, const int z) const;
  std::vector<Block2D> GetOverlap(const Block2D &block) const;
  float GetRoughDistance(const Block2D &block, const int delta_x) const;
  bool GetRangeAtY(const int y, RangeVoxel *range) const;
};

class Block2DWrapper {
public:
  Block2D block_2d_;
  RangeVoxel plug_;
  Block2DWrapper() = default;
  Block2DWrapper(const Block2DWrapper &rhs) = default;
  Block2DWrapper(const RangeVoxel &plug) : plug_(plug){};
  Block2DWrapper(const Block2D &block_2d, const RangeVoxel &plug)
      : block_2d_(block_2d), plug_(plug){};
  bool operator<(const Block2DWrapper &rhs) const {
    return plug_.min_ < rhs.plug_.min_;
  }
};

class Block3D {
public:
  int block_id_ = 0;
  // Bounding box of the block in xyz space.
  int x_min_ = 0;
  int x_max_ = 0;
  int y_min_ = 0;
  int y_max_ = 0;
  int z_min_ = 0;
  int z_max_ = 0;
  // Block2D in the Block3D.
  std::vector<Block2D> blocks_;

public:
  Block3D() = default;
  Block3D(const Block3D &rhs) = default;
  Block3D(const int x, const Block2D &block);
  Block3D(const int x_min, const int x_max, const int y_min, const int y_max,
          const int z_min, const int z_max);
  bool EmplaceBlockBack(const Block2D &block);
  bool IsInBlock(const int x, const int y, const int z) const;
};

class Block3DWrapper {
public:
  Block3D block_3d_;
  Block2D plug_;
  Block3DWrapper() = default;
  Block3DWrapper(const Block3DWrapper &rhs) = default;
  Block3DWrapper(const Block2D &plug) : plug_(plug){};
  Block3DWrapper(const Block3D &block_3d, const Block2D &plug,
                 const int block_id);
  bool operator<(const Block3DWrapper &rhs) const {
    return plug_.y_min_ != rhs.plug_.y_min_ ? plug_.y_min_ < rhs.plug_.y_min_
                                            : plug_.z_min_ < rhs.plug_.z_min_;
  }
};

#endif
