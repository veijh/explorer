#include "m3_explorer/block.h"
#include "m3_explorer/time_track.hpp"
#include <cmath>

Block2D::Block2D(const int y, const RangeVoxel &range)
    : y_min_(y), y_max_(y), z_min_(range.min_), z_max_(range.max_) {
  ranges_.emplace_back(range);
}

Block2D::Block2D(const int y_min, const int y_max, const int z_min,
                 const int z_max)
    : y_min_(y_min), y_max_(y_max), z_min_(z_min), z_max_(z_max) {}

bool Block2D::EmplaceRangeBack(const RangeVoxel &range) {
  if (ranges_.empty()) {
    return false;
  }
  ++y_max_;
  z_min_ = std::min(z_min_, range.min_);
  z_max_ = std::max(z_max_, range.max_);
  ranges_.emplace_back(range);
  return true;
}

bool Block2D::IsInBlock(const int y, const int z) const {
  if (y < y_min_ || y > y_max_ || z < z_min_ || z > z_max_) {
    return false;
  }
  const int y_index = y - y_min_;
  const RangeVoxel range_at_y = ranges_[y_index];
  if (z >= range_at_y.min_ && z <= range_at_y.max_) {
    return true;
  }
  return false;
}

std::vector<Block2D> Block2D::GetOverlap(const Block2D &block) const {
  std::vector<Block2D> overlapped_blocks;
  if (y_max_ < block.y_min_ || y_min_ > block.y_max_ || z_max_ < block.z_min_ ||
      z_min_ > block.z_max_) {
    return overlapped_blocks;
  }
  const int overlap_y_min = std::max(y_min_, block.y_min_);
  const int overlap_y_max = std::min(y_max_, block.y_max_);
  int state = 0;
  Block2D overlapped_block;
  for (int y = overlap_y_min; y <= overlap_y_max; ++y) {
    const RangeVoxel range_at_y = ranges_[y - y_min_];
    const RangeVoxel block_range_at_y = block.ranges_[y - block.y_min_];
    const int z_min = std::max(range_at_y.min_, block_range_at_y.min_);
    const int z_max = std::min(range_at_y.max_, block_range_at_y.max_);
    switch (state) {
    case 0:
      if (z_min <= z_max) {
        overlapped_block = Block2D(y, RangeVoxel(z_min, z_max));
        state = 1;
      }
      break;
    case 1:
      if (z_min > z_max || y == overlap_y_max) {
        state = 0;
        overlapped_blocks.emplace_back(overlapped_block);
      } else {
        overlapped_block.EmplaceRangeBack(RangeVoxel(z_min, z_max));
      }
      break;
    }
  }
  return overlapped_blocks;
}

float Block2D::GetRoughDistance(const Block2D &block, const int delta_x) const {
  std::vector<Block2D> overlapped_blocks = GetOverlap(block);
  if (overlapped_blocks.empty()) {
    int delta_y = 0.5 * (y_max_ + y_min_ - block.y_max_ - block.y_min_);
    int delta_z = 0.5 * (z_max_ + z_min_ - block.z_max_ - block.z_min_);
    return std::hypot(delta_x, delta_y, delta_z);
  } else {
    return std::abs(static_cast<float>(delta_x));
  }
}

bool Block2D::GetRangeAtY(const int y, RangeVoxel *range) const {
  if (y < y_min_ || y > y_max_) {
    return false;
  }
  const int y_index = y - y_min_;
  *range = ranges_[y_index];
  return true;
}

Block3D::Block3D(const int x, const Block2D &block)
    : x_min_(x), x_max_(x), y_min_(block.y_min_), y_max_(block.y_max_),
      z_min_(block.z_min_), z_max_(block.z_max_) {
  blocks_.emplace_back(block);
}

Block3D::Block3D(const int x_min, const int x_max, const int y_min,
                 const int y_max, const int z_min, const int z_max)
    : x_min_(x_min), x_max_(x_max), y_min_(y_min), y_max_(y_max), z_min_(z_min),
      z_max_(z_max) {}

bool Block3D::EmplaceBlockBack(const Block2D &block) {
  if (blocks_.empty()) {
    return false;
  }
  ++x_max_;
  y_min_ = std::min(y_min_, block.y_min_);
  y_max_ = std::max(y_max_, block.y_max_);
  z_min_ = std::min(z_min_, block.z_min_);
  z_max_ = std::max(z_max_, block.z_max_);
  blocks_.emplace_back(block);
  return true;
}

bool Block3D::IsInBlock(const int x, const int y, const int z) const {
  if (x < x_min_ || x > x_max_ || y < y_min_ || y > y_max_ || z < z_min_ ||
      z > z_max_) {
    return false;
  }
  const int x_index = x - x_min_;
  const Block2D block_at_x = blocks_[x_index];
  return block_at_x.IsInBlock(y, z);
}

Block3DWrapper::Block3DWrapper(const Block3D &block_3d, const Block2D &plug,
                               const int block_id)
    : block_3d_(block_3d), plug_(plug) {
  block_3d_.block_id_ = block_id;
}
