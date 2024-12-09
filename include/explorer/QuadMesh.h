#ifndef QUADMESH_H
#define QUADMESH_H
#include <octomap/octomap.h>
struct QuadMesh
{
  double size;
  octomap::point3d center;
  octomap::point3d normal;
  bool operator < (const QuadMesh& item) const{
    if(size != item.size){
      return size < item.size;
    }
    else if(center.x() != item.center.x()){
      return center.x() < item.center.x();
    }
    else if(center.y() != item.center.y()){
      return center.y() < item.center.y();
    }
    else if(center.z() != item.center.z()){
      return center.z() < item.center.z();
    }
    else if(normal.x() != item.normal.x()){
      return normal.x() < item.normal.x();
    }
    else if(normal.y() != item.normal.y()){
      return normal.y() < item.normal.y();
    }
    else{
      return normal.z() < item.normal.z();
    }
  }
};
#endif
