#ifndef _VOROPOINT3D_H_
#define _VOROPOINT3D_H_

#define INTPOINT3D IntPoint3D

/*! A light-weight integer point with fields x,y,z. */
class IntPoint3D {
public:
  IntPoint3D() : x(0), y(0), z(0) {}
  IntPoint3D(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
  int x, y, z;
  bool operator==(const IntPoint3D &p) const {
    return x == p.x && y == p.y && z == p.z;
  }
};

#endif
