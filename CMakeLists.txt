cmake_minimum_required(VERSION 3.0.2)
project(explorer)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
# add_definitions("-g")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  octomap
  octomap_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  visualization_msgs
  mavros_msgs
  lkh_ros
  message_generation
)
find_package(Eigen3 REQUIRED)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED)

find_package(yaml-cpp REQUIRED)

find_package(OpenMP REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  Traj.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  mavros_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES explorer
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

## Declare a C++ library
add_library(${PROJECT_NAME}_lib
  src/frontier_detector.cpp
  src/frontier_cluster.cpp
  src/path_planning.cpp
  src/kd_tree.cpp
  src/hastar.cpp
  src/astar.cpp
  src/octo_astar.cpp
)

add_library(${PROJECT_NAME}_block_lib
  src/grid_astar.cpp
  src/block.cpp
)

add_library(${PROJECT_NAME}_voronoi_lib
  src/dynamicvoronoi.cpp
  src/dynamicvoronoi3D.cpp
)

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
add_dependencies(${PROJECT_NAME}_lib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(${PROJECT_NAME} src/explorer.cpp)
add_executable(analyze src/analyze.cpp)
add_executable(control_test src/control_test.cpp src/hastar.cpp src/astar.cpp)
add_executable(mavros_ctrl src/mavros_ctrl.cpp)
add_executable(map_merge src/map_merge.cpp)
add_executable(frontier_test src/frontier_test.cpp)
add_executable(octo_astar_test src/octo_astar_test.cpp)
add_executable(grid_astar_test src/grid_astar_test.cpp)
add_executable(block_astar_test src/block_astar_test.cpp)
add_executable(building_block_astar_test src/building_block_astar_test.cpp)
add_executable(create_map_test src/create_map_test.cpp)
add_executable(create_s3dis_map_test src/create_s3dis_map_test.cpp)
add_executable(create_building_test src/create_building_test.cpp)
add_executable(voronoi_test src/voronoi_test.cpp)
add_executable(voronoi3D_test src/voronoi3D_test.cpp)
add_executable(voronoi3D_trajectory_test src/voronoi3D_trajectory_test.cpp)
add_executable(voronoi2DS_test src/voronoi2DS_test.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(mavros_ctrl ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${tf2_LIBRARIES}
  ${tf2_ros_LIBRARIES}
  ${PROJECT_NAME}_lib
)

target_link_libraries(analyze
  ${catkin_LIBRARIES}
)

target_link_libraries(control_test
  ${catkin_LIBRARIES}
)

target_link_libraries(mavros_ctrl
  ${catkin_LIBRARIES}
)

target_link_libraries(map_merge
  ${catkin_LIBRARIES}
)

target_link_libraries(frontier_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
)

target_link_libraries(octo_astar_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
)

target_link_libraries(grid_astar_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_block_lib
)

target_link_libraries(block_astar_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_block_lib
)

target_link_libraries(building_block_astar_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_block_lib
)

target_link_libraries(voronoi_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_voronoi_lib
)

target_link_libraries(voronoi3D_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_voronoi_lib
  ${PROJECT_NAME}_block_lib
)

target_link_libraries(voronoi3D_trajectory_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_voronoi_lib
  ${PROJECT_NAME}_block_lib
)

target_link_libraries(voronoi2DS_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${PROJECT_NAME}_voronoi_lib
  OpenMP::OpenMP_CXX
)

target_link_libraries(create_map_test
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}_lib
  ${Boost_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${PROJECT_NAME}_block_lib
)

target_link_libraries(create_s3dis_map_test
  ${catkin_LIBRARIES}
)

target_link_libraries(create_building_test
  ${catkin_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_explorer.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
