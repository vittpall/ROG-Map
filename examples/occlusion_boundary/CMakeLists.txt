cmake_minimum_required(VERSION 3.10)
project(occlusion_boundary)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rog_map
        sensor_msgs
        visualization_msgs
        pcl_ros
        pcl_conversions
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
        INCLUDE_DIRS include
        CATKIN_DEPENDS roscpp rog_map sensor_msgs visualization_msgs
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
)

add_executable(occlusion_boundary_node Apps/occlusion_boundary_node.cpp)

target_link_libraries(occlusion_boundary_node
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
)
