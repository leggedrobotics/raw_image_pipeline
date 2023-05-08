#
# Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
#

cmake_minimum_required(VERSION 3.1)
project(raw_image_pipeline_ros)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -O3 -DNDEBUG")

set(PACKAGE_DEPENDENCIES
  raw_image_pipeline
  cv_bridge
  glog_catkin
  eigen_conversions
  image_transport
  nodelet
  roscpp
)

find_package(catkin REQUIRED
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)

find_package(Eigen3 REQUIRED)
message("Eigen Version:" ${EIGEN3_VERSION_STRING})
message("Eigen Path:" ${Eigen3_DIR})

find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_Version}")
message("OpenCV path: ${OpenCV_DIR}")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${PACKAGE_DEPENDENCIES}
  DEPENDS
    Eigen3
    OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/raw_image_pipeline_ros.cpp
)
target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)

add_library(${PROJECT_NAME}_nodelet
    src/raw_image_pipeline_ros_nodelet.cpp
)
target_link_libraries(${PROJECT_NAME}_nodelet
    ${PROJECT_NAME}
)

add_executable(${PROJECT_NAME}_node
    src/raw_image_pipeline_ros_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
)
