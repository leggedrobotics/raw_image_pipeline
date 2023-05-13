#
# Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for details.
# 

cmake_minimum_required(VERSION 3.1)
project(raw_image_pipeline_white_balance)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -O3 -DNDEBUG")

set(PACKAGE_DEPENDENCIES
  roscpp
  roslib
  dynamic_reconfigure
)

find_package(catkin REQUIRED
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)

find_package(OpenCV REQUIRED)

# Check if OpenCV was compiled with CUDA
if(${OpenCV_CUDA_VERSION})
  add_definitions (-DHAS_CUDA)
endif()

generate_dynamic_reconfigure_options(
  cfg/RawImagePipelineWhiteBalance.cfg
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${PACKAGE_DEPENDENCIES}
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/raw_image_pipeline_white_balance/convolutional_color_constancy.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)


# Add executable
add_executable(${PROJECT_NAME}_node
    src/raw_image_pipeline_white_balance_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
)

add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg)

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
                 PATTERN "*.h"
)

# Mark other files for installation
install(
  DIRECTORY model
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)