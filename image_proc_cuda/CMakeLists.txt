cmake_minimum_required(VERSION 3.1)
project(image_proc_cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -O3 -DNDEBUG")

set(PACKAGE_DEPENDENCIES
  eigen_conversions
  ffcc_catkin
  pybind11_catkin
)

find_package(catkin REQUIRED
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)
# YAML-cpp
find_package(yaml-cpp REQUIRED)

find_package(CUDA REQUIRED)
message("CUDA Version:" ${CUDA_VERSION_STRING})
message("CUDA Path:" ${CUDA_DIR})

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
    YAML_CPP
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(CUDA_LIBRARIES "-L/usr/local/cuda/lib64 -lcudart -lcuda -lcudadevrt")
set(SYSTEM_LIBS "-L/usr/lib/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu/tegra -lnppicc -lnppig -lnppial -lnppif -lnppist -lnppidei -lnpps -lnppisu")

add_library(${PROJECT_NAME} SHARED
    src/image_proc_cuda.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${STATIC_NVENC}
    ${SYSTEM_LIBS}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
)
