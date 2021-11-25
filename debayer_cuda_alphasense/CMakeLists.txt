cmake_minimum_required(VERSION 3.1)
project(debayer_cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -O3 -DNDEBUG")

set(LOCAL_OPENCV_INSTALL false)

set(PACKAGE_DEPENDENCIES
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

find_package(CUDA REQUIRED)
message("CUDA Version:" ${CUDA_VERSION_STRING})
message("CUDA Path:" ${CUDA_DIR})

find_package(Eigen3 REQUIRED)
message("Eigen Version:" ${EIGEN3_VERSION_STRING})
message("Eigen Path:" ${Eigen3_DIR})

if(LOCAL_OPENCV_INSTALL)
  set("OpenCV_DIR" "~/.local_opencv_xphoto/lib/cmake/opencv4")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc xphoto)
message("OpenCV version: ${OpenCV_Version}")
message("OpenCV path: ${OpenCV_DIR}")

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS
    ${PACKAGE_DEPENDENCIES}
  DEPENDS EIGEN3
)

include_directories(include ${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

set(CUDA_LIBRARIES "-L/usr/local/cuda/lib64 -lcudart -lcuda -lcudadevrt")
set(SYSTEM_LIBS "-L/usr/lib/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu/tegra -lnppicc -lnppig -lnppial -lnppif -lnppist -lnppidei -lnpps -lnppisu")

add_library(${PROJECT_NAME}
    src/debayer_cuda.cpp
)

add_library(${PROJECT_NAME}_nodelet
    src/debayer_cuda_nodelet.cpp
)

add_executable(${PROJECT_NAME}_node
    src/debayer_cuda_node.cpp
)

target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${STATIC_NVENC}
    ${SYSTEM_LIBS}
    ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME}_nodelet
    ${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${STATIC_NVENC}
    ${SYSTEM_LIBS}
    ${OpenCV_LIBS}
)
