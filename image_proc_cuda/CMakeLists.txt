cmake_minimum_required(VERSION 3.1)
project(image_proc_cuda)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -O3 -DNDEBUG")

set(PACKAGE_DEPENDENCIES
  eigen_conversions
  image_proc_white_balance
  pybind11_catkin
)

find_package(catkin REQUIRED
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)
# YAML-cpp
find_package(yaml-cpp REQUIRED)

find_package(Eigen3 REQUIRED)
message("Eigen Version:" ${EIGEN3_VERSION_STRING})
message("Eigen Path:" ${Eigen3_DIR})

find_package(OpenCV REQUIRED)
message("OpenCV version: ${OpenCV_Version}")
message("OpenCV path: ${OpenCV_DIR}")

# Check if OpenCV was compiled with CUDA
if(${OpenCV_CUDA_VERSION})
  add_definitions (-DHAS_CUDA)
  find_package(CUDA)
  set(CUDA_npp_LIBRARIES
    ${CUDA_nppc_LIBRARY}
    ${CUDA_nppial_LIBRARY}
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppif_LIBRARY}
    ${CUDA_nppig_LIBRARY}
    ${CUDA_nppim_LIBRARY}
    ${CUDA_nppist_LIBRARY}
    ${CUDA_nppisu_LIBRARY}
    ${CUDA_nppitc_LIBRARY}
    ${CUDA_npps_LIBRARY}
  )
else()
  set(CUDA_npp_LIBRARIES "")
  set(CUDA_INCLUDE_DIRS "")
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    ${PACKAGE_DEPENDENCIES}
  DEPENDS
    OpenCV
    YAML_CPP
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
  src/image_proc_cuda/modules/color_calibration.cpp
  src/image_proc_cuda/modules/color_enhancer.cpp
  src/image_proc_cuda/modules/debayer.cpp
  src/image_proc_cuda/modules/flip.cpp
  src/image_proc_cuda/modules/gamma_correction.cpp
  src/image_proc_cuda/modules/undistortion.cpp
  src/image_proc_cuda/modules/vignetting_correction.cpp
  src/image_proc_cuda/modules/white_balance.cpp
  src/image_proc_cuda/image_proc_cuda.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}
    ${CUDA_npp_LIBRARIES}
    ${OpenCV_LIBS}
    ${YAML_CPP_LIBRARIES}
)

# Mark executables and/or libraries for installation
install(
  TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
  DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
)

# Mark other files for installation
install(
  DIRECTORY model
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)