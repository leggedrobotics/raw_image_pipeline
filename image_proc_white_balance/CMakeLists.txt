cmake_minimum_required(VERSION 3.1)
project(image_proc_white_balance CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

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
find_package(CUDA REQUIRED)

if(${CUDA_FOUND})
  add_definitions (-DHAS_CUDA)
endif()

message("CUDA found: " ${CUDA_FOUND})
message("CUDA Version: " ${CUDA_VERSION_STRING})
message("CUDA include dir: " ${CUDA_INCLUDE_DIRS})
set(CUDA_nppicom_LIBRARY "")

generate_dynamic_reconfigure_options(
  cfg/image_proc_white_balance.cfg
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
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/image_proc_white_balance/convolutional_color_constancy.cpp
)

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

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${CUDA_npp_LIBRARIES}  
  ${OpenCV_LIBRARIES}
)


# Add executable
add_executable(${PROJECT_NAME}_node
    src/image_proc_white_balance_node.cpp
)
target_link_libraries(${PROJECT_NAME}_node
    ${PROJECT_NAME}
)

add_dependencies(${PROJECT_NAME}_node ${PROJECT_NAME}_gencfg)

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