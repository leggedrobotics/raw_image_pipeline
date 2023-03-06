cmake_minimum_required(VERSION 3.1)
project(raw_image_pipeline_python)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -Wextra -O3 -DNDEBUG")

set(PACKAGE_DEPENDENCIES
  raw_image_pipeline
  pybind11_catkin
)

find_package(catkin REQUIRED
    COMPONENTS
    ${PACKAGE_DEPENDENCIES}
)

catkin_package(
  LIBRARIES
  CATKIN_DEPENDS
    ${PACKAGE_DEPENDENCIES}
)

# Add cvnp to add support for opencv-numpy conversions
add_subdirectory(thirdparty/cvnp)

# Based on https://github.com/ipab-slmc/exotica/blob/master/exotica_python/CMakeLists.txt
pybind11_add_module(_py_raw_image_pipeline MODULE 
  src/raw_image_pipeline_python.cpp
)
target_link_libraries(_py_raw_image_pipeline PRIVATE cvnp ${PYTHON_LIBRARIES} ${catkin_LIBRARIES} ${STATIC_NVENC} ${SYSTEM_LIBS})
target_compile_options(_py_raw_image_pipeline PRIVATE "-Wno-deprecated-declarations")  # because we are refactoring and triggering this ourselves.
target_include_directories(_py_raw_image_pipeline PUBLIC ${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
add_dependencies(_py_raw_image_pipeline ${catkin_EXPORTED_TARGETS})

set_target_properties(_py_raw_image_pipeline PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_GLOBAL_PYTHON_DESTINATION}/py_raw_image_pipeline)
set(PYTHON_LIB_DIR ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION}/py_raw_image_pipeline)
add_custom_command(TARGET _py_raw_image_pipeline
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${PYTHON_LIB_DIR}
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:_py_raw_image_pipeline> ${PYTHON_LIB_DIR}/_py_raw_image_pipeline.so
    WORKING_DIRECTORY ${CATKIN_DEVEL_PREFIX}
COMMENT "Copying library files to python directory")

catkin_python_setup()
catkin_install_python(PROGRAMS scripts/apply_pipeline.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS _py_raw_image_pipeline LIBRARY DESTINATION ${CATKIN_GLOBAL_PYTHON_DESTINATION}/py_raw_image_pipeline)