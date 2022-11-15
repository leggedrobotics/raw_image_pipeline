> :warning: **The instructions below are not recommended. Use the [`opencv_catkin`](../README.md##requirements-and-compilation) approach instead.**


## Setting up the Jetson

Instructions on how to set up the Jetson for use with alphasense_rsl resources

### Dependencies

```sh
sudo apt install ros-melodic-eigen-conversions
```

### Compilation and installation of xphoto from OpenCV 4.2.0

Note: This is needed if compiling `debayer_cuda` on Jetson machines, that have Jetpack 4.5 with ROS Melodic installed (OpenCV version must be > 4.2.0).

#### Requirements

```sh
git clone git@github.com:opencv/opencv_contrib.git -b 4.2.0
git clone git@github.com:opencv/opencv.git -b 4.2.0
```

#### Compilation and installation
This installs only xphoto and its dependencies in `.local_opencv_xphoto`

```sh
$ cd ~/git
$ ./git_make_shared.sh
$ mkdir ~/git/opencv/build && cd ~/git/opencv/build
$ mkdir ~/.local_opencv_xphoto
$ cmake -DOPENCV_EXTRA_MODULES_PATH=~/git/opencv_contrib/modules -DBUILD_LIST=xphoto -DCMAKE_INSTALL_PREFIX=~/.local_opencv_xphoto -DENABLE_PRECOMPILED_HEADERS=OFF ..
$ make -j5
$ make install
```

Then set the variable `LOCAL_OPENCV_INSTALL` in CMakeLists.txt of `debayer_cuda` to true.