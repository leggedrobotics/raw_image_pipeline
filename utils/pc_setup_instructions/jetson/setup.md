Instructions on how to set up the Jetson for use with alphasense_rsl resources

# Debayer Cuda Alphasense dependency (needed by debayer_cuda_alphasense)

```
$ sudo apt install ros-melodic-eigen-conversions
```

# Compile, install and depend on xphoto from OpenCV 4.2.0 (needed by debayer_cuda_alphasense)

Note: This is needed if compiling debayer cuda on Jetson machines, that have Jetpack 4.5 with ROS Melodic installed (OpenCV version must be > 4.2.0).

Clone required repos:

```
git clone git@github.com:opencv/opencv_contrib.git -b 4.2.0
git clone git@github.com:opencv/opencv.git -b 4.2.0
```

This installs only xphoto and it's dependencies in .local_opencv_xphoto

```
$ cd ~/git
$ ./git_make_shared.sh
$ mkdir ~/git/opencv/build && cd ~/git/opencv/build
$ mkdir ~/.local_opencv_xphoto
$ cmake -DOPENCV_EXTRA_MODULES_PATH=~/git/opencv_contrib/modules -DBUILD_LIST=xphoto -DCMAKE_INSTALL_PREFIX=~/.local_opencv_xphoto -DENABLE_PRECOMPILED_HEADERS=OFF ..
$ make -j5
$ make install
```

Then set variable 'LOCAL_OPENCV_INSTALL' in CMakeLists.txt of debayer_cuda_alphasense to true.