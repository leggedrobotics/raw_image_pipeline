Instructions on how to compile xphoto of OpenCV used for Whitebalance of the Alphasense color cameras

(all steps are to be executed as subt user)

# Compile and install xphoto from OpenCV 4.2.0

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

Set the `LOCAL_OPENCV_INSTALL` flag in the CMakeLists.txt of this package to true.