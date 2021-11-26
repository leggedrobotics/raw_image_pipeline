# Alphasense RSL


Authors: Marco Tranzatto, Samuel Zimmermann, Lorenz Wellhausen, Timon Homberger, Shehryar Khattak, Gabriel Waibel.

Maintainers: Marco Tranzatto, Timon Homberger.

## License
This source code is released under a [proprietary license](LICENSE).

## Build

[![Build Status](https://ci.leggedrobotics.com/buildStatus/icon?job=bitbucket_leggedrobotics/darpa_subt/master)](https://ci.leggedrobotics.com/job/bitbucket_leggedrobotics/job/darpa_subt/job/master/)

## Overview

This repository provides utility packages for the 7sense Alphasense sensor on the ANYmal C100 robots.

### Packages

1. **`debayer_cuda`**: This is a debayering tool, allowing to convert raw bayer images to color images. This includes white balance, gamma correction and histogram equalization.

#### Requirements

* For `debayer_cuda`: On Jetson Xavier systems that are set up with Jetpack version > 4.5.1, an additional installation setp is required to ensure that the correct version of xphoto from OpenCV is installed (OpenCV version needs to be > 4.2.0). Follow the steps in the setup instructions to install it locally. Note that this requires `LOCAL_OPENCV_INSTALL` to be set in order to provide the correct OpenCV path - [link](LINKAddedHereOnceMerged)

## Documentation

## Building

