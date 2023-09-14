//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#pragma once

#include <opencv2/opencv.hpp>

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace raw_image_pipeline {

class FlipModule {
 public:
  FlipModule(bool use_gpu);
  void enable(bool enabled);
  bool enabled() const;

    //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setAngle(const int& angle);

  //-----------------------------------------------------------------------------
  // Getters
  //-----------------------------------------------------------------------------
  cv::Mat getImage() const;

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image) {
    if (!enabled_) {
      saveFlippedImage(image);
      return false;
    }
    flip(image);
    saveFlippedImage(image);
    return true;
  }

  //-----------------------------------------------------------------------------
  // Helper methods (CPU)
  //-----------------------------------------------------------------------------
 private:
  void flip(cv::Mat& image);
  void saveFlippedImage(cv::Mat& image);

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  void flip(cv::cuda::GpuMat& image);
  void saveFlippedImage(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  bool use_gpu_;
  cv::Mat image_;

  int angle_;
};

}  // namespace raw_image_pipeline