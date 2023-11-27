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

class GammaCorrectionModule {
 public:
  GammaCorrectionModule(bool use_gpu);
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setMethod(const std::string& method);
  void setK(const double& k);

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image) {
    if (!enabled_) {
      return false;
    }
    if (method_ == "custom") {
      gammaCorrectCustom(image);
    } else {
      gammaCorrectDefault(image);
    }
    return true;
  }

  //-----------------------------------------------------------------------------
  // Wrapper methods (CPU)
  //-----------------------------------------------------------------------------
  void gammaCorrectCustom(cv::Mat& image);
  void gammaCorrectDefault(cv::Mat& image);

  //-----------------------------------------------------------------------------
  // Wrapper methods (GPU)
  //-----------------------------------------------------------------------------
#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  void gammaCorrectCustom(cv::cuda::GpuMat& image);
  void gammaCorrectDefault(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Helper methods (CPU)
  //-----------------------------------------------------------------------------
  void init();

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  bool use_gpu_;

  std::string method_;
  bool is_forward_;
  double k_;

  // LUT
  cv::Mat cpu_lut_;

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  cv::Ptr<cv::cuda::LookUpTable> gpu_lut_;
#endif
};

}  // namespace raw_image_pipeline