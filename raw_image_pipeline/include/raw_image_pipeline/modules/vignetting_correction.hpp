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

class VignettingCorrectionModule {
 public:
  VignettingCorrectionModule(bool use_gpu);
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image) {
    if (!enabled_) {
      return false;
    }
    correct(image);
    return true;
  }

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setParameters(const double& scale, const double& a2, const double& a4);

  //-----------------------------------------------------------------------------
  // Helper methods (CPU)
  //-----------------------------------------------------------------------------
  void precomputeVignettingMask(int height, int width);

 private:
  void correct(cv::Mat& image);
#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  void correct(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  bool use_gpu_;

  double vignetting_correction_scale_;
  double vignetting_correction_a2_;
  double vignetting_correction_a4_;
  cv::Mat vignetting_mask_f_;
  cv::Mat image_f_;
#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  cv::cuda::GpuMat gpu_vignetting_mask_f_;
  cv::cuda::GpuMat gpu_image_f_;
#endif
};

}  // namespace raw_image_pipeline