//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

#include <raw_image_pipeline/utils.hpp>
#include <raw_image_pipeline_white_balance/convolutional_color_constancy.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace raw_image_pipeline {

class WhiteBalanceModule {
 public:
  WhiteBalanceModule(bool use_gpu);
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setMethod(const std::string& method);
  void setSaturationPercentile(const double& percentile);
  void setSaturationThreshold(const double& bright_thr, const double& dark_thr);
  void setTemporalConsistency(bool enabled);

  void resetTemporalConsistency();

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image) {
    if (!enabled_) {
      return false;
    }
    if (image.channels() != 3) {
      return false;
    }
    if (method_ == "simple") {
      balanceWhiteSimple(image);
      return true;

    } else if (method_ == "gray_world" || method_ == "grey_world") {
      balanceWhiteGreyWorld(image);
      return true;

    } else if (method_ == "learned") {
      balanceWhiteLearned(image);
      return true;

    } else if (method_ == "ccc") {
      // CCC white balancing - this works directly on GPU
      // cccWBPtr_->setSaturationThreshold(saturation_bright_thr_, saturation_dark_thr_);
      // cccWBPtr_->setTemporalConsistency(temporal_consistency_);
      // cccWBPtr_->setDebug(false);
      // cccWBPtr_->balanceWhite(image, image);
      ccc_->setSaturationThreshold(saturation_bright_thr_, saturation_dark_thr_);
      ccc_->setTemporalConsistency(temporal_consistency_);
      ccc_->setDebug(false);
      ccc_->balanceWhite(image, image);
      return true;

    } else if (method_ == "pca") {
      balanceWhitePca(image);
      return true;

    } else {
      throw std::invalid_argument("White Balance method [" + method_ +
                                  "] not supported. "
                                  "Supported algorithms: 'simple', 'gray_world', 'learned', 'ccc', 'pca'");
    }
  }

  //-----------------------------------------------------------------------------
  // White balance wrapper methods (CPU)
  //-----------------------------------------------------------------------------
 private:
  void balanceWhiteSimple(cv::Mat& image);
  void balanceWhiteGreyWorld(cv::Mat& image);
  void balanceWhiteLearned(cv::Mat& image);
  void balanceWhitePca(cv::Mat& image);

//-----------------------------------------------------------------------------
// White balance wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  void balanceWhiteSimple(cv::cuda::GpuMat& image);
  void balanceWhiteGreyWorld(cv::cuda::GpuMat& image);
  void balanceWhiteLearned(cv::cuda::GpuMat& image);
  void balanceWhitePca(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  bool use_gpu_;

  std::string method_;
  double clipping_percentile_;
  double saturation_bright_thr_;
  double saturation_dark_thr_;
  bool temporal_consistency_;

  // Pointers
  std::unique_ptr<raw_image_pipeline_white_balance::ConvolutionalColorConstancyWB> ccc_;
  // raw_image_pipeline_white_balance::ConvolutionalColorConstancyWB ccc_;
};

}  // namespace raw_image_pipeline