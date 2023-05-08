//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <raw_image_pipeline/modules/gamma_correction.hpp>

namespace raw_image_pipeline {

GammaCorrectionModule::GammaCorrectionModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu) {
  init();
}

void GammaCorrectionModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool GammaCorrectionModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void GammaCorrectionModule::setMethod(const std::string& method) {
  method_ = method;
}

void GammaCorrectionModule::setK(const double& k) {
  k_ = k;
  is_forward_ = (k_ <= 1.0 ? true : false);
  init();
}

void GammaCorrectionModule::init() {
  cpu_lut_ = cv::Mat(1, 256, CV_8U);

  for (int i = 0; i < 256; i++) {
    float f = i / 255.0;
    f = pow(f, k_);
    cpu_lut_.at<uchar>(i) = cv::saturate_cast<uchar>(f * 255.0);
  }

#ifdef HAS_CUDA
  if (use_gpu_) {
    gpu_lut_ = cv::cuda::createLookUpTable(cpu_lut_);
  }
#endif
}

//-----------------------------------------------------------------------------
// White balance wrapper methods (CPU)
//-----------------------------------------------------------------------------
void GammaCorrectionModule::gammaCorrectCustom(cv::Mat& image) {
  cv::LUT(image, cpu_lut_, image);
}

void GammaCorrectionModule::gammaCorrectDefault(cv::Mat& image) {
  gammaCorrectCustom(image);
}

//-----------------------------------------------------------------------------
// White balance wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void GammaCorrectionModule::gammaCorrectCustom(cv::cuda::GpuMat& image) {
  gpu_lut_->transform(image, image);
}

void GammaCorrectionModule::gammaCorrectDefault(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat out;
  cv::cuda::gammaCorrection(image, out, is_forward_);
  image = out;
}
#endif
}  // namespace raw_image_pipeline