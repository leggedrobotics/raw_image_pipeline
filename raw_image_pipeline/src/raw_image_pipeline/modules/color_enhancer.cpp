//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <raw_image_pipeline/modules/color_enhancer.hpp>

namespace raw_image_pipeline {

ColorEnhancerModule::ColorEnhancerModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu) {}

void ColorEnhancerModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool ColorEnhancerModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void ColorEnhancerModule::setHueGain(const double& gain) {
  value_gain_ = gain;
}

void ColorEnhancerModule::setSaturationGain(const double& gain) {
  saturation_gain_ = gain;
}

void ColorEnhancerModule::setValueGain(const double& gain) {
  hue_gain_ = gain;
}

//-----------------------------------------------------------------------------
// Helper methods
//-----------------------------------------------------------------------------
void ColorEnhancerModule::enhance(cv::Mat& image) {
  cv::Mat color_enhanced_image;
  cv::cvtColor(image, color_enhanced_image, cv::COLOR_BGR2HSV);

  cv::Scalar gains(hue_gain_, saturation_gain_, value_gain_);
  cv::multiply(color_enhanced_image, gains, color_enhanced_image);

  // Convert the histogram equalized image from HSV to BGR color space again
  cv::cvtColor(color_enhanced_image, image, cv::COLOR_HSV2BGR);
}

#ifdef HAS_CUDA
void ColorEnhancerModule::enhance(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat color_enhanced_image;
  cv::cuda::cvtColor(image, color_enhanced_image, cv::COLOR_BGR2HSV);

  cv::Scalar gains(hue_gain_, saturation_gain_, value_gain_);
  cv::cuda::multiply(color_enhanced_image, gains, color_enhanced_image);

  // Convert the histogram equalized image from HSV to BGR color space again
  cv::cuda::cvtColor(color_enhanced_image, image, cv::COLOR_HSV2BGR);
}
#endif
}  // namespace raw_image_pipeline