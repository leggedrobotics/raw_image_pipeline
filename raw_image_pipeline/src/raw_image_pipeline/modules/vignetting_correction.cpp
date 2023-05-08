//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <raw_image_pipeline/modules/vignetting_correction.hpp>

namespace raw_image_pipeline {

VignettingCorrectionModule::VignettingCorrectionModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu) {}

void VignettingCorrectionModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool VignettingCorrectionModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void VignettingCorrectionModule::setParameters(const double& scale, const double& a2, const double& a4) {
  vignetting_correction_scale_ = scale;
  vignetting_correction_a2_ = a2;
  vignetting_correction_a4_ = a4;
}

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------
void VignettingCorrectionModule::precomputeVignettingMask(int height, int width) {
  if (height == vignetting_mask_f_.rows && width == vignetting_mask_f_.cols) return;

  // Initialize mask
  vignetting_mask_f_.create(width, height, CV_32F);

  double cx = width / 2.0;
  double cy = height / 2.0;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double r = std::sqrt(std::pow(y - cy, 2) + std::pow(x - cx, 2));
      double k = pow(r, 2) * vignetting_correction_a2_ + pow(r, 4) * vignetting_correction_a4_;
      vignetting_mask_f_.at<float>(x, y) = k;
      // mask.at<float>(x, y) = k;
    }
  }
  double max;
  cv::minMaxLoc(vignetting_mask_f_, NULL, &max, NULL, NULL);
  vignetting_mask_f_ = max > 0 ? vignetting_mask_f_ / max : vignetting_mask_f_;
  // Scale correction
  vignetting_mask_f_ = vignetting_mask_f_ * vignetting_correction_scale_;
  // Add 1
  vignetting_mask_f_ += 1.0;

// Upload to gpu
#ifdef HAS_CUDA
  if (use_gpu_) {
    gpu_vignetting_mask_f_.upload(vignetting_mask_f_);
  }
#endif
}

//-----------------------------------------------------------------------------
// Wrapper methods (CPU)
//-----------------------------------------------------------------------------
void VignettingCorrectionModule::correct(cv::Mat& image) {
  precomputeVignettingMask(image.cols, image.rows);

  // COnvert to Lab to apply correction to L channel
  cv::Mat corrected_image;
  cv::cvtColor(image, corrected_image, cv::COLOR_BGR2Lab);

  // Split the image into 3 channels
  std::vector<cv::Mat> vec_channels(3);
  cv::split(corrected_image, vec_channels);

  // Floating point version
  // Convert image to float
  cv::Mat image_f_;
  vec_channels[0].convertTo(image_f_, CV_32FC1);
  // Multiply by vignetting mask
  cv::multiply(image_f_, vignetting_mask_f_, image_f_, 1.0, CV_32FC1);
  vec_channels[0].release();
  image_f_.convertTo(vec_channels[0], CV_8UC1);

  // Merge 3 channels in the vector to form the color image in LAB color space.
  cv::merge(vec_channels, corrected_image);

  // Convert the histogram equalized image from LAB to BGR color space again
  cv::cvtColor(corrected_image, image, cv::COLOR_Lab2BGR);
}

//-----------------------------------------------------------------------------
// Wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void VignettingCorrectionModule::correct(cv::cuda::GpuMat& image) {
  precomputeVignettingMask(image.cols, image.rows);

  // COnvert to Lab to apply correction to L channel
  cv::cuda::GpuMat corrected_image;
  cv::cuda::cvtColor(image, corrected_image, cv::COLOR_BGR2Lab);

  // Split the image into 3 channels
  std::vector<cv::cuda::GpuMat> vec_channels(3);
  cv::cuda::split(corrected_image, vec_channels);

  // Floating point version
  // Convert image to float
  cv::cuda::GpuMat image_f_;
  vec_channels[0].convertTo(image_f_, CV_32FC1);
  // Multiply by vignetting mask
  cv::cuda::multiply(image_f_, gpu_vignetting_mask_f_, image_f_, 1.0, CV_32FC1);
  vec_channels[0].release();
  image_f_.convertTo(vec_channels[0], CV_8UC1);

  // Merge 3 channels in the vector to form the color image in LAB color space.
  cv::cuda::merge(vec_channels, corrected_image);

  // Convert the histogram equalized image from LAB to BGR color space again
  cv::cuda::cvtColor(corrected_image, image, cv::COLOR_Lab2BGR);
}
#endif

}  // namespace raw_image_pipeline