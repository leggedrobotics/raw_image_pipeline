//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <raw_image_pipeline/modules/flip.hpp>

namespace raw_image_pipeline {

FlipModule::FlipModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu) {}

void FlipModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool FlipModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void FlipModule::setAngle(const int& angle) {
  angle_ = angle;
}

//-----------------------------------------------------------------------------
// Getters
//-----------------------------------------------------------------------------
cv::Mat FlipModule::getImage() const {
  return image_.clone();
}

//-----------------------------------------------------------------------------
// Wrapper methods (CPU)
//-----------------------------------------------------------------------------
void FlipModule::flip(cv::Mat& image) {
  // From https://stackoverflow.com/a/42359233
  cv::Mat out;
  if (angle_ == 90) { // clockwise rotation
    cv::transpose(image, out);
    cv::flip(out, out, 1);
    image = out;

  } else if (angle_ == 180) {
    cv::flip(image, out, -1);  // negative numbers flip x and y
    image = out;

  } else if (angle_ == 270) { // counter-clockwise rotation
    cv::transpose(image, out);
    cv::flip(out, out, 0);
    image = out;

  } else {
    // Do nothing
    
  }
}

void FlipModule::saveFlippedImage(cv::Mat& image) {
  image.copyTo(image_);
}

//-----------------------------------------------------------------------------
// Wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void FlipModule::flip(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat out;

  if (angle_ == 90) { // clockwise rotation
    cv::cuda::transpose(image, out);
    cv::cuda::flip(out, out, 1);
    image = out;

  } else if (angle_ == 180) {
    cv::cuda::flip(image, out, -1);  // negative numbers flip x and y
    image = out;
 
  } else if (angle_ == 270) { // counter-clockwise rotation
    cv::cuda::transpose(image, out);
    cv::cuda::flip(out, out, 0);
    image = out;
  
  } else {
    // Do nothing
  }
}

void FlipModule::saveFlippedImage(cv::cuda::GpuMat& image) {
  image.download(image_);
}
#endif

}  // namespace raw_image_pipeline