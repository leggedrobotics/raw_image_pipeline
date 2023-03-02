#include <image_proc_cuda/modules/color_enhancer.hpp>

namespace image_proc_cuda {

ColorEnhancerModule::ColorEnhancerModule() : enabled_(true) {}

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
}  // namespace image_proc_cuda