// Author: Matias Mattamala

#pragma once

#include <opencv2/opencv.hpp>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace image_proc_cuda {

class ColorEnhancerModule {
 public:
  ColorEnhancerModule();
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setHueGain(const double& gain);
  void setSaturationGain(const double& gain);
  void setValueGain(const double& gain);

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image);

  //-----------------------------------------------------------------------------
  // Helper methods (CPU)
  //-----------------------------------------------------------------------------
 private:
  void enhance(cv::Mat& image);
#ifdef HAS_CUDA
  void enhance(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  double value_gain_;
  double saturation_gain_;
  double hue_gain_;

};

}  // namespace image_proc_cuda