// Author: Matias Mattamala

#pragma once

#include <opencv2/opencv.hpp>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace image_proc_cuda {

class GammaCorrectionModule {
 public:
  GammaCorrectionModule();
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
  bool apply(T& image);

  //-----------------------------------------------------------------------------
  // Wrapper methods (CPU)
  //-----------------------------------------------------------------------------
  void gammaCorrectCustom(cv::Mat& image);
  void gammaCorrectDefault(cv::Mat& image);

//-----------------------------------------------------------------------------
// Wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
  void gammaCorrectCustom(cv::cuda::GpuMat& image);
  void gammaCorrectDefault(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;

  std::string method_;
  bool is_forward_;
  double k_;
};  // namespace image_proc_cuda

}  // namespace image_proc_cuda