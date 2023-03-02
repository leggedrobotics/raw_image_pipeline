// Author: Matias Mattamala

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

#include <image_proc_cuda/utils.hpp>
#include <image_proc_white_balance/convolutional_color_constancy.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace image_proc_cuda {

class WhiteBalanceModule {
 public:
  WhiteBalanceModule();
  WhiteBalanceModule(const std::string& method);
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
  bool apply(T& image, std::string& encoding) {
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
      cccWBPtr_->setSaturationThreshold(saturation_bright_thr_, saturation_dark_thr_);
      cccWBPtr_->setTemporalConsistency(temporal_consistency_);
      cccWBPtr_->setDebug(false);
      cccWBPtr_->balanceWhite(image, image);
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
  void balanceWhiteSimple(cv::cuda::GpuMat& image);
  void balanceWhiteGreyWorld(cv::cuda::GpuMat& image);
  void balanceWhiteLearned(cv::cuda::GpuMat& image);
  void balanceWhitePca(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  std::string method_;
  double clipping_percentile_;
  double saturation_bright_thr_;
  double saturation_dark_thr_;
  bool temporal_consistency_;

  // Pointers
  std::shared_ptr<image_proc_white_balance::ConvolutionalColorConstancyWB> cccWBPtr_;
};

}  // namespace image_proc_cuda