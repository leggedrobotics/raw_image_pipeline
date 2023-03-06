/**
@brief raw_image_pipeline_white_balance.hpp
Class for White Balance built upon "AutoWhiteBalance" code by Shane Yuan,
inspired by Jonathan Barron's "Fast Fourier Color Constancy", CVPR, 2017

Author: Matias Mattamala
*/

#pragma once

#include <stdio.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>

// OpenCV
#include <opencv2/opencv.hpp>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

namespace raw_image_pipeline_white_balance {
class ConvolutionalColorConstancyWB {
 private:
  // Model
  template <typename T>
  struct Model {
    int width_;
    int height_;
    T hist_;
    T filter_;
    T bias_;
    T response_;
    T hist_fft_;
    T filter_fft_;
    T bias_fft_;
    T response_fft_;
  };

  Model<cv::Mat> model_;
#ifdef HAS_CUDA
  Model<cv::cuda::GpuMat> gpu_model_;
#endif

  // Model filename
  bool use_gpu_;
  std::string model_filename_;

  // Image histogram
  cv::Size small_size_;
  float bin_size_;
  float uv0_;

  // Output variables
  cv::Point uv_pos_;
  cv::Point uv_pos_prev_;
  float gain_r_, gain_g_, gain_b_;

  // kalman filter use to smooth result
  cv::Mat kf_measurement_;
  cv::Mat kf_estimate_;
  cv::Mat kf_inv_H_cov_;  // Inverse of measurement model covariance
  cv::Mat kf_H_;          // Observation model
  std::shared_ptr<cv::KalmanFilter> kf_ptr_;

  // Parameters
  float bright_thr_;
  float dark_thr_;
  bool use_temporal_consistency_;
  bool debug_;
  bool debug_uv_offset_;

  // Debugging
  float Lu_debug_;
  float Lv_debug_;

  // Other helper variables
  bool first_frame_;
  size_t idx_;

 private:
  // Load pretrained model
  int loadModel(const std::string& model_file);

  // Calculate histogram of input image
  void calculateHistogramFeature(const cv::Mat& src, std::string out_name = "input");

  // Enforces temporal consistency with a Kalman filter
  int kalmanFiltering();

  // Compute gains
  void computeGains();

  // Computes the filter response to obtain the UV offset
  int computeResponse();
  // Apply gains
  void applyGains(cv::Mat& image);

// Compute and apply RGB gains
#ifdef HAS_CUDA
  int computeResponseCuda();
  void applyGainsCuda(cv::cuda::GpuMat& image);
#endif

 public:
  ConvolutionalColorConstancyWB(bool use_gpu);
  ConvolutionalColorConstancyWB(bool use_gpu, const std::string& filename);
  ~ConvolutionalColorConstancyWB();

// Applies white balance
#ifdef HAS_CUDA
  void balanceWhite(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);
#endif
  void balanceWhite(const cv::Mat& src, cv::Mat& dst);

  // Set threshold
  void resetTemporalConsistency();
  void setSaturationThreshold(float bright_thr, float dark_thr);
  void setTemporalConsistency(bool enable);
  void setUV0(float uv0);
  void setDebug(bool debug);
  void setDebugUVOffset(float Lu, float Lv, float uv0);
};

}  // namespace raw_image_pipeline_white_balance