//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//
// Author: Timon Homberger

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>
#include <raw_image_pipeline/utils.hpp>

// Modules
#include <raw_image_pipeline/modules/color_calibration.hpp>
#include <raw_image_pipeline/modules/color_enhancer.hpp>
#include <raw_image_pipeline/modules/debayer.hpp>
#include <raw_image_pipeline/modules/flip.hpp>
#include <raw_image_pipeline/modules/gamma_correction.hpp>
#include <raw_image_pipeline/modules/undistortion.hpp>
#include <raw_image_pipeline/modules/vignetting_correction.hpp>
#include <raw_image_pipeline/modules/white_balance.hpp>

namespace raw_image_pipeline {

class RawImagePipeline {
 public:
  // Constructor & destructor
  RawImagePipeline(bool use_gpu);
  RawImagePipeline(bool use_gpu, const std::string& params_path, const std::string& calibration_path,
                   const std::string& color_calibration_path);
  ~RawImagePipeline();

  //-----------------------------------------------------------------------------
  // Main interfaces
  //-----------------------------------------------------------------------------
  bool apply(cv::Mat& image, std::string& encoding);

  // Alternative pipeline that returns a copy
  cv::Mat process(const cv::Mat& image, std::string& encoding);

  // Loaders
  void loadParams(const std::string& file_path);
  void loadCameraCalibration(const std::string& file_path);
  void loadColorCalibration(const std::string& file_path);
  void initUndistortion();

  // Other interfaces
  void resetWhiteBalanceTemporalConsistency();
  void setGpu(bool use_gpu);
  void setDebug(bool debug);

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setDebayer(bool enabled);
  void setDebayerEncoding(const std::string& encoding);

  void setFlip(bool enabled);
  void setFlipAngle(int angle);

  void setWhiteBalance(bool enabled);
  void setWhiteBalanceMethod(const std::string& method);
  void setWhiteBalancePercentile(const double& percentile);
  void setWhiteBalanceSaturationThreshold(const double& bright_thr, const double& dark_thr);
  void setWhiteBalanceTemporalConsistency(bool enabled);
  void setColorCalibration(bool enabled);
  void setColorCalibrationMatrix(const std::vector<double>& color_calibration_matrix);
  void setColorCalibrationBias(const std::vector<double>& color_calibration_bias);
  cv::Mat getColorCalibrationMatrix() const;
  cv::Mat getColorCalibrationBias() const;

  void setGammaCorrection(bool enabled);
  void setGammaCorrectionMethod(const std::string& method);
  void setGammaCorrectionK(const double& k);

  void setVignettingCorrection(bool enabled);
  void setVignettingCorrectionParameters(const double& scale, const double& a2, const double& a4);

  void setColorEnhancer(bool enabled);
  void setColorEnhancerHueGain(const double& gain);
  void setColorEnhancerSaturationGain(const double& gain);
  void setColorEnhancerValueGain(const double& gain);

  void setUndistortion(bool enabled);
  void setUndistortionImageSize(int width, int height);
  void setUndistortionNewImageSize(int width, int height);
  void setUndistortionBalance(double balance);
  void setUndistortionFovScale(double fov_scale);
  void setUndistortionCameraMatrix(const std::vector<double>& camera_matrix);
  void setUndistortionDistortionCoefficients(const std::vector<double>& coefficients);
  void setUndistortionDistortionModel(const std::string& model);
  void setUndistortionRectificationMatrix(const std::vector<double>& rectification_matrix);
  void setUndistortionProjectionMatrix(const std::vector<double>& projection_matrix);

  //-----------------------------------------------------------------------------
  // Getters
  //-----------------------------------------------------------------------------
  bool isDebayerEnabled() const;
  bool isFlipEnabled() const;
  bool isWhiteBalanceEnabled() const;
  bool isColorCalibrationEnabled() const;
  bool isGammaCorrectionEnabled() const;
  bool isVignettingCorrectionEnabled() const;
  bool isColorEnhancerEnabled() const;
  bool isUndistortionEnabled() const;

  int getDistImageHeight() const;
  int getDistImageWidth() const;
  std::string getDistDistortionModel() const;
  cv::Mat getDistCameraMatrix() const;
  cv::Mat getDistDistortionCoefficients() const;
  cv::Mat getDistRectificationMatrix() const;
  cv::Mat getDistProjectionMatrix() const;

  int getRectImageHeight() const;
  int getRectImageWidth() const;
  std::string getRectDistortionModel() const;
  cv::Mat getRectCameraMatrix() const;
  cv::Mat getRectDistortionCoefficients() const;
  cv::Mat getRectRectificationMatrix() const;
  cv::Mat getRectProjectionMatrix() const;

  cv::Mat getDistDebayeredImage() const;
  cv::Mat getDistColorImage() const;
  cv::Mat getRectMask() const;
  cv::Mat getProcessedImage() const;

 private:
  //-----------------------------------------------------------------------------
  // Pipeline
  //-----------------------------------------------------------------------------
  template <typename T>
  void pipeline(T& image, std::string& encoding) {
    // Run pipeline
    debayer_->apply(image, encoding);
    saveDebugImage(image, "/tmp/00_debayer.png");

    flipper_->apply(image);
    saveDebugImage(image, "/tmp/01_flip.png");

    white_balancer_->apply(image);
    saveDebugImage(image, "/tmp/02_white_balancing.png");

    color_calibrator_->apply(image);
    saveDebugImage(image, "/tmp/03_color_calibration.png");

    gamma_corrector_->apply(image);
    saveDebugImage(image, "/tmp/04_gamma_correction.png");

    vignetting_corrector_->apply(image);
    saveDebugImage(image, "/tmp/05_vignetting_correction.png");

    color_enhancer_->apply(image);
    saveDebugImage(image, "/tmp/06_color_enhancer.png");

    undistorter_->apply(image);
    saveDebugImage(image, "/tmp/07_undistortion.png");

    // Save processed output
    saveOutput(image);
  }

  void saveOutput(const cv::Mat& image) {
    // copy to internal image
    image.copyTo(image_);
  }

  void saveDebugImage(const cv::Mat& image, const std::string& filename) const {
    if (debug_) {
      cv::Mat tmp;
      image.copyTo(tmp);
      cv::normalize(tmp, tmp, 0, 255.0, cv::NORM_MINMAX);
      cv::imwrite(filename, tmp);
    }
  }

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  void saveDebugImage(const cv::cuda::GpuMat& image, const std::string& filename) const {
    cv::Mat tmp;
    image.download(tmp);
    saveDebugImage(tmp, filename);
  }

  void saveOutput(const cv::cuda::GpuMat& image) {
    // download to internal image
    image.download(image_);
  }
#endif

  //-----------------------------------------------------------------------------
  // Modules
  //-----------------------------------------------------------------------------
  std::unique_ptr<ColorCalibrationModule> color_calibrator_;
  std::unique_ptr<ColorEnhancerModule> color_enhancer_;
  std::unique_ptr<DebayerModule> debayer_;
  std::unique_ptr<FlipModule> flipper_;
  std::unique_ptr<GammaCorrectionModule> gamma_corrector_;
  std::unique_ptr<UndistortionModule> undistorter_;
  std::unique_ptr<VignettingCorrectionModule> vignetting_corrector_;
  std::unique_ptr<WhiteBalanceModule> white_balancer_;

  //-----------------------------------------------------------------------------
  // Other variables
  //-----------------------------------------------------------------------------
  // Pipeline options
  bool use_gpu_;
  bool debug_;

  // Internal variables
  cv::Mat image_;
};

}  // namespace raw_image_pipeline