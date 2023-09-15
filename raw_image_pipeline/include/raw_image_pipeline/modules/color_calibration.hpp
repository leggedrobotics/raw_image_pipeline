//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#pragma once

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <raw_image_pipeline/utils.hpp>
#include <opencv2/opencv.hpp>

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
#include <npp.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace raw_image_pipeline {

class ColorCalibrationModule {
 public:
  ColorCalibrationModule(bool use_gpu);
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setCalibrationMatrix(const std::vector<double>& color_calibration_matrix);
  void setCalibrationBias(const std::vector<double>& color_calibration_bias);

  //-----------------------------------------------------------------------------
  // Getters
  //-----------------------------------------------------------------------------
  cv::Mat getCalibrationMatrix() const;
  cv::Mat getCalibrationBias() const;

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image) {
    if (!enabled_) {
      return false;
    }
    if (image.channels() != 3) {
      return false;
    }
    if (!calibration_available_) {
      std::cout << "No calibration available!" << std::endl;
      return false;
    }
    colorCorrection(image);
    return true;
  }

  //-----------------------------------------------------------------------------
  // Helper methods (CPU)
  //-----------------------------------------------------------------------------
 public:
  void loadCalibration(const std::string& file_path);

 private:
  void initCalibrationMatrix(const cv::Matx33d& matrix);

  void colorCorrection(cv::Mat& image);
#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  void colorCorrection(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  bool use_gpu_;

  // Calibration & undistortion
  bool calibration_available_;
  cv::Matx33f color_calibration_matrix_;
  cv::Scalar color_calibration_bias_;

#ifdef HAS_CUDA
#include <opencv2/cudaarithm.hpp>
  Npp32f gpu_color_calibration_matrix_[3][4] = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
#endif
};

}  // namespace raw_image_pipeline