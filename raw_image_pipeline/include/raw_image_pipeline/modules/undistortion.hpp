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
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace raw_image_pipeline {

class UndistortionModule {
 public:
  UndistortionModule(bool use_gpu);
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setImageSize(int width, int height);
  void setNewImageSize(int width, int height);
  void setBalance(double balance);
  void setFovScale(double fov_scale);

  void setCameraMatrix(const std::vector<double>& camera_matrix);
  void setDistortionCoefficients(const std::vector<double>& coefficients);
  void setDistortionModel(const std::string& model);
  void setRectificationMatrix(const std::vector<double>& rectification_matrix);
  void setProjectionMatrix(const std::vector<double>& projection_matrix);

  //-----------------------------------------------------------------------------
  // Getters
  //-----------------------------------------------------------------------------
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

  cv::Mat getDistImage() const;
  cv::Mat getRectMask() const;

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image) {
    saveUndistortedImage(image);
    if (!enabled_) {
      return false;
    }
    if (!calibration_available_) {
      std::cout << "No calibration available!" << std::endl;
      return false;
    }
    if (dist_distortion_model_ != "none") {
      undistort(image);
    }
    return true;
  }

  //-----------------------------------------------------------------------------
  // Helper methods
  //-----------------------------------------------------------------------------
 public:
  void loadCalibration(const std::string& file_path);
  void init();

 private:
  void undistort(cv::Mat& image);
  void saveUndistortedImage(cv::Mat& image);

#ifdef HAS_CUDA
  void undistort(cv::cuda::GpuMat& image);
  void saveUndistortedImage(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;
  bool use_gpu_;

  // Calibration & undistortion
  bool calibration_available_;

  // Original - "distorted" parameters
  std::string dist_distortion_model_;
  cv::Matx33d dist_camera_matrix_;
  cv::Matx14d dist_distortion_coeff_;
  cv::Matx33d dist_rectification_matrix_;
  cv::Matx34d dist_projection_matrix_;
  cv::Size dist_image_size_;

  // Undistorted parameters
  std::string rect_distortion_model_;
  cv::Matx33d rect_camera_matrix_;
  cv::Matx14d rect_distortion_coeff_;
  cv::Matx33d rect_rectification_matrix_;
  cv::Matx34d rect_projection_matrix_;
  cv::Size rect_image_size_;

  double rect_balance_;
  double rect_fov_scale_;

  cv::Mat undistortion_map_x_;
  cv::Mat undistortion_map_y_;

#ifdef HAS_CUDA
  cv::cuda::GpuMat gpu_undistortion_map_x_;
  cv::cuda::GpuMat gpu_undistortion_map_y_;
#endif

  // Original image
  cv::Mat dist_image_;
  cv::Mat rect_mask_;
};

}  // namespace raw_image_pipeline