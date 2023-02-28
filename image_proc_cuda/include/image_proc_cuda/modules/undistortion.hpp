// Author: Matias Mattamala

#pragma once

#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <image_proc_cuda/utils.hpp>
#include <opencv2/opencv.hpp>

#ifdef HAS_CUDA
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

namespace image_proc_cuda {

class UndistortionModule {
 public:
  UndistortionModule();
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setImageSize(int width, int height);
  void setCameraMatrix(const std::vector<double>& camera_matrix);
  void setDistortionCoefficients(const std::vector<double>& coefficients);
  void setDistortionModel(const std::string& model);
  void setRectificationMatrix(const std::vector<double>& rectification_matrix);
  void setProjectionMatrix(const std::vector<double>& projection_matrix);

  //-----------------------------------------------------------------------------
  // Getters
  //-----------------------------------------------------------------------------
  int getImageHeight() const;
  int getImageWidth() const;
  std::string getDistortionModel() const;
  cv::Mat getCameraMatrix() const;
  cv::Mat getDistortionCoefficients() const;
  cv::Mat getRectificationMatrix() const;
  cv::Mat getProjectionMatrix() const;
  std::vector<double> getColorCalibrationMatrix() const;

  std::string getOriginalDistortionModel() const;
  cv::Mat getOriginalCameraMatrix() const;
  cv::Mat getOriginalDistortionCoefficients() const;
  cv::Mat getOriginalRectificationMatrix() const;
  cv::Mat getOriginalProjectionMatrix() const;

  cv::Mat getDistortedImage() const;

  //-----------------------------------------------------------------------------
  // Main interface
  //-----------------------------------------------------------------------------
  template <typename T>
  bool apply(T& image);

  //-----------------------------------------------------------------------------
  // Helper methods (CPU)
  //-----------------------------------------------------------------------------
 public:
  void loadCalibration(const std::string& file_path);

 private:
  void initRectifyMap();

  void undistort(cv::Mat& image);
#ifdef HAS_CUDA
  void undistort(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;

  // Calibration & undistortion
  bool calibration_available_;
  std::string distortion_model_;

  // Original - "distorted" parameters
  cv::Matx33d camera_matrix_;
  cv::Matx14d distortion_coeff_;
  cv::Matx33d rectification_matrix_;
  cv::Matx34d projection_matrix_;

  // Undistorted parameters
  cv::Matx33d undistorted_camera_matrix_;
  cv::Matx14d undistorted_distortion_coeff_;
  cv::Matx33d undistorted_rectification_matrix_;
  cv::Matx34d undistorted_projection_matrix_;
  cv::Size image_size_;

  cv::Mat undistortion_map_x_;
  cv::Mat undistortion_map_y_;

#ifdef HAS_CUDA
  cv::cuda::GpuMat gpu_undistortion_map_x_;
  cv::cuda::GpuMat gpu_undistortion_map_y_;
#endif

  // Distorted image
  cv::Mat original_image_;
};

}  // namespace image_proc_cuda