// Author: Matias Mattamala

#pragma once

#include <opencv2/opencv.hpp>
#include <image_proc_cuda/utils.hpp>
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>

#ifdef HAS_CUDA
#include <npp.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

namespace image_proc_cuda {

class ColorCalibrationModule {
 public:
  ColorCalibrationModule();
  void enable(bool enabled);
  bool enabled() const;

  //-----------------------------------------------------------------------------
  // Setters
  //-----------------------------------------------------------------------------
  void setCalibrationMatrix(const std::vector<double>& color_calibration_matrix);

  //-----------------------------------------------------------------------------
  // Getters
  //-----------------------------------------------------------------------------
  cv::Mat getCalibrationMatrix() const;

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
  void initCalibrationMatrix(const cv::Matx33d& matrix);

  void colorCorrection(cv::Mat& image);
#ifdef HAS_CUDA
  void colorCorrection(cv::cuda::GpuMat& image);
#endif

  //-----------------------------------------------------------------------------
  // Variables
  //-----------------------------------------------------------------------------
  bool enabled_;

  // Calibration & undistortion
  bool calibration_available_;
  cv::Matx33d color_calibration_matrix_;

#ifdef HAS_CUDA
  Npp32f gpu_color_calibration_matrix_[3][4] = {{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}};
#endif

};

}  // namespace image_proc_cuda