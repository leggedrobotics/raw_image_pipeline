#include <raw_image_pipeline/modules/color_calibration.hpp>

namespace raw_image_pipeline {

ColorCalibrationModule::ColorCalibrationModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu) {
  cv::Matx33d matrix = cv::Matx33d::eye();
  initCalibrationMatrix(matrix);
}

void ColorCalibrationModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool ColorCalibrationModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void ColorCalibrationModule::setCalibrationMatrix(const std::vector<double>& color_calibration_matrix) {
  cv::Matx33d matrix = cv::Matx33d(color_calibration_matrix.data());
  initCalibrationMatrix(matrix);
}

//-----------------------------------------------------------------------------
// Getters
//-----------------------------------------------------------------------------
cv::Mat ColorCalibrationModule::getCalibrationMatrix() const {
  return cv::Mat(color_calibration_matrix_);
}

//-----------------------------------------------------------------------------
// Helper methods
//-----------------------------------------------------------------------------
void ColorCalibrationModule::loadCalibration(const std::string& file_path) {
  std::cout << "Loading color calibration from file " << file_path << std::endl;

  // Check if file exists
  if (boost::filesystem::exists(file_path)) {
    // Load calibration
    YAML::Node node = YAML::LoadFile(file_path);
    // Camera matrix
    cv::Matx33d matrix = utils::get<cv::Matx33d>(node["color_calibration_matrix"], "data", cv::Matx33d::eye());
    initCalibrationMatrix(matrix);

    calibration_available_ = true;
  }
  // If not, disable calibration available flag
  else {
    calibration_available_ = false;
    std::cout << "Warning: Color calibration file doesn't exist" << std::endl;
  }
}

void ColorCalibrationModule::initCalibrationMatrix(const cv::Matx33d& matrix) {
  color_calibration_matrix_ = matrix;

#ifdef HAS_CUDA
  if (use_gpu_) {
    for (size_t i = 0; i < 3; i++)
      for (size_t j = 0; j < 3; j++) {
        gpu_color_calibration_matrix_[i][j] = static_cast<float>(matrix(i, j));
      }
  }
#endif
}

void ColorCalibrationModule::colorCorrection(cv::Mat& image) {
  // https://stackoverflow.com/a/12678457
  cv::Mat flat_image = image.reshape(1, image.rows * image.cols);
  cv::Mat flat_image_f;
  flat_image.convertTo(flat_image_f, CV_32F);

  // Mix
  cv::Mat mixed_image = flat_image_f * color_calibration_matrix_.t();
  cv::Mat image_f = mixed_image.reshape(3, image.rows);

  image_f.convertTo(image, CV_8UC3);
}

#ifdef HAS_CUDA
void ColorCalibrationModule::colorCorrection(cv::cuda::GpuMat& image) {
  NppiSize image_size;
  image_size.width = image.cols;
  image_size.height = image.rows;

  // Apply calibration
  nppiColorTwist32f_8u_C3IR(image.data, static_cast<int>(image.step), image_size, gpu_color_calibration_matrix_);
}
#endif

}  // namespace raw_image_pipeline