#include <image_proc_cuda/modules/undistortion.hpp>

namespace image_proc_cuda {

UndistortionModule::UndistortionModule() : enabled_(true) {}

void UndistortionModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool UndistortionModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void UndistortionModule::setImageSize(int width, int height) {
  image_size_ = cv::Size(width, height);
}

void UndistortionModule::setCameraMatrix(const std::vector<double>& camera_matrix) {
  camera_matrix_ = cv::Matx33d(camera_matrix.data());
}

void UndistortionModule::setDistortionCoefficients(const std::vector<double>& coefficients) {
  distortion_coeff_ = cv::Matx14d(coefficients.data());
}

void UndistortionModule::setDistortionModel(const std::string& model) {
  distortion_model_ = model;
}

void UndistortionModule::setRectificationMatrix(const std::vector<double>& rectification_matrix) {
  rectification_matrix_ = cv::Matx33d(rectification_matrix.data());
}

void UndistortionModule::setProjectionMatrix(const std::vector<double>& projection_matrix) {
  projection_matrix_ = cv::Matx34d(projection_matrix.data());
}

//-----------------------------------------------------------------------------
// Getters
//-----------------------------------------------------------------------------

int UndistortionModule::getImageHeight() const {
  return image_size_.height;
}

int UndistortionModule::getImageWidth() const {
  return image_size_.width;
}

std::string UndistortionModule::getDistortionModel() const {
  if (calibration_available_) {
    if (enabled_) {
      return "none";
    } else {
      return distortion_model_;
    }
  } else {
    return "none";
  }
}

std::string UndistortionModule::getOriginalDistortionModel() const {
  if (calibration_available_) {
    return distortion_model_;
  } else {
    return "none";
  }
}

cv::Mat UndistortionModule::getCameraMatrix() const {
  cv::Rect slice(0, 0, 3, 3);
  return cv::Mat(projection_matrix_)(slice).clone();
}

cv::Mat UndistortionModule::getOriginalCameraMatrix() const {
  return cv::Mat(camera_matrix_).clone();
}

cv::Mat UndistortionModule::getDistortionCoefficients() const {
  if (calibration_available_) {
    if (enabled_) {
      // Image was undistorted, so it's all zeros
      return cv::Mat::zeros(1, 4, CV_64F);
    } else {
      // Return original distortion vector
      return cv::Mat(distortion_coeff_).clone();
    }
  } else {
    // Return just zeros
    return cv::Mat::zeros(1, 4, CV_64F);
  }
}

cv::Mat UndistortionModule::getOriginalDistortionCoefficients() const {
  if (calibration_available_) {
    // Return original distortion vector
    return cv::Mat(distortion_coeff_).clone();
  } else {
    // Return just zeros
    return cv::Mat::zeros(1, 4, CV_64F);
  }
}

cv::Mat UndistortionModule::getRectificationMatrix() const {
  return cv::Mat(rectification_matrix_).clone();
}

cv::Mat UndistortionModule::getOriginalRectificationMatrix() const {
  return cv::Mat(rectification_matrix_).clone();
}

cv::Mat UndistortionModule::getProjectionMatrix() const {
  return cv::Mat(projection_matrix_).clone();
}

cv::Mat UndistortionModule::getOriginalProjectionMatrix() const {
  return cv::Mat(projection_matrix_).clone();
}

cv::Mat UndistortionModule::getDistortedImage() const {
  return original_image_.clone();
}

//-----------------------------------------------------------------------------
// Helper methods
//-----------------------------------------------------------------------------
void UndistortionModule::loadCalibration(const std::string& file_path) {
  std::cout << "Loading camera calibration from file " << file_path << std::endl;

  // Check if file exists
  if (boost::filesystem::exists(file_path)) {
    // Load calibration
    YAML::Node node = YAML::LoadFile(file_path);
    // Camera matrix
    camera_matrix_ = utils::get<cv::Matx33d>(node["camera_matrix"], "data", cv::Matx33d::eye());
    distortion_coeff_ = utils::get<cv::Matx14d>(node["distortion_coefficients"], "data", cv::Matx14d::zeros());
    rectification_matrix_ = utils::get<cv::Matx33d>(node["rectification_matrix"], "data", cv::Matx33d::eye());
    projection_matrix_ = utils::get<cv::Matx34d>(node["projection_matrix"], "data", cv::Matx34d::eye());
    distortion_model_ = utils::get<std::string>(node, "distortion_model", "none");
    int width = utils::get<int>(node, "image_width", 320);
    int height = utils::get<int>(node, "image_height", 240);
    image_size_ = cv::Size(width, height);

    // Init rectify map
    initRectifyMap();
    calibration_available_ = true;
  }
  // If not, disable calibration available flag
  else {
    calibration_available_ = false;
    std::cout << "Warning: Calibration file doesn't exist" << std::endl;
  }
}

void UndistortionModule::initRectifyMap() {
  cv::fisheye::initUndistortRectifyMap(camera_matrix_,         // Intrinsics
                                       distortion_coeff_,      // Distortion
                                       rectification_matrix_,  // Rectification
                                       projection_matrix_,     // New projection matrix
                                       image_size_,            // Image resolution
                                       CV_32F,                 // Map type
                                       undistortion_map_x_,    // Undistortion map for X axis
                                       undistortion_map_y_     // Undistortion map for Y axis
  );

#ifdef HAS_CUDA
  // Upload everything to GPU
  gpu_undistortion_map_x_.upload(undistortion_map_x_);
  gpu_undistortion_map_y_.upload(undistortion_map_y_);
#endif
}

void UndistortionModule::undistort(cv::Mat& image) {
  image.copyTo(original_image_);

  cv::Mat out;
  cv::remap(image, out, undistortion_map_x_, undistortion_map_y_, cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_REPLICATE);
  image = out;
}

#ifdef HAS_CUDA
void UndistortionModule::undistort(cv::cuda::GpuMat& image) {
  image.download(original_image_);

  cv::cuda::GpuMat out;
  cv::cuda::remap(image, out, gpu_undistortion_map_x_, gpu_undistortion_map_y_, cv::InterpolationFlags::INTER_LINEAR,
                  cv::BorderTypes::BORDER_REPLICATE);
  image = out;
}
#endif

//-----------------------------------------------------------------------------
// Apply method
//-----------------------------------------------------------------------------
template <typename T>
bool UndistortionModule::apply(T& image) {
  if (!enabled_) {
    return false;
  }

  if (!calibration_available_) {
    std::cout << "No calibration available!" << std::endl;
    return false;
  }

  if (distortion_model_ != "none") {
    undistort(image);
  }

  return true;
}

}  // namespace image_proc_cuda