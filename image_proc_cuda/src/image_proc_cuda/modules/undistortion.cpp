#include <image_proc_cuda/modules/undistortion.hpp>

namespace image_proc_cuda {

UndistortionModule::UndistortionModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu), rect_balance_(0.0), rect_fov_scale_(1.0) {}

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
  dist_image_size_ = cv::Size(width, height);
  rect_image_size_ = cv::Size(width, height);
}

void UndistortionModule::setNewImageSize(int width, int height) {
  rect_image_size_ = cv::Size(width, height);
  init();
}

void UndistortionModule::setBalance(double balance) {
  rect_balance_ = balance;
  init();
}

void UndistortionModule::setFovScale(double fov_scale) {
  rect_fov_scale_ = fov_scale;
  init();
}

void UndistortionModule::setCameraMatrix(const std::vector<double>& camera_matrix) {
  dist_camera_matrix_ = cv::Matx33d(camera_matrix.data());
  rect_camera_matrix_ = cv::Matx33d(camera_matrix.data());
}

void UndistortionModule::setDistortionCoefficients(const std::vector<double>& coefficients) {
  dist_distortion_coeff_ = cv::Matx14d(coefficients.data());
  rect_distortion_coeff_ = cv::Matx14d(coefficients.data());
}

void UndistortionModule::setDistortionModel(const std::string& model) {
  dist_distortion_model_ = model;
  rect_distortion_model_ = model;
}

void UndistortionModule::setRectificationMatrix(const std::vector<double>& rectification_matrix) {
  dist_rectification_matrix_ = cv::Matx33d(rectification_matrix.data());
  rect_rectification_matrix_ = cv::Matx33d(rectification_matrix.data());
}

void UndistortionModule::setProjectionMatrix(const std::vector<double>& projection_matrix) {
  dist_projection_matrix_ = cv::Matx34d(projection_matrix.data());
  rect_projection_matrix_ = cv::Matx34d(projection_matrix.data());
}

//-----------------------------------------------------------------------------
// Getters
//-----------------------------------------------------------------------------

int UndistortionModule::getRectImageHeight() const {
  return rect_image_size_.height;
}

int UndistortionModule::getRectImageWidth() const {
  return rect_image_size_.width;
}

int UndistortionModule::getDistImageHeight() const {
  return dist_image_size_.height;
}

int UndistortionModule::getDistImageWidth() const {
  return dist_image_size_.width;
}

std::string UndistortionModule::getRectDistortionModel() const {
  if (calibration_available_) {
    if (enabled_) {
      return "none";
    } else {
      return rect_distortion_model_;
    }
  } else {
    return "none";
  }
}

std::string UndistortionModule::getDistDistortionModel() const {
  if (calibration_available_) {
    return dist_distortion_model_;
  } else {
    return "none";
  }
}

cv::Mat UndistortionModule::getRectCameraMatrix() const {
  return cv::Mat(rect_camera_matrix_).clone();
}

cv::Mat UndistortionModule::getDistCameraMatrix() const {
  return cv::Mat(dist_camera_matrix_).clone();
}

cv::Mat UndistortionModule::getRectDistortionCoefficients() const {
  return cv::Mat(rect_distortion_coeff_).clone();
}

cv::Mat UndistortionModule::getDistDistortionCoefficients() const {
  return cv::Mat(dist_distortion_coeff_).clone();
}

cv::Mat UndistortionModule::getRectRectificationMatrix() const {
  return cv::Mat(rect_rectification_matrix_).clone();
}

cv::Mat UndistortionModule::getDistRectificationMatrix() const {
  return cv::Mat(dist_rectification_matrix_).clone();
}

cv::Mat UndistortionModule::getRectProjectionMatrix() const {
  return cv::Mat(rect_projection_matrix_).clone();
}

cv::Mat UndistortionModule::getDistProjectionMatrix() const {
  return cv::Mat(dist_projection_matrix_).clone();
}

cv::Mat UndistortionModule::getDistImage() const {
  return dist_image_.clone();
}

cv::Mat UndistortionModule::getRectMask() const {
  return rect_mask_.clone();
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
    setImageSize(utils::get<int>(node, "image_width", 320), utils::get<int>(node, "image_height", 240));
    setCameraMatrix(utils::get<std::vector<double>>(node["camera_matrix"], "data", std::vector<double>()));
    setDistortionCoefficients(utils::get<std::vector<double>>(node["distortion_coefficients"], "data", std::vector<double>()));
    setDistortionModel(utils::get<std::string>(node, "distortion_model", "none"));
    setRectificationMatrix(utils::get<std::vector<double>>(node["rectification_matrix"], "data", std::vector<double>()));
    setProjectionMatrix(utils::get<std::vector<double>>(node["projection_matrix"], "data", std::vector<double>()));

    // Init rectify map
    init();
    calibration_available_ = true;
  }
  // If not, disable calibration available flag
  else {
    std::cout << "Warning: Calibration file doesn't exist" << std::endl;
    calibration_available_ = false;

    setImageSize(320, 240);
    setCameraMatrix({1.0, 0.0, 0.0, 0.0,  //
                     0.0, 1.0, 0.0, 0.0,  //
                     0.0, 0.0, 1.0, 0.0,  //
                     0.0, 0.0, 0.0, 1.0});
    setDistortionCoefficients({0.0, 0.0, 0.0, 0.0});
    setDistortionModel("none");
    setRectificationMatrix({1.0, 0.0, 0.0,  //
                            0.0, 1.0, 0.0,  //
                            0.0, 0.0, 1.0});
    setProjectionMatrix({1.0, 0.0, 0.0, 0.0,  //
                         0.0, 1.0, 0.0, 0.0,  //
                         0.0, 0.0, 1.0, 0.0});
  }
}

void UndistortionModule::init() {
  cv::Mat new_camera_matrix;
  cv::fisheye::estimateNewCameraMatrixForUndistortRectify(dist_camera_matrix_,         // Intrinsics (distorted image)
                                                          dist_distortion_coeff_,      // Distortion (distorted image)
                                                          dist_image_size_,            // Image size (distorted image)
                                                          dist_rectification_matrix_,  // Rectification (distorted image)
                                                          new_camera_matrix,           // Intrinsics (new, undistorted image)
                                                          rect_balance_,  // Sets the new focal length in range between the min focal length
                                                                          // and the max focal length. Balance is in range of [0, 1]
                                                          rect_image_size_,  // Image size (new, undistorted image)
                                                          rect_fov_scale_    // Divisor for new focal length
  );
  rect_camera_matrix_ = cv::Matx33d((double*)new_camera_matrix.ptr());

  // Initialize undistortion maps
  cv::fisheye::initUndistortRectifyMap(dist_camera_matrix_,         // Intrinsics (distorted image)
                                       dist_distortion_coeff_,      // Distortion (distorted image)
                                       dist_rectification_matrix_,  // Rectification (distorted image)
                                       rect_camera_matrix_,         // New camera matrix (undistorted image)
                                       dist_image_size_,            // Image resolution (distorted image)
                                       CV_32F,                      // Map type
                                       undistortion_map_x_,         // Undistortion map for X axis
                                       undistortion_map_y_          // Undistortion map for Y axis
  );

  // Update other undistorted (rect) values
  rect_distortion_coeff_ = cv::Matx14d::zeros();
  rect_rectification_matrix_ = cv::Matx33d::eye();
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      rect_projection_matrix_(i, j) = rect_camera_matrix_(i, j);
    }
  }

#ifdef HAS_CUDA
  if (use_gpu_) {
    // Upload everything to GPU
    gpu_undistortion_map_x_.upload(undistortion_map_x_);
    gpu_undistortion_map_y_.upload(undistortion_map_y_);
  }
#endif
}

void UndistortionModule::undistort(cv::Mat& image) {
  cv::Mat out;
  cv::remap(image, out, undistortion_map_x_, undistortion_map_y_, cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_CONSTANT, 0);
  image = out;
}

void UndistortionModule::saveUndistortedImage(cv::Mat& image) {
  image.copyTo(dist_image_);
}

#ifdef HAS_CUDA
void UndistortionModule::undistort(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat out;
  cv::cuda::remap(image, out, gpu_undistortion_map_x_, gpu_undistortion_map_y_, cv::InterpolationFlags::INTER_LINEAR, cv::BORDER_CONSTANT, 0);
  image = out;
}

void UndistortionModule::saveUndistortedImage(cv::cuda::GpuMat& image) {
  image.download(dist_image_);
}
#endif

}  // namespace image_proc_cuda