#include <image_proc_cuda/image_proc_cuda.hpp>

#include <boost/filesystem.hpp>
#define FILE_FOLDER (boost::filesystem::path(__FILE__).parent_path().string())
#define DEFAULT_PARAMS_PATH (FILE_FOLDER + "/../../config/pipeline_params_example.yaml")
#define DEFAULT_CALIBRATION_PATH (FILE_FOLDER + "/../../config/alphasense_calib_example.yaml")
#define DEFAULT_COLOR_CALIBRATION_PATH (FILE_FOLDER + "/../../config/alphasense_color_calib_example.yaml")

namespace image_proc_cuda {

ImageProcCuda::ImageProcCuda() : use_gpu_(true), dump_images_(false), idx_(0), white_balancer_() {
  // Load parameters
  loadParams(DEFAULT_PARAMS_PATH);
  undistorter_.loadCalibration(DEFAULT_CALIBRATION_PATH);
  color_calibrator_.loadCalibration(DEFAULT_COLOR_CALIBRATION_PATH);
}

ImageProcCuda::ImageProcCuda(const std::string& params_path, const std::string& calibration_path, const std::string& color_calibration_path,
                             bool use_gpu)
    : use_gpu_(use_gpu), dump_images_(false), idx_(0), white_balancer_() {

  // Load parameters
  if (params_path.empty())
    loadParams(DEFAULT_PARAMS_PATH);
  else
    loadParams(params_path);

  // Load calibration
  if (!calibration_path.empty()) undistorter_.loadCalibration(calibration_path);

  // Load color calibration
  if (color_calibration_path.empty())
    color_calibrator_.loadCalibration(DEFAULT_COLOR_CALIBRATION_PATH);
  else
    color_calibrator_.loadCalibration(color_calibration_path);
}

ImageProcCuda::~ImageProcCuda() {}

void ImageProcCuda::loadParams(const std::string& file_path) {
  std::cout << "Loading image_proc_cuda params from file " << file_path << std::endl;

  // Check if file exists
  if (boost::filesystem::exists(file_path)) {
    // Load parameters
    YAML::Node node = YAML::LoadFile(file_path);

    // Pipeline options
    // Debayer Params
    {
      std::cout << "Loading debayer params" << std::endl;
      bool enabled = utils::get(node["debayer"], "enabled", true);
      std::string encoding = utils::get<std::string>(node["debayer"], "encoding", "auto");
      debayer_.enable(enabled);
      debayer_.setEncoding(encoding);
    }

    // Flip params
    {
      std::cout << "Loading flip params" << std::endl;
      bool enabled = utils::get(node["flip"], "enabled", false);
      flipper_.enable(enabled);
    }

    // White balance params
    {
      std::cout << "Loading white_balance params" << std::endl;
      bool enabled = utils::get(node["white_balance"], "enabled", false);
      std::string method = utils::get<std::string>(node["white_balance"], "method", "ccc");
      double clipping_percentile = utils::get(node["white_balance"], "clipping_percentile", 20.0);
      double saturation_bright_thr = utils::get(node["white_balance"], "saturation_bright_thr", 0.8);
      double saturation_dark_thr = utils::get(node["white_balance"], "saturation_dark_thr", 0.1);
      bool temporal_consistency = utils::get(node["white_balance"], "temporal_consistency", true);

      white_balancer_.enable(enabled);
      white_balancer_.setMethod(method);
      white_balancer_.setSaturationPercentile(clipping_percentile);
      white_balancer_.setSaturationThreshold(saturation_bright_thr, saturation_dark_thr);
      white_balancer_.setTemporalConsistency(temporal_consistency);
    }

    // Color calibration
    {
      std::cout << "Loading color_calibration params" << std::endl;
      bool enabled = utils::get(node["color_calibration"], "enabled", false);
      color_calibrator_.enable(enabled);
    }

    // Gamma correction params
    {
      std::cout << "Loading gamma_correction params" << std::endl;
      bool enabled = utils::get(node["gamma_correction"], "enabled", false);
      std::string method = utils::get<std::string>(node["gamma_correction"], "method", "custom");
      double k = utils::get(node["gamma_correction"], "k", 0.8);

      gamma_corrector_.enable(enabled);
      gamma_corrector_.setMethod(method);
      gamma_corrector_.setK(k);
    }

    // Vignetting correction
    {
      std::cout << "Loading vignetting_correction params" << std::endl;
      bool enabled = utils::get(node["vignetting_correction"], "enabled", false);
      double scale = utils::get(node["vignetting_correction"], "scale", 1.5);
      double a2 = utils::get(node["vignetting_correction"], "a2", 1e-3);
      double a4 = utils::get(node["vignetting_correction"], "a4", 1e-6);

      vignetting_corrector_.enable(enabled);
      vignetting_corrector_.setParameters(scale, a2, a4);
    }

    // Color enhancer
    {
      std::cout << "Loading color_enhancer params" << std::endl;
      bool enabled = utils::get(node["color_enhancer"], "run_color_enhancer", false);
      double hue_gain = utils::get(node["color_enhancer"], "hue_gain", 1.0);
      double saturation_gain = utils::get(node["color_enhancer"], "saturation_gain", 1.0);
      double value_gain = utils::get(node["color_enhancer"], "value_gain", 1.0);

      color_enhancer_.enable(enabled);
      color_enhancer_.setHueGain(hue_gain);
      color_enhancer_.setHueGain(saturation_gain);
      color_enhancer_.setHueGain(value_gain);
    }

    // Undistortion
    {
      std::cout << "Loading undistortion params" << std::endl;
      bool enabled = utils::get(node["undistortion"], "enabled", false);
      double balance = utils::get(node["undistortion"], "balance", 0.0);
      double fov_scale = utils::get(node["undistortion"], "fov_scale", 1.0);

      undistorter_.enable(enabled);
      undistorter_.setBalance(balance);
      undistorter_.setFovScale(fov_scale);
    }

  } else {
    std::cout << "Warning: parameters file doesn't exist" << std::endl;
  }
}

//-----------------------------------------------------------------------------
// Main interfaces
//-----------------------------------------------------------------------------
void ImageProcCuda::loadCameraCalibration(const std::string& file_path) {
  undistorter_.loadCalibration(file_path);
}

void ImageProcCuda::loadColorCalibration(const std::string& file_path) {
  color_calibrator_.loadCalibration(file_path);
}

void ImageProcCuda::initUndistortion() {
  undistorter_.init();
}

cv::Mat ImageProcCuda::process(const cv::Mat& image, std::string& encoding) {
  cv::Mat out = image.clone();
  // Apply pipeline
  apply(out, encoding);
  // Return copy
  return out.clone();
}

bool ImageProcCuda::apply(cv::Mat& image, std::string& encoding) {
  if (use_gpu_) {
#ifdef HAS_CUDA
    cv::cuda::GpuMat image_d;
    image_d.upload(image);
    pipeline(image_d, encoding);
    image_d.download(image);
#else
    throw std::invalid_argument("use_gpu=true but image_proc_cuda was not compiled with CUDA support");
#endif
  } else {
    pipeline(image, encoding);
  }

  // Increase counter
  idx_++;

  return true;
}

//-----------------------------------------------------------------------------
// Misc interfaces
//-----------------------------------------------------------------------------
void ImageProcCuda::setGpu(bool use_gpu) {
  use_gpu_ = use_gpu;
}

void ImageProcCuda::resetWhiteBalanceTemporalConsistency() {
  white_balancer_.resetTemporalConsistency();
}

cv::Mat ImageProcCuda::getDistImage() const {
  return undistorter_.getDistImage();
}

cv::Mat ImageProcCuda::getRectMask() const {
  return undistorter_.getRectMask();
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Debayer
//-----------------------------------------------------------------------------
void ImageProcCuda::setDebayer(bool enabled) {
  debayer_.enable(enabled);
}

void ImageProcCuda::setDebayerEncoding(const std::string& encoding) {
  debayer_.setEncoding(encoding);
}

//-----------------------------------------------------------------------------
// Flip
//-----------------------------------------------------------------------------
void ImageProcCuda::setFlip(bool enabled) {
  flipper_.enable(enabled);
}

//-----------------------------------------------------------------------------
// White Balance
//-----------------------------------------------------------------------------
void ImageProcCuda::setWhiteBalance(bool enabled) {
  white_balancer_.enable(enabled);
}

void ImageProcCuda::setWhiteBalanceMethod(const std::string& method) {
  white_balancer_.setMethod(method);
}

void ImageProcCuda::setWhiteBalancePercentile(const double& percentile) {
  white_balancer_.setSaturationPercentile(percentile);
}

void ImageProcCuda::setWhiteBalanceSaturationThreshold(const double& bright_thr, const double& dark_thr) {
  white_balancer_.setSaturationThreshold(bright_thr, dark_thr);
}

void ImageProcCuda::setWhiteBalanceTemporalConsistency(bool enabled) {
  white_balancer_.setTemporalConsistency(enabled);
}

//-----------------------------------------------------------------------------
// Color calibration
//-----------------------------------------------------------------------------
void ImageProcCuda::setColorCalibration(bool enabled) {
  color_calibrator_.enable(enabled);
}

void ImageProcCuda::setColorCalibrationMatrix(const std::vector<double>& color_calibration_matrix) {
  color_calibrator_.setCalibrationMatrix(color_calibration_matrix);
}

cv::Mat ImageProcCuda::getColorCalibrationMatrix() const {
  return color_calibrator_.getCalibrationMatrix();
}

//-----------------------------------------------------------------------------
// Gamma correction
//-----------------------------------------------------------------------------
void ImageProcCuda::setGammaCorrection(bool enabled) {
  gamma_corrector_.enable(enabled);
}

void ImageProcCuda::setGammaCorrectionMethod(const std::string& method) {
  gamma_corrector_.setMethod(method);
}

void ImageProcCuda::setGammaCorrectionK(const double& k) {
  gamma_corrector_.setK(k);
}

//-----------------------------------------------------------------------------
// Vignetting
//-----------------------------------------------------------------------------
void ImageProcCuda::setVignettingCorrection(bool enabled) {
  vignetting_corrector_.enable(enabled);
}

void ImageProcCuda::setVignettingCorrectionParameters(const double& scale, const double& a2, const double& a4) {
  vignetting_corrector_.setParameters(scale, a2, a4);
}

//-----------------------------------------------------------------------------
// Color enhancer
//-----------------------------------------------------------------------------
void ImageProcCuda::setColorEnhancer(bool enabled) {
  color_enhancer_.enable(enabled);
}

void ImageProcCuda::setColorEnhancerHueGain(const double& gain) {
  color_enhancer_.setHueGain(gain);
}

void ImageProcCuda::setColorEnhancerSaturationGain(const double& gain) {
  color_enhancer_.setSaturationGain(gain);
}

void ImageProcCuda::setColorEnhancerValueGain(const double& gain) {
  color_enhancer_.setValueGain(gain);
}

//-----------------------------------------------------------------------------
// Undistortion
//-----------------------------------------------------------------------------
void ImageProcCuda::setUndistortion(bool enabled) {
  undistorter_.enable(enabled);
}

void ImageProcCuda::setUndistortionImageSize(int width, int height) {
  undistorter_.setImageSize(width, height);
}

void ImageProcCuda::setUndistortionNewImageSize(int width, int height) {
  undistorter_.setNewImageSize(width, height);
}

void ImageProcCuda::setUndistortionBalance(double balance) {
  undistorter_.setBalance(balance);
}
void ImageProcCuda::setUndistortionFovScale(double fov_scale) {
  undistorter_.setFovScale(fov_scale);
}

void ImageProcCuda::setUndistortionCameraMatrix(const std::vector<double>& camera_matrix) {
  undistorter_.setCameraMatrix(camera_matrix);
}

void ImageProcCuda::setUndistortionDistortionCoefficients(const std::vector<double>& coefficients) {
  undistorter_.setDistortionCoefficients(coefficients);
}

void ImageProcCuda::setUndistortionDistortionModel(const std::string& model) {
  undistorter_.setDistortionModel(model);
}

void ImageProcCuda::setUndistortionRectificationMatrix(const std::vector<double>& rectification_matrix) {
  undistorter_.setRectificationMatrix(rectification_matrix);
}

void ImageProcCuda::setUndistortionProjectionMatrix(const std::vector<double>& projection_matrix) {
  undistorter_.setProjectionMatrix(projection_matrix);
}

//-----------------------------------------------------------------------------
// Undistortion getters
//-----------------------------------------------------------------------------
bool ImageProcCuda::isDebayerEnabled() const {
  return debayer_.enabled();
}

bool ImageProcCuda::isFlipEnabled() const {
  return flipper_.enabled();
}

bool ImageProcCuda::isWhiteBalanceEnabled() const {
  return white_balancer_.enabled();
}

bool ImageProcCuda::isColorCalibrationEnabled() const {
  return color_calibrator_.enabled();
}

bool ImageProcCuda::isGammaCorrectionEnabled() const {
  return gamma_corrector_.enabled();
}

bool ImageProcCuda::isVignettingCorrectionEnabled() const {
  return vignetting_corrector_.enabled();
}

bool ImageProcCuda::isColorEnhancerEnabled() const {
  return color_enhancer_.enabled();
}

bool ImageProcCuda::isUndistortionEnabled() const {
  return undistorter_.enabled();
}

//-----------------------------------------------------------------------------
// Undistortion getters
//-----------------------------------------------------------------------------

int ImageProcCuda::getRectImageHeight() const {
  return undistorter_.getRectImageHeight();
}

int ImageProcCuda::getRectImageWidth() const {
  return undistorter_.getRectImageWidth();
}

int ImageProcCuda::getDistImageHeight() const {
  return undistorter_.getDistImageHeight();
}

int ImageProcCuda::getDistImageWidth() const {
  return undistorter_.getDistImageWidth();
}

std::string ImageProcCuda::getRectDistortionModel() const {
  return undistorter_.getRectDistortionModel();
}
std::string ImageProcCuda::getDistDistortionModel() const {
  return undistorter_.getDistDistortionModel();
}

cv::Mat ImageProcCuda::getRectCameraMatrix() const {
  return undistorter_.getRectCameraMatrix();
}

cv::Mat ImageProcCuda::getDistCameraMatrix() const {
  return undistorter_.getDistCameraMatrix();
}

cv::Mat ImageProcCuda::getRectDistortionCoefficients() const {
  return undistorter_.getRectDistortionCoefficients();
}

cv::Mat ImageProcCuda::getDistDistortionCoefficients() const {
  return undistorter_.getDistDistortionCoefficients();
}

cv::Mat ImageProcCuda::getRectRectificationMatrix() const {
  return undistorter_.getRectRectificationMatrix();
}

cv::Mat ImageProcCuda::getDistRectificationMatrix() const {
  return undistorter_.getDistRectificationMatrix();
}

cv::Mat ImageProcCuda::getRectProjectionMatrix() const {
  return undistorter_.getRectProjectionMatrix();
}

cv::Mat ImageProcCuda::getDistProjectionMatrix() const {
  return undistorter_.getDistProjectionMatrix();
}

// void ImageProcCuda::dumpGpuImage(const std::string& name, const cv::cuda::GpuMat& image) {
//   if (dump_images_) {
//     cv::Mat tmp;
//     image.download(tmp);
//     cv::normalize(tmp, tmp, 0, 255.0, cv::NORM_MINMAX);
//     cv::imwrite("/tmp/" + std::to_string(idx_) + "_" + name + ".png", tmp);
//   }
// }

}  // namespace image_proc_cuda