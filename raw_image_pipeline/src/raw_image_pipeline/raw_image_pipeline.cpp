//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <raw_image_pipeline/raw_image_pipeline.hpp>

#include <boost/filesystem.hpp>
#define FILE_FOLDER (boost::filesystem::path(__FILE__).parent_path().string())
#define DEFAULT_PARAMS_PATH (FILE_FOLDER + "/../../config/pipeline_params_example.yaml")
#define DEFAULT_CALIBRATION_PATH (FILE_FOLDER + "/../../config/alphasense_calib_example.yaml")
#define DEFAULT_COLOR_CALIBRATION_PATH (FILE_FOLDER + "/../../config/alphasense_color_calib_example.yaml")

namespace raw_image_pipeline {

RawImagePipeline::RawImagePipeline(bool use_gpu) : use_gpu_(use_gpu), debug_(false) {
  // Load parameters
  loadParams(DEFAULT_PARAMS_PATH);
  undistorter_->loadCalibration(DEFAULT_CALIBRATION_PATH);
  color_calibrator_->loadCalibration(DEFAULT_COLOR_CALIBRATION_PATH);
}

RawImagePipeline::RawImagePipeline(bool use_gpu, const std::string& params_path, const std::string& calibration_path,
                                   const std::string& color_calibration_path)
    : use_gpu_(use_gpu) {
  // Load parameters
  if (params_path.empty())
    loadParams(DEFAULT_PARAMS_PATH);
  else
    loadParams(params_path);

  // Load calibration
  if (!calibration_path.empty()) undistorter_->loadCalibration(calibration_path);

  // Load color calibration
  if (color_calibration_path.empty())
    color_calibrator_->loadCalibration(DEFAULT_COLOR_CALIBRATION_PATH);
  else
    color_calibrator_->loadCalibration(color_calibration_path);
}

RawImagePipeline::~RawImagePipeline() {}

void RawImagePipeline::loadParams(const std::string& file_path) {
  std::cout << "Loading raw_image_pipeline params from file " << file_path << std::endl;

  // Check if file exists
  if (boost::filesystem::exists(file_path)) {
    // Load parameters
    YAML::Node node = YAML::LoadFile(file_path);

    // Pipeline options
    // Debayer Params
    {
      std::cout << "Loading debayer params" << std::endl;
      debayer_ = std::make_unique<DebayerModule>(use_gpu_);

      bool enabled = utils::get(node["debayer"], "enabled", true);
      std::string encoding = utils::get<std::string>(node["debayer"], "encoding", "auto");
      debayer_->enable(enabled);
      debayer_->setEncoding(encoding);
    }

    // Flip params
    {
      std::cout << "Loading flip params" << std::endl;
      flipper_ = std::make_unique<FlipModule>(use_gpu_);

      bool enabled = utils::get(node["flip"], "enabled", false);
      int angle = utils::get(node["flip"], "angle", 0);

      flipper_->enable(enabled);
      flipper_->setAngle(angle);
    }

    // White balance params
    {
      std::cout << "Loading white_balance params" << std::endl;
      white_balancer_ = std::make_unique<WhiteBalanceModule>(use_gpu_);

      bool enabled = utils::get(node["white_balance"], "enabled", false);
      std::string method = utils::get<std::string>(node["white_balance"], "method", "ccc");
      double clipping_percentile = utils::get(node["white_balance"], "clipping_percentile", 20.0);
      double saturation_bright_thr = utils::get(node["white_balance"], "saturation_bright_thr", 0.8);
      double saturation_dark_thr = utils::get(node["white_balance"], "saturation_dark_thr", 0.1);
      bool temporal_consistency = utils::get(node["white_balance"], "temporal_consistency", true);

      white_balancer_->enable(enabled);
      white_balancer_->setMethod(method);
      white_balancer_->setSaturationPercentile(clipping_percentile);
      white_balancer_->setSaturationThreshold(saturation_bright_thr, saturation_dark_thr);
      white_balancer_->setTemporalConsistency(temporal_consistency);
    }

    // Color calibration
    {
      std::cout << "Loading color_calibration params" << std::endl;
      color_calibrator_ = std::make_unique<ColorCalibrationModule>(use_gpu_);

      bool enabled = utils::get(node["color_calibration"], "enabled", false);
      color_calibrator_->enable(enabled);
    }

    // Gamma correction params
    {
      std::cout << "Loading gamma_correction params" << std::endl;
      gamma_corrector_ = std::make_unique<GammaCorrectionModule>(use_gpu_);

      bool enabled = utils::get(node["gamma_correction"], "enabled", false);
      std::string method = utils::get<std::string>(node["gamma_correction"], "method", "custom");
      double k = utils::get(node["gamma_correction"], "k", 0.8);

      gamma_corrector_->enable(enabled);
      gamma_corrector_->setMethod(method);
      gamma_corrector_->setK(k);
    }

    // Vignetting correction
    {
      std::cout << "Loading vignetting_correction params" << std::endl;
      vignetting_corrector_ = std::make_unique<VignettingCorrectionModule>(use_gpu_);

      bool enabled = utils::get(node["vignetting_correction"], "enabled", false);
      double scale = utils::get(node["vignetting_correction"], "scale", 1.5);
      double a2 = utils::get(node["vignetting_correction"], "a2", 1e-3);
      double a4 = utils::get(node["vignetting_correction"], "a4", 1e-6);

      vignetting_corrector_->enable(enabled);
      vignetting_corrector_->setParameters(scale, a2, a4);
    }

    // Color enhancer
    {
      std::cout << "Loading color_enhancer params" << std::endl;
      color_enhancer_ = std::make_unique<ColorEnhancerModule>(use_gpu_);

      bool enabled = utils::get(node["color_enhancer"], "run_color_enhancer", false);
      double hue_gain = utils::get(node["color_enhancer"], "hue_gain", 1.0);
      double saturation_gain = utils::get(node["color_enhancer"], "saturation_gain", 1.0);
      double value_gain = utils::get(node["color_enhancer"], "value_gain", 1.0);

      color_enhancer_->enable(enabled);
      color_enhancer_->setHueGain(hue_gain);
      color_enhancer_->setHueGain(saturation_gain);
      color_enhancer_->setHueGain(value_gain);
    }

    // Undistortion
    {
      std::cout << "Loading undistortion params" << std::endl;
      undistorter_ = std::make_unique<UndistortionModule>(use_gpu_);

      bool enabled = utils::get(node["undistortion"], "enabled", false);
      double balance = utils::get(node["undistortion"], "balance", 0.0);
      double fov_scale = utils::get(node["undistortion"], "fov_scale", 1.0);

      undistorter_->enable(enabled);
      undistorter_->setBalance(balance);
      undistorter_->setFovScale(fov_scale);
    }

  } else {
    std::cout << "Warning: parameters file doesn't exist" << std::endl;
  }
}

//-----------------------------------------------------------------------------
// Main interfaces
//-----------------------------------------------------------------------------
void RawImagePipeline::loadCameraCalibration(const std::string& file_path) {
  undistorter_->loadCalibration(file_path);
}

void RawImagePipeline::loadColorCalibration(const std::string& file_path) {
  color_calibrator_->loadCalibration(file_path);
}

void RawImagePipeline::initUndistortion() {
  undistorter_->init();
}

cv::Mat RawImagePipeline::process(const cv::Mat& image, std::string& encoding) {
  cv::Mat out = image.clone();
  // Apply pipeline
  apply(out, encoding);
  // Return copy
  return out.clone();
}

bool RawImagePipeline::apply(cv::Mat& image, std::string& encoding) {
  if (use_gpu_) {
#ifdef HAS_CUDA
    cv::cuda::GpuMat image_d;
    image_d.upload(image);
    pipeline(image_d, encoding);
    image_d.download(image);
#else
    throw std::invalid_argument("use_gpu=true but raw_image_pipeline was not compiled with CUDA support");
#endif
  } else {
    pipeline(image, encoding);
  }

  return true;
}

//-----------------------------------------------------------------------------
// Misc interfaces
//-----------------------------------------------------------------------------
void RawImagePipeline::setGpu(bool use_gpu) {
  use_gpu_ = use_gpu;
}

void RawImagePipeline::setDebug(bool debug) {
  debug_ = debug;
}

void RawImagePipeline::resetWhiteBalanceTemporalConsistency() {
  white_balancer_->resetTemporalConsistency();
}

cv::Mat RawImagePipeline::getDistDebayeredImage() const {
  return flipper_->getImage();
}

cv::Mat RawImagePipeline::getDistColorImage() const {
  return undistorter_->getDistImage();
}

cv::Mat RawImagePipeline::getRectMask() const {
  return undistorter_->getRectMask();
}

cv::Mat RawImagePipeline::getProcessedImage() const {
  return image_.clone();
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Debayer
//-----------------------------------------------------------------------------
void RawImagePipeline::setDebayer(bool enabled) {
  debayer_->enable(enabled);
}

void RawImagePipeline::setDebayerEncoding(const std::string& encoding) {
  debayer_->setEncoding(encoding);
}

//-----------------------------------------------------------------------------
// Flip
//-----------------------------------------------------------------------------
void RawImagePipeline::setFlip(bool enabled) {
  flipper_->enable(enabled);
}

void RawImagePipeline::setFlipAngle(int angle) {
  flipper_->setAngle(angle);
}

//-----------------------------------------------------------------------------
// White Balance
//-----------------------------------------------------------------------------
void RawImagePipeline::setWhiteBalance(bool enabled) {
  white_balancer_->enable(enabled);
}

void RawImagePipeline::setWhiteBalanceMethod(const std::string& method) {
  white_balancer_->setMethod(method);
}

void RawImagePipeline::setWhiteBalancePercentile(const double& percentile) {
  white_balancer_->setSaturationPercentile(percentile);
}

void RawImagePipeline::setWhiteBalanceSaturationThreshold(const double& bright_thr, const double& dark_thr) {
  white_balancer_->setSaturationThreshold(bright_thr, dark_thr);
}

void RawImagePipeline::setWhiteBalanceTemporalConsistency(bool enabled) {
  white_balancer_->setTemporalConsistency(enabled);
}

//-----------------------------------------------------------------------------
// Color calibration
//-----------------------------------------------------------------------------
void RawImagePipeline::setColorCalibration(bool enabled) {
  color_calibrator_->enable(enabled);
}

void RawImagePipeline::setColorCalibrationMatrix(const std::vector<double>& color_calibration_matrix) {
  color_calibrator_->setCalibrationMatrix(color_calibration_matrix);
}

void RawImagePipeline::setColorCalibrationBias(const std::vector<double>& color_calibration_bias) {
  color_calibrator_->setCalibrationBias(color_calibration_bias);
}

cv::Mat RawImagePipeline::getColorCalibrationMatrix() const {
  return color_calibrator_->getCalibrationMatrix();
}

cv::Mat RawImagePipeline::getColorCalibrationBias() const {
  return color_calibrator_->getCalibrationBias();
}

//-----------------------------------------------------------------------------
// Gamma correction
//-----------------------------------------------------------------------------
void RawImagePipeline::setGammaCorrection(bool enabled) {
  gamma_corrector_->enable(enabled);
}

void RawImagePipeline::setGammaCorrectionMethod(const std::string& method) {
  gamma_corrector_->setMethod(method);
}

void RawImagePipeline::setGammaCorrectionK(const double& k) {
  gamma_corrector_->setK(k);
}

//-----------------------------------------------------------------------------
// Vignetting
//-----------------------------------------------------------------------------
void RawImagePipeline::setVignettingCorrection(bool enabled) {
  vignetting_corrector_->enable(enabled);
}

void RawImagePipeline::setVignettingCorrectionParameters(const double& scale, const double& a2, const double& a4) {
  vignetting_corrector_->setParameters(scale, a2, a4);
}

//-----------------------------------------------------------------------------
// Color enhancer
//-----------------------------------------------------------------------------
void RawImagePipeline::setColorEnhancer(bool enabled) {
  color_enhancer_->enable(enabled);
}

void RawImagePipeline::setColorEnhancerHueGain(const double& gain) {
  color_enhancer_->setHueGain(gain);
}

void RawImagePipeline::setColorEnhancerSaturationGain(const double& gain) {
  color_enhancer_->setSaturationGain(gain);
}

void RawImagePipeline::setColorEnhancerValueGain(const double& gain) {
  color_enhancer_->setValueGain(gain);
}

//-----------------------------------------------------------------------------
// Undistortion
//-----------------------------------------------------------------------------
void RawImagePipeline::setUndistortion(bool enabled) {
  undistorter_->enable(enabled);
}

void RawImagePipeline::setUndistortionImageSize(int width, int height) {
  undistorter_->setImageSize(width, height);
}

void RawImagePipeline::setUndistortionNewImageSize(int width, int height) {
  undistorter_->setNewImageSize(width, height);
}

void RawImagePipeline::setUndistortionBalance(double balance) {
  undistorter_->setBalance(balance);
}
void RawImagePipeline::setUndistortionFovScale(double fov_scale) {
  undistorter_->setFovScale(fov_scale);
}

void RawImagePipeline::setUndistortionCameraMatrix(const std::vector<double>& camera_matrix) {
  undistorter_->setCameraMatrix(camera_matrix);
}

void RawImagePipeline::setUndistortionDistortionCoefficients(const std::vector<double>& coefficients) {
  undistorter_->setDistortionCoefficients(coefficients);
}

void RawImagePipeline::setUndistortionDistortionModel(const std::string& model) {
  undistorter_->setDistortionModel(model);
}

void RawImagePipeline::setUndistortionRectificationMatrix(const std::vector<double>& rectification_matrix) {
  undistorter_->setRectificationMatrix(rectification_matrix);
}

void RawImagePipeline::setUndistortionProjectionMatrix(const std::vector<double>& projection_matrix) {
  undistorter_->setProjectionMatrix(projection_matrix);
}

//-----------------------------------------------------------------------------
// Undistortion getters
//-----------------------------------------------------------------------------
bool RawImagePipeline::isDebayerEnabled() const {
  return debayer_->enabled();
}

bool RawImagePipeline::isFlipEnabled() const {
  return flipper_->enabled();
}

bool RawImagePipeline::isWhiteBalanceEnabled() const {
  return white_balancer_->enabled();
}

bool RawImagePipeline::isColorCalibrationEnabled() const {
  return color_calibrator_->enabled();
}

bool RawImagePipeline::isGammaCorrectionEnabled() const {
  return gamma_corrector_->enabled();
}

bool RawImagePipeline::isVignettingCorrectionEnabled() const {
  return vignetting_corrector_->enabled();
}

bool RawImagePipeline::isColorEnhancerEnabled() const {
  return color_enhancer_->enabled();
}

bool RawImagePipeline::isUndistortionEnabled() const {
  return undistorter_->enabled();
}

//-----------------------------------------------------------------------------
// Undistortion getters
//-----------------------------------------------------------------------------

int RawImagePipeline::getRectImageHeight() const {
  return undistorter_->getRectImageHeight();
}

int RawImagePipeline::getRectImageWidth() const {
  return undistorter_->getRectImageWidth();
}

int RawImagePipeline::getDistImageHeight() const {
  return undistorter_->getDistImageHeight();
}

int RawImagePipeline::getDistImageWidth() const {
  return undistorter_->getDistImageWidth();
}

std::string RawImagePipeline::getRectDistortionModel() const {
  return undistorter_->getRectDistortionModel();
}
std::string RawImagePipeline::getDistDistortionModel() const {
  return undistorter_->getDistDistortionModel();
}

cv::Mat RawImagePipeline::getRectCameraMatrix() const {
  return undistorter_->getRectCameraMatrix();
}

cv::Mat RawImagePipeline::getDistCameraMatrix() const {
  return undistorter_->getDistCameraMatrix();
}

cv::Mat RawImagePipeline::getRectDistortionCoefficients() const {
  return undistorter_->getRectDistortionCoefficients();
}

cv::Mat RawImagePipeline::getDistDistortionCoefficients() const {
  return undistorter_->getDistDistortionCoefficients();
}

cv::Mat RawImagePipeline::getRectRectificationMatrix() const {
  return undistorter_->getRectRectificationMatrix();
}

cv::Mat RawImagePipeline::getDistRectificationMatrix() const {
  return undistorter_->getDistRectificationMatrix();
}

cv::Mat RawImagePipeline::getRectProjectionMatrix() const {
  return undistorter_->getRectProjectionMatrix();
}

cv::Mat RawImagePipeline::getDistProjectionMatrix() const {
  return undistorter_->getDistProjectionMatrix();
}

}  // namespace raw_image_pipeline