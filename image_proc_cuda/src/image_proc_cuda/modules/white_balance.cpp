#include <image_proc_cuda/modules/white_balance.hpp>

namespace image_proc_cuda {

WhiteBalanceModule::WhiteBalanceModule() : enabled_(true), method_("ccc") {
}

void WhiteBalanceModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool WhiteBalanceModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void WhiteBalanceModule::setMethod(const std::string& method) {
  method_ = method;
  // if (method_ == "ccc") {
  //   cccWBPtr_ = std::make_shared<image_proc_white_balance::ConvolutionalColorConstancyWB>();
  // }
}

void WhiteBalanceModule::setSaturationPercentile(const double& percentile) {
  clipping_percentile_ = percentile;
}

void WhiteBalanceModule::setSaturationThreshold(const double& bright_thr, const double& dark_thr) {
  saturation_bright_thr_ = bright_thr;
  saturation_dark_thr_ = dark_thr;
}

void WhiteBalanceModule::setTemporalConsistency(bool enabled) {
  temporal_consistency_ = enabled;
}

void WhiteBalanceModule::resetTemporalConsistency() {
  if (method_ == "ccc") {
    // cccWBPtr_->resetTemporalConsistency();
    ccc_.resetTemporalConsistency();
  }
}

//-----------------------------------------------------------------------------
// White balance wrapper methods (CPU)
//-----------------------------------------------------------------------------
void WhiteBalanceModule::balanceWhiteSimple(cv::Mat& image) {
  cv::Ptr<cv::xphoto::SimpleWB> wb;
  wb = cv::xphoto::createSimpleWB();
  wb->setP(clipping_percentile_);
  wb->balanceWhite(image, image);
}

void WhiteBalanceModule::balanceWhiteGreyWorld(cv::Mat& image) {
  cv::Ptr<cv::xphoto::GrayworldWB> wb;
  wb = cv::xphoto::createGrayworldWB();
  wb->setSaturationThreshold(saturation_bright_thr_);
  wb->balanceWhite(image, image);
}

void WhiteBalanceModule::balanceWhiteLearned(cv::Mat& image) {
  cv::Ptr<cv::xphoto::LearningBasedWB> wb;
  wb = cv::xphoto::createLearningBasedWB();
  wb->setSaturationThreshold(saturation_bright_thr_);
  wb->balanceWhite(image, image);
}

void WhiteBalanceModule::balanceWhitePca(cv::Mat& image) {
  // Note: BGR input

  // Split channels
  std::vector<cv::Mat> split_img;
  cv::split(image, split_img);
  split_img[0].convertTo(split_img[0], CV_32FC1);
  split_img[2].convertTo(split_img[2], CV_32FC1);

  // Get elementwise squared values
  cv::Mat I_r_2;
  cv::Mat I_b_2;
  cv::multiply(split_img[0], split_img[0], I_b_2);
  cv::multiply(split_img[2], split_img[2], I_r_2);

  // Get summed up channels
  const double sum_I_r_2 = cv::sum(I_r_2)[0];
  const double sum_I_b_2 = cv::sum(I_b_2)[0];
  const double sum_I_g = cv::sum(split_img[1])[0];
  const double sum_I_r = cv::sum(split_img[2])[0];
  const double sum_I_b = cv::sum(split_img[0])[0];

  // Get max values of channels
  double max_I_r, max_I_g, max_I_b, max_I_r_2, max_I_b_2;
  double min_I_r, min_I_g, min_I_b, min_I_r_2, min_I_b_2;
  cv::minMaxLoc(split_img[2], &min_I_r, &max_I_r);  // R
  cv::minMaxLoc(split_img[1], &min_I_g, &max_I_g);  // G
  cv::minMaxLoc(split_img[0], &min_I_b, &max_I_b);  // B
  cv::minMaxLoc(I_r_2, &min_I_r_2, &max_I_r_2);
  cv::minMaxLoc(I_b_2, &min_I_b_2, &max_I_b_2);

  // Prepare Matrices for PCA method
  Eigen::Matrix2f mat_temp_b;
  mat_temp_b << sum_I_b_2, sum_I_b, max_I_b_2, max_I_b;
  Eigen::Matrix2f mat_temp_r;
  mat_temp_r << sum_I_r_2, sum_I_r, max_I_r_2, max_I_r;
  Eigen::Vector2f vec_temp_g;
  vec_temp_g << sum_I_g, max_I_g;

  // PCA method calculation
  Eigen::Vector2f vec_out_b, vec_out_r;
  vec_out_b = mat_temp_b.inverse() * vec_temp_g;
  vec_out_r = mat_temp_r.inverse() * vec_temp_g;
  cv::Mat b_point = vec_out_b[0] * I_b_2 + vec_out_b[1] * split_img[0];
  cv::Mat r_point = vec_out_r[0] * I_r_2 + vec_out_r[1] * split_img[2];

  // Saturate values above 255
  cv::threshold(b_point, b_point, 255, 255, cv::THRESH_TRUNC);
  cv::threshold(r_point, r_point, 255, 255, cv::THRESH_TRUNC);

  // Convert back to UINT8
  b_point.convertTo(b_point, CV_8UC1);
  r_point.convertTo(r_point, CV_8UC1);

  // Merge channels
  std::vector<cv::Mat> channels;
  channels.push_back(b_point);
  channels.push_back(split_img[1]);
  channels.push_back(r_point);
  cv::Mat merged_image;
  cv::merge(channels, merged_image);

  image = merged_image;
}

//-----------------------------------------------------------------------------
// White balance wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void WhiteBalanceModule::balanceWhiteSimple(cv::cuda::GpuMat& image) {
  cv::Mat cpu_image;
  image.download(cpu_image);
  balanceWhiteSimple(cpu_image);
  image.upload(cpu_image);
}

void WhiteBalanceModule::balanceWhiteGreyWorld(cv::cuda::GpuMat& image) {
  cv::Mat cpu_image;
  image.download(cpu_image);
  balanceWhiteGreyWorld(cpu_image);
  image.upload(cpu_image);
}

void WhiteBalanceModule::balanceWhiteLearned(cv::cuda::GpuMat& image) {
  cv::Mat cpu_image;
  image.download(cpu_image);
  balanceWhiteLearned(cpu_image);
  image.upload(cpu_image);
}

void WhiteBalanceModule::balanceWhitePca(cv::cuda::GpuMat& image) {
  cv::Mat cpu_image;
  image.download(cpu_image);
  balanceWhitePca(cpu_image);
  image.upload(cpu_image);
}
#endif

}  // namespace image_proc_cuda