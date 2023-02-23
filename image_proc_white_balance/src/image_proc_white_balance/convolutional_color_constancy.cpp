/**
@brief image_proc_white_balance.cpp
Class for White Balance built upon "AutoWhiteBalance" code by Shane Yuan
based on Barron, "Fast Fourier Color Constancy", CVPR, 2017

Author: Matias Mattamala
*/

#include <image_proc_white_balance/convolutional_color_constancy.hpp>

#include <boost/filesystem.hpp>
#define FILE_FOLDER (boost::filesystem::path(__FILE__).parent_path().string())
#define DEFAULT_MODEL_PATH (FILE_FOLDER + "/../../model/default.bin")

namespace image_proc_white_balance {
ConvolutionalColorConstancyWB::ConvolutionalColorConstancyWB()
    : ConvolutionalColorConstancyWB::ConvolutionalColorConstancyWB(DEFAULT_MODEL_PATH) {}

ConvolutionalColorConstancyWB::ConvolutionalColorConstancyWB(const std::string& filename)
    : model_filename_(filename),
      small_size_(360, 270),
      bin_size_(1.0f / 64.0f),
      uv0_(-1.421875),
      saturation_thr_(0.9),
      use_temporal_consistency_(false),
      debug_(false),
      debug_uv_offset_(false),
      Lu_debug_(0.f),
      Lv_debug_(0.f),
      first_frame_(true),
      idx_(0) {
  loadModel(model_filename_);
}

ConvolutionalColorConstancyWB::~ConvolutionalColorConstancyWB() {}

void ConvolutionalColorConstancyWB::balanceWhite(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst) {
  // Copy input to output
  src.copyTo(dst);

  cv::Mat src_cpu;
  src.download(src_cpu);

  // Resize image
  cv::Mat small_image;
  cv::resize(src_cpu, small_image, small_size_);

  // Convert to floating point
  cv::Mat small_image_f;
  small_image.convertTo(small_image_f, CV_32F);

  // Compute histogram feature
  calculateHistogramFeature(small_image_f);

  // Compute response using the histogram
  computeResponse();

  // Track to keep temporal consistency
  if (use_temporal_consistency_) {
    kalmanFiltering();
  }

  // Apply gains for white balance
  applyWhiteBalance(dst);

  // if (debug_)
  // {
  //     cv::Mat dst_f;
  //     dst.convertTo(dst_f, CV_32F);
  //     calculateHistogramFeature(dst_f, "wb");

  //     cv::Mat src_vis;
  //     cv::normalize(src_cpu, src_vis, 0, 255.0, cv::NORM_MINMAX);
  //     cv::imwrite("/tmp/image_proc_white_balance_src" + std::to_string(idx_) + ".png", src_vis);

  //     cv::Mat dst_vis;
  //     cv::normalize(dst, dst_vis, 0, 255.0, cv::NORM_MINMAX);
  //     cv::imwrite("/tmp/image_proc_white_balance_dst" + std::to_string(idx_) + ".png", dst_vis);
  // }

  idx_++;
}

// Loads the model and initializes variables
int ConvolutionalColorConstancyWB::loadModel(const std::string& model_file) {
  std::cout << "Loading model file " << model_file << std::endl;

  std::fstream fs(model_file, std::ios::in | std::ios::binary);
  // read size
  fs.read(reinterpret_cast<char*>(&model_.width_), sizeof(int));
  fs.read(reinterpret_cast<char*>(&model_.height_), sizeof(int));

  // Read filter
  model_.filter_.create(model_.height_, model_.width_, CV_32F);
  model_.bias_.create(model_.height_, model_.width_, CV_32F);
  fs.read(reinterpret_cast<char*>(model_.filter_.data), sizeof(float) * model_.width_ * model_.height_);
  fs.read(reinterpret_cast<char*>(model_.bias_.data), sizeof(float) * model_.width_ * model_.height_);
  fs.close();
  model_.filter_ = model_.filter_.t();
  model_.bias_ = model_.bias_.t();

  if (debug_) {
    cv::Mat filter_vis;
    cv::normalize(model_.filter_, filter_vis, 0, 255.0, cv::NORM_MINMAX);
    cv::imwrite("/tmp/image_proc_white_balance_filter" + std::to_string(idx_) + ".png", filter_vis);

    cv::Mat bias_vis;
    cv::normalize(model_.bias_, bias_vis, 0, 255.0, cv::NORM_MINMAX);
    cv::imwrite("/tmp/image_proc_white_balance_bias" + std::to_string(idx_) + ".png", bias_vis);
  }

  // upload to GPU
  filter_d_.upload(model_.filter_);
  bias_d_.upload(model_.bias_);

  // Preallocate FFT variables
  filter_fft_.create(model_.height_, model_.width_, CV_32FC2);
  bias_fft_.create(model_.height_, model_.width_, CV_32FC2);
  hist_fft_.create(model_.height_, model_.width_, CV_32FC2);
  response_fft_.create(model_.height_, model_.width_, CV_32FC2);
  response_.create(model_.height_, model_.width_, CV_32FC2);
  cv::cuda::dft(filter_d_, filter_fft_, cv::Size(model_.width_, model_.height_));
  cv::cuda::dft(bias_d_, bias_fft_, cv::Size(model_.width_, model_.height_));

  // Preallocate histograms
  image_histogram_.create(model_.height_, model_.width_, CV_32F);
  image_histogram_.setTo(cv::Scalar(0));
  hist_d_.create(model_.width_, model_.height_, CV_32F);
  hist_d_.setTo(cv::Scalar(0.f));

  // Computed uv coordinates
  uv_pos_ = cv::Point(model_.height_ / 2, model_.width_ / 2);

  // Kalman filter
  kf_ptr_ = std::make_shared<cv::KalmanFilter>(2, 2, 0, CV_32F);

  // Prior
  // Initialize estimate prior
  kf_ptr_->statePre.at<float>(0) = uv_pos_.x;
  kf_ptr_->statePre.at<float>(1) = uv_pos_.y;
  kf_ptr_->statePost.at<float>(0) = uv_pos_.x;
  kf_ptr_->statePost.at<float>(1) = uv_pos_.y;

  // Measurement
  kf_measurement_ = cv::Mat::zeros(2, 1, CV_32F);

  // Process model
  // Initialize transition matrix
  kf_ptr_->transitionMatrix = (cv::Mat_<float>(2, 2) << 1.f, 0, 0, 1.f);
  // Init process noise covariance
  kf_ptr_->processNoiseCov = (cv::Mat_<float>(2, 2) << 1.f, 0, 0, 1.f);
  // Measurement model
  // Init measurement matrix (H)
  kf_H_ = (cv::Mat_<float>(2, 2) << 1.f, 0, 0, 1.f);
  kf_ptr_->measurementMatrix = kf_H_;
  kf_ptr_->measurementNoiseCov = (cv::Mat_<float>(2, 2) << 10.f, 0, 0, 10.f);
  // Preallocate inverse measurement covariance
  kf_inv_H_cov_ = kf_ptr_->measurementNoiseCov.inv();

  return 0;
}

// Calculates histogram in CPU
void ConvolutionalColorConstancyWB::calculateHistogramFeature(const cv::Mat& src, std::string out_name) {
  // Make a mask to remove saturated pixels (bright and dark ones)
  cv::Mat gray;
  cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  cv::Mat upper_mask, lower_mask;
  cv::threshold(gray, upper_mask, 255 * saturation_thr_, 255,
                1);  // inverted threshold, removes large values
  cv::threshold(gray, lower_mask, 255 * (1 - saturation_thr_), 255,
                0);  // normal threshold, removes small values
  cv::Mat mask = upper_mask & lower_mask;

  if (debug_) {
    cv::imwrite("/tmp/image_proc_white_balance_lower_mask_" + out_name + std::to_string(idx_) + ".png", lower_mask);
    cv::imwrite("/tmp/image_proc_white_balance_upper_mask_" + out_name + std::to_string(idx_) + ".png", upper_mask);
    cv::imwrite("/tmp/image_proc_white_balance_mask_" + out_name + std::to_string(idx_) + ".png", mask);
  }

  // Apply logarithm to the whole image to get log-chroma
  cv::log(src, src);

  // Reset image histogram
  image_histogram_.setTo(cv::Scalar(0.f));

  size_t num_samples = 0;
  int u, v;
  float num_pixels = src.rows * src.cols;
  float pixel_weight = 1.0f / num_pixels;
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      cv::Point3f bgr = src.at<cv::Point3f>(i, j);
      const float& log_r = bgr.z;
      const float& log_g = bgr.y;
      const float& log_b = bgr.x;

      if (!std::isfinite(log_r) || !std::isfinite(log_g) || !std::isfinite(log_b)) {
        continue;
      }
      if (mask.at<float>(i, j) < 1.f) {
        // std::cout << "skipping pixel (" << i << ", " << j << ")" << std::endl;
        continue;
      }

      // Compute histogram bin coordinates
      u = round((log_g - log_r - uv0_) / bin_size_);
      v = round((log_g - log_b - uv0_) / bin_size_);

      // Ensure it stays within the histogram bins
      u = std::max<int>(std::min<int>(u, 255), 0);
      v = std::max<int>(std::min<int>(v, 255), 0);

      image_histogram_.at<float>(u, v) += pixel_weight;
      num_samples++;
    }
  }

  if (debug_) {
    cv::Mat hist_vis;
    cv::normalize(image_histogram_, hist_vis, 0, 255.0, cv::NORM_MINMAX);
    cv::imwrite("/tmp/image_proc_white_balance_histogram_" + out_name + std::to_string(idx_) + ".png", hist_vis);
  }
}

int ConvolutionalColorConstancyWB::computeResponse() {
  // Save previous uv point
  uv_pos_prev_ = uv_pos_;

  // upload to gpu
  hist_d_.upload(image_histogram_);

  // Reset
  hist_fft_.setTo(cv::Scalar(0.f));
  response_fft_.setTo(cv::Scalar(0.f));
  response_.setTo(cv::Scalar(0.f));

  // compute FFT of the histogram
  cv::cuda::dft(hist_d_, hist_fft_, cv::Size(model_.width_, model_.height_));

  // Convolve filter and histogram to get the response (FFT multiplication)
  cv::cuda::mulSpectrums(filter_fft_, hist_fft_, response_fft_, 0);

  // Add bias to the response
  cv::cuda::add(response_fft_, bias_fft_, response_fft_);

  // Compute inverse FFT
  cv::cuda::dft(response_fft_, response_, cv::Size(model_.width_, model_.height_), cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT);

  // Get max position to get illuminants
  cv::cuda::minMaxLoc(response_, NULL, NULL, NULL, &uv_pos_);

  if (debug_) {
    cv::Mat response;
    response_.download(response);
    cv::Mat response_vis;
    cv::normalize(response, response_vis, 0, 255.0, cv::NORM_MINMAX);
    cv::imwrite("/tmp/image_proc_white_balance_response" + std::to_string(idx_) + ".png", response_vis);
  }
  return 0;
}

int ConvolutionalColorConstancyWB::kalmanFiltering() {
  if (first_frame_) {
    first_frame_ = false;

    // Set the predicted state to the computed position
    kf_ptr_->statePost.at<float>(0) = uv_pos_.x;
    kf_ptr_->statePost.at<float>(1) = uv_pos_.y;
  } else {
    // Predict step:
    cv::Mat pred = kf_ptr_->predict();
    cv::Point pred_point(pred.at<float>(0), pred.at<float>(1));

    // Update step
    kf_measurement_.at<float>(0) = uv_pos_.x;
    kf_measurement_.at<float>(1) = uv_pos_.y;

    // Outlier detection
    constexpr double chi2_2dof_001 = 11.345;
    cv::Mat innovation = kf_measurement_ - kf_H_ * pred;
    double chi2 = innovation.dot(kf_inv_H_cov_ * innovation);
    // std::cout << "kf_measurement_" << kf_measurement_ << std::endl;
    // std::cout << "pred" << pred << std::endl;
    // std::cout << "kf_H_" << kf_H_ << std::endl;
    // std::cout << "chi2: " << chi2 << std::endl;

    // if (chi2 < chi2_2dof_001)
    // {
    kf_estimate_ = kf_ptr_->correct(kf_measurement_);
    // }
    // else
    // {
    //     kf_estimate_ = pred;
    // }
    cv::Point estimated_point(kf_estimate_.at<float>(0), kf_estimate_.at<float>(1));

    // Update UV
    uv_pos_.x = kf_estimate_.at<float>(0);
    uv_pos_.y = kf_estimate_.at<float>(1);
  }
  return 0;
}

void ConvolutionalColorConstancyWB::applyWhiteBalance(cv::cuda::GpuMat& image) {
  // Convert position in UV space to luminance gain
  float Lu, Lv, z;
  float u_pos_debug = 0.f;
  float v_pos_debug = 0.f;
  // std::cout << "(u,v)   = (" << uv_pos_.x << ", " << uv_pos_.y << ")" << std::endl;

  if (debug_uv_offset_) {
    Lu = Lu_debug_ - uv0_;
    Lv = Lv_debug_ - uv0_;
    u_pos_debug = (Lu) / bin_size_;
    v_pos_debug = (Lv) / bin_size_;
    // std::cout << "Lu: " << Lu << ", std::exp(-2 * Lu): " << std::exp(-2 * Lu) << std::endl;
    // std::cout << "Lv: " << Lv << ", std::exp(-2 * Lv): " << std::exp(-2 * Lv) << std::endl;
    // std::cout << "std::exp(-2 * Lu) + std::exp(-2 * Lv) + 1: " << std::exp(-2 * Lu) +
    // std::exp(-2 * Lv) + 1 << std::endl;
    z = std::sqrt(std::exp(-2 * Lu) + std::exp(-2 * Lv) + 1);
  } else {
    Lu = (uv_pos_.x) * bin_size_ + uv0_;
    Lv = (uv_pos_.y) * bin_size_ + uv0_;
    z = std::sqrt(std::exp(-2 * Lu) + std::exp(-2 * Lv) + 1.0);
  }

  // Compute gains
  // Note: we do not use z, because it satcv::Maturates the image
  float gain_r = 1.0 / std::exp(-Lu);
  float gain_g = 1.0;
  float gain_b = 1.0 / std::exp(-Lv);

  float factor = std::min(std::min(gain_r, gain_g), gain_b);
  gain_r /= factor;
  gain_g /= factor;
  gain_b /= factor;

  Npp32f color_twist[3][4] = {{1.0, 0.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0}, {0.0, 0.0, 1.0, 0.0}};
  bool is_rgb = false;
  if (is_rgb) {
    color_twist[0][0] = gain_r;
    color_twist[1][1] = gain_g;
    color_twist[2][2] = gain_b;
  } else {
    color_twist[0][0] = gain_b;
    color_twist[1][1] = gain_g;
    color_twist[2][2] = gain_r;
  }

  NppiSize image_size;
  image_size.width = image.cols;
  image_size.height = image.rows;
  nppiColorTwist32f_8u_C3IR(image.data, static_cast<int>(image.step), image_size, color_twist);
}

void ConvolutionalColorConstancyWB::resetTemporalConsistency() {
  first_frame_ = true;
}

void ConvolutionalColorConstancyWB::setSaturationThreshold(float threshold) {
  saturation_thr_ = threshold;
}

void ConvolutionalColorConstancyWB::setTemporalConsistency(bool enable) {
  use_temporal_consistency_ = enable;
}

void ConvolutionalColorConstancyWB::setDebug(bool debug) {
  debug_ = debug;
}

void ConvolutionalColorConstancyWB::setUV0(float uv0) {
  uv0_ = uv0;
}

void ConvolutionalColorConstancyWB::setDebugUVOffset(float Lu, float Lv, float uv0) {
  Lu_debug_ = Lu;
  Lv_debug_ = Lv;
  uv0_ = uv0;
  debug_uv_offset_ = true;
}

}  // namespace image_proc_white_balance