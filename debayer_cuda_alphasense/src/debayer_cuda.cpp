#include "debayer_cuda/debayer_cuda.h"


namespace debayer {

DebayerCuda::DebayerCuda(
    const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private), spinner_(1), image_transport_(nh),
      skipped_images_for_slow_topic_(0) {
      google::InstallFailureSignalHandler();
}

DebayerCuda::~DebayerCuda() {}

bool DebayerCuda::run() {
  ROS_INFO_STREAM("[DebayerCuda] Starting...");
  spinner_.start();
  return true;
}

void DebayerCuda::setupROSparams() {
  if (!nh_private_.getParam("input_topic", input_topic_)) ROS_ERROR_STREAM("could not get input topic!");
  else ROS_INFO_STREAM("input topic: " << input_topic_);
  if (!nh_private_.getParam("output_topic", output_topic_)) ROS_ERROR_STREAM("could not get output topic!");
  else ROS_INFO_STREAM("output topic: " << output_topic_);
  if (!nh_private_.param<std::string>("nppi_debayer_mode", nppi_debayer_mode_, "NPPI_BAYER_GRBG"))
      ROS_WARN_STREAM("could not get nppi debayering mode from ROS-params, defaulting to: " << nppi_debayer_mode_);
  if (!nh_private_.param<bool>("red_blue_swap", red_blue_swap_, false))
      ROS_WARN_STREAM("could not get red_blue_swapping param, defaulting to: " << red_blue_swap_);
  if (!nh_private_.param<std::string>("output_encoding", output_encoding_, "BGR"))
      ROS_WARN_STREAM("could not get output encoding, defaulting to: " << output_encoding_);
  if (!nh_private_.param("skip_number_of_images_for_slow_topic", skip_number_of_images_for_slow_topic_, -1))
      ROS_WARN_STREAM("could not get 'skip_number_of_images_for_slow_topic', defaulting to: " << skip_number_of_images_for_slow_topic_);
  if (!nh_private_.param("needs_rotation", needs_rotation_, false))
      ROS_WARN_STREAM("could not get 'needs_rotation', defaulting to: " << needs_rotation_);
  if (!nh_private_.param("white_balance_clipping_percentile", white_balance_clipping_percentile_, 0.2))
      ROS_WARN_STREAM("could not get 'white_balance_clipping_percentile', defaulting to: " << white_balance_clipping_percentile_);
  if (!nh_private_.param("gamma_correction_k", gamma_correction_k_, 0.8))
      ROS_WARN_STREAM("could not get 'gamma_correction_k', defaulting to: " << gamma_correction_k_);
  if (!nh_private_.param("run_clahe", run_clahe_, false))
      ROS_WARN_STREAM("could not get 'run_clahe', defaulting to: " << run_clahe_);
  if (!nh_private_.param("clahe_clip_limit", clahe_clip_limit_, 1.5))
      ROS_WARN_STREAM("could not get 'clahe_clip_limit', defaulting to: " << clahe_clip_limit_);
  if (!nh_private_.param("clahe_tiles_grid_size", clahe_tiles_grid_size_, 8))
      ROS_WARN_STREAM("could not get 'clahe_tiles_grid_size', defaulting to: " << clahe_tiles_grid_size_);
}

void DebayerCuda::setupSubAndPub() {
  constexpr size_t kRosQueueSize = 20u;
  // Set up the raw image subscriber.
  boost::function<void(const sensor_msgs::ImageConstPtr&)> image_callback =
      boost::bind(&DebayerCuda::imageCallback, this, _1);
  sub_raw_image_ = image_transport_.subscribe(
      input_topic_, kRosQueueSize, boost::bind(& DebayerCuda::imageCallback, this, _1));
  // Set up the processed image publisher.
  pub_color_image_ =
      image_transport_.advertise(output_topic_, kRosQueueSize);
  pub_color_image_slow_ =
      image_transport_.advertise(output_topic_ + "/slow", kRosQueueSize);
}

void DebayerCuda::imageCallback(
    const sensor_msgs::ImageConstPtr& image) {
  // Copy Ros msg to opencv
  CHECK_NOTNULL(image);
  cv_bridge::CvImagePtr cv_ptr;
  try {
    if (image->encoding == sensor_msgs::image_encodings::MONO8) {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::MONO8);
    } else if (image->encoding == sensor_msgs::image_encodings::BAYER_GBRG8) {
      cv_ptr = cv_bridge::toCvCopy(image, sensor_msgs::image_encodings::BAYER_GBRG8);
    } else {
      ROS_ERROR_STREAM("Unknown image encoding: " << image->encoding);
    }
  } catch (const cv_bridge::Exception& e) {
    ROS_FATAL_STREAM("cv_bridge exception: " << e.what());
  }
  CHECK_NOTNULL(cv_ptr);

  // Debayering and opt. rotation of images [GPU]
  debayer(&cv_ptr->image, nppi_debayer_mode_);

  // OpenCV simple whitebalance [CPU]
  whiteBalance(cv_ptr, white_balance_clipping_percentile_);

  // Custom gamma correction [CPU]
  gammaCorrection(cv_ptr->image, cv_ptr->image, gamma_correction_k_);

  // Clahe from OpenCV [CPU]
  // Attention, clahe is computationally expensive!
  if (run_clahe_) clahe(cv_ptr->image, cv_ptr->image, clahe_clip_limit_, clahe_tiles_grid_size_);

  // PCA based color correction [CPU]
  colorCorrection(cv_ptr);
  
  // Publish color image
  publishColorImage(cv_ptr, image);
}

void DebayerCuda::setupImageFormatParams(const cv::Mat* raw_image) {
  bayer_image_step_ = raw_image->step;
  color_image_step_ = bayer_image_step_ * 3;
  bayer_image_data_size_ = bayer_image_step_ * raw_image->rows;
  color_image_data_size_ = bayer_image_data_size_ * 3;
  bayer_image_width_ = raw_image->cols;
  bayer_image_height_ = raw_image->rows;
  image_size_.width = bayer_image_width_;
  image_size_.height = bayer_image_height_;
  image_roi_.x = 0;
  image_roi_.y = 0;
  image_roi_.width = bayer_image_width_;
  image_roi_.height = bayer_image_height_;
}

void DebayerCuda::debayer(
      cv::Mat* raw_image, const std::string& debayer_mode) {
  CHECK_NOTNULL(raw_image);

  // Setup image dimension params
  setupImageFormatParams(raw_image);

  // Allocate image sized GPU memory
  Npp8u* dp_image_bayer = 0;
  Npp8u* dp_image_color = 0;
  cudaMalloc(&dp_image_bayer, bayer_image_data_size_);
  cudaMalloc(&dp_image_color, color_image_data_size_);

  // Copy Bayer image to GPU
  cudaMemcpy(dp_image_bayer, raw_image->data, bayer_image_data_size_, cudaMemcpyHostToDevice);

  // Debayering from NPPI
  //https://docs.nvidia.com/cuda/npp/group__image__color__debayer.html
  if (debayer_mode == "NPPI_BAYER_BGGR")
    nppiCFAToRGB_8u_C1C3R(dp_image_bayer, bayer_image_step_, image_size_, image_roi_,
        dp_image_color, color_image_step_, NPPI_BAYER_BGGR, NPPI_INTER_UNDEFINED);
  else if (debayer_mode == "NPPI_BAYER_RGGB")
    nppiCFAToRGB_8u_C1C3R(dp_image_bayer, bayer_image_step_, image_size_, image_roi_,
        dp_image_color, color_image_step_, NPPI_BAYER_RGGB, NPPI_INTER_UNDEFINED);
  else if (debayer_mode == "NPPI_BAYER_GBRG")
    nppiCFAToRGB_8u_C1C3R(dp_image_bayer, bayer_image_step_, image_size_, image_roi_,
        dp_image_color, color_image_step_, NPPI_BAYER_GBRG, NPPI_INTER_UNDEFINED);
  else if (debayer_mode == "NPPI_BAYER_GRBG")
    nppiCFAToRGB_8u_C1C3R(dp_image_bayer, bayer_image_step_, image_size_, image_roi_,
        dp_image_color, color_image_step_, NPPI_BAYER_GRBG, NPPI_INTER_UNDEFINED);
  else
    ROS_ERROR_STREAM("Invalid NPPI debayer pattern chosen!");

  // Swapping color channels R and B for cam 5 (due to rotated cam)
  //https://docs.nvidia.com/cuda/npp/group__image__color__twist.html
  if (red_blue_swap_) {
    const Npp32f aTwist[3][4] {0.0, 0.0, 1.0, 0.0,
                               0.0, 1.0, 0.0, 0.0,
                               1.0, 0.0, 0.0, 0.0};
    nppiColorTwist32f_8u_C3IR(dp_image_color, color_image_step_, image_size_, aTwist);
  }

  // GAMMA CORRECTION (disabled for now)
  //https://docs.nvidia.com/cuda/npp/group__image__color__gamma__correction.html
  bool gamma_corr = false;
  if (gamma_corr) {
    bool forward = true;
    if (forward) {
      nppiGammaFwd_8u_C3IR(dp_image_color, color_image_step_, image_size_);
    }
    else nppiGammaInv_8u_C3IR(dp_image_color, color_image_step_, image_size_);
  }

  // Rotation for cameras that are mounted 180 degrees rotated
  if (needs_rotation_) nppiMirror_8u_C3IR (dp_image_color, color_image_step_, image_size_, NPP_BOTH_AXIS);

  // Copy to CV Mat
  raw_image->create(raw_image->rows, raw_image->cols, CV_8UC3);
  cudaMemcpy(raw_image->data, dp_image_color, color_image_data_size_, cudaMemcpyDeviceToHost);
  cudaFree(dp_image_color);
  cudaFree(dp_image_bayer);
}

void DebayerCuda::publishColorImage(
    const cv_bridge::CvImagePtr& cv_ptr,
    const sensor_msgs::ImageConstPtr& orig_image) {
  // Copy to ROS
  CHECK_NOTNULL(cv_ptr);
  CHECK_NOTNULL(orig_image);
  const sensor_msgs::ImagePtr color_img_msg = cv_ptr->toImageMsg();
  CHECK_NOTNULL(color_img_msg);

  // Set output encoding
  if (output_encoding_ == "RGB")
      color_img_msg->encoding = sensor_msgs::image_encodings::RGB8;
  else if (output_encoding_ == "BGR")
      color_img_msg->encoding = sensor_msgs::image_encodings::BGR8;
  else ROS_ERROR_STREAM("Found invalid image encoding: " << output_encoding_
      << ", make sure to set a supported ouput encoding (either 'RGB' or 'BGR')");

  // Assign the original timestamp and publish
  color_img_msg->header.stamp = orig_image->header.stamp;
  pub_color_image_.publish(color_img_msg);
  if (skipped_images_for_slow_topic_ >= skip_number_of_images_for_slow_topic_ ||
      skip_number_of_images_for_slow_topic_ <= 0) {
    pub_color_image_slow_.publish(color_img_msg);
    skipped_images_for_slow_topic_ = 0;
  } else {
    skipped_images_for_slow_topic_++;
  }
}

void DebayerCuda::whiteBalance(cv_bridge::CvImagePtr& cv_ptr, const float P) {
  // OpenCV white balancing
  cv::Ptr<cv::xphoto::SimpleWB> wb;
  wb = cv::xphoto::createSimpleWB();
  wb->setP(P); // Percentiles of highest and lowest pixel values to ignore; increases robustness
  wb->balanceWhite(cv_ptr->image, cv_ptr->image);
}

// Gamma correction test with own implementation
void DebayerCuda::gammaCorrection(cv::Mat& src, cv::Mat& dst, float K) {
  uchar LUT[256];
  src.copyTo(dst);
  for (int i = 0; i < 256; i++){
    //float f = (i + 0.5f) / 255;
    float f = i  / 255.0;
    f = pow(f, K);
    //LUT[i] = cv::saturate_cast<uchar>(f*255.0f-0.5f);
    LUT[i] = cv::saturate_cast<uchar>(f*255.0);
  }
  if (dst.channels() == 1){
    cv::MatIterator_<uchar> it = dst.begin<uchar>();
    cv::MatIterator_<uchar> it_end = dst.end<uchar>();
    for (; it != it_end; ++it){
      *it = LUT[(*it)];
    }
  }
  else{
    cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> it_end = dst.end<cv::Vec3b>();
    for (; it != it_end; ++it){
      (*it)[0] = LUT[(*it)[0]];
      (*it)[1] = LUT[(*it)[1]];
      (*it)[2] = LUT[(*it)[2]];
    }
  }
}

void DebayerCuda::clahe(cv::Mat& src, cv::Mat& dst, float clip_limit, int tiles_grid_size) {
  // Histogram equalization
  cv::Mat equalized_image;
  cvtColor(src, equalized_image, cv::COLOR_BGR2Lab);
  //Split the image into 3 channels; Y, Cr and Cb channels respectively and store it in a std::vector
  std::vector<cv::Mat> vec_channels(3);
  cv::split(equalized_image, vec_channels);
  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  clahe->setClipLimit(clip_limit);
  clahe->setTilesGridSize(cv::Size(tiles_grid_size, tiles_grid_size));
  clahe->apply(vec_channels[0], equalized_image);
  vec_channels[0].release();
  equalized_image.copyTo(vec_channels[0]);
  //Merge 3 channels in the vector to form the color image in YCrCB color space.
  cv::merge(vec_channels, equalized_image);
  //Convert the histogram equalized image from YCrCb to BGR color space again
  cv::cvtColor(equalized_image, dst, cv::COLOR_Lab2BGR);
}

bool DebayerCuda::isOverexposed(cv::Mat& img, int threshold) {
  auto intensity = getIntensity(img);
  cv::Mat dst_temp;
  cv::threshold(intensity, dst_temp, 250, 255, cv::THRESH_BINARY);
  int no_overexposed = int(cv::sum(dst_temp)[0] / 255.0);
  return (no_overexposed > threshold);
}

cv::Mat DebayerCuda::getIntensity(cv::Mat& img) {
  cv::Mat intensity;
  cv::transform(img, intensity, cv::Matx13f(1.0/3.0, 1.0/3.0, 1.0/3.0));
  cv::threshold(intensity, intensity, 255, 255, cv::THRESH_TRUNC);
  intensity.convertTo(intensity, CV_8UC1);
  return intensity;
}

void DebayerCuda::colorCorrection(cv_bridge::CvImagePtr& cv_ptr) {
  // Split channels
  std::vector<cv::Mat> split_img;
  cv::split(cv_ptr->image, split_img);
  split_img[0].convertTo(split_img[0], CV_32FC1);
  split_img[2].convertTo(split_img[2], CV_32FC1);

  // Get elementwise squared values
  cv::Mat I_r_2;
  cv::Mat I_b_2;
  cv::multiply(split_img[0], split_img[0], I_r_2);
  cv::multiply(split_img[2], split_img[2], I_b_2);

  // Get summed up channels
  const double sum_I_r_2 = cv::sum(I_r_2)[0];
  const double sum_I_b_2 = cv::sum(I_b_2)[0];
  const double sum_I_g = cv::sum(split_img[1])[0];  // Note: the image is actually BGR, due to symmertry in r/b however I believe this does not matter
  const double sum_I_r = cv::sum(split_img[0])[0];
  const double sum_I_b = cv::sum(split_img[2])[0];

  // Get max values of channels
  double max_I_r, max_I_g, max_I_b, max_I_r_2, max_I_b_2;
  double min_I_r, min_I_g, min_I_b, min_I_r_2, min_I_b_2;
  cv::minMaxLoc(split_img[0], &min_I_r, &max_I_r);
  cv::minMaxLoc(split_img[1], &min_I_g, &max_I_g);
  cv::minMaxLoc(split_img[2], &min_I_b, &max_I_b);
  cv::minMaxLoc(I_r_2, &min_I_r_2, &max_I_r_2);
  cv::minMaxLoc(I_b_2, &min_I_b_2, &max_I_b_2);

  // Prepare Matrices for PCA method
  Eigen::Matrix2f mat_temp_b;
  mat_temp_b << sum_I_b_2, sum_I_b,
                max_I_b_2, max_I_b;
  Eigen::Matrix2f mat_temp_r;
  mat_temp_r << sum_I_r_2, sum_I_r,
                max_I_r_2, max_I_r;
  Eigen::Vector2f vec_temp_g;
  vec_temp_g << sum_I_g, max_I_g;

  // PCA method calculation
  Eigen::Vector2f vec_out_b, vec_out_r;
  vec_out_b = mat_temp_b.inverse() * vec_temp_g;
  vec_out_r = mat_temp_r.inverse() * vec_temp_g;
  cv::Mat b_point = vec_out_b[0] * I_b_2 + vec_out_b[1] * split_img[2];
  cv::Mat r_point = vec_out_r[0] * I_r_2 + vec_out_r[1] * split_img[0];

  // Saturate values above 255
  cv::threshold(b_point, b_point, 255, 255, cv::THRESH_TRUNC);
  cv::threshold(r_point, r_point, 255, 255, cv::THRESH_TRUNC);
  
  // Convert back to UINT8
  b_point.convertTo(b_point, CV_8UC1);
  r_point.convertTo(r_point, CV_8UC1);

  // Merge channels
  std::vector<cv::Mat> channels;
  channels.push_back(r_point);
  channels.push_back(split_img[1]);
  channels.push_back(b_point);
  cv::Mat merged_image;
  cv::merge(channels, merged_image);

  cv_ptr->image = merged_image;
}

}  // namespace debayer
