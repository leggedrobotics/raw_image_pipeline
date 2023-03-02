// Author: Matias Mattamala
// Author: Timon Homberger

#include <ros/ros.h>
#include <image_proc_cuda_ros/image_proc_cuda_ros.hpp>

namespace image_proc_cuda {

ImageProcCudaRos::ImageProcCudaRos(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh), nh_private_(nh_private), spinner_(1), image_transport_(nh), skipped_images_for_slow_topic_(0) {
  google::InstallFailureSignalHandler();
}

ImageProcCudaRos::~ImageProcCudaRos() {}

bool ImageProcCudaRos::run() {
  ROS_INFO_STREAM("[ImageProcCudaRos] Starting...");
  spinner_.start();
  return true;
}

void ImageProcCudaRos::setupRosParams() {
  // Topic options
  readRequiredParameter("input_topic", input_topic_);
  readRequiredParameter("output_topic", output_topic_);

  // Get transport hint
  transport_ = getTransportHintFromTopic(input_topic_);

  // Other parameters
  output_encoding_ = readParameter("output_encoding", std::string("BGR"));
  output_frame_ = readParameter("output_frame", std::string("passthrough"));
  skip_number_of_images_for_slow_topic_ = readParameter("skip_number_of_images_for_slow_topic", -1);

  bool use_gpu = readParameter("use_gpu", true);
  image_proc_.setGpu(use_gpu);

  // Debayer
  bool run_debayer = readParameter("debayer/enabled", true);
  image_proc_.setDebayer(run_debayer);

  std::string debayer_encoding = readParameter("debayer/encoding", std::string("auto"));
  image_proc_.setDebayerEncoding(debayer_encoding);

  // Output options
  bool needs_rotation = readParameter("flip/enabled", false);
  image_proc_.setFlip(needs_rotation);

  // White balancing params
  bool run_white_balance = readParameter("white_balance/enabled", false);
  image_proc_.setWhiteBalance(run_white_balance);

  std::string white_balance_method = readParameter("white_balance/method", std::string("simple"));
  image_proc_.setWhiteBalanceMethod(white_balance_method);

  double white_balance_clipping_percentile = readParameter("white_balance/clipping_percentile", 10.0);
  image_proc_.setWhiteBalancePercentile(white_balance_clipping_percentile);

  double white_balance_saturation_bright_thr = readParameter("white_balance/saturation_bright_thr", 0.9);
  double white_balance_saturation_dark_thr = readParameter("white_balance/saturation_dark_thr", 0.1);
  image_proc_.setWhiteBalanceSaturationThreshold(white_balance_saturation_bright_thr, white_balance_saturation_dark_thr);

  bool white_balance_temporal_consistency = readParameter("white_balance/temporal_consistency", false);
  image_proc_.setWhiteBalanceTemporalConsistency(white_balance_temporal_consistency);

  // Color calibration
  bool run_color_calibration = readParameter("color_calibration/enabled", false);
  image_proc_.setColorCalibration(run_color_calibration);

  if (run_color_calibration) {
    // Check if we have the calibration file path
    std::string color_calibration_file = readParameter("color_calibration/calibration_file", std::string(""));
    image_proc_.loadColorCalibration(color_calibration_file);

    if (color_calibration_file.empty()) {
      std::vector<double> color_calibration_matrix =
          readParameter("color_calibration/calibration_matrix/data", std::vector<double>({0.0, 0.0, 0.0, 0.0}));
      image_proc_.setColorCalibrationMatrix(color_calibration_matrix);
    }
  }

  // Gamma correction
  bool run_gamma_correction = readParameter("gamma_correction/enabled", false);
  image_proc_.setGammaCorrection(run_gamma_correction);

  std::string gamma_correction_method = readParameter("gamma_correction/method", std::string("default"));
  image_proc_.setGammaCorrectionMethod(gamma_correction_method);

  double gamma_correction_k = readParameter("gamma_correction/k", 0.8);
  image_proc_.setGammaCorrectionK(gamma_correction_k);

  // Vignetting correction
  bool run_vignetting_correction = readParameter("vignetting_correction/enabled", false);
  image_proc_.setVignettingCorrection(run_vignetting_correction);

  double vignetting_correction_scale = readParameter("vignetting_correction/scale", 1.0);
  double vignetting_correction_a2 = readParameter("vignetting_correction/a2", 1.0);
  double vignetting_correction_a4 = readParameter("vignetting_correction/a4", 1.0);
  image_proc_.setVignettingCorrectionParameters(vignetting_correction_scale, vignetting_correction_a2, vignetting_correction_a4);

  // Color enhancer
  bool run_color_enhancer = readParameter("color_enhancer/enabled", false);
  image_proc_.setColorEnhancer(run_color_enhancer);

  double color_enhancer_hue_gain = readParameter("color_enhancer/hue_gain", 1.0);
  image_proc_.setColorEnhancerHueGain(color_enhancer_hue_gain);

  double color_enhancer_saturation_gain = readParameter("color_enhancer/saturation_gain", 1.0);
  image_proc_.setColorEnhancerSaturationGain(color_enhancer_saturation_gain);

  double color_enhancer_value_gain = readParameter("color_enhancer/value_gain", 1.0);
  image_proc_.setColorEnhancerValueGain(color_enhancer_value_gain);

  // Undistortion parameters
  bool run_undistortion = readParameter("undistortion/enabled", false);
  image_proc_.setUndistortion(run_undistortion);

  double undistortion_balance = readParameter("undistortion/balance", 0.0);
  image_proc_.setUndistortionBalance(undistortion_balance);

  double undistortion_fov_scale = readParameter("undistortion/fov_scale", 1.0);
  image_proc_.setUndistortionFovScale(undistortion_fov_scale);

  // Check if we have the calibration file path
  std::string calibration_file = readParameter("undistortion/calibration_file", std::string(""));
  image_proc_.loadCameraCalibration(calibration_file);

  if (calibration_file.empty()) {
    int image_width = readParameter("undistortion/image_width", 640);
    int image_height = readParameter("undistortion/image_width", 480);
    image_proc_.setUndistortionImageSize(image_width, image_height);

    std::vector<double> camera_matrix =
        readParameter("undistortion/camera_matrix/data", std::vector<double>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    image_proc_.setUndistortionCameraMatrix(camera_matrix);

    std::vector<double> distortion_coeff =
        readParameter("undistortion/distortion_coefficients/data", std::vector<double>({0.0, 0.0, 0.0, 0.0}));
    image_proc_.setUndistortionDistortionCoefficients(distortion_coeff);

    std::string distortion_model = readParameter("undistortion/distortion_model", std::string("none"));
    image_proc_.setUndistortionDistortionModel(distortion_model);

    std::vector<double> rectification_matrix =
        readParameter("undistortion/rectification_matrix/data", std::vector<double>({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}));
    image_proc_.setUndistortionRectificationMatrix(rectification_matrix);

    std::vector<double> projection_matrix = readParameter(
        "undistortion/projection_matrix/data", std::vector<double>({1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}));
    image_proc_.setUndistortionProjectionMatrix(projection_matrix);

    // Init undistortion module
    image_proc_.initUndistortion();
  }
}

void ImageProcCudaRos::setupSubAndPub() {
  constexpr size_t ros_queue_size = 5u;  // We always process the most updated frame

  // Set up the raw image subscriber.
  boost::function<void(const sensor_msgs::ImageConstPtr&)> image_callback = boost::bind(&ImageProcCudaRos::imageCallback, this, _1);

  image_transport::TransportHints transport_hint(transport_);

  // Subscribe image
  sub_raw_image_ = image_transport_.subscribe(input_topic_,                            // topic
                                              ros_queue_size,                          // queue size
                                              &ImageProcCudaRos::imageCallback, this,  // callback
                                              transport_hint                           // hints
  );
  // Set up the processed image publisher.
  if (image_proc_.isUndistortionEnabled()) {
    pub_image_rect_ = image_transport_.advertiseCamera(output_topic_ + "/image_rect", ros_queue_size);
    pub_image_rect_mask_ = image_transport_.advertise(output_topic_ + "/image_mask", ros_queue_size);
    pub_image_rect_slow_ = image_transport_.advertise(output_topic_ + "/image_rect/slow", ros_queue_size);
  }

  pub_image_ = image_transport_.advertiseCamera(output_topic_, ros_queue_size);
  pub_image_slow_ = image_transport_.advertise(output_topic_ + "/slow", ros_queue_size);

  // Setup service calls
  reset_wb_temporal_consistency_server_ =
      nh_private_.advertiseService("reset_white_balance", &ImageProcCudaRos::resetWhiteBalanceHandler, this);
}  // namespace image_proc_cuda

void ImageProcCudaRos::imageCallback(const sensor_msgs::ImageConstPtr& image_msg) {
  // Copy Ros msg to opencv
  CHECK_NOTNULL(image_msg);
  cv_bridge::CvImagePtr cv_ptr_processed;

  if (transport_ != "raw")
    cv_ptr_processed = cv_bridge::toCvCopy(image_msg, "bgr8");
  else
    cv_ptr_processed = cv_bridge::toCvCopy(image_msg, image_msg->encoding);

  CHECK_NOTNULL(cv_ptr_processed);

  if (cv_ptr_processed->image.empty()) {
    ROS_WARN("image empty");
    return;
  }

  // Run image proc cuda pipeline
  image_proc_.apply(cv_ptr_processed->image, cv_ptr_processed->encoding);

  // Publish undistorted
  if (image_proc_.isUndistortionEnabled()) {
    // Publish undistorted
    publishColorImage(cv_ptr_processed,                                                                   // Processed
                      image_msg,                                                                          // Original image
                      image_proc_.getRectMask(),                                                          // Mask
                      image_proc_.getRectImageHeight(), image_proc_.getRectImageWidth(),                  // Dimensions
                      image_proc_.getRectDistortionModel(), image_proc_.getRectDistortionCoefficients(),  // Distortion stuff
                      image_proc_.getRectCameraMatrix(), image_proc_.getRectRectificationMatrix(),
                      image_proc_.getRectProjectionMatrix(),  // Pinhole stuff
                      pub_image_rect_, pub_image_rect_slow_,  // Publishers
                      skipped_images_for_slow_topic_rect_     // Counter to keep track of the number of skipped images
    );

    // Replace image by distorted
    cv_ptr_processed->image = image_proc_.getDistImage();
  }

  // Publish distorted image
  publishColorImage(cv_ptr_processed,                                                                   // Processed
                    image_msg,                                                                          // Original image
                    cv::Mat(),                                                                          // Mask
                    image_proc_.getDistImageHeight(), image_proc_.getDistImageWidth(),                  // Dimensions
                    image_proc_.getDistDistortionModel(), image_proc_.getDistDistortionCoefficients(),  // Distortion stuff
                    image_proc_.getDistCameraMatrix(), image_proc_.getDistRectificationMatrix(),
                    image_proc_.getDistProjectionMatrix(),  // Pinhole stuff
                    pub_image_, pub_image_slow_,            // Publishers
                    skipped_images_for_slow_topic_          // Counter to keep track of the skipped images
  );
}

bool ImageProcCudaRos::resetWhiteBalanceHandler(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res) {
  image_proc_.resetWhiteBalanceTemporalConsistency();
  res.success = true;
  res.message = "White balance resetted";
  return true;
}

void ImageProcCudaRos::publishColorImage(const cv_bridge::CvImagePtr& cv_ptr_processed,                                //
                                         const sensor_msgs::ImageConstPtr& orig_image,                                 //
                                         const cv::Mat& mask,                                                          // Mask
                                         int image_height, int image_width,                                            // Dimensions
                                         const std::string& distortion_model, const cv::Mat& distortion_coefficients,  //
                                         const cv::Mat& camera_matrix, const cv::Mat& rectification_matrix,
                                         const cv::Mat& projection_matrix,                                                                //
                                         image_transport::CameraPublisher& camera_publisher, image_transport::Publisher& slow_publisher,  //
                                         int& skipped_images                                                                              //
) {
  // Note: Image is BGR
  // Convert image to output encoding
  if (output_encoding_ == "RGB") {
    cv::cvtColor(cv_ptr_processed->image, cv_ptr_processed->image, cv::COLOR_BGR2RGB);
  }

  // Copy to ROS
  CHECK_NOTNULL(cv_ptr_processed);
  CHECK_NOTNULL(orig_image);
  const sensor_msgs::ImagePtr color_img_msg = cv_ptr_processed->toImageMsg();
  CHECK_NOTNULL(color_img_msg);

  // Set output encoding
  if (cv_ptr_processed->image.channels() == 3) {
    if (output_encoding_ == "RGB")
      color_img_msg->encoding = sensor_msgs::image_encodings::RGB8;
    else if (output_encoding_ == "BGR")
      color_img_msg->encoding = sensor_msgs::image_encodings::BGR8;
    else
      ROS_ERROR_STREAM("Found invalid image encoding: " << output_encoding_
                                                        << ", make sure to set a supported ouput encoding (either 'RGB' or 'BGR')");
  }

  sensor_msgs::CameraInfoPtr color_camera_info_msg(new sensor_msgs::CameraInfo());

  // Fix output frame if required
  if (output_frame_ == "passthrough") {
    color_camera_info_msg->header.frame_id = orig_image->header.frame_id;
  } else {
    color_img_msg->header.frame_id = output_frame_;
    color_camera_info_msg->header.frame_id = output_frame_;
  }
  color_camera_info_msg->header.stamp = orig_image->header.stamp;
  color_camera_info_msg->height = image_height;
  color_camera_info_msg->width = image_width;

  // Fix distortion model if it's none
  if (color_camera_info_msg->distortion_model == "none") {
    color_camera_info_msg->distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
  } else {
    color_camera_info_msg->distortion_model = distortion_model;
  }

  // Other calibration stuff
  color_camera_info_msg->D = utils::toStdVector<double>(distortion_coefficients);
  color_camera_info_msg->K = utils::toBoostArray<double, 9>(camera_matrix);
  color_camera_info_msg->R = utils::toBoostArray<double, 9>(rectification_matrix);
  color_camera_info_msg->P = utils::toBoostArray<double, 12>(projection_matrix);

  // Assign the original timestamp and publish
  color_img_msg->header.stamp = orig_image->header.stamp;
  camera_publisher.publish(color_img_msg, color_camera_info_msg);

  // Publish to slow topic
  if (skipped_images >= skip_number_of_images_for_slow_topic_ || skip_number_of_images_for_slow_topic_ <= 0) {
    slow_publisher.publish(color_img_msg);
    skipped_images = 0;
  } else {
    skipped_images++;
  }
}

std::string ImageProcCudaRos::getTransportHintFromTopic(std::string& image_topic) {
  // This method modifies the input topic

  std::string transport_hint = "compressed";
  std::size_t ind = image_topic.find(transport_hint);  // Find if compressed is in the topic name

  if (ind != std::string::npos) {
    transport_hint = image_topic.substr(ind, image_topic.length());  // Get the hint as the last part
    image_topic.erase(ind - 1, image_topic.length());                // We remove the hint from the topic
  } else {
    transport_hint = "raw";  // In the default case we assume raw topic
  }
  return transport_hint;
}

std::vector<double> ImageProcCudaRos::readParameter(const std::string& param, std::vector<double> default_value) {
  std::stringstream ss;  // Prepare default string
  for (auto v : default_value) {
    ss << v << " ";
  }
  std::vector<double> value;
  if (!nh_private_.param<std::vector<double>>(param, value, default_value)) {
    ROS_WARN_STREAM("could not get [" << param << "], using default " << ss.str());
  }
  return value;
}

}  // namespace image_proc_cuda