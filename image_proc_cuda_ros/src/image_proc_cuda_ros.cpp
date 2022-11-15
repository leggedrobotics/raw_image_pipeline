#include <image_proc_cuda_ros/image_proc_cuda_ros.hpp>
#include <ros/package.h>

namespace image_proc_cuda
{

ImageProcCudaRos::ImageProcCudaRos(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      spinner_(1),
      image_transport_(nh),
      skipped_images_for_slow_topic_(0)
{
    google::InstallFailureSignalHandler();
}

ImageProcCudaRos::~ImageProcCudaRos() {}

bool ImageProcCudaRos::run()
{
    ROS_INFO_STREAM("[ImageProcCudaRos] Starting...");
    spinner_.start();
    return true;
}

void ImageProcCudaRos::setupROSparams()
{
    // Topic options
    if (!nh_private_.getParam("input_topic", input_topic_))
        ROS_ERROR_STREAM("could not get input topic!");
    else
        ROS_INFO_STREAM("input topic: " << input_topic_);

    if (!nh_private_.param<std::string>("transport", transport_, "raw"))
        ROS_ERROR_STREAM("Assuming topic is raw!");

    if (!nh_private_.getParam("output_topic", output_topic_))
        ROS_ERROR_STREAM("could not get output topic!");
    else
        ROS_INFO_STREAM("output topic: " << output_topic_);

    // Debayer options
    std::string debayer_option;
    if (!nh_private_.param<std::string>("debayer_option", debayer_option, "auto"))
        ROS_WARN_STREAM(
            "could not get encoding_option from ROS-params, defaulting to: " << debayer_option);
    image_proc_.setDebayerOption(debayer_option);

    // Output options
    if (!nh_private_.param<std::string>("output_encoding", output_encoding_, "BGR"))
        ROS_WARN_STREAM("could not get output encoding, defaulting to: " << output_encoding_);

    if (!nh_private_.param<int>("skip_number_of_images_for_slow_topic",
                           skip_number_of_images_for_slow_topic_, -1))
        ROS_WARN_STREAM("could not get 'skip_number_of_images_for_slow_topic', defaulting to: "
                        << skip_number_of_images_for_slow_topic_);

    bool needs_rotation;
    if (!nh_private_.param<bool>("needs_rotation", needs_rotation, false))
        ROS_WARN_STREAM("could not get 'needs_rotation', defaulting to: " << needs_rotation);
    image_proc_.setFlip(needs_rotation);

    // White balancing params
    bool run_white_balance;
    if (!nh_private_.param<bool>("run_white_balance", run_white_balance, false))
        ROS_WARN_STREAM("could not get 'run_white_balance', defaulting to: " << run_white_balance);
    image_proc_.setWhiteBalance(run_white_balance);
    
    std::string white_balance_method;
    if (!nh_private_.param<std::string>("white_balance_method", white_balance_method, "simple"))
        ROS_WARN_STREAM("could not get 'run_white_balance', defaulting to: " << white_balance_method);
    image_proc_.setWhiteBalanceMethod(white_balance_method);

    double white_balance_clipping_percentile;
    if (!nh_private_.param<double>("white_balance_clipping_percentile", white_balance_clipping_percentile,
                           10.0))
        ROS_WARN_STREAM("could not get 'white_balance_clipping_percentile', defaulting to: "
                        << white_balance_clipping_percentile);
    image_proc_.setWhiteBalancePercentile(white_balance_clipping_percentile);
    
    double white_balance_saturation_threshold;
    if (!nh_private_.param<double>("white_balance_saturation_threshold",
                           white_balance_saturation_threshold, 0.9))
        ROS_WARN_STREAM("could not get 'white_balance_saturation_threshold', defaulting to: "
                        << white_balance_saturation_threshold);
    image_proc_.setWhiteBalanceSaturationThreshold(white_balance_saturation_threshold);

    bool white_balance_temporal_consistency = false;
    if (!nh_private_.param<bool>("white_balance_temporal_consistency",
                           white_balance_temporal_consistency, false))
        ROS_WARN_STREAM("could not get 'white_balance_temporal_consistency', defaulting to: "
                        << white_balance_temporal_consistency);
    image_proc_.setWhiteBalanceTemporalConsistency(white_balance_temporal_consistency);

    // Gamma correction
    bool run_gamma_correction;
    if (!nh_private_.param<bool>("run_gamma_correction", run_gamma_correction, false))
        ROS_WARN_STREAM(
            "could not get 'run_white_balance', defaulting to: " << run_gamma_correction);
    image_proc_.setGammaCorrection(run_gamma_correction);

    std::string gamma_correction_method;
    if (!nh_private_.param<std::string>("gamma_correction_method", gamma_correction_method,
                                        "default"))
        ROS_WARN_STREAM(
            "could not get 'gamma_correction_method', defaulting to: " << gamma_correction_method);
    image_proc_.setGammaCorrectionMethod(gamma_correction_method);
    
    double gamma_correction_k;
    if (!nh_private_.param<double>("gamma_correction_k", gamma_correction_k, 0.8))
        ROS_WARN_STREAM("could not get 'gamma_correction_k', defaulting to: "
                        << gamma_correction_k << " ("
                        << (gamma_correction_k >= 0 ? "forward" : "inverse") << ")");
    image_proc_.setGammaCorrectionK(gamma_correction_k);

    // Vignetting correction
    bool run_vignetting_correction;
    if (!nh_private_.param<bool>("run_vignetting_correction", run_vignetting_correction, false))
        ROS_WARN_STREAM("could not get 'run_vignetting_correction', defaulting to: "
                        << run_vignetting_correction);
    image_proc_.setVignettingCorrection(run_vignetting_correction);
    
    double vignetting_correction_scale;
    if (!nh_private_.param<double>("vignetting_correction_scale", vignetting_correction_scale, 1.0))
        ROS_WARN_STREAM("could not get 'vignetting_correction_scale', defaulting to: "
                        << vignetting_correction_scale);
    image_proc_.setVignettingCorrectionScale(vignetting_correction_scale);

    double vignetting_correction_a2;
    if (!nh_private_.param<double>("vignetting_correction_a2", vignetting_correction_a2, 1.0))
        ROS_WARN_STREAM("could not get 'vignetting_correction_a2', defaulting to: "
                        << vignetting_correction_a2);
    image_proc_.setVignettingCorrectionA2(vignetting_correction_a2);

    double vignetting_correction_a4;
    if (!nh_private_.param<double>("vignetting_correction_a4", vignetting_correction_a4, 1.0))
        ROS_WARN_STREAM("could not get 'vignetting_correction_a4', defaulting to: "
                        << vignetting_correction_a4);
    image_proc_.setVignettingCorrectionA4(vignetting_correction_a4);

    // Clahe parameters
    bool run_clahe;
    if (!nh_private_.param<bool>("run_clahe", run_clahe, false))
        ROS_WARN_STREAM("could not get 'run_clahe', defaulting to: " << run_clahe);
    image_proc_.setClahe(run_clahe);

    double clahe_clip_limit;
    if (!nh_private_.param<double>("clahe_clip_limit", clahe_clip_limit, 1.5))
        ROS_WARN_STREAM("could not get 'clahe_clip_limit', defaulting to: " << clahe_clip_limit);
    image_proc_.setClaheLimit(clahe_clip_limit);

    int clahe_tiles_grid_size;
    if (!nh_private_.param<int>("clahe_tiles_grid_size", clahe_tiles_grid_size, 8))
        ROS_WARN_STREAM(
            "could not get 'clahe_tiles_grid_size', defaulting to: " << clahe_tiles_grid_size);
    image_proc_.setClaheGridSize(clahe_tiles_grid_size);

    // Color enhancer
    bool run_color_enhancer;
    if (!nh_private_.param<bool>("run_color_enhancer", run_color_enhancer, false))
        ROS_WARN_STREAM(
            "could not get 'run_color_enhancer', defaulting to: " << run_color_enhancer);
    image_proc_.setColorEnhancer(run_color_enhancer);

    double color_enhancer_hue_gain;
    if (!nh_private_.param<double>("color_enhancer_hue_gain", color_enhancer_hue_gain, 1.0))
        ROS_WARN_STREAM(
            "could not get 'color_enhancer_hue_gain', defaulting to: " << color_enhancer_hue_gain);
    image_proc_.setColorEnhancerHueGain(color_enhancer_hue_gain);

    double color_enhancer_saturation_gain;
    if (!nh_private_.param<double>("color_enhancer_saturation_gain", color_enhancer_saturation_gain, 1.0))
        ROS_WARN_STREAM("could not get 'color_enhancer_saturation_gain', defaulting to: "
                        << color_enhancer_saturation_gain);
    image_proc_.setColorEnhancerSaturationGain(color_enhancer_saturation_gain);

    double color_enhancer_value_gain;
    if (!nh_private_.param<double>("color_enhancer_value_gain", color_enhancer_value_gain, 1.0))
        ROS_WARN_STREAM("could not get 'color_enhancer_value_gain', defaulting to: "
                        << color_enhancer_value_gain);
    image_proc_.setColorEnhancerValueGain(color_enhancer_value_gain);

    // Color calibration
    bool run_color_calibration;
    if (!nh_private_.param<bool>("run_color_calibration", run_color_calibration, false))
        ROS_WARN_STREAM(
            "could not get 'run_color_calibration', defaulting to: " << run_color_calibration);
    image_proc_.setColorCalibration(run_color_calibration);

    if(run_color_calibration)
    {
        // Check if we have the calibration file path
        std::string color_calibration_file;
        if (!nh_private_.param<std::string>("color_calibration_file", color_calibration_file, ""))
            ROS_WARN_STREAM("could not get 'color_calibration_file', will try to load from parameter server");
        image_proc_.loadColorCalibration(color_calibration_file);

        if(color_calibration_file.empty())
        {
            std::vector<double> color_calibration_matrix;
            if (!nh_private_.param<std::vector<double>>("color_calibration_matrix/data", color_calibration_matrix, {0.0, 0.0, 0.0, 0.0}))
                ROS_WARN_STREAM("could not get 'color_calibration_matrix/data', defaulting to zeros vector");
            image_proc_.setColorCalibrationMatrix(color_calibration_matrix);
        }
    }

    // Undistortion parameters
    bool run_undistortion;
    if (!nh_private_.param<bool>("run_undistortion", run_undistortion, false))
        ROS_WARN_STREAM(
            "could not get 'run_undistortion', defaulting to: " << run_undistortion);
    image_proc_.setUndistortion(run_undistortion);

    // if(run_undistortion)
    // {
        // Check if we have the calibration file path
        std::string calibration_file;
        if (!nh_private_.param<std::string>("calibration_file", calibration_file, ""))
            ROS_WARN_STREAM("could not get 'calibration_file', will try to load from parameter server");
        image_proc_.loadCalibration(calibration_file);

        if(calibration_file.empty())
        {
            int image_width;
            if (!nh_private_.param("image_width", image_width, 640))
                ROS_WARN_STREAM("could not get 'image_width', defaulting to: " << image_width);

            int image_height;
            if (!nh_private_.param("image_width", image_height, 480))
                ROS_WARN_STREAM("could not get 'image_height', defaulting to: " << image_height);
            image_proc_.setUndistortionImageSize(image_width, image_height);

            std::vector<double> camera_matrix;
            if (!nh_private_.param<std::vector<double>>("camera_matrix/data", camera_matrix,
                                                                        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}))
                ROS_WARN_STREAM("could not get 'camera_matrix/data', defaulting to Identity matrix");
            image_proc_.setUndistortionCameraMatrix(camera_matrix);

            std::vector<double> distortion_coeff;
            if (!nh_private_.param<std::vector<double>>("distortion_coefficients/data", distortion_coeff, {0.0, 0.0, 0.0, 0.0}))
                ROS_WARN_STREAM("could not get 'distortion_coefficients/data', defaulting to zeros vector");
            image_proc_.setUndistortionDistortionCoefficients(distortion_coeff);

            std::string distortion_model;
            if (!nh_private_.param<std::string>("distortion_model", distortion_model, "none"))
                ROS_WARN_STREAM("could not get 'distortion_model', defaulting to: " << distortion_model);
            image_proc_.setUndistortionDistortionModel(distortion_model);

            std::vector<double> rectification_matrix;
            if (!nh_private_.param<std::vector<double>>("rectification_matrix/data", rectification_matrix, 
                                                                                {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0}))
                ROS_WARN_STREAM("could not get 'rectification_matrix/data', defaulting to Identity matrix");
            image_proc_.setUndistortionRectificationMatrix(rectification_matrix);

            std::vector<double> projection_matrix;
            if (!nh_private_.param<std::vector<double>>("projection_matrix/data", projection_matrix, 
                                                        {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0}))
                ROS_WARN_STREAM("could not get 'projection_matrix/data', defaulting to Identity matrix");
            image_proc_.setUndistortionProjectionMatrix(projection_matrix);

            // Init rectify map
            image_proc_.initRectifyMap();
        }
    // }

    // Debug parameters
    bool dump_images;
    if (!nh_private_.param("dump_images", dump_images, false))
        ROS_WARN_STREAM("could not get 'dump_images', defaulting to: " << dump_images);
    image_proc_.setDumpImages(dump_images);
}

void ImageProcCudaRos::setupSubAndPub()
{
    constexpr size_t ros_queue_size = 1u; // We always process the most updated frame

    // Set up the raw image subscriber.
    boost::function<void(const sensor_msgs::ImageConstPtr&)> image_callback =
        boost::bind(&ImageProcCudaRos::imageCallback, this, _1);

    image_transport::TransportHints transport_hint(transport_);

    // Subscribe image
    sub_raw_image_ = image_transport_.subscribe(input_topic_,                       // topic
                                                ros_queue_size,                     // queue size
                                                &ImageProcCudaRos::imageCallback, this,  // callback
                                                transport_hint                      // hints
    );
    // Set up the processed image publisher.
    pub_color_image_ = image_transport_.advertiseCamera(output_topic_, ros_queue_size);
    pub_color_image_slow_ = image_transport_.advertise(output_topic_ + "/slow", ros_queue_size);

    // Setup service calls
    reset_wb_temporal_consistency_server_ = nh_private_.advertiseService("reset_white_balance", 
                                                    &ImageProcCudaRos::resetWhiteBalanceHandler, this);
}

void ImageProcCudaRos::imageCallback(const sensor_msgs::ImageConstPtr& image_msg)
{
    // Copy Ros msg to opencv
    CHECK_NOTNULL(image_msg);
    cv_bridge::CvImagePtr cv_ptr;
    if (transport_ != "raw")
        cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
    else
        cv_ptr = cv_bridge::toCvCopy(image_msg, image_msg->encoding);

    CHECK_NOTNULL(cv_ptr);

    if (cv_ptr->image.empty())
    {
        ROS_WARN("image empty");
        return;
    }

    // Call debayer cuda
    image_proc_.apply(cv_ptr->image, image_msg->encoding);

    // Publish color image
    publishColorImage(cv_ptr, image_msg);
}

bool ImageProcCudaRos::resetWhiteBalanceHandler(std_srvs::Trigger::Request  &req, std_srvs::Trigger::Response &res)
{
    image_proc_.resetWhiteBalanceTemporalConsistency();
    res.success = true;
    res.message = "White balance resetted";
    return true;
}

void ImageProcCudaRos::publishColorImage(const cv_bridge::CvImagePtr& cv_ptr,
                                    const sensor_msgs::ImageConstPtr& orig_image)
{
    // Note: Image is BGR
    // Convert image to output encoding
    if (output_encoding_ == "RGB")
        cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_BGR2RGB);

    // Copy to ROS
    CHECK_NOTNULL(cv_ptr);
    CHECK_NOTNULL(orig_image);
    const sensor_msgs::ImagePtr color_img_msg = cv_ptr->toImageMsg();
    CHECK_NOTNULL(color_img_msg);

    // Set output encoding
    if(cv_ptr->image.channels() == 3){
        if (output_encoding_ == "RGB")
            color_img_msg->encoding = sensor_msgs::image_encodings::RGB8;
        else if (output_encoding_ == "BGR")
            color_img_msg->encoding = sensor_msgs::image_encodings::BGR8;
        else
            ROS_ERROR_STREAM(
                "Found invalid image encoding: "
                << output_encoding_
                << ", make sure to set a supported ouput encoding (either 'RGB' or 'BGR')");
    }
    
    // Prepare camera info message
    // if(!image_proc_.getUndistortionEnabled())
    // {
    //     ROS_WARN_ONCE("Undistortion is not enabled. "
    //                   "CameraInfo messages will have the default calibration settings. "
    //                   "This message will print once.");
    // }
    sensor_msgs::CameraInfoPtr color_camera_info_msg(new sensor_msgs::CameraInfo());
    
    color_camera_info_msg->header.frame_id = orig_image->header.frame_id;
    color_camera_info_msg->header.stamp    = orig_image->header.stamp;
    color_camera_info_msg->height = image_proc_.getImageHeight();
    color_camera_info_msg->width = image_proc_.getImageWidth();
    color_camera_info_msg->distortion_model = image_proc_.getDistortionModel();
    // Fix distortion model if it's none
    if(color_camera_info_msg->distortion_model == "none")
        color_camera_info_msg->distortion_model = sensor_msgs::distortion_models::PLUMB_BOB;
    // Other calibration stuff
    color_camera_info_msg->D = utils::toStdVector<double>(image_proc_.getDistortionCoefficients());
    color_camera_info_msg->K = utils::toBoostArray<double, 9>(image_proc_.getCameraMatrix());
    color_camera_info_msg->R = utils::toBoostArray<double, 9>(image_proc_.getRectificationMatrix());
    color_camera_info_msg->P = utils::toBoostArray<double, 12>(image_proc_.getProjectionMatrix());

    // Assign the original timestamp and publish
    color_img_msg->header.stamp = orig_image->header.stamp;
    pub_color_image_.publish(color_img_msg, color_camera_info_msg);

    // Publish to slow topic
    if (skipped_images_for_slow_topic_ >= skip_number_of_images_for_slow_topic_ ||
        skip_number_of_images_for_slow_topic_ <= 0)
    {
        pub_color_image_slow_.publish(color_img_msg);
        skipped_images_for_slow_topic_ = 0;
    }
    else
    {
        skipped_images_for_slow_topic_++;
    }
}

} // namespace image_proc_cuda