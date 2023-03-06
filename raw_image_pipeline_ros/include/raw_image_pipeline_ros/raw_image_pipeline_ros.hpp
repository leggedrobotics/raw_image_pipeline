// Author: Matias Mattamala
// Author: Timon Homberger

#pragma once

#include <cv_bridge/cv_bridge.h>
#include <glog/logging.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <sensor_msgs/distortion_models.h>
#include <std_srvs/Trigger.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

#include <raw_image_pipeline/raw_image_pipeline.hpp>

namespace raw_image_pipeline {
class RawImagePipelineRos {
 public:
  // Constructor & destructor
  RawImagePipelineRos(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
  ~RawImagePipelineRos();

  // Starts the node
  bool run();

 private:
  // Setup methods
  void loadParams();
  void setupRos();

  // Main callback method
  void imageCallback(const sensor_msgs::ImageConstPtr& image);

  // Publishers
  void publishColorImage(const cv_bridge::CvImagePtr& cv_ptr_processed,                                // Processed image
                         const sensor_msgs::ImageConstPtr& orig_image,                                 // Original image
                         const cv::Mat& mask,                                                          // Mask
                         int image_height, int image_width,                                            // Dimensions
                         const std::string& distortion_model, const cv::Mat& distortion_coefficients,  // Distortion
                         const cv::Mat& camera_matrix, const cv::Mat& rectification_matrix, const cv::Mat& projection_matrix,  //
                         image_transport::CameraPublisher& camera_publisher, image_transport::Publisher& slow_publisher,       //
                         int& skipped_images);

  // Services
  bool resetWhiteBalanceHandler(std_srvs::Trigger::Request& req, std_srvs::Trigger::Response& res);

  // Helpers
  std::string getTransportHintFromTopic(std::string& image_topic);

  template <typename T>
  void readRequiredParameter(const std::string& param, T& value) {
    if (!nh_private_.getParam(param, value)) {
      ROS_FATAL_STREAM("Could not get [" << param << "]");
      std::exit(-1);
    } else {
      ROS_INFO_STREAM(param << ": " << value);
    }
  }

  template <typename T>
  T readParameter(const std::string& param, T default_value) {
    T value;
    if (!nh_private_.param<T>(param, value, default_value)) {
      ROS_WARN_STREAM("could not get [" << param << "], defaulting to: " << value);
    } else {
      ROS_INFO_STREAM(param << ": " << value);
    }
    return value;
  }

  std::vector<double> readParameter(const std::string& param, std::vector<double> default_value);

  // ROS
  ros::NodeHandle nh_;
  ros::NodeHandle nh_private_;
  ros::AsyncSpinner spinner_;

  // Subscribers
  image_transport::ImageTransport image_transport_;
  image_transport::Subscriber sub_raw_image_;

  // Publisher
  image_transport::CameraPublisher pub_image_;
  image_transport::Publisher pub_image_slow_;

  // Rectified image publisher
  image_transport::CameraPublisher pub_image_rect_;
  image_transport::Publisher pub_image_rect_mask_;
  image_transport::Publisher pub_image_rect_slow_;

  // Services
  ros::ServiceServer reset_wb_temporal_consistency_server_;

  // ROS Params
  std::string input_topic_;
  std::string output_topic_;
  std::string transport_;

  std::string output_encoding_;
  std::string output_frame_;

  // Slow topic
  int skip_number_of_images_for_slow_topic_;
  int skipped_images_for_slow_topic_;
  int skipped_images_for_slow_topic_rect_;

  // Postprocessing pipeline
  std::unique_ptr<RawImagePipeline> raw_image_pipeline_;
};

}  // namespace raw_image_pipeline