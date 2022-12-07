// Author: Matias Mattamala
// Author: Timon Homberger

#pragma once

#include <glog/logging.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/distortion_models.h>
#include <std_srvs/Trigger.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <image_proc_cuda/image_proc_cuda.hpp>

namespace image_proc_cuda
{
class ImageProcCudaRos
{
public:
    // Constructor & destructor
    ImageProcCudaRos(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    ~ImageProcCudaRos();

    // Setup methods
    void setupSubAndPub();
    void setupROSparams();

    // Starts the node
    bool run();

    // Main callback method
    void imageCallback(const sensor_msgs::ImageConstPtr& image);

private:
    // Publishers
    void publishColorImage(const cv_bridge::CvImagePtr& cv_ptr_processed,
                           const sensor_msgs::ImageConstPtr& orig_image,
                           int image_height,
                           int image_width,
                           const std::string& distortion_model,
                           const cv::Mat& distortion_coefficients,
                           const cv::Mat& camera_matrix,
                           const cv::Mat& rectification_matrix,
                           const cv::Mat& projection_matrix,
                           image_transport::CameraPublisher& camera_publisher,
                           image_transport::Publisher& slow_publisher,
                           int& skipped_images);
    
    // Services
    bool resetWhiteBalanceHandler(std_srvs::Trigger::Request  &req, std_srvs::Trigger::Response &res);

    // Postprocessing pipeline
    ImageProcCuda image_proc_;

    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    ros::AsyncSpinner spinner_;
    image_transport::ImageTransport image_transport_;
    image_transport::Subscriber sub_raw_image_;
    image_transport::CameraPublisher pub_color_image_;
    image_transport::Publisher pub_color_image_slow_;
    image_transport::CameraPublisher pub_color_image_distorted_;
    image_transport::Publisher pub_color_image_distorted_slow_;
    ros::ServiceServer reset_wb_temporal_consistency_server_;

    // ROS Params
    std::string input_topic_;
    std::string output_topic_;
    std::string transport_;

    std::string debayer_option_;
    std::string output_encoding_;
    std::string output_frame_;
    bool publish_distorted_; // To publish both distorted and undistorted

    // Slow topic
    int skip_number_of_images_for_slow_topic_;
    int skipped_images_for_slow_topic_;
    int skipped_images_for_slow_topic_distorted_;
};

} // namespace image_proc_cuda