#ifndef DEBAYER_CUDA_H_
#define DEBAYER_CUDA_H_

#include <glog/logging.h>

#include <npp.h>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <opencv2/xphoto/white_balance.hpp>

//#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace debayer {

class DebayerCuda {

  public:

    DebayerCuda(const ros::NodeHandle& nh, const ros::NodeHandle& nh_private);
    ~DebayerCuda();

    bool run();
    void imageCallback(const sensor_msgs::ImageConstPtr& image);

    void setupSubAndPub();
    void setupROSparams();

  private:

    void debayer(cv::Mat* raw_image, const std::string& debayer_mode);
    void setupImageFormatParams(const cv::Mat* raw_image);
    void publishColorImage(
      const cv_bridge::CvImagePtr& cv_ptr,
      const sensor_msgs::ImageConstPtr& orig_image);
    void whiteBalance(cv_bridge::CvImagePtr& cv_ptr, const float P);
    void gammaCorrection(cv::Mat& src, cv::Mat& dst, float K);
    void clahe(cv::Mat& src, cv::Mat& dst, float clip_limit, int tiles_grid_size);
    bool isOverexposed(cv::Mat& img, int threshold);
    cv::Mat getIntensity(cv::Mat& img);
    void colorCorrection(cv_bridge::CvImagePtr& cv_ptr);

    // ROS
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    ros::AsyncSpinner spinner_;
    image_transport::ImageTransport image_transport_;
    image_transport::Subscriber sub_raw_image_;
    image_transport::Publisher pub_color_image_;
    image_transport::Publisher pub_color_image_slow_;

    // ROS Params
    std::string input_topic_;
    std::string output_topic_;
    std::string nppi_debayer_mode_;
    bool red_blue_swap_;
    std::string output_encoding_;
    int skip_number_of_images_for_slow_topic_;
    int skipped_images_for_slow_topic_;
    bool needs_rotation_;
    double white_balance_clipping_percentile_;
    double gamma_correction_k_;
    bool run_clahe_;
    double clahe_clip_limit_;
    int clahe_tiles_grid_size_;

    // Image Format Params
    int bayer_image_step_;
    int color_image_step_;
    int bayer_image_data_size_;
    int color_image_data_size_;
    int bayer_image_width_;
    int bayer_image_height_;
    NppiSize image_size_;
    NppiRect image_roi_;
};

}  // namespace debayer

#endif  // DEBAYER_CUDA_H_
