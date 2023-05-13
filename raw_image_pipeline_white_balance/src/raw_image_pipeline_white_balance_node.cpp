//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
// 

/**
@brief raw_image_pipeline_white_balance_node.cpp
simple test node to control UV using rqt_reconfigure
*/

// ROS
#include <ros/package.h>  // to get the current directory of the package.
#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <raw_image_pipeline_white_balance/convolutional_color_constancy.hpp>

#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <raw_image_pipeline_white_balance/RawImagePipelineWhiteBalanceConfig.h>

class WhiteBalanceRos {
 public:
  dynamic_reconfigure::Server<raw_image_pipeline_white_balance::RawImagePipelineWhiteBalanceConfig> server_;
  dynamic_reconfigure::Server<raw_image_pipeline_white_balance::RawImagePipelineWhiteBalanceConfig>::CallbackType f_;
  std::string model_file_;
  std::string image_file_;
  cv::Mat image_;

#ifdef HAS_CUDA
  cv::cuda::GpuMat image_d_;
#endif

  std::shared_ptr<raw_image_pipeline_white_balance::ConvolutionalColorConstancyWB> wb_;

  WhiteBalanceRos(std::string model_file, std::string image_file) {
    model_file_ = model_file;
    image_file_ = image_file;

    image_ = cv::imread(image_file_, cv::IMREAD_COLOR);
    cv::imshow("original", image_);

    bool use_gpu = true;
    wb_ = std::make_shared<raw_image_pipeline_white_balance::ConvolutionalColorConstancyWB>(use_gpu, model_file_);
    wb_->setUV0(-1.421875);
    wb_->setDebug(false);

    // CPU version
    cv::Mat wb_image;
    wb_->balanceWhite(image_, wb_image);

    // GPU version
#ifdef HAS_CUDA
    image_d_.upload(image_);
    cv::Mat wb_image_gpu;
    cv::cuda::GpuMat wb_image_d_;
    wb_->balanceWhite(image_d_, wb_image_d_);
    wb_image_d_.download(wb_image_gpu);
    cv::imshow("corrected_gpu", wb_image_gpu);
#endif

    // Show
    cv::imshow("original", image_);
    cv::imshow("corrected_cpu", wb_image);

    cv::waitKey(10);

    f_ = boost::bind(&WhiteBalanceRos::callback, this, _1, _2);
    server_.setCallback(f_);
  }

  void callback(raw_image_pipeline_white_balance::RawImagePipelineWhiteBalanceConfig& config, uint32_t) {
    wb_->setSaturationThreshold(config.bright_thr, config.dark_thr);
    // wb_->setDebugUVOffset(config.Lu_offset, config.Lv_offset, config.uv0);
    std::cout << "-- Updating parameters -- " << std::endl;
    std::cout << "   bright_thr: " << config.bright_thr << std::endl;
    std::cout << "   dark_thr:   " << config.dark_thr << std::endl;
    std::cout << "   Lu_offset:  " << config.Lu_offset << std::endl;
    std::cout << "   Lv_offset:  " << config.Lv_offset << std::endl;
    std::cout << "   uv0:        " << config.uv0 << std::endl;

    // CPU version
    cv::Mat wb_image;
    wb_->balanceWhite(image_, wb_image);

// GPU version
#ifdef HAS_CUDA
    cv::Mat wb_image_gpu;
    cv::cuda::GpuMat wb_image_d_;
    wb_->balanceWhite(image_d_, wb_image_d_);
    wb_image_d_.download(wb_image_gpu);
    cv::imshow("corrected_gpu", wb_image_gpu);
#endif

    // Show
    cv::imshow("original", image_);
    cv::imshow("corrected_cpu", wb_image);

    cv::waitKey(10);
  }
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "raw_image_pipeline_white_balance");
  ros::NodeHandle nh_priv("~");
  std::string model_file = ros::package::getPath("raw_image_pipeline_white_balance") + "/model/default.bin";
  std::string image_file = ros::package::getPath("raw_image_pipeline_white_balance") + "/data/alphasense2.png";

  // Get input image path
  nh_priv.param<std::string>("image", image_file, image_file);

  WhiteBalanceRos wb(model_file, image_file);

  ROS_INFO("Spinning node");
  ros::spin();
  return 0;
}
