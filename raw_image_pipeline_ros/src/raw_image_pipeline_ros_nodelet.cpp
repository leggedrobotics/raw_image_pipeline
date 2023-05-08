//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <raw_image_pipeline_ros/raw_image_pipeline_ros.hpp>
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <memory>

class RawImagePipelineNodelet : public nodelet::Nodelet {

  public:
    RawImagePipelineNodelet(){}
    ~RawImagePipelineNodelet(){}

  private:
    virtual void onInit() {

      nh_ = getNodeHandle();
      nh_private_ = getPrivateNodeHandle();

      raw_image_pipeline_ = std::make_shared<raw_image_pipeline::RawImagePipelineRos>(nh_, nh_private_);
      raw_image_pipeline_->run();
    }

    ros::NodeHandle nh_, nh_private_;
    std::shared_ptr<raw_image_pipeline::RawImagePipelineRos> raw_image_pipeline_;
};

//Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(RawImagePipelineNodelet, nodelet::Nodelet);
