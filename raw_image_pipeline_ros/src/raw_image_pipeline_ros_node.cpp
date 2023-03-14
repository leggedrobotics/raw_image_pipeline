#include <raw_image_pipeline_ros/raw_image_pipeline_ros.hpp>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "raw_image_pipeline_ros");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_priv("~");

  raw_image_pipeline::RawImagePipelineRos image_proc(nh, nh_priv);
  image_proc.run();
  
  ros::waitForShutdown();
  return 0;
}
