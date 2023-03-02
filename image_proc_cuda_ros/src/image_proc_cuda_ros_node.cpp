#include <image_proc_cuda_ros/image_proc_cuda_ros.hpp>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_proc_cuda_ros");

  // std::string node_name = ros::this_node::getName()

  ros::NodeHandle nh;
  ros::NodeHandle nh_priv("~");

  image_proc_cuda::ImageProcCudaRos image_proc(nh, nh_priv);

  image_proc.setupRosParams();
  image_proc.setupSubAndPub();

  // Spin
  ros::AsyncSpinner spinner(1); // Use n threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
