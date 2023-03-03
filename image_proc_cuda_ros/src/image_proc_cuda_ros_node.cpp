#include <image_proc_cuda_ros/image_proc_cuda_ros.hpp>


int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_proc_cuda_ros");
  
  ros::NodeHandle nh;
  ros::NodeHandle nh_priv("~");

  image_proc_cuda::ImageProcCudaRos image_proc(nh, nh_priv);

  image_proc.setupRosParams();
  image_proc.setupSubAndPub();
  image_proc.run();
  
  ros::waitForShutdown();
  return 0;
}
