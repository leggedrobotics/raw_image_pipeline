#include "debayer_cuda/debayer_cuda.h"


int main(int argc, char** argv)
{
  ros::init(argc, argv, "debayer_cuda");
  ros::NodeHandle nh;
  ros::NodeHandle nh_priv("~");

  debayer::DebayerCuda debayer_obj(nh, nh_priv);

  debayer_obj.setupROSparams();
  debayer_obj.setupSubAndPub();

  // Spin
  ros::AsyncSpinner spinner(1); // Use n threads
  spinner.start();
  ros::waitForShutdown();
  return 0;
}
