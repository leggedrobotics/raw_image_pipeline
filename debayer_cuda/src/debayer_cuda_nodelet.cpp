#include "debayer_cuda/debayer_cuda.h"

#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>


class DebayerCudaNodelet : public nodelet::Nodelet {

  public:
    DebayerCudaNodelet(){}
    ~DebayerCudaNodelet(){}

  private:
    virtual void onInit() {

      nh_ = getNodeHandle();
      nh_private_ = getPrivateNodeHandle();

      debayer_ = new debayer::DebayerCuda(nh_, nh_private_);

      debayer_->setupROSparams();
      debayer_->setupSubAndPub();
    }

    ros::NodeHandle nh_, nh_private_;
    debayer::DebayerCuda* debayer_;
};

//Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(DebayerCudaNodelet, nodelet::Nodelet);
