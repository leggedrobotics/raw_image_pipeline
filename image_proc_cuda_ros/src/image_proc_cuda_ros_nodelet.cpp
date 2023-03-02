#include <image_proc_cuda_ros/image_proc_cuda_ros.hpp>
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>
#include <memory>

class ImageProcCudaNodelet : public nodelet::Nodelet {

  public:
    ImageProcCudaNodelet(){}
    ~ImageProcCudaNodelet(){}

  private:
    virtual void onInit() {

      nh_ = getNodeHandle();
      nh_private_ = getPrivateNodeHandle();

      image_proc_ = std::make_shared<image_proc_cuda::ImageProcCudaRos>(nh_, nh_private_);

      image_proc_->setupRosParams();
      image_proc_->setupSubAndPub();
    }

    ros::NodeHandle nh_, nh_private_;
    std::shared_ptr<image_proc_cuda::ImageProcCudaRos> image_proc_;
};

//Declare as a Plug-in
PLUGINLIB_EXPORT_CLASS(ImageProcCudaNodelet, nodelet::Nodelet);
