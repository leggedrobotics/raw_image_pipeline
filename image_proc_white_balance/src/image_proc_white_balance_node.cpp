/**
@brief image_proc_white_balance_node.cpp
simple test node to control UV using rqt_reconfigure

author: Matias Mattamala

*/

// ROS
#include <ros/ros.h>
#include <ros/package.h>  // to get the current directory of the package.

#include <opencv2/opencv.hpp>
#include <image_proc_white_balance/convolutional_color_constancy.hpp>

#include <ros/ros.h>

#include <dynamic_reconfigure/server.h>
#include <image_proc_white_balance/ImageProcWhiteBalanceConfig.h>

class WhiteBalanceRos
{
public:
    dynamic_reconfigure::Server<image_proc_white_balance::ImageProcWhiteBalanceConfig> server_;
    dynamic_reconfigure::Server<image_proc_white_balance::ImageProcWhiteBalanceConfig>::CallbackType f_;
    std::string model_file_;
    std::string image_file_;
		cv::Mat image_;
    cv::cuda::GpuMat image_d_;
		std::unique_ptr<image_proc_white_balance::ConvolutionalColorConstancyWB> wb_;

    WhiteBalanceRos(std::string model_file, std::string image_file)
    {
        model_file_ = model_file;
        image_file_ = image_file;

				image_ = cv::imread(image_file_, cv::IMREAD_COLOR);
				wb_ = std::make_unique<image_proc_white_balance::ConvolutionalColorConstancyWB>(model_file_);
				wb_->setUV0(-1.421875);
        wb_->setDebug(true);
        image_d_.upload(image_);

				cv::Mat wb_image;
        cv::cuda::GpuMat wb_image_d_;
        wb_->balanceWhite(image_d_, wb_image_d_);
        wb_image_d_.download(wb_image);

        cv::imshow("original", image_);
        cv::imshow("corrected", wb_image);
        cv::waitKey(10);

				f_ = boost::bind(&WhiteBalanceRos::callback, this, _1, _2);
        server_.setCallback(f_);
    }

    void callback(image_proc_white_balance::ImageProcWhiteBalanceConfig &config, uint32_t level)
    {
        wb_->setSaturationThreshold(config.saturation_threshold);
        wb_->setDebugUVOffset(config.Lu_offset, config.Lv_offset, config.uv0);

        cv::Mat wb_image;
        cv::cuda::GpuMat wb_image_d_;
        wb_->balanceWhite(image_d_, wb_image_d_);
        wb_image_d_.download(wb_image);

        cv::imshow("original", image_);
        cv::imshow("corrected", wb_image);
        cv::waitKey(10);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "image_proc_white_balance");
    ros::NodeHandle nh_priv("~");
    std::string model_file = ros::package::getPath("image_proc_white_balance") + "/model/default.bin";
    std::string image_file = ros::package::getPath("image_proc_white_balance") + "/data/alphasense.png";

    // Get input image path
    nh_priv.param<std::string>("image", image_file, image_file);

    WhiteBalanceRos wb(model_file, image_file);

    ROS_INFO("Spinning node");
    ros::spin();
    return 0;
}
