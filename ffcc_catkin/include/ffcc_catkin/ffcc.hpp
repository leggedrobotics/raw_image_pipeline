/**
@brief ffcc.hpp
Class for White Balance built upon "AutoWhiteBalance" code by Shane Yuan
based on Barron, "Fast Fourier Color Constancy", CVPR, 2017

Author: Matias Mattamala
*/

#pragma once

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <fstream>
#include <cmath>
#include <npp.h>
#include <stdio.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

namespace ffcc
{
class FastFourierColorConstancyWB
{
private:
    // Model
    struct Model
    {
        int width_;
        int height_;
        cv::Mat filter_;
        cv::Mat bias_;
    } model_;

    // Model filename
    std::string model_filename_;

    // Histogram feature
    int width;
    int height;
    cv::Mat image_histogram_;

    // input image
    // cv::cuda::GpuMat img_d;
    cv::Size small_size_;

    // histogram step
    float bin_size_;
    float uv0_;

    // Convolution variables (CUDA)
    cv::cuda::GpuMat hist_d_;
    cv::cuda::GpuMat filter_d_;
    cv::cuda::GpuMat bias_d_;
    cv::cuda::GpuMat response_fft_;
    cv::cuda::GpuMat response_;
    cv::cuda::GpuMat filter_fft_;
    cv::cuda::GpuMat bias_fft_;
    cv::cuda::GpuMat hist_fft_;

    // Output variables
    cv::Point uv_pos_;
    cv::Point uv_pos_prev_;
    float gain_r, gain_g, gain_b;

    // kalman filter use to smooth result
    cv::Mat kf_measurement_;
    cv::Mat kf_estimate_;
    cv::Mat kf_inv_H_cov_; // Inverse of measurement model covariance
    cv::Mat kf_H_; // Observation model
    std::shared_ptr<cv::KalmanFilter> kf_ptr_;

    // Parameters
    float saturation_thr_;
    bool use_temporal_consistency_;
    bool debug_;
    bool debug_uv_offset_;

    // Debugging
    float Lu_debug_;
    float Lv_debug_;

    // Other helper variables
    bool first_frame_;
    size_t idx_;

private:
    // Load pretrained model
    int loadModel(const std::string& model_file);

    // Calculate histogram of input image
    void calculateHistogramFeature(const cv::Mat& src, std::string out_name = "input");

    // Computes the filter response to obtain the UV offset
    int computeResponse();

    // Enforces temporal consistency with a Kalman filter
    int kalmanFiltering();

    // Compute and apply RGB gains
    void applyWhiteBalance(cv::cuda::GpuMat& dst);

public:
    FastFourierColorConstancyWB(const std::string& filename);
    FastFourierColorConstancyWB();
    ~FastFourierColorConstancyWB();

    // Applies white balance
    void balanceWhite(const cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst);

    // Set threshold
    void resetTemporalConsistency();
    void setSaturationThreshold(float threshold);
    void setTemporalConsistency(bool enable);
    void setUV0(float uv0);
    void setDebug(bool debug);
    void setDebugUVOffset(float Lu, float Lv, float uv0);
    
};

}  // namespace ffcc