// Author: Matias Mattamala
// Author: Timon Homberger

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/xphoto.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <npp.h>
#include <yaml-cpp/yaml.h>

#include <ffcc_catkin/ffcc.hpp>
#include <image_proc_cuda/utils.hpp>

namespace image_proc_cuda
{

class ImageProcCuda
{
    // Private list of bayer types
    std::vector<std::string> BAYER_TYPES = {
        "bayer_bggr8",
        "bayer_gbrg8",
        "bayer_grbg8",
        "bayer_rggb8"
        "bayer_bggr16",
        "bayer_gbrg16",
        "bayer_grbg16",
        "bayer_rggb16"
    };

public:
    // Constructor & destructor
    ImageProcCuda(const std::string& params_path = "", 
                  const std::string& calibration_path = "",
                  const std::string& color_calibration_path = "");
    ~ImageProcCuda();

    // Main interface to apply the pipeline
    bool apply(cv::Mat& image, const std::string& encoding);

    // Alternative pipeline that returns a copy
    cv::Mat process(const cv::Mat& image, const std::string& encoding);

    // Loaders
    void loadParams(const std::string& file_path);
    void loadCalibration(const std::string& file_path);
    void loadColorCalibration(const std::string& file_path);
    void initRectifyMap();
    void initColorCalibrationMatrix(const cv::Matx33d& matrix);

    // Setters
    void setDebayerOption(const std::string option);
    void setFlip(bool enabled);
    void setWhiteBalance(bool enabled);
    void setWhiteBalanceMethod(const std::string& method);
    void setWhiteBalancePercentile(const double& percentile);
    void setWhiteBalanceSaturationThreshold(const double& threshold);
    void setWhiteBalanceTemporalConsistency(bool enabled);
    void setGammaCorrection(bool enabled);
    void setGammaCorrectionMethod(const std::string& method);
    void setGammaCorrectionK(const double& k);
    void setVignettingCorrection(bool enabled);
    void setVignettingCorrectionScale(const double& scale);
    void setVignettingCorrectionA2(const double& a2);
    void setVignettingCorrectionA4(const double& a4);
    void setClahe(bool enabled);
    void setClaheLimit(const double& limit);
    void setClaheGridSize(const double& grid_size);
    void setColorEnhancer(bool enabled);
    void setColorEnhancerHueGain(const double& gain);
    void setColorEnhancerSaturationGain(const double& gain);
    void setColorEnhancerValueGain(const double& gain);
    void setColorCalibration(bool enabled);
    void setColorCalibrationMatrix(const std::vector<double>& color_calibration_matrix);
    void setUndistortion(bool enabled);
    void setUndistortionImageSize(int width, int height);
    void setUndistortionCameraMatrix(const std::vector<double>& camera_matrix);
    void setUndistortionDistortionCoefficients(const std::vector<double>& coefficients);
    void setUndistortionDistortionModel(const std::string& model);
    void setUndistortionRectificationMatrix(const std::vector<double>& rectification_matrix);
    void setUndistortionProjectionMatrix(const std::vector<double>& projection_matrix);
    void setDumpImages(bool enabled);

    // Other interfaces
    void resetWhiteBalanceTemporalConsistency();

    // Getters
    bool getUndistortionEnabled() const;
    int getImageHeight() const;
    int getImageWidth() const;
    std::string getDistortionModel() const;
    cv::Mat getCameraMatrix() const;
    cv::Mat getDistortionCoefficients() const;
    cv::Mat getRectificationMatrix() const;
    cv::Mat getProjectionMatrix() const;
    std::vector<double> getColorCalibrationMatrix() const;
    
private:
    // Applies demosaicing
    void debayer(cv::cuda::GpuMat& image, const std::string& image_encoding,
                 const std::string& encoding_option);
    // Check if the format is Bayer
    bool isBayerEncoding(const std::string& encoding);

    // Flips the image (180 deg)
    void flip(cv::cuda::GpuMat& image);

    // White balancing
    void whiteBalance(cv::cuda::GpuMat& image, const std::string& wb_method);

    // Gamma correction
    void gammaCorrection(cv::cuda::GpuMat& image, const std::string& method);
    // OpenCV's gamma correction
    void defaultGammaCorrection(cv::cuda::GpuMat& image, bool is_forward);
    // Manual gamma detection
    void customGammaCorrection(cv::cuda::GpuMat& image, float k);

    // Vignetting correction
    void precomputeVignettingMask(int height, int width);
    void vignettingCorrection(cv::cuda::GpuMat& image);

    // Histogram equalization
    void clahe(cv::cuda::GpuMat& src, float clip_limit, int tiles_grid_size);

    // Color enhancer
    void colorEnhancer(cv::cuda::GpuMat& image);

    // Color calibration
    void colorCalibration(cv::cuda::GpuMat& image);

    // Fisheye distortion correction
    void undistort(cv::cuda::GpuMat& image);

    // PCA-based color correction
    void pcaBalanceWhite(cv::Mat& image);

    // Helper to save images
    void dumpGpuImage(const std::string& name, const cv::cuda::GpuMat& image);

    // Pointers
    std::unique_ptr<ffcc::FastFourierColorConstancyWB> ffccWBPtr_;

    // Pipeline options
    bool needs_rotation_;
    bool run_white_balance_;
    bool run_gamma_correction_;
    bool run_vignetting_correction_;
    bool run_clahe_;
    bool run_color_enhancer_;
    bool run_color_calibration_;
    bool run_undistortion_;

    // Debayer Params
    std::string debayer_option_;

    // White balance
    std::string white_balance_method_;
    double white_balance_clipping_percentile_;
    double white_balance_saturation_threshold_;
    bool white_balance_temporal_consistency_;

    // Gamma correction
    std::string gamma_correction_method_;
    bool is_gamma_correction_forward_;
    double gamma_correction_k_;

    // Vignetting correction
    cv::cuda::GpuMat vignetting_mask_f_;
    cv::cuda::GpuMat image_f_;
    double vignetting_correction_scale_;
    double vignetting_correction_a2_;
    double vignetting_correction_a4_;

    // CLAHE
    double clahe_clip_limit_;
    int clahe_tiles_grid_size_;

    // Color enhancer
    double color_enhancer_value_gain_;
    double color_enhancer_saturation_gain_;
    double color_enhancer_hue_gain_;

    // Color calibration
    bool color_calibration_available_;
    Npp32f color_calibration_matrix_[3][4] = {{1.f, 0.f, 0.f, 0.f},
                                              {0.f, 1.f, 0.f, 0.f},
                                              {0.f, 0.f, 1.f, 0.f}};

    // Calibration & undistortion
    bool calibration_available_;
    std::string distortion_model_;
    cv::Matx33d camera_matrix_;
    cv::Matx14d distortion_coeff_;
    cv::Matx33d rectification_matrix_;
    cv::Matx34d projection_matrix_;
    cv::Size image_size_;
    cv::cuda::GpuMat undistortion_map_x_;
    cv::cuda::GpuMat undistortion_map_y_;

    // Debug
    bool dump_images_;
    size_t idx_;
};

} // namespace image_proc_cuda