#include <image_proc_cuda/image_proc_cuda.hpp>

#include <boost/filesystem.hpp>
#define FILE_FOLDER (boost::filesystem::path(__FILE__).parent_path().string())
#define DEFAULT_PARAMS_PATH (FILE_FOLDER + "/../config/pipeline_params_example.yaml")

namespace image_proc_cuda 
{

ImageProcCuda::ImageProcCuda(const std::string& params_path, 
                             const std::string& calibration_path,
                             const std::string& color_calibration_path)
:   // Debug
    dump_images_(false),
    idx_(0)
{
    // Initialize FFCC
    ffccWBPtr_ = std::make_unique<ffcc::FastFourierColorConstancyWB>();

    // Load parameters
    if(params_path.empty())
        loadParams(DEFAULT_PARAMS_PATH);
    else
        loadParams(params_path);
    
    // Load calibration
    if(!calibration_path.empty())
        loadCalibration(calibration_path);
    
    // Load color calibration
    if(color_calibration_path.empty())
        loadColorCalibration(DEFAULT_PARAMS_PATH);
    else
        loadColorCalibration(color_calibration_path);
}

ImageProcCuda::~ImageProcCuda() {}

void ImageProcCuda::loadParams(const std::string& file_path)
{
    std::cout << "Loading image_proc_cuda params from file " << file_path << std::endl;

    // Check if file exists
    if(boost::filesystem::exists(file_path))
    {
        // Load parameters
        YAML::Node node = YAML::LoadFile(file_path);
        // Pipeline options
        needs_rotation_       = utils::get(node, "needs_rotation", false);
        run_white_balance_    = utils::get(node, "run_white_balance", false);
        run_gamma_correction_ = utils::get(node, "run_gamma_correction", false);
        run_vignetting_correction_ = utils::get(node, "run_vignetting_correction", false);
        run_clahe_          = utils::get(node, "run_clahe", false);
        run_color_enhancer_ = utils::get(node, "run_color_enhancer", false);
        run_color_calibration_ = utils::get(node, "run_color_calibration", false);
        run_undistortion_   = utils::get(node, "run_undistortion", false);
        keep_distorted_     = utils::get(node, "keep_distorted", false);
        // Debayer Params
        debayer_option_ = utils::get<std::string>(node, "debayer_option", "auto");
        // White balance
        white_balance_method_ = utils::get<std::string>(node, "white_balance_method", "ffcc");
        white_balance_clipping_percentile_  = utils::get(node, "white_balance_clipping_percentile", 20.0);
        white_balance_saturation_threshold_ = utils::get(node, "white_balance_saturation_threshold", 0.8);
        white_balance_temporal_consistency_ = utils::get(node, "white_balance_temporal_consistency", true);
        // Gamma correction
        gamma_correction_method_ = utils::get<std::string>(node, "gamma_correction_method", "custom");
        gamma_correction_k_      = utils::get(node, "gamma_correction_k", 0.8);
        is_gamma_correction_forward_ = (gamma_correction_k_ <= 1.0 ? true : false);
        // Vignetting correction
        vignetting_correction_scale_ = utils::get(node, "vignetting_correction_scale", 1.5);
        vignetting_correction_a2_    = utils::get(node, "vignetting_correction_a2", 1e-3);
        vignetting_correction_a4_    = utils::get(node, "vignetting_correction_a4", 1e-6);
        // CLAHE
        clahe_clip_limit_      = utils::get(node, "clahe_clip_limit", 1.5);
        clahe_tiles_grid_size_ = utils::get(node, "clahe_tiles_grid_size", 8);
        // Color enhancer
        color_enhancer_value_gain_      = utils::get(node, "color_enhancer_value_gain", 1.0);
        color_enhancer_saturation_gain_ = utils::get(node, "color_enhancer_saturation_gain", 1.0);
        color_enhancer_hue_gain_        = utils::get(node, "color_enhancer_hue_gain", 1.0);
    }
    else
    {
        std::cout << "Warning: parameters file doesn't exist" << std::endl;
    }
}

void ImageProcCuda::loadCalibration(const std::string& file_path)
{
    std::cout << "Loading camera calibration from file " << file_path << std::endl;

    // Check if file exists
    if(boost::filesystem::exists(file_path))
    {
        // Load calibration
        YAML::Node node = YAML::LoadFile(file_path);
        // Camera matrix
        camera_matrix_         = utils::get<cv::Matx33d>(node["camera_matrix"], "data", cv::Matx33d::eye());
        distortion_coeff_      = utils::get<cv::Matx14d>(node["distortion_coefficients"], "data", cv::Matx14d::zeros());
        rectification_matrix_  = utils::get<cv::Matx33d>(node["rectification_matrix"], "data", cv::Matx33d::eye());
        projection_matrix_     = utils::get<cv::Matx34d>(node["projection_matrix"], "data", cv::Matx34d::eye());
        distortion_model_      = utils::get<std::string>(node, "distortion_model", "none");
        int width  = utils::get<int>(node, "image_width", 320);
        int height = utils::get<int>(node, "image_height", 240);
        image_size_ = cv::Size(width, height);

        // Init rectify map
        initRectifyMap();
        calibration_available_ = true;
    }
    // If not, disable calibration available flag
    else
    {
        calibration_available_ = false;
        std::cout << "Warning: Calibration file doesn't exist" << std::endl;
    }
}

void ImageProcCuda::loadColorCalibration(const std::string& file_path)
{
    std::cout << "Loading color calibration from file " << file_path << std::endl;

    // Check if file exists
    if(boost::filesystem::exists(file_path))
    {
        // Load calibration
        YAML::Node node = YAML::LoadFile(file_path);
        // Camera matrix
        cv::Matx33d matrix = utils::get<cv::Matx33d>(node["color_calibration_matrix"], "data", cv::Matx33d::eye());
        initColorCalibrationMatrix(matrix);

        color_calibration_available_ = true;
    }
    // If not, disable calibration available flag
    else
    {
        color_calibration_available_ = false;
        std::cout << "Warning: Color calibration file doesn't exist" << std::endl;
    }
}

void ImageProcCuda::initRectifyMap()
{

    cv::Mat undistortion_map_x;
    cv::Mat undistortion_map_y;
    cv::fisheye::initUndistortRectifyMap(camera_matrix_,    // Intrinsics
                                        distortion_coeff_,     // Distortion
                                        rectification_matrix_, // Rectification
                                        projection_matrix_,    // New projection matrix
                                        image_size_,           // Image resolution
                                        CV_32F,                // Map type
                                        undistortion_map_x,    // Undistortion map for X axis
                                        undistortion_map_y     // Undistortion map for Y axis
    );
    // Upload everything to GPU
    undistortion_map_x_.upload(undistortion_map_x);
    undistortion_map_y_.upload(undistortion_map_y);
}

void ImageProcCuda::initColorCalibrationMatrix(const cv::Matx33d& matrix)
{
    for(size_t i=0; i<3; i++)
        for(size_t j=0; j<3; j++)
        {
            color_calibration_matrix_[i][j] = static_cast<float>(matrix(i,j));
        }
}

void ImageProcCuda::setDebayerOption(const std::string option) 
{
    debayer_option_ = option; 
}

void ImageProcCuda::setKeepDistorted(bool enabled)
{
    keep_distorted_ = enabled;
}

void ImageProcCuda::setFlip(bool enabled)
{
    needs_rotation_ = enabled;
}

void ImageProcCuda::setWhiteBalance(bool enabled)
{
    run_white_balance_ = enabled;
}

void ImageProcCuda::setWhiteBalanceMethod(const std::string& method)
{
    white_balance_method_ = method;
}

void ImageProcCuda::setWhiteBalancePercentile(const double& percentile)
{
    white_balance_clipping_percentile_ = percentile;
}

void ImageProcCuda::setWhiteBalanceSaturationThreshold(const double& threshold)
{
    white_balance_saturation_threshold_ = threshold;
}

void ImageProcCuda::setWhiteBalanceTemporalConsistency(bool enabled){
    white_balance_temporal_consistency_ = enabled;
}

void ImageProcCuda::setGammaCorrection(bool enabled){
    run_gamma_correction_ = enabled;
}

void ImageProcCuda::setGammaCorrectionMethod(const std::string& method)
{
    gamma_correction_method_ = method;
}

void ImageProcCuda::setGammaCorrectionK(const double& k)
{
    gamma_correction_k_ = k; 
    is_gamma_correction_forward_ = (gamma_correction_k_ <= 1.0 ? true : false);
}

void ImageProcCuda::setVignettingCorrection(bool enabled)
{
    run_vignetting_correction_ = enabled;
}

void ImageProcCuda::setVignettingCorrectionScale(const double& scale)
{
    vignetting_correction_scale_ = scale;
}

void ImageProcCuda::setVignettingCorrectionA2(const double& a2)
{
    vignetting_correction_a2_ = a2;
}

void ImageProcCuda::setVignettingCorrectionA4(const double& a4)
{
    vignetting_correction_a4_ = a4;
}

void ImageProcCuda::setClahe(bool enabled)
{
    run_clahe_ = enabled;
}

void ImageProcCuda::setClaheLimit(const double& limit)
{
    clahe_clip_limit_ = limit;
}

void ImageProcCuda::setClaheGridSize(const double& grid_size)
{
    clahe_tiles_grid_size_ = grid_size;
}

void ImageProcCuda::setColorEnhancer(bool enabled)
{
    run_color_enhancer_ = enabled;
}

void ImageProcCuda::setColorEnhancerHueGain(const double& gain)
{
    color_enhancer_value_gain_ = gain;
}

void ImageProcCuda::setColorEnhancerSaturationGain(const double& gain)
{
    color_enhancer_saturation_gain_ = gain;
}

void ImageProcCuda::setColorEnhancerValueGain(const double& gain)
{
    color_enhancer_hue_gain_ = gain;
}

void ImageProcCuda::setColorCalibration(bool enabled)
{
    run_color_calibration_ = enabled;
}

void ImageProcCuda::setColorCalibrationMatrix(const std::vector<double>& color_calibration_matrix)
{
    cv::Matx33d matrix = cv::Matx33d(color_calibration_matrix.data());
    initColorCalibrationMatrix(matrix);
}

void ImageProcCuda::setUndistortion(bool enabled)
{
    run_undistortion_ = enabled;
    calibration_available_ = true;
}

void ImageProcCuda::setUndistortionImageSize(int width, int height)
{
    image_size_ = cv::Size(width, height);
}

void ImageProcCuda::setUndistortionCameraMatrix(const std::vector<double>& camera_matrix)
{
    camera_matrix_ = cv::Matx33d(camera_matrix.data());
}

void ImageProcCuda::setUndistortionDistortionCoefficients(const std::vector<double>& coefficients)
{
    distortion_coeff_ = cv::Matx14d(coefficients.data());
}

void ImageProcCuda::setUndistortionDistortionModel(const std::string& model)
{
    distortion_model_ = model;
}

void ImageProcCuda::setUndistortionRectificationMatrix(const std::vector<double>& rectification_matrix)
{
    rectification_matrix_ = cv::Matx33d(rectification_matrix.data());
}

void ImageProcCuda::setUndistortionProjectionMatrix(const std::vector<double>& projection_matrix)
{
    projection_matrix_ = cv::Matx34d(projection_matrix.data());
}

void ImageProcCuda::setDumpImages(bool enabled)
{
    dump_images_ = enabled;
}

void ImageProcCuda::resetWhiteBalanceTemporalConsistency()
{
    ffccWBPtr_->resetTemporalConsistency();
}

int ImageProcCuda::getImageHeight() const
{
    return image_size_.height;
}

int ImageProcCuda::getImageWidth() const
{
    return image_size_.width;
}

std::string ImageProcCuda::getDistortionModel() const
{
    if(calibration_available_)
    {
        if(run_undistortion_)
        {
            return "none";
        }
        else
        {
            return distortion_model_;
        }
    }
    else {
        return "none";
    }
}
std::string ImageProcCuda::getOriginalDistortionModel() const
{
    if(calibration_available_)
    {
        return distortion_model_;
    }
    else {
        return "none";
    }
}

cv::Mat ImageProcCuda::getCameraMatrix() const
{
    cv::Rect slice(0, 0, 3, 3);
    return cv::Mat(projection_matrix_)(slice).clone();
}

cv::Mat ImageProcCuda::getOriginalCameraMatrix() const
{
    return cv::Mat(camera_matrix_).clone();
}

cv::Mat ImageProcCuda::getDistortionCoefficients() const
{
    if(calibration_available_)
    {
        if(run_undistortion_)
        {
            // Image was undistorted, so it's all zeros
            return cv::Mat::zeros(1, 4, CV_64F);
        }
        else
        {
            // Return original distortion vector
            return cv::Mat(distortion_coeff_).clone();
        }
    } 
    else
    {
        // Return just zeros
        return cv::Mat::zeros(1, 4, CV_64F);
    }
}

cv::Mat ImageProcCuda::getOriginalDistortionCoefficients() const
{
    if(calibration_available_)
    {
        // Return original distortion vector
        return cv::Mat(distortion_coeff_).clone();
    } 
    else
    {
        // Return just zeros
        return cv::Mat::zeros(1, 4, CV_64F);
    }
}

cv::Mat ImageProcCuda::getRectificationMatrix() const
{
    return cv::Mat(rectification_matrix_).clone();
}

cv::Mat ImageProcCuda::getOriginalRectificationMatrix() const
{
    return cv::Mat(rectification_matrix_).clone();
}


cv::Mat ImageProcCuda::getProjectionMatrix() const
{
    return cv::Mat(projection_matrix_).clone();
}

cv::Mat ImageProcCuda::getOriginalProjectionMatrix() const
{
    return cv::Mat(projection_matrix_).clone();
}

bool ImageProcCuda::getUndistortionEnabled() const 
{
    return run_undistortion_;
}

std::vector<double> ImageProcCuda::getColorCalibrationMatrix() const
{
    std::vector<double> out;
    out.reserve(9);
    for (size_t i=0; i < 3; i++)
        for (size_t j=0; j<3; j++)
            out.push_back(color_calibration_matrix_[i][j]);
    return out;
}

cv::Mat ImageProcCuda::getDistortedImage() const
{
    return distorted_image_.clone();
}

cv::Mat ImageProcCuda::process(const cv::Mat& image, const std::string& encoding) {
    cv::Mat out = image.clone();
    // Apply pipeline
    apply(out, encoding);
    // Return copy
    return out.clone();
}

bool ImageProcCuda::apply(cv::Mat& image, const std::string& encoding)
{
    // Send image to GPU
    cv::cuda::GpuMat image_d;
    image_d.upload(image);
    dumpGpuImage("bayered", image_d);

    // Debayer to change the encoding
    debayer(image_d, encoding, debayer_option_);
    dumpGpuImage("debayered", image_d);

    // Flip image 180 deg
    if (needs_rotation_)
        flip(image_d);
    dumpGpuImage("flipped", image_d);

    // White balance
    if (run_white_balance_)
        whiteBalance(image_d, white_balance_method_);
    dumpGpuImage("white_balanced", image_d);

    // Apply color calibration
    if (run_color_calibration_)
        colorCalibration(image_d);
    dumpGpuImage("color_calibrated", image_d);

    // Gamma Correction to adjust illumination
    if (run_gamma_correction_)
        gammaCorrection(image_d, gamma_correction_method_);
    dumpGpuImage("gamma_corrected", image_d);

    // Vignetting Correction
    if (run_vignetting_correction_)
        vignettingCorrection(image_d);
    dumpGpuImage("vignetting_corrected", image_d);

    // Contrast Limited Adaptive Histogram Equalization
    if (run_clahe_)
        clahe(image_d, clahe_clip_limit_, clahe_tiles_grid_size_);
    dumpGpuImage("clahe", image_d);

    // To increase saturation of colors
    if (run_color_enhancer_)
        colorEnhancer(image_d);
    dumpGpuImage("color_enhanced", image_d);

    // Undistort
    if (keep_distorted_)
    {
        image_d.download(distorted_image_);
    }

    if (run_undistortion_)
        undistort(image_d);
    dumpGpuImage("undistorted", image_d);

    // Get back to CPU
    image_d.download(image);

    // Increase counter
    idx_++;

    return true;
}

void ImageProcCuda::debayer(cv::cuda::GpuMat& image, const std::string& image_encoding,
                          const std::string& encoding_option)
{
    // First check if we selected auto as the encoding option, otherwise we use the value given by
    // the user
    std::string encoding = encoding_option == "auto" ? image_encoding : encoding_option;
    cv::cuda::GpuMat out;

    // We only apply demosaicing (debayer) if the format is valid
    if (encoding == "bayer_bggr8")
    {
        cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerBG2BGR_MHT);
        image = out;
    }
    else if (encoding == "bayer_gbrg8")
    {
        cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerGB2BGR_MHT);
        image = out;
    }
    else if (encoding == "bayer_grbg8")
    {
        cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerGR2BGR_MHT);
        image = out;
    }
    else if (encoding == "bayer_rggb8")
    {
        cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerRG2BGR_MHT);
        image = out;
    }
    // We ignore non-bayer encodings
    else if (isBayerEncoding(encoding))
    {
        throw std::invalid_argument("Encoding [" + encoding + "] is a valid pattern but is not supported!");
    }
}

bool ImageProcCuda::isBayerEncoding(const std::string& encoding)
{
    // Find if encoding is in list of Bayer types
    return std::find(BAYER_TYPES.begin(), BAYER_TYPES.end(), encoding) != BAYER_TYPES.end();
}

void ImageProcCuda::flip(cv::cuda::GpuMat& image)
{
    cv::cuda::GpuMat out;
    cv::cuda::flip(image, out, -1);  // negative numbers flip x and y
    image = out;
}

void ImageProcCuda::whiteBalance(cv::cuda::GpuMat& image, const std::string& wb_method)
{
    if(image.channels() != 3){
        return;
    }

    if (wb_method == "simple")
    {
        cv::Mat tmp;
        image.download(tmp);

        // Simple OpenCV white balancing
        cv::Ptr<cv::xphoto::SimpleWB> wb;
        wb = cv::xphoto::createSimpleWB();
        // Percentiles of highest and lowest pixel values to ignore
        wb->setP(white_balance_clipping_percentile_);
        wb->balanceWhite(tmp, tmp);
        image.upload(tmp);
    }
    else if (wb_method == "gray_world" || wb_method == "grey_world")
    {
        cv::Mat tmp;
        image.download(tmp);

        // Gray world OpenCV white balancing
        cv::Ptr<cv::xphoto::GrayworldWB> wb;
        wb = cv::xphoto::createGrayworldWB();
        wb->setSaturationThreshold(white_balance_saturation_threshold_);
        wb->balanceWhite(tmp, tmp);
        image.upload(tmp);
    }
    else if (wb_method == "learned")
    {
        cv::Mat tmp;
        image.download(tmp);

        // Gray world OpenCV white balancing
        cv::Ptr<cv::xphoto::LearningBasedWB> wb;
        wb = cv::xphoto::createLearningBasedWB();
        wb->setSaturationThreshold(white_balance_saturation_threshold_);
        wb->balanceWhite(tmp, tmp);
        image.upload(tmp);
    }
    else if (wb_method == "ffcc")
    {
        // cv::Mat tmp;
        // image.download(tmp);

        // FFCC white balancing - this works directly on GPU
        ffccWBPtr_->setSaturationThreshold(white_balance_saturation_threshold_);
        ffccWBPtr_->setTemporalConsistency(white_balance_temporal_consistency_);
        ffccWBPtr_->setDebug(false);
        cv::Mat out;
        ffccWBPtr_->balanceWhite(image, image);
        // image.upload(out);
    }
    else if (wb_method == "pca")
    {
        cv::Mat tmp;
        image.download(tmp);

        // FFCC white balancing
        pcaBalanceWhite(tmp);

        image.upload(tmp);
    }
    else
    {
        throw std::invalid_argument(
            "White Balance method ["
            + wb_method
            + "] not supported. "
               "Supported algorithms: 'simple', 'gray_world', 'learned', 'ffcc', 'pca'");
    }
}

void ImageProcCuda::gammaCorrection(cv::cuda::GpuMat& image, const std::string& method)
{
    if (method == "custom")
    {
        customGammaCorrection(image, gamma_correction_k_);
    }
    else
    {
        defaultGammaCorrection(image, is_gamma_correction_forward_);
    }
}

void ImageProcCuda::defaultGammaCorrection(cv::cuda::GpuMat& image, bool is_forward)
{
    cv::cuda::GpuMat out;
    cv::cuda::gammaCorrection(image, out, is_forward);
    image = out;
}

// Gamma correction test with own implementation
void ImageProcCuda::customGammaCorrection(cv::cuda::GpuMat& image, float k)
{
    cv::Mat tmp;
    image.download(tmp);

    cv::Mat dst;

    uchar LUT[256];
    tmp.copyTo(dst);
    for (int i = 0; i < 256; i++)
    {
        float f = i / 255.0;
        f = pow(f, k);
        LUT[i] = cv::saturate_cast<uchar>(f * 255.0);
    }
    if (dst.channels() == 1)
    {
        cv::MatIterator_<uchar> it = dst.begin<uchar>();
        cv::MatIterator_<uchar> it_end = dst.end<uchar>();
        for (; it != it_end; ++it)
        {
            *it = LUT[(*it)];
        }
    }
    else
    {
        cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
        cv::MatIterator_<cv::Vec3b> it_end = dst.end<cv::Vec3b>();
        for (; it != it_end; ++it)
        {
            (*it)[0] = LUT[(*it)[0]];
            (*it)[1] = LUT[(*it)[1]];
            (*it)[2] = LUT[(*it)[2]];
        }
    }
    image.upload(dst);
}

void ImageProcCuda::precomputeVignettingMask(int height, int width)
{
    if (height == vignetting_mask_f_.rows && width == vignetting_mask_f_.cols)
        return;

    // Initialize mask
    cv::Mat mask(width, height, CV_32F);

    double cx = width / 2.0;
    double cy = height / 2.0;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double r = std::sqrt(std::pow(y - cy, 2) + std::pow(x - cx, 2));
            double k = pow(r, 2) * vignetting_correction_a2_ +
                       pow(r, 4) * vignetting_correction_a4_;
            mask.at<float>(x, y) = k;
            // mask.at<float>(x, y) = k;
        }
    }
    double max;
    cv::minMaxLoc(mask, NULL, &max, NULL, NULL);
    mask = max > 0? mask / max : mask;
    // Scale correction
    mask = mask * vignetting_correction_scale_;
    // Add 1
    mask += 1.0;

    // Upload to gpu
    vignetting_mask_f_.upload(mask);
}

void ImageProcCuda::vignettingCorrection(cv::cuda::GpuMat& image)
{
    precomputeVignettingMask(image.cols, image.rows);
    dumpGpuImage("vignetting_mask", vignetting_mask_f_);

    // Histogram equalization
    cv::cuda::GpuMat corrected_image;
    cv::cuda::cvtColor(image, corrected_image, cv::COLOR_BGR2Lab);

    // Split the image into 3 channels
    std::vector<cv::cuda::GpuMat> vec_channels(3);
    cv::cuda::split(corrected_image, vec_channels);

    // Floating point version
    // Convert image to float
    cv::cuda::GpuMat image_f_;
    vec_channels[0].convertTo(image_f_, CV_32FC1);
    // Multiply by vignetting mask
    cv::cuda::multiply(image_f_, vignetting_mask_f_, image_f_, 1.0, CV_32FC1);
    vec_channels[0].release();
    image_f_.convertTo(vec_channels[0], CV_8UC1);

    // Merge 3 channels in the vector to form the color image in LAB color space.
    cv::cuda::merge(vec_channels, corrected_image);

    // Convert the histogram equalized image from LAB to BGR color space again
    cv::cuda::cvtColor(corrected_image, image, cv::COLOR_Lab2BGR);
}

void ImageProcCuda::clahe(cv::cuda::GpuMat& image, float clip_limit, int tiles_grid_size)
{
    // Histogram equalization
    cv::cuda::GpuMat equalized_image;
    cv::cuda::cvtColor(image, equalized_image, cv::COLOR_BGR2Lab);

    // Split the image into 3 channels
    std::vector<cv::cuda::GpuMat> vec_channels(3);
    cv::cuda::split(equalized_image, vec_channels);

    cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE();
    clahe->setClipLimit(clip_limit);
    clahe->setTilesGridSize(cv::Size(tiles_grid_size, tiles_grid_size));

    // Apply CLAHE to Y channel
    clahe->apply(vec_channels[0], equalized_image);
    vec_channels[0].release();
    equalized_image.copyTo(vec_channels[0]);

    // Merge 3 channels in the vector to form the color image in LAB color space.
    cv::cuda::merge(vec_channels, equalized_image);

    // Convert the histogram equalized image from LAB to BGR color space again
    cv::cuda::cvtColor(equalized_image, image, cv::COLOR_Lab2BGR);
}

void ImageProcCuda::colorEnhancer(cv::cuda::GpuMat& image)
{
    if(image.channels() != 3){
        return;
    }
    
    // Color enhancer
    cv::cuda::GpuMat color_enhanced_image;
    cv::cuda::cvtColor(image, color_enhanced_image, cv::COLOR_BGR2HSV);

    // Split the image into 3 channels: H (hue), S (saturation), V (value)
    std::vector<cv::cuda::GpuMat> vec_channels(3);
    cv::cuda::split(color_enhanced_image, vec_channels);

    // Apply gains per channel
    cv::cuda::multiply(vec_channels[0], color_enhancer_hue_gain_, vec_channels[0]);
    cv::cuda::multiply(vec_channels[1], color_enhancer_saturation_gain_, vec_channels[1]);
    cv::cuda::multiply(vec_channels[2], color_enhancer_value_gain_, vec_channels[2]);

    // Merge 3 channels in the vector to form the color image in HSV color space.
    cv::cuda::merge(vec_channels, color_enhanced_image);

    // Convert the histogram equalized image from HSV to BGR color space again
    cv::cuda::cvtColor(color_enhanced_image, image, cv::COLOR_HSV2BGR);
}

void ImageProcCuda::colorCalibration(cv::cuda::GpuMat& image)
{
    if(!color_calibration_available_)
    {
        std::cout << "No color calibration available!" << std::endl;
        return;
    }

    NppiSize image_size;
    image_size.width  = image.cols;
    image_size.height = image.rows;

    // Apply calibration
    // nppiColorTwist32f_8u_C3IR(image.ptr<Npp8u>(), static_cast<int>(image.step), image_size, color_calibration_matrix_);
    nppiColorTwist32f_8u_C3IR(image.data, static_cast<int>(image.step), image_size, color_calibration_matrix_);

}

void ImageProcCuda::undistort(cv::cuda::GpuMat& image)
{
    cv::cuda::GpuMat out;

    if(!calibration_available_)
    {
        std::cout << "No calibration available!" << std::endl;
        return;
    }

    if (distortion_model_ == "equidistant")
    {    
        cv::cuda::remap(image, out, undistortion_map_x_, undistortion_map_y_,
                  cv::InterpolationFlags::INTER_LINEAR, cv::BorderTypes::BORDER_REPLICATE);
        image = out;
        return;
    }
    else 
    {
        // do nothing
        return;
    }
}

void ImageProcCuda::dumpGpuImage(const std::string& name, const cv::cuda::GpuMat& image)
{
    if (dump_images_)
    {
        cv::Mat tmp;
        image.download(tmp);
        cv::normalize(tmp, tmp, 0, 255.0, cv::NORM_MINMAX);
        cv::imwrite("/tmp/" + std::to_string(idx_) + "_" + name + ".png", tmp);
    }
}

void ImageProcCuda::pcaBalanceWhite(cv::Mat& image)
{
    // Note: BGR input

    // Split channels
    std::vector<cv::Mat> split_img;
    cv::split(image, split_img);
    split_img[0].convertTo(split_img[0], CV_32FC1);
    split_img[2].convertTo(split_img[2], CV_32FC1);

    // Get elementwise squared values
    cv::Mat I_r_2;
    cv::Mat I_b_2;
    cv::multiply(split_img[0], split_img[0], I_b_2);
    cv::multiply(split_img[2], split_img[2], I_r_2);

    // Get summed up channels
    const double sum_I_r_2 = cv::sum(I_r_2)[0];
    const double sum_I_b_2 = cv::sum(I_b_2)[0];
    const double sum_I_g = cv::sum(split_img[1])[0];
    const double sum_I_r = cv::sum(split_img[2])[0];
    const double sum_I_b = cv::sum(split_img[0])[0];

    // Get max values of channels
    double max_I_r, max_I_g, max_I_b, max_I_r_2, max_I_b_2;
    double min_I_r, min_I_g, min_I_b, min_I_r_2, min_I_b_2;
    cv::minMaxLoc(split_img[2], &min_I_r, &max_I_r);  // R
    cv::minMaxLoc(split_img[1], &min_I_g, &max_I_g);  // G
    cv::minMaxLoc(split_img[0], &min_I_b, &max_I_b);  // B
    cv::minMaxLoc(I_r_2, &min_I_r_2, &max_I_r_2);
    cv::minMaxLoc(I_b_2, &min_I_b_2, &max_I_b_2);

    // Prepare Matrices for PCA method
    Eigen::Matrix2f mat_temp_b;
    mat_temp_b << sum_I_b_2, sum_I_b, max_I_b_2, max_I_b;
    Eigen::Matrix2f mat_temp_r;
    mat_temp_r << sum_I_r_2, sum_I_r, max_I_r_2, max_I_r;
    Eigen::Vector2f vec_temp_g;
    vec_temp_g << sum_I_g, max_I_g;

    // PCA method calculation
    Eigen::Vector2f vec_out_b, vec_out_r;
    vec_out_b = mat_temp_b.inverse() * vec_temp_g;
    vec_out_r = mat_temp_r.inverse() * vec_temp_g;
    cv::Mat b_point = vec_out_b[0] * I_b_2 + vec_out_b[1] * split_img[0];
    cv::Mat r_point = vec_out_r[0] * I_r_2 + vec_out_r[1] * split_img[2];

    // Saturate values above 255
    cv::threshold(b_point, b_point, 255, 255, cv::THRESH_TRUNC);
    cv::threshold(r_point, r_point, 255, 255, cv::THRESH_TRUNC);

    // Convert back to UINT8
    b_point.convertTo(b_point, CV_8UC1);
    r_point.convertTo(r_point, CV_8UC1);

    // Merge channels
    std::vector<cv::Mat> channels;
    channels.push_back(b_point);
    channels.push_back(split_img[1]);
    channels.push_back(r_point);
    cv::Mat merged_image;
    cv::merge(channels, merged_image);

    image = merged_image;
}

} // namespace image_proc_cuda