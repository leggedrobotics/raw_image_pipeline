#include <image_proc_cuda/image_proc_cuda.hpp>
#include <pybind11/pybind11.h>
#include <cvnp/cvnp.h>

namespace py = pybind11;
using namespace image_proc_cuda;

PYBIND11_MODULE(_py_image_proc_cuda, m) {
    m.doc() = "Image Proc Cuda bindings"; // module docstring
    py::class_<ImageProcCuda>(m, "ImageProcCuda")
        .def(py::init<const std::string &, const std::string &, const std::string &>(), 
                py::arg("params_path") = "", 
                py::arg("calibration_path") = "",
                py::arg("color_calibration_path") = "")
        .def("apply", &ImageProcCuda::apply)
        .def("process", &ImageProcCuda::process)
        .def("load_params", &ImageProcCuda::loadParams)
        .def("load_calibration", &ImageProcCuda::loadCalibration)
        .def("load_color_calibration", &ImageProcCuda::loadColorCalibration)
        .def("init_rectify_map", &ImageProcCuda::initRectifyMap)
        .def("set_debayer_option", &ImageProcCuda::setDebayerOption)
        .def("set_flip", &ImageProcCuda::setFlip)
        .def("set_white_balance", &ImageProcCuda::setWhiteBalance)
        .def("set_white_balance_method", &ImageProcCuda::setWhiteBalanceMethod)
        .def("set_white_balance_percentile", &ImageProcCuda::setWhiteBalancePercentile)
        .def("set_white_balance_saturation_threshold", &ImageProcCuda::setWhiteBalanceSaturationThreshold)
        .def("set_white_balance_temporal_consistency", &ImageProcCuda::setWhiteBalanceTemporalConsistency)
        .def("set_gamma_correction", &ImageProcCuda::setGammaCorrection)
        .def("set_gamma_correction_method", &ImageProcCuda::setGammaCorrectionMethod)
        .def("set_gamma_correction_k", &ImageProcCuda::setGammaCorrectionK)
        .def("set_vignetting_correction", &ImageProcCuda::setVignettingCorrection)
        .def("set_vignetting_correction_scale", &ImageProcCuda::setVignettingCorrectionScale)
        .def("set_vignetting_correction_a2", &ImageProcCuda::setVignettingCorrectionA2)
        .def("set_vignetting_correction_a4", &ImageProcCuda::setVignettingCorrectionA4)
        .def("set_clahe", &ImageProcCuda::setClahe)
        .def("set_clahe_limit", &ImageProcCuda::setClaheLimit)
        .def("set_clahe_grid_size", &ImageProcCuda::setClaheGridSize)
        .def("set_color_enhancer", &ImageProcCuda::setColorEnhancer)
        .def("set_color_enhancer_hue_gain", &ImageProcCuda::setColorEnhancerHueGain)
        .def("set_color_enhancer_saturation_gain", &ImageProcCuda::setColorEnhancerSaturationGain)
        .def("set_color_enhancer_value_gain", &ImageProcCuda::setColorEnhancerValueGain)
        .def("set_color_calibration", &ImageProcCuda::setColorCalibration)
        .def("set_color_calibration_matrix", &ImageProcCuda::setColorCalibrationMatrix)
        .def("set_undistortion", &ImageProcCuda::setUndistortion)
        .def("set_undistortion", &ImageProcCuda::setUndistortionImageSize)
        .def("set_undistortion_image_size", &ImageProcCuda::setUndistortionCameraMatrix)
        .def("set_undistortion_camera_matrix", &ImageProcCuda::setUndistortionDistortionCoefficients)
        .def("set_undistortion_distortion_coeffs", &ImageProcCuda::setUndistortionDistortionModel)
        .def("set_undistortion_distortion_model", &ImageProcCuda::setUndistortionRectificationMatrix)
        .def("set_undistortion_projection_matrix", &ImageProcCuda::setUndistortionProjectionMatrix)
        .def("set_dump_images", &ImageProcCuda::setDumpImages)
        .def("reset_white_balance_temporal_consistency", &ImageProcCuda::resetWhiteBalanceTemporalConsistency)
        .def("get_undistortion_enabled", &ImageProcCuda::getUndistortionEnabled)
        .def("get_image_height", &ImageProcCuda::getImageHeight)
        .def("get_image_width", &ImageProcCuda::getImageWidth)
        .def("get_distortion_model", &ImageProcCuda::getDistortionModel)
        .def("get_camera_matrix", &ImageProcCuda::getCameraMatrix)
        .def("get_distortion_coefficients", &ImageProcCuda::getDistortionCoefficients)
        .def("get_rectification_matrix", &ImageProcCuda::getRectificationMatrix)
        .def("get_projection_matrix", &ImageProcCuda::getProjectionMatrix)
        ;
}