#include <cvnp/cvnp.h>
#include <pybind11/pybind11.h>
#include <image_proc_cuda/image_proc_cuda.hpp>

namespace py = pybind11;
using namespace image_proc_cuda;

PYBIND11_MODULE(_py_image_proc_cuda, m) {
  m.doc() = "Image Proc Cuda bindings";  // module docstring
  py::class_<ImageProcCuda>(m, "ImageProcCuda")
      .def(py::init<bool>(),  //
           py::arg("use_gpu"))
      .def(py::init<bool, const std::string&, const std::string&, const std::string&>(),  //
           py::arg("use_gpu") = true, py::arg("params_path") = "",
           py::arg("calibration_path") = "",        //
           py::arg("color_calibration_path") = "")
      .def("apply", &ImageProcCuda::apply)
      .def("process", &ImageProcCuda::process)
      .def("load_params", &ImageProcCuda::loadParams)
      .def("set_gpu", &ImageProcCuda::setGpu)
      .def("set_debayer", &ImageProcCuda::setDebayer)
      .def("set_debayer_encoding", &ImageProcCuda::setDebayerEncoding)
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
      .def("set_vignetting_correction_parameters", &ImageProcCuda::setVignettingCorrectionParameters)
      .def("set_color_enhancer", &ImageProcCuda::setColorEnhancer)
      .def("set_color_enhancer_hue_gain", &ImageProcCuda::setColorEnhancerHueGain)
      .def("set_color_enhancer_saturation_gain", &ImageProcCuda::setColorEnhancerSaturationGain)
      .def("set_color_enhancer_value_gain", &ImageProcCuda::setColorEnhancerValueGain)
      .def("set_color_calibration", &ImageProcCuda::setColorCalibration)
      .def("set_color_calibration_matrix", &ImageProcCuda::setColorCalibrationMatrix)
      .def("set_undistortion", &ImageProcCuda::setUndistortion)
      .def("set_undistortion_image_size", &ImageProcCuda::setUndistortionImageSize)
      .def("set_undistortion_new_image_size", &ImageProcCuda::setUndistortionNewImageSize)
      .def("set_undistortion_balance", &ImageProcCuda::setUndistortionBalance)
      .def("set_undistortion_fov_scale", &ImageProcCuda::setUndistortionFovScale)
      .def("set_undistortion_camera_matrix", &ImageProcCuda::setUndistortionDistortionCoefficients)
      .def("set_undistortion_distortion_coeffs", &ImageProcCuda::setUndistortionDistortionModel)
      .def("set_undistortion_distortion_model", &ImageProcCuda::setUndistortionRectificationMatrix)
      .def("set_undistortion_projection_matrix", &ImageProcCuda::setUndistortionProjectionMatrix)
      .def("get_dist_image_height", &ImageProcCuda::getDistImageHeight)
      .def("get_dist_image_height", &ImageProcCuda::getDistImageHeight)
      .def("get_dist_image_width", &ImageProcCuda::getDistImageWidth)
      .def("get_dist_distortion_model", &ImageProcCuda::getDistDistortionModel)
      .def("get_dist_camera_matrix", &ImageProcCuda::getDistCameraMatrix)
      .def("get_dist_distortion_coefficients", &ImageProcCuda::getDistDistortionCoefficients)
      .def("get_dist_rectification_matrix", &ImageProcCuda::getDistRectificationMatrix)
      .def("get_dist_projection_matrix", &ImageProcCuda::getDistProjectionMatrix)
      .def("get_rect_image_height", &ImageProcCuda::getRectImageHeight)
      .def("get_rect_image_height", &ImageProcCuda::getRectImageHeight)
      .def("get_rect_image_width", &ImageProcCuda::getRectImageWidth)
      .def("get_rect_distortion_model", &ImageProcCuda::getRectDistortionModel)
      .def("get_rect_camera_matrix", &ImageProcCuda::getRectCameraMatrix)
      .def("get_rect_distortion_coefficients", &ImageProcCuda::getRectDistortionCoefficients)
      .def("get_rect_rectification_matrix", &ImageProcCuda::getRectRectificationMatrix)
      .def("get_rect_projection_matrix", &ImageProcCuda::getRectProjectionMatrix)
      .def("reset_white_balance_temporal_consistency", &ImageProcCuda::resetWhiteBalanceTemporalConsistency);
}