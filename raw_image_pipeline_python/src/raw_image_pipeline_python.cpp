//
// Copyright (c) 2021-2023, ETH Zurich, Robotic Systems Lab, Matias Mattamala. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//

#include <cvnp/cvnp.h>
#include <pybind11/pybind11.h>
#include <raw_image_pipeline/raw_image_pipeline.hpp>

namespace py = pybind11;
using namespace raw_image_pipeline;

PYBIND11_MODULE(_py_raw_image_pipeline, m) {
  m.doc() = "Image Proc Cuda bindings";  // module docstring
  py::class_<RawImagePipeline>(m, "RawImagePipeline")
      .def(py::init<bool>(),  //
           py::arg("use_gpu"))
      .def(py::init<bool, const std::string&, const std::string&, const std::string&>(),  //
           py::arg("use_gpu") = false, py::arg("params_path") = "",
           py::arg("calibration_path") = "",        //
           py::arg("color_calibration_path") = "")
      .def("apply", &RawImagePipeline::apply)
      .def("process", &RawImagePipeline::process)
      .def("load_params", &RawImagePipeline::loadParams)
      .def("set_gpu", &RawImagePipeline::setGpu)
      .def("set_debug", &RawImagePipeline::setDebug)
      .def("set_debayer", &RawImagePipeline::setDebayer)
      .def("set_debayer_encoding", &RawImagePipeline::setDebayerEncoding)
      .def("set_flip", &RawImagePipeline::setFlip)
      .def("set_white_balance", &RawImagePipeline::setWhiteBalance)
      .def("set_white_balance_method", &RawImagePipeline::setWhiteBalanceMethod)
      .def("set_white_balance_percentile", &RawImagePipeline::setWhiteBalancePercentile)
      .def("set_white_balance_saturation_threshold", &RawImagePipeline::setWhiteBalanceSaturationThreshold)
      .def("set_white_balance_temporal_consistency", &RawImagePipeline::setWhiteBalanceTemporalConsistency)
      .def("set_gamma_correction", &RawImagePipeline::setGammaCorrection)
      .def("set_gamma_correction_method", &RawImagePipeline::setGammaCorrectionMethod)
      .def("set_gamma_correction_k", &RawImagePipeline::setGammaCorrectionK)
      .def("set_vignetting_correction", &RawImagePipeline::setVignettingCorrection)
      .def("set_vignetting_correction_parameters", &RawImagePipeline::setVignettingCorrectionParameters)
      .def("set_color_enhancer", &RawImagePipeline::setColorEnhancer)
      .def("set_color_enhancer_hue_gain", &RawImagePipeline::setColorEnhancerHueGain)
      .def("set_color_enhancer_saturation_gain", &RawImagePipeline::setColorEnhancerSaturationGain)
      .def("set_color_enhancer_value_gain", &RawImagePipeline::setColorEnhancerValueGain)
      .def("set_color_calibration", &RawImagePipeline::setColorCalibration)
      .def("set_color_calibration_matrix", &RawImagePipeline::setColorCalibrationMatrix)
      .def("set_color_calibration_bias", &RawImagePipeline::setColorCalibrationBias)
      .def("set_undistortion", &RawImagePipeline::setUndistortion)
      .def("set_undistortion_image_size", &RawImagePipeline::setUndistortionImageSize)
      .def("set_undistortion_new_image_size", &RawImagePipeline::setUndistortionNewImageSize)
      .def("set_undistortion_balance", &RawImagePipeline::setUndistortionBalance)
      .def("set_undistortion_fov_scale", &RawImagePipeline::setUndistortionFovScale)
      .def("set_undistortion_camera_matrix", &RawImagePipeline::setUndistortionDistortionCoefficients)
      .def("set_undistortion_distortion_coeffs", &RawImagePipeline::setUndistortionDistortionModel)
      .def("set_undistortion_distortion_model", &RawImagePipeline::setUndistortionRectificationMatrix)
      .def("set_undistortion_projection_matrix", &RawImagePipeline::setUndistortionProjectionMatrix)
      .def("get_dist_image_height", &RawImagePipeline::getDistImageHeight)
      .def("get_dist_image_height", &RawImagePipeline::getDistImageHeight)
      .def("get_dist_image_width", &RawImagePipeline::getDistImageWidth)
      .def("get_dist_distortion_model", &RawImagePipeline::getDistDistortionModel)
      .def("get_dist_camera_matrix", &RawImagePipeline::getDistCameraMatrix)
      .def("get_dist_distortion_coefficients", &RawImagePipeline::getDistDistortionCoefficients)
      .def("get_dist_rectification_matrix", &RawImagePipeline::getDistRectificationMatrix)
      .def("get_dist_projection_matrix", &RawImagePipeline::getDistProjectionMatrix)
      .def("get_rect_image_height", &RawImagePipeline::getRectImageHeight)
      .def("get_rect_image_height", &RawImagePipeline::getRectImageHeight)
      .def("get_rect_image_width", &RawImagePipeline::getRectImageWidth)
      .def("get_rect_distortion_model", &RawImagePipeline::getRectDistortionModel)
      .def("get_rect_camera_matrix", &RawImagePipeline::getRectCameraMatrix)
      .def("get_rect_distortion_coefficients", &RawImagePipeline::getRectDistortionCoefficients)
      .def("get_rect_rectification_matrix", &RawImagePipeline::getRectRectificationMatrix)
      .def("get_rect_projection_matrix", &RawImagePipeline::getRectProjectionMatrix)
      .def("reset_white_balance_temporal_consistency", &RawImagePipeline::resetWhiteBalanceTemporalConsistency);
}