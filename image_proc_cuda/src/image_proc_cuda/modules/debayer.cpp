#include <image_proc_cuda/modules/debayer.hpp>

namespace image_proc_cuda {

DebayerModule::DebayerModule(bool use_gpu) : enabled_(true), use_gpu_(use_gpu) {}

void DebayerModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool DebayerModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void DebayerModule::setEncoding(const std::string& encoding) {
  encoding_ = encoding;
}

//-----------------------------------------------------------------------------
// Helper methods
//-----------------------------------------------------------------------------
bool DebayerModule::isBayerEncoding(const std::string& encoding) const {
  // Find if encoding is in list of Bayer types
  return std::find(BAYER_TYPES.begin(), BAYER_TYPES.end(), encoding) != BAYER_TYPES.end();
}

//-----------------------------------------------------------------------------
// Wrapper methods (CPU)
//-----------------------------------------------------------------------------
void DebayerModule::debayer(cv::Mat& image, std::string& encoding) {
  cv::Mat out;
  // We only apply demosaicing (debayer) if the format is valid
  if (encoding == "bayer_bggr8") {
    cv::demosaicing(image, out, cv::COLOR_BayerBG2BGR);
    image = out;
  } else if (encoding == "bayer_gbrg8") {
    cv::demosaicing(image, out, cv::COLOR_BayerGB2BGR);
    image = out;
  } else if (encoding == "bayer_grbg8") {
    cv::demosaicing(image, out, cv::COLOR_BayerGR2BGR);
    image = out;
  } else if (encoding == "bayer_rggb8") {
    cv::demosaicing(image, out, cv::COLOR_BayerRG2BGR);
    image = out;
  }
  // We ignore non-bayer encodings
  else if (isBayerEncoding(encoding)) {
    throw std::invalid_argument("Encoding [" + encoding + "] is a valid pattern but is not supported!");
  }

  // Update encoding
  encoding = "bgr8";
  cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
}

//-----------------------------------------------------------------------------
// Wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void DebayerModule::debayer(cv::cuda::GpuMat& image, std::string& encoding) {
  cv::cuda::GpuMat out;
  // We only apply demosaicing (debayer) if the format is valid
  if (encoding == "bayer_bggr8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerBG2BGR_MHT);
    image = out;
  } else if (encoding == "bayer_gbrg8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerGB2BGR_MHT);
    image = out;
  } else if (encoding == "bayer_grbg8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerGR2BGR_MHT);
    image = out;
  } else if (encoding == "bayer_rggb8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerRG2BGR_MHT);
    image = out;
  }
  // We ignore non-bayer encodings
  else if (isBayerEncoding(encoding)) {
    throw std::invalid_argument("Encoding [" + encoding + "] is a valid pattern but is not supported!");
  }

  // Update encoding
  encoding = "bgr8";
}
#endif
}  // namespace image_proc_cuda