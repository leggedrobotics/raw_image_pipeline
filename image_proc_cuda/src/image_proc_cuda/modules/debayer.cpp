#include <image_proc_cuda/modules/debayer.hpp>

namespace image_proc_cuda {

DebayerModule::DebayerModule() : enabled_(true) {}

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
// Wrapper methods (CPU)
//-----------------------------------------------------------------------------
void DebayerModule::debayer(cv::Mat& image) {
  cv::Mat out;
  // We only apply demosaicing (debayer) if the format is valid
  if (encoding_ == "bayer_bggr8") {
    cv::demosaicing(image, out, cv::COLOR_BayerBG2BGR_EA);
    image = out;
  } else if (encoding_ == "bayer_gbrg8") {
    cv::demosaicing(image, out, cv::COLOR_BayerGB2BGR_EA);
    image = out;
  } else if (encoding_ == "bayer_grbg8") {
    cv::demosaicing(image, out, cv::COLOR_BayerGR2BGR_EA);
    image = out;
  } else if (encoding_ == "bayer_rggb8") {
    cv::demosaicing(image, out, cv::COLOR_BayerRG2BGR_EA);
    image = out;
  }
  // We ignore non-bayer encodings
  else if (isBayerEncoding(encoding_)) {
    throw std::invalid_argument("Encoding [" + encoding_ + "] is a valid pattern but is not supported!");
  }
}

//-----------------------------------------------------------------------------
// Wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void DebayerModule::debayer(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat out;
  // We only apply demosaicing (debayer) if the format is valid
  if (encoding_ == "bayer_bggr8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerBG2BGR_MHT);
    image = out;
  } else if (encoding_ == "bayer_gbrg8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerGB2BGR_MHT);
    image = out;
  } else if (encoding_ == "bayer_grbg8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerGR2BGR_MHT);
    image = out;
  } else if (encoding_ == "bayer_rggb8") {
    cv::cuda::demosaicing(image, out, cv::cuda::COLOR_BayerRG2BGR_MHT);
    image = out;
  }
  // We ignore non-bayer encodings
  else if (isBayerEncoding(encoding_)) {
    throw std::invalid_argument("Encoding [" + encoding_ + "] is a valid pattern but is not supported!");
  }
}
#endif

//-----------------------------------------------------------------------------
// Apply method
//-----------------------------------------------------------------------------
template <typename T>
bool DebayerModule::apply(T& image) {
  if (!enabled_) {
    return false;
  }

  debayer(image);
  return true;
}
}  // namespace image_proc_cuda