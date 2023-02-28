#include <image_proc_cuda/modules/gamma_correction.hpp>

namespace image_proc_cuda {

GammaCorrectionModule::GammaCorrectionModule() : enabled_(true) {}

void GammaCorrectionModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool GammaCorrectionModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
void GammaCorrectionModule::setMethod(const std::string& method) {
  method_ = method;
}

void GammaCorrectionModule::setK(const double& k) {
  k_ = k;
  is_forward_ = (k_ <= 1.0 ? true : false);
}

//-----------------------------------------------------------------------------
// White balance wrapper methods (CPU)
//-----------------------------------------------------------------------------
void GammaCorrectionModule::gammaCorrectCustom(cv::Mat& image) {
  cv::Mat dst;

  uchar LUT[256];
  image.copyTo(dst);
  for (int i = 0; i < 256; i++) {
    float f = i / 255.0;
    f = pow(f, k_);
    LUT[i] = cv::saturate_cast<uchar>(f * 255.0);
  }

  if (dst.channels() == 1) {
    cv::MatIterator_<uchar> it = dst.begin<uchar>();
    cv::MatIterator_<uchar> it_end = dst.end<uchar>();
    for (; it != it_end; ++it) {
      *it = LUT[(*it)];
    }
  } else {
    cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
    cv::MatIterator_<cv::Vec3b> it_end = dst.end<cv::Vec3b>();
    for (; it != it_end; ++it) {
      (*it)[0] = LUT[(*it)[0]];
      (*it)[1] = LUT[(*it)[1]];
      (*it)[2] = LUT[(*it)[2]];
    }
  }
  image = dst;
}

void GammaCorrectionModule::gammaCorrectDefault(cv::Mat& image) {
  gammaCorrectCustom(image);
}

//-----------------------------------------------------------------------------
// White balance wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void GammaCorrectionModule::gammaCorrectCustom(cv::cuda::GpuMat& image) {
  cv::Mat cpu_image;
  image.download(cpu_image);
  gammaCorrectCustom(cpu_image);
  image.upload(cpu_image);
}

void GammaCorrectionModule::gammaCorrectDefault(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat out;
  cv::cuda::gammaCorrection(image, out, is_forward_);
  image = out;
}
#endif

//-----------------------------------------------------------------------------
// Apply method
//-----------------------------------------------------------------------------
template <typename T>
bool GammaCorrectionModule::apply(T& image) {
  if (!enabled_) {
    return false;
  }

  if (method_ == "custom") {
    gammaCorrectCustom(image);

  } else {
    gammaCorrectDefault(image);
  }

  return true;
}
}  // namespace image_proc_cuda