#include <image_proc_cuda/modules/flip.hpp>

namespace image_proc_cuda {

FlipModule::FlipModule() : enabled_(true) {}

void FlipModule::enable(bool enabled) {
  enabled_ = enabled;
}

bool FlipModule::enabled() const {
  return enabled_;
}

//-----------------------------------------------------------------------------
// Wrapper methods (CPU)
//-----------------------------------------------------------------------------
void FlipModule::flip(cv::Mat& image) {
  cv::Mat out;
  cv::flip(image, out, -1);  // negative numbers flip x and y
  image = out;
}

//-----------------------------------------------------------------------------
// Wrapper methods (GPU)
//-----------------------------------------------------------------------------
#ifdef HAS_CUDA
void FlipModule::flip(cv::cuda::GpuMat& image) {
  cv::cuda::GpuMat out;
  cv::cuda::flip(image, out, -1);  // negative numbers flip x and y
  image = out;
}
#endif

}  // namespace image_proc_cuda