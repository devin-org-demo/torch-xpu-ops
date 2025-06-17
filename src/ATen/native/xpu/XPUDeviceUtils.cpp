#include <chrono>
#include <thread>
#include <torch/library.h>

namespace at::native::xpu {

void _sleep(int64_t milliseconds) {
  std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

} // namespace at::native::xpu

TORCH_LIBRARY(xpu, m) {
  m.def("_sleep", &at::native::xpu::_sleep);
}
