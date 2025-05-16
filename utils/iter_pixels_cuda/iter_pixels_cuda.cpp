#include <torch/extension.h>
#include <vector>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> iter_pixels_cuda_forward(
    torch::Tensor depth_0,
    torch::Tensor depth_1,
    torch::Tensor conf,
    torch::Tensor I,
    torch::Tensor J);


std::vector<torch::Tensor> iter_pixels_cuda_backward(
    torch::Tensor depth_1_grad,
    torch::Tensor conf_grad,
    torch::Tensor I_1,
    torch::Tensor J_1);


std::vector<torch::Tensor> iter_pixels_forward(
    torch::Tensor depth_0,
    torch::Tensor depth_1,
    torch::Tensor conf,
    torch::Tensor I,
    torch::Tensor J) {
  CHECK_INPUT(depth_0);
  CHECK_INPUT(depth_1);
  CHECK_INPUT(conf);
  CHECK_INPUT(I);
  CHECK_INPUT(J);

  return iter_pixels_cuda_forward(depth_0, depth_1, conf, I, J);
}


std::vector<torch::Tensor> iter_pixels_backward(
    torch::Tensor depth_1_grad,
    torch::Tensor conf_grad,
    torch::Tensor I_1,
    torch::Tensor J_1) {
  CHECK_INPUT(depth_1_grad);
  CHECK_INPUT(conf_grad);
  CHECK_INPUT(I_1);
  CHECK_INPUT(J_1);

  return iter_pixels_cuda_backward(depth_1_grad, conf_grad, I_1, J_1);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &iter_pixels_forward, "iter_pixels forward (CUDA)");
  m.def("backward", &iter_pixels_backward, "iter_pixels forward (CUDA)");
}
