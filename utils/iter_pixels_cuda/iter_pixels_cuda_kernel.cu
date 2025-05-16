#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void iter_pixels_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depth_0,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depth_1,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> conf,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> I,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> J,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> I_1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> J_1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depth_out,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> conf_out) {
  //batch index
  const int b = blockIdx.z;
  // column index
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (h < depth_1.size(1) && w < depth_1.size(2)){
    if(depth_0[b][h][w]>0&&depth_1[b][h][w]<depth_out[b][I[b][h][w]][J[b][h][w]]){
      depth_out[b][I[b][h][w]][J[b][h][w]] = depth_1[b][h][w];
      conf_out[b][I[b][h][w]][J[b][h][w]] = conf[b][h][w];
      I_1[b][I[b][h][w]][J[b][h][w]] = h;
      J_1[b][I[b][h][w]][J[b][h][w]] = w;
    }
  }
}


template <typename scalar_t>
__global__ void iter_pixels_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depth_1_grad,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> conf_grad,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> I_1,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> J_1,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> depth_grad_out,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> conf_grad_out) {
  //batch index
  const int b = blockIdx.z;
  // column index
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  const int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (h < depth_1_grad.size(1) && w < depth_1_grad.size(2)){
    if(I_1[b][h][w]>=0){
        depth_grad_out[b][I_1[b][h][w]][J_1[b][h][w]] = depth_1_grad[b][h][w];
        conf_grad_out[b][I_1[b][h][w]][J_1[b][h][w]] = conf_grad[b][h][w];
    }
  }
}


std::vector<torch::Tensor> iter_pixels_cuda_forward(
    torch::Tensor depth_0,
    torch::Tensor depth_1,
    torch::Tensor conf,
    torch::Tensor I,
    torch::Tensor J) {

  const auto batch_size = depth_1.size(0);
  const auto height_size = depth_1.size(1);
  const auto width_size = depth_1.size(2);

  auto depth_out = torch::ones_like(depth_1)*100000;
  auto conf_out = torch::zeros_like(depth_1);
  auto I_1 = -torch::ones_like(depth_1);
  auto J_1 = -torch::ones_like(depth_1);

//   const int threads = 1024;
  const dim3 threads(32, 32);
  const dim3 blocks((width_size + threads.x - 1) / threads.x, (height_size + threads.y - 1) / threads.y, batch_size);

  AT_DISPATCH_FLOATING_TYPES(depth_1.type(), "iter_pixels_forward_cuda", ([&] {
    iter_pixels_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        depth_0.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        depth_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        conf.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        I.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        J.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        I_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        J_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        depth_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        conf_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  return {depth_out, conf_out, I_1, J_1};
}


std::vector<torch::Tensor> iter_pixels_cuda_backward(
    torch::Tensor depth_1_grad,
    torch::Tensor conf_grad,
    torch::Tensor I_1,
    torch::Tensor J_1) {

  const auto batch_size = depth_1_grad.size(0);
  const auto height_size = depth_1_grad.size(1);
  const auto width_size = depth_1_grad.size(2);

  auto depth_grad_out = torch::zeros_like(depth_1_grad);
  auto conf_grad_out = torch::zeros_like(depth_1_grad);

//   const int threads = 1024;
  const dim3 threads(32, 32);
  const dim3 blocks((width_size + threads.x - 1) / threads.x, (height_size + threads.y - 1) / threads.y, batch_size);

  AT_DISPATCH_FLOATING_TYPES(depth_1_grad.type(), "iter_pixels_backward_cuda", ([&] {
    iter_pixels_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        depth_1_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        conf_grad.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        I_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        J_1.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        depth_grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
        conf_grad_out.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
  }));

  return {depth_grad_out, conf_grad_out};
}
