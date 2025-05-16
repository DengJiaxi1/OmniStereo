import torch

# Our module!
import iter_pixels_cuda

class iter_pixels(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth_0, depth_1, conf, I, J):
        depth_output, conf_output, I_1, J_1 = iter_pixels_cuda.forward(depth_0, depth_1, conf, I, J)
        ctx.save_for_backward(I_1, J_1)
        return depth_output, conf_output

    @staticmethod
    def backward(ctx, depth_grad, conf_grad):
        I_1, J_1 = ctx.saved_tensors
        depth_output, conf_output = iter_pixels_cuda.backward(depth_grad, conf_grad, I_1, J_1)
        return None, depth_output, conf_output, None, None

