from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='iter_pixels_cuda',
    ext_modules=[
        CUDAExtension('iter_pixels_cuda', [
            'iter_pixels_cuda.cpp',
            'iter_pixels_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
