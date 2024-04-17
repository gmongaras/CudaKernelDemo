from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='kernel_go_brrrrr',
    ext_modules=[
        CUDAExtension('kernel_go_brrrrr.Dot1', [
            'Cuda/Dot1.cu',
        ]),
        CUDAExtension('kernel_go_brrrrr.Dot2', [
            'Cuda/Dot2.cu',
        ]),
        CUDAExtension('kernel_go_brrrrr.Dot3', [
            'Cuda/Dot3.cu',
        ]),
        CUDAExtension('kernel_go_brrrrr.Dot4', [
            'Cuda/Dot4.cu',
        ]),
        
        
        CUDAExtension('kernel_go_brrrrr.Matmul1', [
            'Cuda/Matmul1.cu',
        ]),
        CUDAExtension('kernel_go_brrrrr.Matmul2', [
            'Cuda/Matmul2.cu',
        ]),
        CUDAExtension('kernel_go_brrrrr.Matmul3', [
            'Cuda/Matmul3.cu',
        ]),
        CUDAExtension('kernel_go_brrrrr.Matmul4', [
            'Cuda/Matmul4.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })