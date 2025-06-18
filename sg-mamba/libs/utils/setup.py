import torch
import os

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension


setup(
    name='nms_1d_cpu',
    ext_modules=[
        CppExtension(
            name = 'nms_1d_cpu',
            sources = ['./csrc/nms_cpu.cpp'],
            extra_compile_args=['-fopenmp'],
            include_dirs=[os.path.join(torch.utils.cpp_extension.include_paths()[0], 'torch')],
            extra_link_args=['-L' + os.path.join(torch.utils.cpp_extension.library_paths()[0]), '-ltorch', '-lc10']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
