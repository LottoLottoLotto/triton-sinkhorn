from setuptools import setup, find_packages

setup(
    name="mhc-fused",
    version="0.1.0",
    description="Memory-efficient Fused Sinkhorn Kernel for MHC/DeepSeek Architectures",
    author="Victor-1B Team",
    packages=find_packages(),
    install_requires=[
        "torch",
        "triton",
    ],
)
