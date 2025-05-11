from setuptools import setup, find_packages

setup(
    name="codegen",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Code generation and embedding utilities",
    python_requires=">=3.8",
) 